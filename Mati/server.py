"""
The one process that owns the field.

    python3 -m Mati.server                    # start it
    python3 -m Mati.server --port 8756
    curl -N localhost:8756/chat -d '{"text":"hello"}'

WHY A SERVER. θ₀ is mutable state backed by a file. Two processes holding it
would write over each other, so exactly one may own it — and Mati is two apps
(a chat app for people, a monitoring/training terminal for pro users) that both
need it at once. That makes the owner a server and both apps its clients.

WHAT IT ACTUALLY BUYS. Not the reload: loading the field is 1.45s, about 2% of
a turn. Measured profile of one turn, no LLM call :

    import + load_field       1.45s    2.0%     <- all a persistent process saves
    address + recall          0.51s    0.7%
    imprint_text              9.50s   13.3%
    field_state               3.34s    4.7%
    consolidate_reply         0.49s    0.7%
    remember (episode)       27.48s   38.4%     <- after the reply
    reindex (15 notes)       28.86s   40.3%     <- after the reply, and growing
                             ------
                             71.64s

78.7% of a turn is consolidation the user has no reason to wait for. A
persistent process can move it off the critical path; a per-invocation script
cannot. That is the point of this file.

So a turn here streams its reply and hands the episode write to a worker.
Reindex does not run per turn at all — it re-addresses stored notes after θ₀
moves, costs ~1.9s per note and grows with the corpus, so the worker batches it
into idle time instead.

CONCURRENCY. One lock guards the field; every mutation serialises through it.
Monitoring does NOT take the lock — the turn publishes telemetry as it goes and
readers get the last snapshot, so watching Mati think never blocks it. The
worker releases the lock between reindex batches, so a message arriving during
a reindex waits for one batch (~8s) rather than the whole pass (~60s).

BINDING. 127.0.0.1 only, and not configurable to a public interface: "nothing
leaves your machine" is a property of the binding, not a promise in a README.
"""

import argparse
import inspect
import json
import os
import queue
import sqlite3
import sys
import numpy as np
import threading
import traceback
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .field import notes as N

from . import activity
from . import interruption
from . import notify
from . import bridge
from . import exclusive
from . import expression as expr
from . import inquiry
from . import predict
from . import schedule
from . import signal as sig
from . import study
from . import vault
from . import config as C
from .llm import LLMUnavailable, Refusal, make_llm
from . import orient
from . import reading
from . import speaker
from . import gatelog
from . import learning
from .ollin import SYSTEM_PROMPT, system_prompt, turn


def _takes_should_stop(fn):
    """Does this task accept a `should_stop` keyword?"""
    try:
        return "should_stop" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False



# ─────────────────────────────────────────────────────────────────────────────
# Telemetry — what the pro terminal watches
# ─────────────────────────────────────────────────────────────────────────────

# ⚠️ THE SERVER HAD BEEN RUNNING THE SAME CODE FOR ELEVEN AND A HALF DAYS.
# Found while tracing why episodes stopped recording what Mati said.
# Every filter, writer and gate on that path passed a direct offline test —
# because the code on disk was correct. It had simply never been loaded:
# `Mati.server` was started on Aug 16 and 12 commits had landed since,
# including the temperature change, the reading rewrite and the search fix.
#
# Two symptoms pointed at it and neither was legible on its own. Episodes
# lost their `I said:` line; and `gate_log.reply_words` was NULL on rows
# written days after that column was added — a value the running code could
# only have written as an integer, so the running code was not this code.
#
# The cost was not the stale process. It was ELEVEN DAYS OF CONCLUSIONS drawn
# against it: "I've let the terminal run all day, its recall never improved"
# was a true report about code from before the fixes. A long-lived process
# that silently serves old behaviour turns every subsequent measurement into
# a measurement of the past, and nothing in `/health` said so.
#
# Module mtimes rather than the git hash on purpose: it catches an
# uncommitted edit too, which is the state most of this work is in.

def _vault_state():
    """True / False / None — unlocked, locked, or no vault at all.

    ⚠️ NEVER PROMPTS. This is a status read on a health route that polls;
    asking for Touch ID here would put a fingerprint prompt on a timer.
    """
    try:
        from . import vault
        if not vault.is_set_up():
            return None
        return bool(vault.unlocked())
    except Exception:
        return None


def _code_stamp():
    """Newest mtime across the modules a running server would have imported."""
    import glob
    here = os.path.dirname(os.path.abspath(__file__))
    newest = 0.0
    # ⚠️ MATI'S OWN MODULES ONLY. This used to reach into the parent
    # directory for a sibling project's terminal script, which a running
    # Mati server never imports — so editing that project marked MATI's
    # code stale, in the one field
    # people check before trusting a behaviour report.
    # ⚠️ AND `field/` WAS MISSING while that was included: it IS imported.
    # The stamp under-covered Mati and over-covered a neighbour.
    for pat in (os.path.join(here, "*.py"),
                os.path.join(here, "field", "*.py")):
        for f in glob.glob(pat):
            try:
                newest = max(newest, os.path.getmtime(f))
            except OSError:
                pass
    return newest


_CODE_AT_START = _code_stamp()



# ⚠️ THESE BOUND *WAITING*, NEVER WORK. Reindexing to completion legitimately
# took 183s and must be allowed to finish — a half-packed install
# is worse than a slow one. What must never be unbounded is waiting for
# something that may never come. See `FieldService.pack`.
JOIN_TIMEOUT = 120.0     # the work queue should drain in seconds, not minutes
FIELD_TIMEOUT = 120.0    # if another owner holds the field, it is not letting go
RELEASE_TIMEOUT = 180.0  # the exit path. ⚠️ BOUNDED ON PURPOSE —
                         # unbounded, a process wedges mid-exit.


class FieldBusy(RuntimeError):
    """The field could not be taken in time.

    ⚠️ NOT a failure to save — it means somebody ELSE holds it, and the honest
    answer is to say so rather than wait forever. `exclusive.release()` frees
    the SERVER's claim; it does not take the field back from an admin terminal
    that has loaded it. That distinction cost a 900-second hang .
    """


class Telemetry:
    """Last-known state, plus a fan-out of events to live subscribers.

    Readers never touch the field, so the monitoring app can watch a turn
    without contending with it. The snapshot is whatever the turn last
    published, which is exactly what a gauge should show.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._snapshot = {"stage": "starting", "since": time.time()}
        self._subscribers = []

    def publish(self, **fields):
        # Derive the face from the stage unless the caller chose one. Done here
        # rather than at each call site because there are fifteen of them and
        # "a publish site nobody remembered to update" is precisely how the
        # correction moments went unpublished in the first place.
        if "stage" in fields and "expression" not in fields:
            fields["expression"] = expr.read(stage=fields["stage"])
        event = {"t": time.time(), **fields}
        with self._lock:
            self._snapshot.update(event)
            if "stage" in fields:
                self._snapshot["since"] = event["t"]
            dead = []
            for q in self._subscribers:
                try:
                    q.put_nowait(event)
                except queue.Full:
                    dead.append(q)          # a client that stopped reading
            for q in dead:
                self._subscribers.remove(q)

    def snapshot(self):
        with self._lock:
            return dict(self._snapshot)

    def subscribe(self):
        q = queue.Queue(maxsize=256)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q):
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)


# ─────────────────────────────────────────────────────────────────────────────
# The field, and the single lock that guards it
# ─────────────────────────────────────────────────────────────────────────────

class FieldService:
    def __init__(self, provider=None, verbose=True, llm=None):
        """`llm` overrides the provider — used by tests to keep a turn fast and
        deterministic. Everything else about the turn stays real."""
        self.verbose = verbose
        self.telemetry = Telemetry()
        self.started = time.time()

        self.llm = llm if llm is not None else make_llm(system_prompt(), provider)
        self.provider = provider or C.PROVIDER
        if hasattr(self.llm, "available"):
            ok, detail = self.llm.available()
            if not ok:
                raise LLMUnavailable(detail)
            self._llm_detail = detail
        else:
            self._llm_detail = self.provider

        # TWO clients on purpose. `self.llm` carries the bare system prompt and
        # is what the study/dream/inquiry worker uses. `self.chat_llm` carries
        # the system prompt PLUS the waking block, and is the only one that
        # talks to a person.
        #
        # They are separate because the whole saving depends on the chat prefix
        # never changing: Ollama re-prefills only what changed, at ~35 ms per
        # new token against ~0.1 ms per cached one. Giving the worker the same
        # client would not corrupt anything, it would just carry ~600 tokens it
        # has no use for on every chunk it reads. (The caches do not evict each
        # other — measured: a worker call between two chat turns left the chat
        # prefix still warm at 0.19 s.)
        # `chat_llm` is a PROPERTY, not an attribute — see below. Tests
        # reassign `service.llm` after construction, and an attribute captured
        # at init would silently keep talking to the client they replaced.
        self._oriented_llm = None
        # Tests inject a fake client to keep a turn fast and deterministic.
        # Rebuilding `chat_llm` from the real provider on wake would throw that
        # away and quietly put the suite back on a 3B — which is exactly what
        # happened: two voice checks started failing with a real model's words.
        self._llm_injected = llm is not None
        self.orientation = None
        self.slept_at = None
        # ⚠️ RESTING IS NOT LEAVING. The user: "/sleep should go into slow field
        # mode, not disconnect from terminal." Sleeping used to mean handing
        # the field over and letting go — which is what /quit is for. Resting
        # keeps the field, keeps answering, and simply does less: the idle loop
        # stops starting new work and only finishes what it owes.
        self.slow = False
        # Exchanges cut off mid-reply, per session. See interruption.py.
        self._pending = {}
        self.was_shutdown = False

        # Unlock BEFORE loading the field. If the memories are encrypted and
        # the key cannot be had, there is nothing useful to start — better to
        # say so than to come up and answer every question with "I don't
        # remember", which is what a locked vault looks like from outside.
        # ⚠️ CLAIM θ₀ BEFORE OPENING IT. The old check only asked whether a
        # SERVER was running, so a terminal holding the field with `/load` was
        # invisible here and both would have written it back. Refusing to start
        # is the right failure: the person can see the terminal they left open.
        if not exclusive.take("Mati server"):
            cur = exclusive.held_by() or {}
            raise SystemExit(
                f"\n  {cur.get('holder', 'Something')} (pid {cur.get('pid')}) "
                f"is holding the field.\n"
                f"  Close it first — two processes writing θ₀ overwrite each "
                f"other.\n")

        # ⚠️⚠️ A LOCKED VAULT MUST NOT STOP THE SERVER FROM STARTING, and this
        # used to. `vault.unlock()` RAISES on a passphrase-only vault — there
        # is no keychain key by design, so there is nothing to unlock
        # unprompted — and the exception came straight out of `__init__`. The
        # effect: on a Mac without Touch ID, setup offered "passphrase only",
        # completed successfully, and then the server crashed with a traceback
        # on every launch. A completed setup and a product that cannot run.
        #
        # ⭐ IT DEGRADES INSTEAD, which is what every other reader of sealed
        # data already does: `daylog.rows()` returns [], `presets._open()`
        # returns None, `charter_chunks()` returns []. Mati comes up, answers,
        # and cannot read its sealed memories until someone unlocks it — which
        # is a true thing it can say, and far better than refusing to boot.
        #
        # ⚠️ Found by actually running the install rather than by reading it.
        # `test_setup_run` covers the passphrase-only SETUP branch and passes;
        # nothing had ever started a SERVER against the vault it produces.
        self.locked_at_start = False
        if vault.is_set_up() and not vault.unlocked():
            self.telemetry.publish(stage="locked")
            try:
                vault.unlock(reason="unlock Mati's memories")
            except Exception as e:                           # noqa: BLE001
                self.locked_at_start = True
                if verbose:
                    print(f"  [mati] memories are locked ({e.__class__.__name__}"
                          f") — starting anyway; recall will be empty until "
                          f"`python3 -m Mati.vault --unlock`", file=sys.stderr)

        self.telemetry.publish(stage="loading field")
        self.lm, self.tok = bridge.load_field()

        # ⚠️ REENTRANT, AND THE NON-REENTRANT VERSION DEADLOCKED THE WORKER
        # TWICE. `_worker_loop` takes the field
        # around every queued task (line ~1448), and `_remember_action` — one
        # of those tasks, queued at line ~1082 after any media action — takes
        # it AGAIN. A plain Lock cannot be re-entered by the thread that holds
        # it, so the worker blocked forever: 0% CPU, `worker_alive: True`
        # because the thread existed, `task_done` never reached because it sat
        # before the `finally`, and the queue growing instead of draining.
        #
        # Reproduced minimally before changing anything: plain Lock never
        # returns, RLock completes.
        #
        # ⚠️ An RLock is right here rather than removing one of the two
        # acquisitions, because there are SIXTEEN `with self.field()` sites and
        # any future task that takes the field would hit the same wall. The
        # invariant "only one thread touches the field" is preserved — RLock
        # still excludes OTHER threads.
        #
        # ⚠️ RLock HAS NO `.locked()`. `/health`'s `busy` used it; see there.
        self._lock = threading.RLock()
        self._waiting = 0                  # threads queued for the field
        self._waiting_lock = threading.Lock()

        self.asleep = False
        self.sessions = {}
        self.last_episode = {}    # session -> note_id, so a
                                  # correction knows its target
        self.last_challenged = {} # session -> did Mati push back last turn?
                                  # A second correction after one means defer.
        self._sessions_lock = threading.Lock()

        self.work = queue.Queue()
        self.last_activity = time.time()
        # `last_activity` is moved by the worker too, so it cannot answer "how
        # long since anyone talked to me". This one is set ONLY by a real turn.
        self.last_turn = time.time()
        self.stats = {"turns": 0, "episodes": 0, "reindexed": 0,
                      "last_reindex": None}
        self._stop = threading.Event()
        # Worker liveness, reported by /health. It went unreported through two
        # separate thread deaths, and both were found only by
        # noticing a number that would not move.
        self.worker_beat = time.time()
        self.last_worker_error = None
        # Start ready to dream rather than an hour out: a fresh process that
        # has been idle since boot has nothing owed and may as well rest.
        self.last_dream = 0.0
        # When the person has asked for reading to be the priority. Bounded:
        # full speed starves dreaming and deepening, which is the whole reason
        # the budget exists — so this is a request, not a new default.
        # ⚠️ PERSISTED, because it used to be memory-only and a restart threw
        # it away in silence. Ask for eight hours of reading, restart Mati for
        # any reason, and the window is gone with nothing said — which is
        # exactly what happened. An overnight window is the case
        # most likely to span a restart and least likely to be watched.
        self.study_priority_until = self._load_focus()
        # Starts "already silent", so a fresh server does not wait 6h before
        # self-talk is allowed to resume behind a long queue.
        self._last_inquiry = 0.0
        # Text the turn handed over rather than imprinting while someone
        # waited. Drained on the SHORT idle tier, ahead of everything else —
        # the field must catch up with the conversation before it does
        # anything more speculative with it.
        self._to_imprint = []
        # Who each session says it is. NULL until told; the owner is assumed
        # only after they confirm, never before — see `speaker`.
        self._speaker = {}
        speaker.ensure_column()
        gatelog.ensure()
        learning.ensure_column()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True,
                                        name="mati-worker")
        # Separate from the worker on purpose: the worker spends minutes inside
        # a single reindex, and a heartbeat that shared its loop would stop for
        # exactly as long as the interesting part lasted.
        self._pulse = threading.Thread(target=self._pulse_loop, daemon=True,
                                       name="mati-pulse")
        # ⚠️ DO NOT ADD A SIGUSR1 faulthandler HERE. Tried to find a
        # stalled worker without sudo: it dumped two frames and then KILLED the
        # server mid-write. Dumping all threads from a signal handler while
        # numpy holds the GIL in C is not safe in this process. If a stack is
        # needed, use `sudo py-spy dump --pid`, which is out-of-process.
        self._worker.start()
        self._pulse.start()
        self.check_addresses("startup")
        # ⚠️ THE FIRST SEARCH OF A PROCESS COSTS ~15s AND EVERY ONE AFTER IT
        # COSTS 0.06s. `bridge.search` spawns openssl per body, so the first
        # call decrypts the whole store; the cache makes the rest free.
        # : 14.94s cold, 0.06s warm. Paying it here means
        # the person never does — and after a restart they were paying it in
        # their first sentence.
        try:
            t_warm = time.time()
            bridge.search("warm the body cache", k=1)
            # ⚠️ ONE SEGMENT, NOT THE YEAR. The newest
            # `notes.VERBATIM_SEGMENT_DAYS` of exact words, bounded by
            # `warm_verbatim`'s budget to ~3s, so Mati can be asked about
            # what was literally said this month from its first sentence. The
            # rest of the year arrives a segment per idle pass.
            #
            # ⚠️ AND NOT THE WHOLE YEAR HERE, DELIBERATELY. An earlier cut had
            # `search` decrypt every verbatim inside the year on demand; a
            # cold search measured 73.0s against 0.36s warm, because in a
            # ten-week-old store "inside the year" is the entire store. The
            # year bounds the archive; it does not make this year cheap. That
            # is what the user's "don't decrypt everything at once" rules out.
            from . import notes as _nwarm
            _nwarm.warm_verbatim(segment=0)
            if self.verbose:
                print(f"  [mati] memory cache warm "
                      f"({time.time()-t_warm:.0f}s)", file=sys.stderr)
        except Exception as e:
            if self.verbose:
                print(f"  [mati] cache warm skipped: {e}", file=sys.stderr)
        # ⚠️ STARTING IS A KIND OF WAKING, and it did not used to be. The
        # server comes up with `asleep = False`, so it never passed through
        # `wake()` — the one place that builds the orientation. A freshly
        # started Mati therefore had `orientation = None`, `turn(oriented=
        # False)`, and every turn paid the full ~70s of prefill this change
        # exists to remove. It looked healthy from outside; it was simply slow
        # again, which is the hardest kind of regression to notice.
        #
        # `slept_at` is None here on purpose: a cold start genuinely does not
        # know how long it was down, and the block says so rather than
        # inventing a duration.
        self.build_orientation()
        self.telemetry.publish(stage="idle", ready=True)

    # ── field lock ───────────────────────────────────────────────────────────

    class _Held:
        def __init__(self, svc, timeout=None):
            self.svc = svc
            # ⚠️ None means wait forever, which is right for ordinary work.
            # Only the shutdown path passes a bound — see `pack`.
            self.timeout = timeout

        def __enter__(self):
            # `_waiting` counts threads BLOCKED on the field, never the one
            # holding it. Decrementing on exit instead of on acquire made a
            # holder count itself, so `contended` was True exactly while the
            # worker was working — and every `should_stop=lambda: self.contended`
            # fired on its first check. Study silently did nothing for hours;
            # reindex did the same once it was given the same predicate.
            with self.svc._waiting_lock:
                self.svc._waiting += 1
            got = (self.svc._lock.acquire() if self.timeout is None
                   else self.svc._lock.acquire(timeout=self.timeout))
            with self.svc._waiting_lock:
                self.svc._waiting -= 1      # no longer waiting, either way
            if not got:
                # ⚠️ RAISED, NOT RETURNED. A caller that got no field must not
                # run its body against `self.lm` and then claim it succeeded.
                # ⚠️ REPORTS WHAT IT OBSERVED, AND GUESSES NOTHING. The
                # first version said "most likely an admin terminal with the
                # field loaded". There was once no terminal — the
                # log showed `POST /study_now`, so Mati's OWN study task was
                # holding it. A guessed cause stated confidently in an error
                # message sends the reader after the wrong thing, which is
                # exactly what it did here.
                raise FieldBusy(
                    f"could not take the field within {self.timeout}s — "
                    f"another holder has it. Check the log for the last "
                    f"request before this, and `worker_alive` / "
                    f"`holds_field` in /health; an admin terminal with the "
                    f"field loaded and a long-running study or imprint are "
                    f"both possible")
            return self.svc

        def __exit__(self, *exc):
            self.svc._lock.release()
            return False

    def field(self, timeout=None):
        """`with svc.field():` — exclusive access to lm/tok.

        ⚠️ `timeout` raises `FieldBusy` instead of waiting forever. Only the
        shutdown path uses it; ordinary work should wait its turn.
        """
        return self._Held(self, timeout=timeout)

    @property
    def contended(self):
        """Is anyone else waiting? The worker yields when this is true."""
        with self._waiting_lock:
            return self._waiting > 0

    # ── sleep / wake ─────────────────────────────────────────────────────────
    #
    # The pro terminal is an EXCLUSIVE tool, not a second client: it loads the
    # ── the focus window outlives the process ────────────────────────────────

    def _focus_path(self):
        return os.path.join(os.path.dirname(C.NOTES_DB), "study_focus.json")

    def _save_focus(self):
        try:
            with open(self._focus_path(), "w") as fh:
                json.dump({"until": self.study_priority_until}, fh)
        except OSError:
            pass                       # a lost window is not worth failing on

    def _load_focus(self):
        """Resume a window that outlived the process. Expired ones are 0."""
        try:
            with open(self._focus_path()) as fh:
                until = float(json.load(fh).get("until") or 0)
        except (OSError, ValueError, TypeError):
            return 0.0
        if until <= time.time():
            return 0.0
        if self.verbose:
            print(f"  [mati] resuming reading focus, "
                  f"{(until - time.time())/60:.0f} min left", file=sys.stderr)
        return until

    @property
    def chat_llm(self):
        """The client that talks to a PERSON: system prompt + waking block.

        Falls back to the plain `llm` until a waking block exists, and whenever
        one was injected — that is what keeps the suite deterministic and what
        lets a test swap `service.llm` mid-run and have chat follow it.
        """
        return self._oriented_llm or self.llm

    # field itself, trains on it and writes θ₀ back. Two owners silently
    # overwrite each other, so it may only run when Mati is shut down — or
    # asleep, which is what this pair of calls means. Asleep, Mati has
    # finished its queued work, persisted θ₀ and let go of the field; the
    # process stays up so it can be woken, which costs 1.45s.

    def consolidate(self):
        """The day-boundary work: tomorrow's prediction, and the day's rapport.

        ⚠️ THIS USED TO LIVE INSIDE THE HANDOFF, WHICH IS WHY IT COST THE FIELD.
        the user: *"It is supposed to go to sleep again tonight, and I don't want
        it to disconnect — I want it to create the prediction and then dream."*
        Consolidating and letting go were the same function, so the only way to
        get a prediction was to give up the field. They are separate now:
        `release()` still consolidates on its way out, and `rest()` consolidates
        and stays.
        """
        # ⚠️ RAPPORT IS BUILT HERE, NOT DECIDED IN THE MOMENT.
        # The user: "the rapport isn't something that happens in the moment, it
        # should be part of the prediction." The day's settled exchanges are
        # learned from the record while nobody is waiting, and the turn path
        # only ever looks them up.
        #
        # Idempotent and day-stamped: called again on the same day it does
        # nothing, because "if something happens against the prediction, the
        # prediction is held until the end of the day — it may just be
        # delayed."
        # ⚠️ CONSOLIDATION HAPPENS HERE, AND ONLY HERE. The user: the prediction
        # "is produced while humans sleep". Moving this to the dream meant they
        # put it back — sleep is the boundary of a day, and a prediction OF THE
        # DAY has to be made at one.
        #
        # ⚠️ `_maybe_predict` had NO CALLER anywhere until now, which is the
        # real reason the record held zero predictions after 26 days. I had
        # blamed the evidence gates.
        #
        # Nothing here runs on a turn or on idle. Noticing happens through the
        # day; deciding what tomorrow is about happens at the end of one.
        try:
            # ⚠️ THINKING ABOUT THE DAY COMES BEFORE PREDICTING IT. The user:
            # "Mati should talk to itself about its day, and let the field
            # bring up memories based on what occurred… humans spend half
            # their waking day recomputing everything they experienced in
            # relation to themselves and everyone around them, to make
            # predictions faster and more likely." The reflection is the
            # recomputing; the prediction reads what it turned up. Ordered,
            # not merely both present.
            self._maybe_reflect()
            self._maybe_predict()
            from . import rapport
            n = rapport.build(verbose=self.verbose)
            if n and self.verbose:
                print(f"  [rapport] settled {n} exchanges for the day",
                      file=sys.stderr)
            return True
        except Exception as e:
            self.telemetry.publish(stage="asleep", rapport_error=repr(e))
            return False

    def rest(self):
        """Sleep for the night: consolidate, then dream, KEEPING THE FIELD.

        ⚠️ THIS IS WHAT A QUIET NIGHT DOES NOW, AND IT IS NOT `release()`.
        Nothing here hands the field over, drops `self.lm` or touches the
        exclusive lock, so the terminal stays connected through the night and
        Mati can still be spoken to — it simply stops starting new work.

        Dreaming continues on purpose: `_maybe_dream` is not gated on `slow`,
        so the idle loop goes on turning over what is held once an hour. That
        is the "and then dream" half of the request, and it is the half that
        the old auto-sleep made impossible — a released field cannot dream,
        because there is no field in memory to dream with.

        `self.slow` is its own idempotence guard: the idle pass only calls this
        when not already slow.
        """
        self.slow = True                      # keep the field, do less
        self.consolidate()
        # ⚠️ THE DAY'S LEARNING BECOMES THE ADDRESS SPACE HERE, AND NOWHERE
        # ELSE. Through the day Mati addresses against a frozen θ₀ and
        # imprints into the live one, so what it has learned today is not yet
        # in the space it recalls through. Promoting at night is what lets the
        # day in; it is deliberately once-a-night because promotion is the only
        # thing that makes an address stale, and the reindex that repairs them
        # costs ~3.8s a note.
        #
        # ⚠️ AFTER `consolidate()`, NEVER BEFORE — the queue still holds
        # episodes to imprint, and promoting first would publish a space that
        # is missing the last of the day and then mark nothing stale for it.
        #
        # ⚠️ AND THE REINDEX GOES WITH IT, IN THE SAME BREATH. The first cut of
        # this promoted and left the repair to the idle worker, which is
        # SWITCHED OFF (`C.REINDEX_ON_IDLE` defaults to 0, retired by
        # measurement). Promotion is the one thing that stales every address,
        # so promoting with nothing to repair it would have invalidated the
        # whole working set every night and never fixed it — strictly worse
        # than not promoting at all, and silent. `promote()`'s own docstring
        # says "⚠️ REINDEX AFTER THIS"; it has to be the code that says it too.
        #
        # ⚠️ THIS IS NOT THE LOOP THAT WAS RETIRED. That one queued a batch of
        # 4 whenever anything was stale, re-checked against a live fingerprint
        # that had moved again, and burned 10.5 hours. This is ONE bounded pass
        # after ONE deliberate invalidation, against a fingerprint that holds
        # still until the next `rest()` — so it converges by construction.
        # It runs on the worker and yields to an arriving turn.
        # ⚠️ UNCONDITIONAL. This used to be skipped under
        # `MATI_FROZEN_ADDRESSING=0`, which pinned no space and so had
        # nothing to publish. That flag is gone — see `bridge.pin_address_space`
        # for why a two-mode memory architecture cannot ship.
        try:
            with self.field():
                bridge.promote(self.lm)
                promoted = {"promoted": True,
                            "address_gs": bridge.address_gs(self.lm)}
            self.work.put(("reindex after promote",
                           self._reindex_after_promote))
        except Exception as e:                # never let this block the night
            promoted = {"promoted": False, "error": repr(e)}
            if self.verbose:
                print(f"  [mati] promote failed: {e!r}", file=sys.stderr)
        activity.log(activity.SLEEP, "night — field kept")
        self.telemetry.publish(stage="resting", slow=True, **promoted)
        if self.verbose:
            print("  [mati] resting for the night — field KEPT, dreaming on",
                  file=sys.stderr)
        return {"slow": True, "asleep": False, "holds_field": True, **promoted}

    def _verbatim_warm_field(self):
        """`"warm/total"` for the RAG-ready year, or None if unavailable.

        ⚠️ A STRING, NOT A RATIO. "1829/2362" says both how far warming got
        and how much there is; a percentage hides a growing denominator,
        which is how `169/202 stale` became a figure that could never reach
        zero. Never raises — `/health` must answer even with a locked vault.
        """
        try:
            from . import notes as _n
            warm, total = _n.verbatim_warmth()
            out = f"{warm}/{total}"
            # ⚠️ SAY WHEN THE FLOOR COULD NOT BE HELD. The user's rule is that a
            # year is the floor and capacity the ceiling; if a year of exact
            # words does not fit in this machine's budget, that is a fact
            # about the machine and the owner should see it rather than
            # wonder why old conversations only come back as summaries.
            if _n.year_overflowed:
                out += " ⚠️ year does not fit in budget"
            return out
        except Exception:
            return None

    def _verbatim_budget_field(self):
        """`"held/budget MB"` for the plaintext cache. Never raises."""
        try:
            from . import notes as _n
            return (f"{_n._cache_bytes() / 1048576:.1f}/"
                    f"{_n.cache_budget_bytes() / 1048576:.0f} MB")
        except Exception:
            return None

    def _maybe_warm_verbatim(self):
        """Make one older segment's exact words searchable. True if it worked.

        Walks outward from the hot window, newest segment first. ⚠️ Stops at
        the first segment that is already warm rather than scanning to the end
        of history every pass — `_verbatim_segment` remembers where it got to,
        and a segment that reports nothing left advances it.

        ⚠️ GIVES UP QUIETLY ON A LOCKED VAULT. Warming needs the key; without
        it every row raises, and a background job must not turn that into a
        log full of failures for something nobody asked for.
        """
        seg = getattr(self, "_verbatim_segment", 0)
        # ⚠️ The edge of the RAG-ready year is checked below, once `notes`
        # is imported — past it, warming would walk the whole archive one
        # SELECT per idle pass forever, and older material is what the
        # two-tier design says to find by gist instead.
        try:
            from . import notes as _n
            warm, total = _n.verbatim_warmth()
            edge = _n.warming_edge()
            # ⚠️ RE-WARM WHEN THE CACHE HAS EVICTED. `notes._evict` gives up
            # the oldest segments when the machine's budget is reached, and
            # without this the counter would already be past the edge and
            # warming would never resume — `verbatim_warm` would sit short of
            # total and stay there, which reads as "working".
            if total and warm < total and seg >= edge:
                seg = self._verbatim_segment = 0
            if seg >= edge:
                return False
            done, left = _n.warm_verbatim(segment=seg)
        except Exception:
            return False
        if left:
            return done > 0            # same segment again next pass
        self._verbatim_segment = seg + 1
        if done and self.verbose:
            warm, tot = _n.verbatim_warmth()
            print(f"  [mati] warmed {done} verbatims, segment {seg} "
                  f"(days {seg * _n.VERBATIM_SEGMENT_DAYS:.0f}–"
                  f"{(seg + 1) * _n.VERBATIM_SEGMENT_DAYS:.0f}) — "
                  f"{warm}/{tot} of the year searchable by exact words",
                  file=sys.stderr)
        return done > 0

    def _reindex_after_promote(self):
        """Re-address the working set into the space `rest()` just promoted.

        ⚠️ A LOOP, BECAUSE ONE PASS IS NOT ENOUGH. `bridge.reindex` yields to
        an arriving turn between notes and commits what it finished, so a
        single call can return with a remainder — and with
        `C.REINDEX_ON_IDLE` off there is nothing else that would ever pick it
        up. Every address in the store would stay filed in last night's space,
        quietly, with the log reporting a promotion that happened.

        It terminates for the reason the retired idle loop could not: the
        fingerprint it is repairing toward holds still until the next
        `rest()`, so each pass strictly reduces what is owed. The cap is
        belt-and-braces against a row that cannot be addressed at all.
        """
        total, passes = 0, 0
        while not self._stop.is_set() and passes < 50:
            passes += 1
            with self.field():
                if self.asleep or self.lm is None:
                    break                   # the field was handed over
                n = bridge.reindex(self.lm, self.tok, limit=0,
                                   should_stop=lambda: self.contended)
            total += n
            self.worker_beat = time.time()
            self.telemetry.publish(stage="worker: reindex after promote",
                                   reindexed=total)
            if not n:
                break                       # nothing stale left
        if total:
            with self.field():
                # ⚠️ RE-CHECK UNDER THE LOCK. `/release` can set `self.lm` to
                # None between the loop and this save; it raised
                # AttributeError on None.mf once already, in `_maybe_reindex`.
                if not self.asleep and self.lm is not None:
                    bridge.save_notes_field(self.lm)
            self.stats["reindexed"] = self.stats.get("reindexed", 0) + total
            self.stats["last_reindex"] = time.time()
        if self.verbose:
            print(f"  [mati] re-addressed {total} notes into the promoted "
                  f"space in {passes} pass(es)", file=sys.stderr)
        return {"reindexed": total, "passes": passes}

    def release(self):
        """Finish up, persist, and LET GO OF THE FIELD. Idempotent.

        ⚠️ RENAMED FROM `sleep()` AT THE USER'S INSTRUCTION, because the old name
        was the whole misunderstanding. "Sleep" is what a person does nightly
        and expects to keep their mind through; this drops `self.lm` and hands
        θ₀ to whoever takes the exclusive lock next. Nightly rest is `rest()`.
        This is for an explicit handoff — the pro terminal taking the field —
        and for process exit. Nothing calls it on a timer any more.
        """
        if self.asleep:
            return {"asleep": True, "already": True}
        self.asleep = True                    # refuse new turns from here on
        # Releasing IS handing the field over, so the lock goes with it.
        exclusive.release()
        self.consolidate()
        self.work.join()                      # let queued episodes finish
        with self.field():
            bridge.save_notes_field(self.lm)
            self.lm, self.tok = None, None
        self.slept_at = time.time()
        activity.log(activity.SLEEP, "released — field handed over")
        self.telemetry.publish(stage="asleep", asleep=True)
        if self.verbose:
            print("  [mati] released — the field is free", file=sys.stderr)
        return {"asleep": True}

    # ⚠️ The old name, kept ONLY so a terminal running an older build does not
    # break mid-session. It is the handoff, which is no longer what "sleep"
    # means anywhere else in this file.
    def sleep(self):
        return self.release()

    def pack(self):
        """Make this install carryable. ⚠️ DOES NOT STOP THE PROCESS.

        ⚠️ RENAMED FROM `shut_down` , because the old name was a
        lie and the lie cost a 900-second hang. The user: *"I always thought
        shutdown meant the field, because quit is for the terminal."* That is
        a fair reading of the word, and the terminal's own help text said
        "/shutdown STOPS Mati for real" — which it never did. It packs.

        The four verbs, now each meaning one thing:

            rest()      nightly. KEEPS the field, still dreams, still answers.
            release()   hands the field over. Process keeps running.
            pack()      makes the directory copyable. Process keeps running.
            stop()      packs, then actually STOPS the process.

        Stop for real, and leave something that can be carried away.

        ⚠️ **THIS IS NOT `sleep()`, AND THE DIFFERENCE MATTERS.** Sleep drains,
        snapshots and lets go of the field so the pro terminal can take it —
        the process stays up and is expected to wake. Historically `sleep` was
        the only stop there was, which is why this reads as a new command: what
        was needed was a stop that leaves a **portable install**, because Mati
        is a few megabytes that can move to another machine on a drive.

        Two things sleep does not do, both of which produce a copy that looks
        fine and is not:

          - **A full reindex.** Addresses go stale faster than the idle worker
            repairs them (measured: 22 of 23 after ~9 turns), so an install
            handed over mid-session carries recall that has quietly collapsed
            toward recency. Unbounded here; a transfer is worth the seconds.
          - **A WAL checkpoint.** Committed rows can sit in
            `mati_notes.db-wal`. Dragging the `.db` alone to a drive loses
            them silently at both ends.
        """
        from . import transfer
        out = {"packed": True, "shutdown": True}
        exclusive.release()          # stopping for real; the field is free
        if not self.asleep:
            # ⚠️ THE WAITS ARE BOUNDED; THE WORK IS NOT. Reindexing to
            # completion legitimately took 183s and must be
            # allowed to finish — a half-packed install is worse than a slow
            # one. What must never be unbounded is WAITING for something that
            # may never come: a queue that cannot drain, or a field somebody
            # else is holding. A 900s hang was the second.
            if not self._join_work(JOIN_TIMEOUT):
                out["owed_work"] = self.work.unfinished_tasks
                out["warning"] = (
                    f"the work queue did not drain in {JOIN_TIMEOUT}s; "
                    f"{self.work.unfinished_tasks} task(s) still owed")
                print(f"  [pack] {out['warning']}", file=sys.stderr)
            try:
                with self.field(timeout=FIELD_TIMEOUT):
                    out["transfer"] = transfer.prepare(self.lm, self.tok,
                                                       verbose=self.verbose)
                    bridge.save_notes_field(self.lm)
                    self.lm, self.tok = None, None
                self.asleep = True
            except FieldBusy as e:
                # ⚠️ NOT PORTABLE, AND IT SAYS SO. θ₀ was never saved, so the
                # folder must not be advertised as copyable. Returning is
                # still right: the caller gets an answer and a reason instead
                # of a process that never replies.
                out["packed"] = False
                out["portable"] = False
                out["error"] = str(e)
                out["hint"] = ("find the holder before acting: an admin "
                               "terminal (close it, or `/revert` there first "
                               "if it imprinted), or a study/imprint task "
                               "still running — the log's last request says "
                               "which")
                print(f"  [pack] {e}", file=sys.stderr)
                self.telemetry.publish(stage="idle", pack_failed=str(e))
                return out
        else:
            # Already let go of the field, so reindexing is impossible — but
            # the checkpoint is the part that loses data and still applies.
            out["transfer"] = transfer.prepare(verbose=self.verbose)
        ok, findings = transfer.check()
        out["portable"] = ok
        out["findings"] = [{"level": l, "note": n} for l, n in findings]
        # A stage of its own, NOT "asleep" — asleep is a rest Mati returns
        # from and the avatar shows it resting underwater. Shutdown is Mati
        # not running, and the avatar shows nothing. Publishing "asleep" here
        # would draw a sleeping Axolotl for a process that has gone.
        self.telemetry.publish(stage="shutdown", asleep=True, shutdown=True,
                               portable=ok)
        return out

    def stop(self):
        """Pack, then STOP THE PROCESS. The verb that was missing.

        ⚠️ Nothing stopped the server before this existed. `/shutdown` packed
        and kept serving, so every restart needed a manual `kill` — twice in
        one day , and the second time only after a 900s hang.

        ⚠️ THE STOP IS DEFERRED TO ANOTHER THREAD ON PURPOSE. `httpd.shutdown()`
        blocks until `serve_forever` returns, and calling it from inside a
        request handler would wait on a loop that is waiting on this handler.
        So the response goes out first and the stop lands a moment later.
        """
        out = self.pack()
        httpd = getattr(self, "httpd", None)
        if httpd is None:
            out["stopped"] = False
            out["why"] = ("no server attached — packed only; stop the process "
                          "yourself")
            return out

        def _stop_after_reply():
            time.sleep(0.5)          # let the response flush to the client
            try:
                httpd.shutdown()     # breaks serve_forever; `finally` cleans up
            except Exception:                                # noqa: BLE001
                pass

        threading.Thread(target=_stop_after_reply, daemon=True).start()
        out["stopped"] = True
        return out

    def restart(self):
        """Stop, and bring a fresh one up behind it. (ok, detail).

        ⚠️ THE SUCCESSOR IS LAUNCHED BEFORE THE STOP, DELIBERATELY. If it is
        spawned after, a failure to launch leaves the person with no server
        and no message — the one outcome a restart button must never produce.
        Launching first means a failure is reported while the current server
        is still answering, and nothing is lost.

        ⚠️ AND THE SUCCESSOR WAITS FOR THE PORT. Two servers cannot hold the
        field at once, so it polls until this process has actually let go
        rather than racing it and dying on the exclusive lock.

        ⚠️ `start_new_session=True` DETACHES IT. Without it the child is in
        this process group and dies with the parent — which is a restart that
        only stops.
        """
        import subprocess as _sp
        # ⚠️ NEITHER THE PORT NOR THE PID IS A RELIABLE SIGNAL, and both were tried
        # both. Waiting on the PORT launches too early: the socket closes at
        # `httpd.shutdown()` while the FIELD is released later, in the cleanup
        # after it, so the successor is refused with "Mati server is holding
        # the field" and exits — leaving nothing running. Waiting on the PID
        # is worse: a dead process stays visible to `os.kill(pid, 0)` as a
        # zombie until it is reaped, so the poll can hang on a process that
        # has already exited.
        #
        # ⚠️ SO IT RETRIES INSTEAD OF PREDICTING. The successor tries to
        # start, checks whether it is answering, and tries again if not. That
        # needs no model of the shutdown's timing — which is the part I kept
        # getting wrong — and it recovers from a slow release rather than
        # dying on one.
        # ⚠️ `SERVER_PORT`, NOT `PORT`. `C.PORT` DOES NOT EXIST, so this read
        # the 8756 fallback on every machine and the successor-waiter polled
        # the wrong socket whenever MATI_PORT was set. On the default port it
        # worked by coincidence — the fallback happened to be the right
        # number. On any other port the waiter never waited for the old server
        # to release the field, so the first successor died on the exclusive
        # lock, and it never saw a healthy successor come up either.
        # Found while testing restart on a SPARE port, which is the one
        # configuration the bug was invisible in before.
        port = C.SERVER_PORT
        waiter = (
            "import socket,subprocess,sys,time\n"
            "def up():\n"
            "    s=socket.socket(); s.settimeout(1)\n"
            "    try: s.connect(('127.0.0.1',%d)); return True\n"
            "    except OSError: return False\n"
            "    finally: s.close()\n"
            "for _ in range(120):\n"          # wait out the old one
            "    if not up(): break\n"
            "    time.sleep(0.5)\n"
            "for attempt in range(10):\n"      # then try until it sticks
            "    p=subprocess.Popen([sys.executable,'-m','Mati.server'],cwd=%r)\n"
            "    for _ in range(40):\n"
            "        time.sleep(0.5)\n"
            "        if up(): raise SystemExit(0)\n"
            "        if p.poll() is not None: break\n"
            "    else:\n"
            "        raise SystemExit(0)\n"
            "    time.sleep(2.0)\n"
            % (port, C.ROOT))
        try:
            _sp.Popen([sys.executable, "-c", waiter], cwd=C.ROOT,
                      stdout=_sp.DEVNULL, stderr=_sp.DEVNULL,
                      start_new_session=True)
        except Exception as e:                               # noqa: BLE001
            # ⚠️ NOTHING IS STOPPED. The current server keeps serving.
            return {"restarting": False,
                    "detail": f"could not launch a successor ({e}); "
                              "nothing was stopped"}
        out = self.stop()
        out["restarting"] = True
        out["detail"] = ("stopping now; a fresh one will take about two "
                         "minutes to wake")
        # ⚠️ SAY IF THE SUCCESSOR MAY COME BACK UNABLE TO READ ANYTHING.
        # Bodies are sealed and the successor starts with nobody present, so
        # it cannot answer a Touch ID prompt. If the keychain does not release
        # the key unprompted it will run, look healthy on every other
        # measure, and hold no memories — and the person asked for a restart,
        # not for amnesia. : a background start died with
        # "Touch ID declined or unavailable".
        try:
            from . import vault
            if vault.is_set_up() and not vault.adopted():
                out["warning"] = (
                    "this Mac's keychain does not hold the key, so the new "
                    "server will not be able to read your memories until you "
                    "unlock it")
        except Exception:
            pass
        return out

    def _join_work(self, timeout):
        """Wait for the work queue to drain, but not forever. True if it did.

        ⚠️ `Queue.join()` TAKES NO TIMEOUT, which is why a wedged queue could
        hang a shutdown indefinitely. Waiting on it in a throwaway daemon
        thread is the standard way to bound it: if the join never returns the
        thread is simply abandoned, and the caller reports what is still owed
        instead of blocking the process forever.
        """
        done = threading.Event()

        def _wait():
            try:
                self.work.join()
            finally:
                done.set()

        threading.Thread(target=_wait, daemon=True).start()
        return done.wait(timeout)

    def check_addresses(self, why=""):
        """Are the stored addresses still in step with the field just loaded?

        THE ASYMMETRY THIS CLOSES. `sleep()` snapshots θ₀ on the way out, so
        something has to check it on the way back in — and waking is precisely
        when θ₀ is most likely to have moved. Sleep exists to hand the field to
        the pro terminal, and the pro terminal exists to TRAIN it. So the
        field that comes back is routinely not the field that left. (Measured
        historically: a terminal load/save pair once pointed at fields 1.87 rad
        apart.)

        Addresses filed against the old θ₀ are then stale, and stale addresses
        do not fail loudly — retrieval quietly collapses toward recency. Mati
        would come up looking healthy and remembering the wrong things.

        This DOES NOT reindex inline. Reindexing costs ~29s and grows with the
        corpus; blocking a wake on it would trade a silent fault for a startup
        that looks hung. It measures, reports, and lets the idle worker — which
        already selects most-stale-first — do the repair.
        """
        try:
            gs_now = bridge.address_gs(self.lm)   # the frozen space's figure
            conn = sqlite3.connect(bridge.NOTES_DB_PATH)
            try:
                bridge._ensure_schema(conn)
                # ⚠️ WORKING SET ONLY. This counted every note, including the
                # archived ones that `bridge.reindex` never touches by design —
                # so with 537 of 608 filed it could never report fewer than 537
                # stale, and it queues a reindex whenever stale > 0. Every
                # reindex finished, re-checked, still saw 537, and queued
                # another: 26 in 6.3 hours, a loop that cannot terminate.
                #
                # That loop is the startup delay, and the reason Mati looked
                # like it was re-learning itself on every load. It was chasing
                # a number it could not move.
                rows = conn.execute(
                    "SELECT feat_gs FROM notes WHERE archived IS NULL"
                ).fetchall()
            finally:
                conn.close()
        except Exception as e:                 # never block a wake on a check
            self.telemetry.publish(stage="idle", address_check=f"failed: {e!r}")
            return {"checked": False, "error": repr(e)}

        # A NULL feat_gs was never addressed at all, which is maximal staleness
        # rather than none — counted, but it cannot contribute a drift number.
        never = sum(1 for (f,) in rows if f is None)
        drifts = [abs(gs_now - float(f)) for (f,) in rows if f is not None]
        worst = max(drifts) if drifts else 0.0
        stale = never + sum(1 for d in drifts if d > C.STALE_EPS)

        out = {"checked": True, "why": why, "episodes": len(rows),
               "stale": stale, "never_indexed": never, "worst_drift": worst,
               "gs_std": gs_now}
        if stale:
            # Ask for repair rather than performing it. The worker rotates by
            # staleness, so the worst rows go first without being told.
            # Batched and yielding: this fires right after startup/wake, which
            # is exactly when someone is likely to type. Without should_stop it
            # held the field for ~122s (32 notes x 3.8s) and every arriving turn
            # queued behind it — .
            # ⚠️ MEASURED, NOT ACTED ON — see C.REINDEX_ON_IDLE. Queueing here
            # fed a loop that could not converge and pegged the process for
            # 10.5 hours. The number is still reported so `/addresses` tells
            # the truth about the index; it just no longer starts work.
            if C.REINDEX_ON_IDLE:
                self.work.put(("reindex", lambda: {"reindexed": bridge.reindex(
                    self.lm, self.tok, limit=C.REINDEX_BATCH,
                    should_stop=lambda: self.contended)}))
                out["queued"] = True
            else:
                out["queued"] = False
                out["note"] = ("field reindexing retired — lexical is the "
                               "primary index (ARCHITECTURE.md)")
        self.telemetry.publish(stage="idle", address_check=out)
        if self.verbose and stale:
            # ⚠️ SAY WHAT ACTUALLY HAPPENED. This printed "reindex queued"
            # unconditionally, and went on printing it after reindexing was
            # retired — a log line asserting work that was no longer being
            # started. The same shape as every other fault this week.
            tail = ("reindex queued" if out.get("queued")
                    else "not repaired — field reindexing is retired")
            print(f"  [mati] {stale}/{len(rows)} addresses stale after {why} "
                  f"(worst {worst:.3e}) — {tail}", file=sys.stderr)
        return out

    def wake(self):
        """Take the field back. Costs one load (~1.5s).

        Symmetric with `sleep()`: that one snapshots and lets go, this one
        picks up and then CHECKS what it picked up. See `check_addresses`.
        """
        if not self.asleep:
            return {"asleep": False, "already": True}
        # Whatever we handed the field to on sleep may still have it. Waking on
        # top of it would give θ₀ two owners — the exact thing sleeping avoided.
        if not exclusive.take("Mati server"):
            cur = exclusive.held_by() or {}
            return {"asleep": True, "refused": True,
                    "held_by": cur.get("holder"), "pid": cur.get("pid")}
        self.telemetry.publish(stage="waking")
        with self.field():
            self.lm, self.tok = bridge.load_field()
        self.asleep = False
        self.slow = False
        self.last_activity = time.time()
        activity.log(activity.WAKE)
        # The field may have been trained by the terminal while we were away.
        self.check_addresses("wake")
        # ⚠️ CATCH UP ANYTHING WRITTEN WHILE LOCKED. `seal_soft` stores in
        # the clear rather than dropping a reminder when there is no key, so
        # the exposure window is one process lifetime — and only if something
        # closes it. This closes it. Idempotent and cheap when there is
        # nothing to do.
        try:
            n = vault.migrate_tables(verbose=False)
            if n and self.verbose:
                print(f"  [vault] sealed {n} values written while locked",
                      file=sys.stderr)
        except Exception:
            pass                      # never block a wake on housekeeping

        # Read everything ONCE, here, where nobody is waiting on it.
        self.build_orientation()
        self.telemetry.publish(stage="idle", asleep=False)
        if self.verbose:
            print("  [mati] awake", file=sys.stderr)
        return {"asleep": False}

    def build_orientation(self, prewarm=True):
        """
        Read the field, the record and the clock once, and hold the result.

        This is the work `turn()` used to redo on every single turn: 2,001
        tokens of field state, self-knowledge and recalled memories, rebuilt
        fresh each time and therefore re-prefilled each time at ~35 ms/token —
        ~70 s before the model could say a word. Built once and held constant,
        the same material is cached after its first send and costs ~0.1 s.

        `prewarm` pays even that first send here, in the background, so the
        person's first turn does not wear it. It is one throwaway generation of
        a single token; what is being bought is the prefill, not the reply.
        """
        coherence = None
        try:
            with self.field():
                _t, coherence = bridge.field_state(self.lm, self.tok, "waking")
        except Exception as e:                 # never let waking fail on this
            if self.verbose:
                print(f"  [mati] orientation field read failed: {e}",
                      file=sys.stderr)
        try:
            block = orient.orientation(
                lm=self.lm, tok=self.tok, since=self.slept_at,
                was_shutdown=self.was_shutdown, coherence=coherence)
        except Exception as e:
            if self.verbose:
                print(f"  [mati] orientation failed: {e}", file=sys.stderr)
            return None
        self.orientation = block
        n, within = orient.size(block)
        if not self._llm_injected:
            self._oriented_llm = make_llm(system_prompt() + "\n\n" + block,
                                          self.provider)
        if self.verbose:
            print(f"  [mati] waking block: {n} tokens"
                  f"{'' if within else ' — OVER BUDGET, it eats history'}",
                  file=sys.stderr)
        if prewarm:
            self._prewarm()
        # ⚠️ THE ENCYCLOPEDIA WARMS HERE OR IT STALLS A REPLY. Mounting a
        # full-text index over a few hundred megabytes takes seconds, and the
        # first lookup of a session is exactly when nobody is willing to wait.
        # Backgrounded, so a missing binary or a missing file costs nothing.
        try:
            from . import encyclopedia as _enc0
            if _enc0.warm() and self.verbose:
                print("  [mati] encyclopedia warming", file=sys.stderr)
        except Exception:
            pass
        return block

    def _prewarm(self):
        """Pay the first prefill now rather than on the person's first turn."""
        try:
            t = time.time()
            self.chat_llm.reply([{"role": "user", "content": "."}],
                                max_tokens=1)
            if self.verbose:
                print(f"  [mati] prompt cache warm ({time.time()-t:.0f}s)",
                      file=sys.stderr)
        except Exception as e:
            if self.verbose:
                print(f"  [mati] prewarm skipped: {e}", file=sys.stderr)

    # ── sessions ─────────────────────────────────────────────────────────────

    def history(self, session_id):
        with self._sessions_lock:
            return list(self.sessions.get(session_id, []))

    def append_history(self, session_id, user_text, reply):
        with self._sessions_lock:
            h = self.sessions.setdefault(session_id, [])
            h.append({"role": "user", "content": user_text})
            h.append({"role": "assistant", "content": reply})
            # Local context is only LOCAL_NUM_CTX tokens and the system prompt
            # already occupies part of it, so this trim is a hard requirement.
            keep = 2 * C.SESSION_MAX_TURNS
            if len(h) > keep:
                del h[:len(h) - keep]

    def clear_session(self, session_id):
        with self._sessions_lock:
            gone = self.sessions.pop(session_id, None) is not None
        # The self-knowledge memo tracks what THIS session has been told; a
        # cleared session has been told nothing, so it must hear it again.
        bridge.forget_self_knowledge(session_id)
        return gone

    # ── the turn ─────────────────────────────────────────────────────────────

    class Asleep(Exception):
        """The field has been handed over; nothing can run until it's back."""

    def _require_awake(self):
        """Wake if asleep. Refuse only when something else holds the field.

        ⚠️ THIS USED TO REFUSE, AND AUTO-SLEEP WOULD HAVE MADE THAT A TRAP.
        A sleeping server turned away every turn until someone typed /wake —
        harmless while nothing ever slept on its own, and a locked door the
        moment something did. The user's whole reason for a schedule is that
        Mati should be READY BEFORE THE APP OPENS; being told to knock twice
        is the opposite of that.
        """
        # ⚠️ STIRRING IS NOT WAKING, AND ONLY ONE OF THEM WAS HANDLED. Resting
        # keeps the field, so `asleep` is False and none of the code below
        # runs — nothing else in the file clears `slow` except `wake()`, which
        # a resting Mati never reaches. Left alone it would go slow on the
        # first quiet night and never study, inquire or acquire again: awake,
        # answering, and quietly doing none of its own work. The person
        # speaking is the signal that the night is over.
        if self.slow:
            self.slow = False
            activity.log(activity.STIR, "spoken to")
            self.telemetry.publish(stage="awake", slow=False)
        if self.asleep or self.lm is None:
            out = self.wake()
            if out.get("refused"):
                who = out.get("held_by") or "something else"
                raise self.Asleep(
                    f"the field is held by {who} — close it and Mati will "
                    f"take it back.")
            if self.asleep or self.lm is None:
                raise self.Asleep(
                    "Mati could not take the field back. POST /wake.")

    def _remember_action(self, session_id, asked, said):
        """An action Mati took is a lived moment, and is recorded as one.

        ⚠️ It skips the coherence gate deliberately: the gate decides whether
        a REPLY settled the field well enough to keep, and an action did not
        come from the field at all. What happened, happened.
        """
        try:
            from . import ollin as _O
            with self.field():
                bridge.remember(self.lm, self.tok,
                                f"I was asked: {asked}\nI did it: {said}",
                                speaker=self._speaker.get(session_id),
                                verbatim=_O._verbatim_of(asked, said,
                                                         who=self._speaker.get(session_id)))
        except Exception as e:                          # noqa: BLE001
            if self.verbose:
                print(f"  [dj] not recorded: {e}", file=sys.stderr)
        return {}

    def interrupt(self, session_id, said):
        """The person started typing over the reply. Hold what was said.

        ⚠️ THIS ABANDONS A REPLY AND NOTHING ELSE. No field write is cancelled,
        no queued episode is dropped, no lock is touched — once θ₀ has started
        moving it finishes. Interruption is a conversational event, and the
        invariant that two owners of θ₀ is the unrecoverable mistake is older
        than this feature and does not bend for it.
        """
        ex = self._pending.get(session_id)
        if ex is None:
            return None
        ex.cut(said)
        activity.log(activity.INTERRUPTED, f"after {ex.cut_at} words")
        return interruption.summarise(ex)

    def chat(self, session_id, text, on_chunk=None, remember=True):
        """Run one turn. Blocks until the reply is complete; episode deferred."""
        self._require_awake()
        self.last_activity = time.time()
        t0 = time.time()
        # "Stop asking me questions" has to take effect on the turn it is said,
        # not after the next sleep — so it is read here, before anything else.
        switched = schedule.reads_ask_control(text)
        if switched and self.verbose:
            print(f"  [ask] questions turned {switched} by the person",
                  file=sys.stderr)
        self.telemetry.publish(stage="waiting for field", session=session_id)

        def sink(thunk):
            thunk.session = session_id       # so the worker can file the id
            self.work.put(("episode", thunk))

        def chunk(piece):
            self.last_activity = time.time()
            if on_chunk:
                on_chunk(piece)

        # ⚠️ SPOKEN FIRST, BEFORE ANY FIELD WORK. : the
        # first real token arrives after 77.4s — 61% of the turn is a blank
        # screen. The user: "that wait-time needs to be filled by the conscious
        # agent." This is a lookup and an echo, no model call, so it appears
        # in about a second while the field takes its 77.
        #
        # ⚠️ It is SPEECH and is therefore part of the record. Everything in it
        # is copied from the store or is affect about the act itself, so there
        # is nothing here that can be false.
        # ⚠️ AN ASKED-FOR ACTION IS ANSWERED IMMEDIATELY, NOT IN 120 SECONDS.
        # "play some jazz" does not need the field, the memory or the model —
        # it needs a station and a play command, and both take about two
        # seconds. Routing it through the ordinary turn would make the person
        # wait two minutes to hear music.
        #
        # ⚠️ ONLY ON A LITERAL MATCH. `dj.hears_request` returns None unless
        # there is an explicit verb; playing is the first irreversible thing
        # Mati does, so a false positive is audible and a false negative
        # costs one sentence.
        try:
            # ⚠️ ONE DOOR. This called `dj.hears_request` directly, which made
            # the MUSIC module the front door for all media and assumed music
            # unless it spotted a video word — so "play a podcast about
            # history" and "play radio from Mexico" both went to the Apple
            # Music library. `media.hears` classifies first and routes second,
            # and an unclear request is asked about rather than guessed.
            from . import media
            req = media.hears(text)
            if req:
                said = media.answer(req)
                # A film answers with a URL for the page to play, not a launch.
                play_video = play_reel = None
                if isinstance(said, dict):
                    play_video = said.get("play_video")
                    # ⚠️ A LIST, AND THE PAGE LOOPS IT MUTED. Silent-era
                    # shorts are meant to run WHILE the radio does.
                    play_reel = said.get("play_reel")
                    said = said.get("say")
                # ⚠️ A MEDIA ANSWER IS CANNED AND NEVER REACHES `turn()`,
                # so it was the third silent path. "Playing Blue Note
                # Radio" is something Mati DID, and it should say that
                # aloud as readily as anything a model generated.
                if said and isinstance(said, str):
                    try:
                        from . import speech as _sp
                        _sp.say(said)
                    except Exception:
                        pass
                if said:
                    chunk(said)
                    self.last_turn = time.time()
                    self.stats["turns"] = self.stats.get("turns", 0) + 1
                    if remember:
                        sink(lambda do_reindex=False, should_stop=None:
                             self._remember_action(session_id, text, said))
                    info = {"action": "media", "seconds": 0.0,
                            "kind": req.get("kind"), "verb": req.get("verb")}
                    if play_video:
                        info["play_video"] = play_video
                    if play_reel:
                        info["play_reel"] = play_reel
                    return said, info
        except Exception as e:                          # noqa: BLE001
            if self.verbose:
                print(f"  [dj] skipped: {e}", file=sys.stderr)

        opener = ""
        try:
            from . import smalltalk, rapport
            opener = smalltalk.opening(
                text, fast=bool(rapport.is_fixed(text)))
        except Exception as e:                          # noqa: BLE001
            if self.verbose:
                print(f"  [smalltalk] skipped: {e}", file=sys.stderr)
        if opener:
            chunk(opener + "\n\n")
            # ⚠️ THE OPENER IS SPOKEN TOO, and leaving it out was the gap
            # reported as "it does not speak the canned responses". This is
            # the patience line — "Hm.", "Let me sit with that a second." —
            # and it is the single most MATI thing in a reply: nothing
            # generated it, Mati says it because a person is waiting.
            #
            # ⚠️ ENQUEUED, NOT SAID. `say()` interrupts; the reply's own
            # sentences are queued right behind this one, and interrupting
            # would cut the opener off mid-word as the first sentence landed.
            try:
                from . import speech as _sp
                _sp.enqueue(opener)
            except Exception:
                pass

        # ⚠️ WHO IS TYPING. An episode used to record nothing about the
        # speaker, so a third party's words were filed as the owner's —
        # , when four episodes about "my AI assistant" left
        # Mati unable to say whether the person in front of it was the user.
        #
        # ⚠️ MATI NO LONGER ASKS WHO IS THERE. The user: "Mati's assumption
        # should always be that the peer will be speaking to them. It is up to
        # Mati to figure out on their own if that is truly the case."
        #
        # It used to open a new session by asking for a name. That is the wrong
        # posture twice over: it greets the person who built it as a stranger,
        # and it puts the work of identification on them when it belongs to
        # Mati. The default is the owner, and an introduction is TAKEN when
        # offered rather than solicited.
        #
        # ⚠️ Nothing about attribution is weakened. `named_in` still records a
        # name the moment someone gives one, and `attribute` still marks a
        # memory as somebody else's. What is gone is the interrogation.
        told = speaker.named_in(text)
        if told:
            self._speaker[session_id] = told

        # ⚠️ CONTINUE THE EXCHANGE, DO NOT START A SECOND ONE. If the last turn
        # in this session was cut off, this text is another PART of it — the
        # abandoned prompt, how far the reply got and the new words become one
        # memory. `text` stays the latest part, because that is what is being
        # answered; the exchange is what gets remembered.
        ex = self._pending.get(session_id)
        if ex is not None and ex.cut_at is not None:
            ex.add(text)
        else:
            ex = interruption.Exchange(text)
        self._pending[session_id] = ex

        with self.field():
            self.telemetry.publish(stage="thinking", session=session_id)
            history = self.history(session_id)
            reply, info = turn(self.lm, self.tok, self.chat_llm, text,
                               # ⚠️ THE ONE CALLER WITH A PERSON WAITING.
                               # `turn(speak=...)` defaults to False so
                               # anything else that runs a turn stays silent.
                               speak=True,
                               history=history, verbose=self.verbose,
                               remember=remember, on_chunk=chunk,
                               episode_sink=sink,
                               prev_episode_id=self.last_episode.get(session_id),
                               prev_challenged=self.last_challenged.get(
                                   session_id, False),
                               session=session_id,
                               oriented=self.orientation is not None,
                               speaker=self._speaker.get(session_id),
                               exchange=ex)

        # ⚠️ A COMPLETED REPLY CLOSES THE EXCHANGE. Without this the session
        # would accumulate parts forever and every later turn would be filed as
        # a continuation of the first thing ever said in it.
        if reply is not None:
            self.append_history(session_id, text, reply)
            self._pending.pop(session_id, None)
        self.last_challenged[session_id] = bool(info.get("challenging"))
        self.stats["turns"] += 1
        self.last_turn = time.time()          # the person is here
        self.last_activity = time.time()
        info["seconds"] = time.time() - t0
        self.telemetry.publish(
            stage="idle", session=session_id,
            basin=list(info.get("triplet", [])), coherence=info.get("coherence"),
            # The colour signal, computed here so every client renders the same
            # thing rather than each inventing its own mapping.
            signal=(sig.encode(info["coherence"])
                    if info.get("coherence") is not None else None),
            n_memories=info.get("n_memories"), computed=info.get("computed"),
            # The face, chosen here for the same reason as the colour: one
            # clip, decided in one place, every client agreeing.
            expression=expr.read(stage="idle", info=info),
            # The socially meaningful flags, raw, for the pro app. Without
            # these the avatar cannot see the moments it exists to express.
            correcting=info.get("correcting"),
            challenging=bool(info.get("challenging")),
            deferred=bool(info.get("deferred")),
            will_research=info.get("will_research"),
            turn_seconds=info["seconds"], queue_depth=self.work.qsize())
        return reply, info

    # ── worker: deferred writes, then idle reindex ───────────────────────────

    def _pulse_loop(self):
        """Publish the LIVE phases a few times a second, so the panel can show
        that the field is running without implying anything is being learned.

        ⚠️ WHY THIS IS HONEST TO ANIMATE. The free-running dynamics drift the
        live phases ~1.22 rad while θ₀ moves exactly 0.00000000 — measured, and
        recorded in `expression.py`. So this is motion that means "awake", not
        motion that means "changing". Between turns the panel's basin and
        coherence are the LAST TURN's numbers, frozen; animating those would be
        putting movement on stale values, which is the same lie as a progress
        bar that keeps going after the work stopped.

        ⚠️ IT NEVER WAITS FOR THE FIELD. A visual that queues behind reindex
        would slow the work it depicts, so the lock is taken non-blocking and
        the tick is simply skipped when the worker has it. A missing frame
        costs nothing; a contended one costs the thing being watched.
        """
        while not self._stop.is_set():
            time.sleep(C.PULSE_SECONDS)
            if self.asleep or self.lm is None or self.contended:
                continue
            if not self._lock.acquire(blocking=False):
                continue                       # the worker is using it
            try:
                states = self.lm.mf.get_all_states()
                phases = np.asarray(states[0])[0]        # mode 0, phase row
                coh = float(self.lm.output_coherence(states))
            except Exception:
                continue                       # a visual must never raise
            finally:
                self._lock.release()
            # Per-bin resultant length: how aligned that stretch of the field
            # is, in [0,1]. Structure rather than raw angle, so the strip means
            # something as well as moving.
            try:
                bins = np.array_split(phases, C.PULSE_BINS)
                strip = [float(abs(np.mean(np.exp(1j * b)))) for b in bins]
            except Exception:
                continue
            self.telemetry.publish(pulse={"strip": [round(x, 3) for x in strip],
                                          "coherence": round(coh, 4)})

    _idle_after_cache = (0.0, 0.0)

    def _maybe_keep_schedule(self):
        """Wake before the day usually starts; rest after it usually ends."""
        try:
            from . import rhythm
            sched = rhythm.daily_schedule()
        except Exception:
            return False
        if not sched:
            return False              # no pattern; nothing to honour
        now = datetime.now()
        hour = now.hour + now.minute / 60.0
        wake, rest = sched["wake_hour"], sched["sleep_hour"]
        # Ready BEFORE the app opens, not when it does.
        daytime = (wake - 0.5) <= hour < rest
        if self.asleep and daytime:
            self.wake()
            return True
        # ⚠️ THE MORNING BRANCH ONLY KNEW ABOUT `asleep`. Now that a night is a
        # rest rather than a handoff, Mati arrives at morning SLOW, not
        # asleep — so without this the schedule could put it to rest at night
        # and never bring it back.
        if self.slow and daytime:
            self.slow = False
            activity.log(activity.STIR, "morning")
            return True
        if not self.asleep and not self.slow and not daytime:
            self.rest()               # keep the field, consolidate, dream
            return True
        return False

    def _idle_after(self):
        """How long a silence means this person has gone. Learned, cached."""
        now = time.time()
        when, val = self._idle_after_cache
        if now - when < 60 and val:
            return val
        try:
            from . import rhythm
            val = rhythm.idle_after()
        except Exception:
            val = C.SLEEP_AFTER_SECONDS
        type(self)._idle_after_cache = (now, val)
        return val

    def _idle_pass(self):
        """One pass of everything that happens when nobody is talking.

        Split out of `_worker_loop` so the loop can guard it — see the note at
        the call site. Returns as soon as one piece of work is done, so the
        next pass re-checks contention rather than running the whole list
        while a turn waits.

        ⚠️ `idle_work` GATES THE WHOLE PASS, and that is the point of asking
        as ONE question. Dreaming, studying, self-talk and warming are four
        switches internally, but the person's concern is singular — is this
        spending my battery — so one answer has to actually stop all four.
        A consent that turned off one of them would be the worst kind of
        setting: it looks like it worked.
        """
        # ⚠️ PHASE 3: A LONG GAP IS THE MOMENT, because this pass is the thing
        # being consented to. Raised here rather than on a turn — a turn is by
        # definition not a quiet spell, and asking "may I work while you are
        # away" to someone who is plainly present is the wrong question at the
        # wrong time. It waits in `pending_ask` for whenever they come back.
        try:
            from . import asks as _asks0
            _asks0.ask_idle_if_needed()
        except Exception:
            pass

        try:
            if C._settings().get("idle_work") is False:
                # ⚠️ DUE ITEMS STILL RUN. A reminder the person set is not
                # speculative work — declining "may I think while you are
                # away" must not silently cancel something they asked for.
                self._maybe_keep_schedule()
                self._maybe_announce()
                return
        except Exception:
            pass                     # unreadable settings must not stop work
        # Due items are checked on every pass, not only on idle — a reminder
        # set for 15:00 should not wait for a quiet spell.
        # ⚠️ SCHEDULED, SO MATI IS READY BEFORE THE APP OPENS. The user: sleep
        # and wake "should be scheduled in the beginning and then allowed to
        # reschedule at a learned set time every day, this way Mati is ready
        # before the app is opened."
        #
        # ⚠️ INERT UNTIL THE PATTERN IS REAL, and it is not yet. Over 17 days
        # the first turn has a standard deviation of 8.5 HOURS — median 12:31,
        # ranging 00:01 to 22:36. `rhythm.daily_schedule` returns None on that
        # and this does nothing, because waking at a median drawn from noise
        # would put Mati to sleep mid-conversation. When the days settle, it
        # starts working with no further change.
        self._maybe_keep_schedule()
        self._maybe_announce()
        # ⚠️ HERE, NOT IN THE IDLE BLOCK BELOW, AND NOT GATED ON `slow`.
        # Placed after `_maybe_deepen` first, it sat behind a 45-minute idle
        # gate AND behind an early return that fires whenever there are debts
        # — the starvation pattern this file has recorded five times. It rate
        # limits itself to WORLD_EVERY_SECONDS, so it needs no other gate, and
        # it must not be starved: the whole point is that the news is already
        # there before anyone asks.
        self._maybe_fetch_world()
        # ⚠️ THE SHORT TIER, AND IT COMES FIRST. Turns no longer imprint while
        # the person waits — 11.4s of a 55s turn, for a field reading nothing
        # in the reply consumed. The debt is settled here instead, within
        # seconds of the exchange rather than ten minutes later, because the
        # field being behind the conversation is a different thing from the
        # field being idle.
        #
        # Everything else — studying, dreaming, questions, inquiry — waits for
        # the long tier below. Catching up beats acquiring.
        if self._drain_imprints():
            self.last_activity = time.time()
            return
        # SLEEP — left alone, not merely paused. The expensive work lives here
        # so a gap in conversation is never spent imprinting a document chunk
        # (~60s) while the person waits to type again. Ordered: absorb what was
        # queued, finish what was written thin, only then acquire anything new.
        # ⚠️ RESTING STILL PERCOLATES. The user: "the solution came from the
        # desire to leave Mati running even if the apps are closed, for
        # continuity — but I want to leave it open for something that needs
        # percolating."
        #
        # My first version returned here and stopped everything, dreaming
        # included, which is the opposite of percolating. The division that
        # matters is not cheap-versus-expensive, it is OUTWARD versus INWARD:
        #
        #   outward — reading a new document, asking itself new questions.
        #             Acquisition. Stops while resting.
        #   inward  — dreaming. Turning over what is already held. Continues.
        #
        # ⚠️ The PREDICTION and RAPPORT are rebuilt in `sleep()`, not here. A
        # prediction of the day has to be made at the boundary of one, and idle
        # is a quiet spell in the middle of a day rather than the end of it.
        #
        # A person resting is not learning a new subject and is very much still
        # digesting the day.
        # ⚠️ THE THRESHOLD IS THE PERSON'S, NOT A CONSTANT. `rhythm.idle_after`
        # reads their own p95 gap, floored at 45 minutes. Cached for a minute
        # so this stays a comparison rather than a query.
        # ⚠️ NOTHING PUT MATI TO SLEEP. `/health` has reported "sleeping":
        # true after two quiet hours since long before tonight — a claim
        # nothing acted on. The only callers of `sleep()` were the raw
        # endpoint, which the terminal stopped using when /sleep became REST,
        # and process exit.
        #
        # That matters more than it sounds: CONSOLIDATION LIVES IN `sleep()`.
        # The prediction and the day's rapport are rebuilt there and nowhere
        # else, so on a machine simply left running they would never have been
        # rebuilt at all.
        # ⚠️ THIS CALLED `sleep()` — THE FULL HANDOFF — AND SO DISCONNECTED THE
        # TERMINAL EVERY NIGHT. The user found it the next morning: "the terminal
        # was disconnected from the field. Sleep is not supposed to disconnect
        # from the field anymore, only quit." It now rests: consolidates the
        # day, goes slow, keeps the field and goes on dreaming.
        if (not self.asleep and not self.slow
                and time.time() - self.last_turn >= C.SLEEP_AFTER_SECONDS):
            self.rest()
            return True

        if time.time() - self.last_turn >= self._idle_after():
            # ⚠️ FIRST, AND THAT IS THE POINT. The user: *"split the verbatim
            # into time segments, so we don't have to decrypt everything at
            # once"* and *"the last year should be RAG ready"*. One segment is
            # ~3s and needs NO FIELD LOCK, so it is the cheapest useful thing
            # an idle pass can do — and putting it after the debts below would
            # starve it exactly as dreaming was starved: `_maybe_study` and
            # `_maybe_deepen` return early to finish their own work, take
            # minutes doing it, and a job behind them simply never runs while
            # there is anything to read. See `notes.warm_verbatim`.
            if self._maybe_warm_verbatim():
                self.last_activity = time.time()
                return
            if not self.slow and self._maybe_study():
                activity.log(activity.STUDY)
                self.last_activity = time.time()
                return                    # more to read before repairing
            if self._maybe_deepen():
                self.last_activity = time.time()
                return                    # finish debts before acquiring
            # After absorbing, before acquiring: a memory of HAVING READ. It
            # goes here so it is written from a settled field rather than
            # mid-batch, and so the counts it states are the ones that were
            # true when it was written.
            # Condense over-long memories so focusing on them later returns a
            # summary rather than a cut-off head. Off the turn, bounded per
            # pass, and after the debts above — see Mati/digest.py.
            if not self.slow and self._maybe_digest():
                self.last_activity = time.time()
                return
            if self._maybe_record_reading():
                self.last_activity = time.time()
                return
            if self._maybe_resolve():
                self.last_activity = time.time()
                return                    # they already answered it, unasked
            if self._maybe_log_question():
                self.last_activity = time.time()
                return                    # ask the person, who actually knows
            # ⚠️ BEFORE INQUIRY, AFTER THE DEBTS. Dreaming went last at first,
            # and that reproduced the bug it was meant to fix: `_maybe_inquire`
            # fires on nearly every sleep pass while topics stay open, so the
            # pass returned before ever reaching the dream. Starved, not
            # missing — which looks identical from outside.
            #
            # It is safe here because it rate-limits itself to once an hour
            # (DREAM_EVERY_SECONDS), so it cannot starve inquiry in turn. The
            # order is: repay what is owed, rest, then acquire something new.
            # Dreaming turns over what is held. It does NOT rebuild the
            # prediction — that happens in `sleep()`, at the end of a day.
            if self._maybe_dream():
                activity.log(activity.DREAM)
                self.last_activity = time.time()
                return
            if (not self.slow and C.INQUIRE_ON_IDLE
                    and self._maybe_inquire()):
                self.last_activity = time.time()
                return                    # it asked; absorb next pass
        # Bound the work BEFORE doing it: anything past WORKING_SET drops to
        # the lexical archive, so the reindex below is O(WORKING_SET) rather
        # than O(everything ever said). Cheap — one UPDATE, no field work.
        # Bounded growth: a season of demand is enough to see a pattern in,
        # and the log is written on every turn.
        bridge.prune_hits()
        archived, restored = bridge.rebalance_tiers()
        if (archived or restored) and self.verbose:
            print(f"  [tiers] archived {archived}, restored {restored}",
                  file=sys.stderr)
        # IDLE — a gap in conversation. Only what is quick, and what recall
        # cannot do without: one turn stales every stored address (measured
        # 100% -> 0% self-retrieval), so reindex cannot wait for sleep or
        # recall is broken mid-conversation. It is bounded and yields per note,
        # so it fits in a gap.
        self._maybe_reindex()

    def _worker_loop(self):
        while not self._stop.is_set():
            try:
                kind, thunk = self.work.get(timeout=1.0)
            except queue.Empty:
                # ⚠️ WRAPPED, AND THAT IS THE POINT. This branch sits outside
                # the try/except that guards task execution below, so for most
                # of this project's life ANY exception in the idle or sleep
                # work propagated out of the `while` and killed this thread.
                # It happened TWICE — `int(None)` in
                # `deepen_shallow`, then a save racing `/sleep` — and both
                # times `/health` went on reporting ready, awake and holding
                # the field while Mati did no background work whatsoever: no
                # reindexing, no deepening, no dreaming, no questions. The
                # only tells were a stuck `queue_depth` and a reindex count
                # that never moved.
                #
                # A broken idle pass must cost one pass, not the worker.
                try:
                    self._idle_pass()
                except Exception:                       # noqa: BLE001
                    self.stats["worker_errors"] = (
                        self.stats.get("worker_errors", 0) + 1)
                    self.last_worker_error = traceback.format_exc(limit=4)
                    print(f"  [worker] idle pass failed, continuing:\n"
                          f"{self.last_worker_error}", file=sys.stderr)
                self.worker_beat = time.time()
                continue
            try:
                self.telemetry.publish(stage=f"worker: {kind}",
                                       queue_depth=self.work.qsize())
                with self.field():
                    # An episode write imprints for ~55s at TRIPLET_CAP=64 and
                    # used to hold the field for all of it, so a turn arriving
                    # mid-write waited the whole time (measured 73s). Tasks that
                    # accept `should_stop` yield at a safe boundary; those that
                    # do not are called as before.
                    # Ask the signature rather than catching TypeError: a
                    # TypeError raised INSIDE the thunk would otherwise trigger
                    # the fallback and run the whole write a SECOND time —
                    # imprinting the same episode twice.
                    if _takes_should_stop(thunk):
                        info = thunk(should_stop=lambda: self.contended) or {}
                    else:
                        info = thunk() or {}
                # Tasks are plain callables and not all of them return a dict —
                # `bridge.remember` hands back (note_id, triplet), which used to
                # blow up the telemetry publish below with "'tuple' object has
                # no attribute 'get'" AFTER the work had already succeeded, so
                # the log claimed a failure that never happened.
                if not isinstance(info, dict):
                    info = {"episode_id": info[0]} if (
                        isinstance(info, tuple) and info) else {}
                if kind == "episode":
                    self.stats["episodes"] += 1
                    # Remember which episode this was, so that if the NEXT turn
                    # is a correction it knows what it is correcting.
                    sid, nid = getattr(thunk, "session", None), info.get("episode_id")
                    if sid and nid:
                        self.last_episode[sid] = nid
                self.telemetry.publish(stage="idle", worker_done=kind,
                                       episode_id=info.get("episode_id"),
                                       queue_depth=self.work.qsize())
            except Exception as e:                  # a worker crash must not
                self.telemetry.publish(             # take the server with it
                    stage="idle", worker_error=f"{type(e).__name__}: {e}")
                if self.verbose:
                    print(f"  [worker] {kind} failed: {e!r}", file=sys.stderr)
            finally:
                self.work.task_done()

    def _drain_imprints(self):
        """Imprint what the turns handed over. One per pass, oldest first.

        One at a time so an arriving turn waits at most a single imprint
        (~11s) rather than a whole backlog, and so `self.contended` is
        re-checked between them.
        """
        if self.asleep or self.lm is None or not self._to_imprint:
            return 0
        text = self._to_imprint.pop(0)
        try:
            with self.field():
                bridge.imprint_text(self.lm, self.tok, text)
            self.stats["imprinted"] = self.stats.get("imprinted", 0) + 1
            if self.verbose:
                print(f"  [worker] imprinted a turn ({len(self._to_imprint)} "
                      f"waiting)", file=sys.stderr)
        except bridge.TooShortToImprint:
            pass
        except Exception as e:
            if self.verbose:
                print(f"  [worker] imprint failed: {e}", file=sys.stderr)
        return 1

    def _maybe_reflect(self):
        """Mati talks to itself about the day. Never raises, never blocks.

        ⚠️ EVERY OUTCOME IS LOGGED, INCLUDING NOTHING-HAPPENED, and that is
        the whole point of this shape. This logged only SUCCESS and printed
        failures only under `--verbose`, so on a normally-started server a
        reflection that raised left no trace anywhere: no activity row, no
        `mine` note, no complaint. Measured  — the newest `mine`
        row was FOURTEEN DAYS OLD while `sleep` and `dream` were logging
        nightly, and nothing in the system could say why.

        ⚠️ `reflect()` ITSELF WAS FINE. Called directly it returned a round
        on the first try. The defect was never the introspection; it was that
        its absence was undetectable. An invisible failure in a nightly path
        nobody watches is the same shape as `world.fetch_all` having no caller
        for fifteen days and the verbatim search being write-only.
        """
        try:
            from . import reflect
            with self.field():
                wrote = reflect.reflect(self.lm, self.tok, self.chat_llm,
                                        verbose=self.verbose)
            if wrote:
                activity.log(activity.REFLECT, f"{len(wrote)} rounds")
            else:
                # ⚠️ NOT an error — the gates are allowed to say no (too few
                # episodes, already done today). But "it ran and declined" and
                # "it never ran" must be distinguishable after the fact.
                activity.log(activity.REFLECT, "nothing to reflect on")
            return bool(wrote)
        except Exception as e:                      # noqa: BLE001
            activity.log(activity.REFLECT, f"FAILED: {type(e).__name__}: {e}")
            print(f"  [reflect] failed: {e!r}", file=sys.stderr)
            return False

    def _maybe_digest(self):
        """Condense one or two over-long memories. Never raises."""
        try:
            from . import digest
            return digest.step(llm=self.chat_llm, verbose=self.verbose)
        except Exception as e:                      # noqa: BLE001
            activity.log(activity.REFLECT,
                         f"_maybe_digest FAILED: {type(e).__name__}: {e}")
            print(f"  [digest] skipped: {e}", file=sys.stderr)
            return False

    def _maybe_record_reading(self):
        """Write a memory of having read, at each quarter of a document.

        ⚠️ 844 passages absorbed and no memory of the reading. The content was
        stored and labelled `studied`, so Mati could say "I read that" when a
        passage was handed to it — but asked what it had BEEN reading, it had
        nothing, because studying produced text and no experience.
        """
        if self.asleep or self.lm is None:
            return 0
        try:
            with self.field():
                wrote = reading.record_due(self.lm, self.tok,
                                           verbose=self.verbose)
        except Exception as e:
            activity.log(activity.REFLECT,
                         f"_maybe_record_reading FAILED: {type(e).__name__}: {e}")
            print(f"  [reading] failed: {e}", file=sys.stderr)
            return 0
        if wrote:
            self.stats["reading"] = self.stats.get("reading", 0) + len(wrote)
        return len(wrote)

    def _maybe_study(self):
        """Read a little of whatever is queued, while nobody is talking.

        Uploading a document is fast — it chunks and queues. The imprinting is
        ~30s a chunk and deforms θ₀, so it happens here, in the gaps, a couple
        of chunks at a time. Studying takes priority over reindexing: a chunk
        that has not been imprinted yet cannot be recalled at all, whereas a
        stale address still retrieves, just worse.
        """
        if self.asleep or self.lm is None:
            return 0
        if study.status().get("pending", 0) == 0:
            return 0
        # ⚠️ YIELD THE PASS. Study returns early from `_idle_pass`, so without
        # this a queued document starves everything below it — measured, a
        # 1,759-chunk upload stopped self-talk dead and left `dreams` at 0 for
        # a whole night. Taking only 1 pass in STUDY_EVERY lets deepening,
        # resolving, question-forming, inquiry and dreaming keep their turns
        # while the reading still gets most of the wall clock, because a study
        # pass is ~90s and theirs are seconds.
        self._study_pass = getattr(self, "_study_pass", 0) + 1
        # Asked for directly, reading takes every pass again — the old speed,
        # on purpose and with an end. Outside that window it yields.
        focused = time.time() < self.study_priority_until
        if not focused:
            if self._study_pass % C.STUDY_EVERY:
                return 0
        # A bigger bite while focused, because the ~89s reindex is per BATCH.
        # Two chunks per reindex is ~80% index repair; twenty is ~18%.
        batch = C.STUDY_PRIORITY_BATCH if focused else C.STUDY_BATCH
        with self.field():
            done = study.study(self.lm, self.tok, budget=batch,
                               similarity_gate=C.STUDY_SIMILARITY_GATE,
                               should_stop=lambda: self.contended,
                               verbose=self.verbose)
        if done:
            learned = sum(1 for d in done if d["status"] == "learned")
            self.stats["studied"] = self.stats.get("studied", 0) + len(done)
            self.telemetry.publish(stage="idle", studied=len(done),
                                   learned=learned,
                                   study_pending=study.status()["pending"])
        return len(done)

    def _maybe_deepen(self):
        """Finish memories that were imprinted thin because a turn was waiting.

        Ordered AFTER study and BEFORE inquiry, for the reason study is first:
        absorb what is owed before acquiring anything new. A shallow memory is
        a debt already taken on — an unasked question costs nothing.

        One memory per pass, yielding between them, because each owed pass is
        ~1s/token. θ₀ moves, so reindex follows on the next idle pass as it
        does after any deformation.
        """
        if self.asleep or self.lm is None:
            return 0
        if self.contended or not self.work.empty():
            return 0
        with self.field():
            n = bridge.deepen_shallow(self.lm, self.tok, limit=1,
                                      should_stop=lambda: self.contended)
        if n:
            owed = bridge.shallow_count()
            self.stats["deepened"] = self.stats.get("deepened", 0) + n
            self.telemetry.publish(stage="idle", deepened=n, shallow_owed=owed)
            if self.verbose:
                print(f"  [worker] deepened {n} shallow memory"
                      f"{'' if n == 1 else 'ies'}; {owed} still owed",
                      file=sys.stderr)
        return n

    def _maybe_resolve(self):
        """Answer a waiting question from something the person said later.

        Runs BEFORE forming new questions: clearing a debt the conversation
        already settled beats adding another. Nothing reaches the person —
        a question answered this way is simply never asked.
        """
        if self.asleep or self.llm is None:
            return 0
        if self.contended or not self.work.empty():
            return 0
        got = inquiry.resolve_from_later(self.llm, verbose=self.verbose)
        return 1 if got else 0

    def _maybe_log_question(self):
        """LOG a question about one recent memory. Does not ask it.

        Sleep is when the question is FORMED; chat is when it is asked. The
        two are deliberately separate: forming needs the model and the field
        and takes seconds, and there is nobody there to ask anyway. So it goes
        into the queue with the memory's subject attached and waits — possibly
        for days — until the person raises that subject themselves.

        Nothing here reaches the person. `_maybe_announce` cannot deliver it
        either: it calls `schedule.pending()` with no `user_text`, which
        excludes every topic-gated item, so a question has exactly one route
        out and that route requires the person to speak first.


        `inquiry.step` asks Mati's OWN language model and files the answer as
        `studied:asked about …` — the weakest thing it holds, by the README's
        own account. That design assumed somewhere to go and look; on a device
        with no corpus it can only manufacture more of the weakest kind. The
        person is right there and actually knows, and their answer arrives as a
        LIVED episode — the primary training signal.

        The question is QUEUED, never asked now. It waits until the person
        raises the subject: at the top of a conversation it is an interruption,
        when they brought the topic up it is interest. There is no cap on how
        many may wait, because an unraised question costs nothing — what makes
        them bearable is when they surface, not how few exist.

        Off the moment the person says so (`schedule.may_ask`).
        """
        if self.asleep or self.lm is None:
            return 0
        if not C.ASK_USER_ON_SLEEP or not schedule.may_ask():
            return 0
        if self.contended or not self.work.empty():
            return 0
        pick = bridge.episode_to_ask_about()
        if not pick:
            return 0
        note_id, body = pick
        # ⚠️ `note_id` WAS NOT PASSED, FOR THE LIFE OF THIS FEATURE. It is on
        # the line above and `question_for_user` takes it, so `detail` was
        # NULL on all 862 queued questions — and `detail` is what lets a later
        # conversation that answers a question be FILED against the memory it
        # came from rather than merely retired. The link the code documents as
        # load-bearing had never once been populated.
        q = inquiry.question_for_user(body, self.llm, verbose=self.verbose,
                                      note_id=note_id)
        # Marked either way: a memory that yields no usable question must not
        # be retried every sleep cycle forever.
        bridge.mark_asked_about(note_id)
        if q:
            self.telemetry.publish(stage="idle", asked_about=note_id)
            if self.verbose:
                print(f"  [worker] logged a question, to ask if it comes up: "
                      f"{q}", file=sys.stderr)
            return 1
        return 0

    def _maybe_dream(self):
        """Free-dream once, if everything else is settled.

        ⚠️ THIS NEEDS A CALLER OR IT IS DECORATION. `bridge.dream` and the `/dream`
        endpoint both existed and the worker invoked neither, so an install
        left running for hours never dreamed. Found by leaving the terminal
        open and waiting for it.

        LAST in the sleep cycle, deliberately. Study, deepening, resolving and
        inquiry all repay debts; this one creates them — dreaming deforms θ₀,
        every stored address is measured against θ₀, and so a dream buys a
        reindex. Running it before the debts are paid means reindexing twice.
        """
        if self.asleep or self.lm is None:
            return 0
        if not C.DREAM_ON_IDLE:
            return 0
        if self.contended or not self.work.empty():
            return 0
        if time.time() - self.last_dream < C.DREAM_EVERY_SECONDS:
            return 0                     # let the field settle between dreams
        self.last_dream = time.time()
        self.telemetry.publish(stage="dreaming", cycles=C.DREAM_CYCLES)
        try:
            with self.field():
                bridge.dream(self.lm, self.tok, cycles=C.DREAM_CYCLES,
                             verbose=self.verbose)
        finally:
            self.telemetry.publish(stage="idle")
        self.stats["dreams"] = self.stats.get("dreams", 0) + 1
        if self.verbose:
            print(f"  [mati] dreamed {C.DREAM_CYCLES} cycle(s)",
                  file=sys.stderr)
        self.check_addresses("dream")
        return 1

    def _maybe_announce(self):
        """Raise something Mati has been holding — only if allowed to.

        Refusing does not lose the item: it stays queued and is delivered at
        the top of the next conversation. Consent is checked per kind and quiet
        hours override it, so turning reminders on does not license 3am.
        """
        due = schedule.pending()   # topic-gated items excluded
        if not due:
            return 0
        announced = 0
        for item in due:
            ok, why = schedule.may_interrupt(item["kind"])
            if not ok:
                # ⚠️ PHASE 3: THE FIRST TIME SOMETHING IS ACTUALLY DUE is the
                # only honest moment to ask about interruptions — before that
                # it is a hypothetical, and `schedule.consent` has shipped
                # default-off with nothing ever asking to change it, so the
                # whole feature was unreachable. The item is NOT lost either
                # way; it waits for the next conversation.
                try:
                    from . import asks as _asks
                    _asks.ask_interrupt_if_needed()
                except Exception:
                    pass
                continue                      # waits for the user to speak
            self.telemetry.publish(stage="idle", announce={
                "id": item["id"], "kind": item["kind"], "body": item["body"]})
            # ⚠️ AND ACTUALLY DELIVER IT. Publishing alone reached nobody:
            # `announce` is consumed by no client, so this loop ran on every
            # idle pass, respected consent and quiet hours perfectly, and had
            # no effect. A due reminder arrived only when the person happened
            # to start a conversation — rent due on the 1st, told on the 3rd.
            #
            # ⚠️ MARKED DELIVERED ONLY IF IT WAS. If the notification could
            # not be shown the item stays queued and surfaces at the top of
            # the next conversation, which is exactly the old behaviour. A
            # reminder marked delivered but never seen is worse than a late
            # one.
            if notify.send(item["body"], subtitle=item.get("kind")):
                schedule.mark_delivered([item["id"]])
                announced += 1
        if announced and self.verbose:
            print(f"  [announce] raised {announced} item(s)", file=sys.stderr)
        return announced

    def _maybe_inquire(self):
        """Look into one of the user's topics, while nobody is talking.

        Ordering on idle is: study what is already queued, then ask a new
        question, then reindex, then (elsewhere) free-dream. Absorbing beats
        acquiring — queued material that has not been imprinted cannot be
        recalled at all, whereas one more unasked question costs nothing.
        """
        if self.asleep or self.lm is None:
            return None
        # Absorbing beats acquiring — but not indefinitely. See
        # C.INQUIRE_MAX_SILENCE_H: a big enough document used to turn this
        # subsystem off for months, which is not a priority, it is a mute.
        if study.status().get("pending", 0):
            silent_h = (time.time() - self._last_inquiry) / 3600.0
            if silent_h < C.INQUIRE_MAX_SILENCE_H:
                return None
        if inquiry.status().get("open", 0) == 0:
            return None
        with self.field():
            out = inquiry.step(self.lm, self.tok, self.llm,
                               verbose=self.verbose)
        self._last_inquiry = time.time()
        if out:
            self.stats["asked"] = self.stats.get("asked", 0) + 1
            self.telemetry.publish(stage="idle", asked=out.get("topic"),
                                   queued=out.get("queued"))
        return out

    def _maybe_fetch_world(self):
        """Read the feeds, so the news is already there when it is asked for.

        ⚠️ THE ONE THING MATI DOES ON THE NETWORK UNASKED, so it gets its own
        switch. The README now lists every connection and says this one can be
        turned off; a promise in the README that has no setting behind it is
        the overclaiming the release plan warns about, one layer down.
        `idle_work=False` already stops it with everything else — this is the
        narrower control for someone who wants the quiet work and not the
        feeds.

        ⚠️ ON by default, and that is defensible precisely because it sends
        NOTHING about the person: a GET to four public feeds with a fixed
        User-Agent. The thing being defaulted on is a download, not a
        disclosure.
        """
        try:
            if C._settings().get("headlines") is False:
                return 0
        except Exception:
            pass
        return self._fetch_world_now()

    def _fetch_world_now(self):
        """The fetch itself. Split out so the gate above reads as a gate.

        the user: *"if I ask 'Have you heard any interesting news?'
        I shouldn't have to wait for Mati to search for news, it already
        looked it up overnight, unless there was no interesting news found
        overnight."*

        ⚠️ `world.fetch_all` HAD NO CALLER ANYWHERE. Its own docstring says
        "For sleep and idle, never a turn" and nothing ever ran it — so the
        store held news from ****, fifteen days stale, from a
        one-off manual fetch. `expire()` runs inside `fetch_all`, so nothing
        was being aged out either.

        ⚠️ IT RUNS WHILE RESTING, UNLIKE THE OTHER OUTWARD WORK. Studying and
        inquiring stop overnight because they are acquisition that competes
        with the person. Reading a feed is seconds of network and no field
        work at all, and the whole point is that it has happened BEFORE
        morning. This is the first instance of the rule the user stated:
        a prediction should cause PREPARATION, not an announcement.
        """
        now = time.time()
        if now - getattr(self, "_world_at", 0) < C.WORLD_EVERY_SECONDS:
            return False
        self._world_at = now
        try:
            from . import world
            out = world.fetch_all()
        except Exception as e:                               # noqa: BLE001
            if self.verbose:
                print(f"  [world] fetch skipped: {e}", file=sys.stderr)
            return False
        if self.verbose:
            print(f"  [world] {out.get('kept')} new from "
                  f"{out.get('feeds')} feeds", file=sys.stderr)
        return bool(out.get("kept"))

    def _maybe_predict(self):
        """Resolve anything due, and record what the record now justifies.

        Cheap — SQL over timestamps, no field and no model. Runs on idle for
        tidiness rather than cost.

        ⚠️ NOTHING CONSULTS THE RESULT YET. `predict.may_influence()` gates
        that, and it is False until enough predictions have resolved with a
        good enough hit rate. Until then this only builds the record that will
        decide whether Mati can predict anything at all. :
        the store supports zero predictions, and that is the correct output.
        """
        try:
            decided = predict.resolve()
            made = []
            for p in predict.propose():
                made.append(predict.record(p))
            if decided or made:
                self.telemetry.publish(
                    stage="idle", predictions_resolved=len(decided),
                    predictions_made=len(made), predict_score=predict.score())
            return bool(decided or made)
        except Exception as e:              # a guess must never break a turn
            self.telemetry.publish(stage="idle", predict_error=repr(e))
            return False

    def _maybe_reindex(self):
        """Batch stale addresses back into the current space, during quiet time.

        Not per-turn: it costs ~1.9s per note and grows with the corpus, so
        every turn would pay ~29s of machine time for freshness the next turn
        may not need. Between batches it checks for a waiting turn and stops,
        which bounds how long an arriving message can be blocked.
        """
        if self.asleep or self.lm is None:
            return                          # not ours to touch while asleep
        if time.time() - self.last_activity < C.IDLE_SECONDS:
            return
        if self.contended or not self.work.empty():
            return
        n_total = 0
        while not self._stop.is_set() and not self.contended and self.work.empty():
            with self.field():
                n = bridge.reindex(self.lm, self.tok, limit=C.REINDEX_BATCH,
                                   should_stop=lambda: self.contended)
            if not n:
                break
            n_total += n
            # The heartbeat has to advance DURING long work, not only between
            # passes. A full reindex holds the worker for minutes, and a
            # liveness figure that climbs the whole time reports a healthy
            # worker as a hung one — the exact misdiagnosis this field exists
            # to prevent. `worker_silent_for` means "not at a pass boundary",
            # and is only alarming when the stage is idle as well.
            self.worker_beat = time.time()
            self.telemetry.publish(stage="worker: reindex", reindexed=n_total)
            if n < C.REINDEX_BATCH and not self.contended:
                break                              # nothing stale left
            if self.contended:
                break                              # a turn is waiting — go now
        if n_total:
            with self.field():
                # ⚠️ RE-CHECK AFTER TAKING THE LOCK. `/sleep` sets `self.lm` to
                # None to hand the field over, and it can land between the
                # batch loop above and this save — , where
                # it raised AttributeError on None.mf. The addresses are
                # already committed by `bridge.reindex`; only persisting θ₀ is
                # skipped here, and sleep persists it anyway.
                if self.asleep or self.lm is None:
                    return
                bridge.save_notes_field(self.lm)
            self.stats["reindexed"] += n_total
            self.stats["last_reindex"] = time.time()
            self.telemetry.publish(stage="idle", reindexed_total=n_total)
            # Republish the staleness figures. `check_addresses` used to run
            # ONLY at startup and wake, so `/addresses` and the terminal's FIELD
            # panel kept showing the startup snapshot while the worker quietly
            # repaired hundreds of addresses behind them — a monitor that
            # disagreed with `/queue` and `/health` and was simply frozen.
            # It is a plain SELECT over feat_gs, no field forwards, so it is
            # cheap enough to run after every pass.
            self.check_addresses("after reindex")
            if self.verbose:
                print(f"  [worker] reindexed {n_total} on idle", file=sys.stderr)
        # Whether or not anything was stale, the pass counts as activity —
        # otherwise an empty corpus re-enters this every second forever.
        self.last_activity = time.time()

    # ── read-only views ──────────────────────────────────────────────────────

    def episodes(self, limit=50):
        conn = sqlite3.connect(N.NOTES_DB_PATH)
        try:
            bridge._ensure_schema(conn)
            rows = conn.execute(
                "SELECT note_id, body, source, gs_std, wall_created "
                "FROM notes ORDER BY note_id DESC LIMIT ?", (limit,)).fetchall()
        finally:
            conn.close()
        return [{"note_id": r[0], "body": bridge.vault.open_body(r[1]),
                 "source": r[2],
                 "gs_std": r[3], "created": r[4]} for r in rows]

    def n_episodes(self):
        if not os.path.exists(N.NOTES_DB_PATH):
            return 0
        conn = sqlite3.connect(N.NOTES_DB_PATH)
        try:
            bridge._ensure_schema(conn)
            return conn.execute("SELECT count(*) FROM notes").fetchone()[0]
        finally:
            conn.close()

    def health(self):
        return {
            "ready": True,
            "asleep": self.asleep,
            "slow": self.slow,
            "holds_field": not self.asleep and self.lm is not None,
            # ⚠️ REPORTED, BECAUSE A LOCKED VAULT IS NOW A RUNNING STATE
            # RATHER THAN A CRASH. Mati starts and answers with its sealed
            # memories unreadable, which looks healthy on every other measure
            # — exactly the invisible degradation this project keeps paying
            # for. `memories_locked` is how someone finds out without
            # guessing why recall is empty.
            # ⚠️ `_vault_state()` ALREADY EXISTS AND NEVER PROMPTS — it is a
            # status read on a route that polls, so asking for Touch ID here
            # would put a fingerprint prompt on a timer. True = unlocked,
            # False = locked, None = no vault, so `is False` is the only
            # correct test: `not None` would report an unencrypted install as
            # locked.
            "memories_locked": _vault_state() is False,
            "provider": self.provider,
            "llm": self._llm_detail,
            "uptime_seconds": round(time.time() - self.started, 1),
            # Stale means: the files changed after this process imported them,
            # so what is answering is not what is on disk. See `_code_stamp`.
            "code_stale": _code_stamp() > _CODE_AT_START + 1.0,
            # ⚠️ WITHOUT THIS, AN AMNESIAC SERVER LOOKS HEALTHY. Bodies are
            # sealed; a process that cannot unlock reads none of them, and
            # every other field here would still say it is fine. Found
            # found when a server started in the background died with
            # "Touch ID declined or unavailable" — Touch ID needs someone
            # present, and nothing started without a person can count on it.
            # That is exactly the condition a restart creates.
            "vault_unlocked": _vault_state(),
            "code_age_hours": round(
                max(0.0, time.time() - _CODE_AT_START) / 3600.0, 1),
            "episodes": self.n_episodes(),
            "sessions": len(self.sessions),
            "queue_depth": self.work.qsize(),
            # ⚠️ A dead worker is invisible without this: `ready`, `asleep`
            # and `holds_field` all keep saying the right thing while no
            # background work happens at all.
            # Demand data, as COMPUTED figures — the form self-diagnosis may
            # speak but never store. `self_knowledge` carries only the
            # qualitative version.
            # What is being read, for the terminal's TRAINING panel. Only
            # documents with work left — a finished one is not progress.
            "reading": [d for d in (study.status().get("documents") or [])
                        if d.get("pending")],
            "needed": bridge.hit_summary(days=14),
            "working_set": bridge.tier_counts()[0],
            "archived": bridge.tier_counts()[1],
            # ⚠️ THE ONLY WAY TO SEE THE FREEZE FROM OUTSIDE, and the thing to
            # check before trusting any report about recall. "loaded" means the
            # persisted space came back and stored addresses are still in it;
            # "seeded" after a restart that should have loaded means the file
            # was missing or corrupt, and addressing silently fell back to the
            # live field — degraded in exactly the way that looks like nothing.
            # `address_drift` is how far the live field has moved from the
            # space, in RADIANS: it should grow through the day and drop to 0
            # at `rest()`. Pinned at 0 across a day of turns means promotion
            # stopped happening and the day's learning is never reaching what
            # Mati recalls through — the one failure mode of this design that
            # does not announce itself.
            "address_space": getattr(self.lm, "_address_space", None),
            "address_frozen": self.lm is not None and bridge.thawed(self.lm),
            "address_drift": (round(bridge.address_drift(self.lm), 6)
                              if self.lm is not None else None),
            # How much of the RAG-ready year can be found by its EXACT words
            # rather than only by Mati's gist of it. Climbs as the idle
            # worker warms segments; `warm` well below `total` long after a
            # restart means warming is starved — check `_verbatim_segment`.
            "verbatim_warm": self._verbatim_warm_field(),
            "verbatim_budget": self._verbatim_budget_field(),
            "study_priority_left": max(
                0, round(self.study_priority_until - time.time())),
            "dreams": self.stats.get("dreams", 0),
            "since_dream": (round(time.time() - self.last_dream, 1)
                            if self.last_dream else None),
            # ⚠️ WATCH THIS. Turns hand their imprint to the worker now; if
            # this number climbs and does not fall, the field is drifting
            # behind the conversation and nothing else will say so.
            "waiting_to_imprint": len(self._to_imprint),
            "worker_alive": self._worker.is_alive(),
            "worker_silent_for": round(time.time() - self.worker_beat, 1),
            "worker_errors": self.stats.get("worker_errors", 0),
            "last_worker_error": self.last_worker_error,
            "waiting_to_say": len(schedule.pending(include_held=True)),
            "when": schedule.sense_line(),
            # ⚠️ NOT `.locked()` — RLock does not have it, and the field
            # lock became an RLock to stop the worker deadlocking on itself.
            # `_waiting` already counts threads BLOCKED on the field, which is
            # what "busy" was reporting anyway: someone is waiting for it.
            "busy": self.contended,
            "idle_seconds": round(time.time() - self.last_activity, 1),
            # Since the PERSON last spoke, and whether that is long enough that
            # Mati considers itself asleep and will do the expensive work.
            "since_turn": round(time.time() - self.last_turn, 1),
            "sleeping": (time.time() - self.last_turn) >= C.SLEEP_AFTER_SECONDS,
            "stats": dict(self.stats),
        }

    def shutdown(self):
        self._stop.set()


# ─────────────────────────────────────────────────────────────────────────────
# HTTP
# ─────────────────────────────────────────────────────────────────────────────

# Routes that belong to the pro terminal rather than to conversation.
# ⚠️ CHAT IS NOT IN HERE and must never be: `/chat`, `/health`, `/playing`,
# `/stations`, `/silence`, `/interrupt` are what the app itself speaks, and
# gating them would switch Mati off rather than the terminal.
# What the app may read and write. ⚠️ A CLOSED LIST: a settings route that
# accepts any key lets the page write config the measurements depend on.
# ⚠️ `assistant_name` IS HERE SO THE APP CAN SET IT. The mechanism
# existed and nothing could reach it — settable only by hand-editing
# the settings JSON, which is not a choice anyone makes.
_APP_SETTINGS = ("terminal_enabled", "notify_on_reply", "voice_on",
                 "assistant_name", "headlines", "voice_rate", "local_model",
                 "voice_name", "avatar", "reply_length")

# ⚠️ `/sleep`, `/wake` and `/rest` LEFT THIS SET.
# They are things a person does to their own Mati — "go to sleep
# now", "wake up" — not field mechanics. The gate answers "should
# a stranger reach this over HTTP"; it had been answering "is this
# a user action" as well, which put sleeping in the same bucket as
# packing the store for transfer.
_TERMINAL_ONLY = frozenset({
    # ⚠️ THE LIBRARY LEFT THIS SET. Managing what Mati has read
    # — adding, archiving, reshelving — is a person tending their own
    # assistant's specialisation, not an inspection of the mechanism. It has
    # its own drawer in the app now, and gating it behind the terminal switch
    # would hide an ordinary task behind a developer control. The gate still
    # answers "should a stranger reach this over HTTP"; `_terminal_on` is not
    # the authority for "is this a user action".
    #
    # ⚠️ `/library/deshelve` goes with them DESPITE DELETING, because it
    # already refuses without an explicit confirm carrying the count. The
    # protection is the confirmation, not the obscurity.
    "/reindex", "/dream", "/imprint", "/study_now", "/study",
    "/episodes", "/pack", "/release",
})


def _terminal_on():
    """Has the person switched the terminal on? Default NO.

    ⚠️ READ FRESH EVERY TIME, not cached at import. Toggling it in the app
    must take effect without restarting the server, or the setting is a lie
    until the next launch.
    """
    try:
        from . import config as _c
        return bool(_c._settings().get("terminal_enabled", False))
    except Exception:
        return False


class Handler(BaseHTTPRequestHandler):
    service = None                     # set by serve()
    protocol_version = "HTTP/1.1"
    server_version = "Mati"

    # ── plumbing ─────────────────────────────────────────────────────────────

    def log_message(self, fmt, *args):
        if self.service and self.service.verbose:
            sys.stderr.write(f"  [http] {fmt % args}\n")

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode() or "{}")
        except json.JSONDecodeError:
            return None

    def _json(self, obj, status=200):
        payload = json.dumps(obj, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    # ── Serving the app ────────────────────────────────────────────────
    # ⚠️ THE ONLY REASON THIS EXISTS. A page opened from file:// is a
    # different origin, so every fetch to 127.0.0.1 is blocked — no
    # auto-wake, no conversation, nothing. Serving the app from the same
    # origin as the API removes the problem rather than working around it
    # with CORS headers, which would widen who may call this server.
    _TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css",
              ".js": "text/javascript", ".json": "application/json",
              ".svg": "image/svg+xml", ".png": "image/png",
              ".jpg": "image/jpeg", ".webp": "image/webp",
              ".mp4": "video/mp4", ".webmanifest": "application/manifest+json",
              ".woff2": "font/woff2", ".ico": "image/x-icon"}

    def _serve_app(self, path):
        """Serve one file from APP_DIR. False if it is not ours to serve."""
        import mimetypes  # noqa: F401  (kept for future types)
        root = os.path.realpath(C.APP_DIR)
        rel = "index.html" if path in ("", "/") else path.lstrip("/")
        target = os.path.realpath(os.path.join(root, rel))
        # ⚠️ PATH TRAVERSAL. `..` in a URL must not reach the notes DB, the
        # vault descriptor or θ₀ — all of which sit one directory up. The
        # check is on the RESOLVED path, because a symlink inside the app
        # folder would defeat a check on the string.
        if not (target == root or target.startswith(root + os.sep)):
            return False
        if not os.path.isfile(target):
            return False
        ext = os.path.splitext(target)[1].lower()
        ctype = self._TYPES.get(ext)
        if not ctype:
            return False              # only known kinds, never arbitrary files
        try:
            with open(target, "rb") as fh:
                body = fh.read()
        except OSError:
            return False
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # The app is local and changes while it is being built.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)
        return True

    def _sse_open(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

    def _sse(self, event, data):
        """One SSE frame, flushed — a chunk the client cannot see is not a stream."""
        self.wfile.write(f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"
                         .encode())
        self.wfile.flush()

    # ── routes ───────────────────────────────────────────────────────────────

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/") or "/"
        svc = self.service
        # ⚠️ `/` NOW SERVES THE APP, AND `/health` STILL ANSWERS JSON. The two
        # were the same endpoint; `exclusive.holds_field` and every probe use
        # `/health` explicitly, so nothing that checks whether Mati is up is
        # affected. If the app folder is absent, `/` falls through to health
        # exactly as before — an install without the front end still works.
        if path == "/health":
            return self._json(svc.health())
        if path in ("/", "/index.html") and self._serve_app("/"):
            return
        if path == "/":
            return self._json(svc.health())
        if path.startswith("/app/") and self._serve_app(path[4:]):
            return
        if "." in path.rsplit("/", 1)[-1] and self._serve_app(path):
            return
        # ⚠️ ITS OWN ROUTE, NOT A CHAT TURN. The indicator polls this; routing
        # it through `/chat` would spend a turn, an episode and the field on
        # the question "is sound coming out".
        # ⚠️ THE TERMINAL IS OFF UNTIL IT IS SWITCHED ON, AND THE GATE IS
        # HERE RATHER THAN IN THE APP. The user: *"the terminal becomes only
        # available if the user switches on the setting in the Mati app."*
        #
        # The app's "Show terminal" toggle only ever hid a BUTTON — every
        # route stayed reachable with curl. Hiding a door is not locking it,
        # and these are the routes that reindex the field, imprint text, and
        # hand the field over. Loopback-only is not an argument: anything
        # running as this user could reach them.
        if path in _TERMINAL_ONLY and not _terminal_on():
            return self._json(
                {"error": "the terminal is switched off",
                 "how": "Settings → Show terminal, in the Mati app"}, 403)
        if path == "/models":
            # ⚠️ WHAT OLLAMA ACTUALLY HAS, not a list we believe in. Offering
            # a model that is not pulled produces a failure at the first turn
            # rather than at the moment of choosing.
            try:
                from . import machine as _m
                ok, names = _m.ollama_ready()
                from . import config as _c, setup as _s
                # ⚠️ SIZES AND A CEILING, BECAUSE THE FIELD RUNS ALONGSIDE.
                # The model is not the only thing in memory: the field
                # simulation sits beside it, so "will this fit" is not the
                # same question Ollama answers. On 8 GB a 3B is the
                # comfortable ceiling, and a 7B will swap rather than fail —
                # which looks like Mati becoming unbearably slow rather than
                # like a bad choice.
                #
                # ⚠️ NO FIELD IS INVALIDATED BY SWITCHING. Addresses come from
                # `bridge.address_vector(lm, tok, …)` — the FIELD's own lm and
                # tokenizer — so the Ollama model is the language faculty
                # only. Nothing needs reindexing and `restart` packs on the
                # way out, so no separate pack is needed either.
                sizes = {}
                try:
                    pulled, _err = _s._pulled_models()
                    sizes = {n: round((sz or 0) / 1e9, 1) for n, sz in pulled or []}
                except Exception:
                    pass
                prof = _m.profile()
                return self._json({"models": names, "reachable": ok,
                                   "current": _c.LOCAL_MODEL, "sizes": sizes,
                                   "ram_gb": prof.get("ram_gb"),
                                   "comfortable_gb": round(
                                       max(1.0, (prof.get("ram_gb") or 8) * 0.3), 1)})
            except Exception as e:                           # noqa: BLE001
                return self._json({"models": [], "error": str(e)})
        if path == "/voices":
            # ⚠️ THE MACHINE'S OWN LIST, not a hardcoded one. macOS ships a
            # different set per install and per language; a fixed list would
            # offer voices that are not there.
            try:
                from . import speech as _sp
                return self._json({"voices": _sp.voices()})
            except Exception as e:                           # noqa: BLE001
                return self._json({"voices": [], "error": str(e)})
        if path == "/settings":
            # ⚠️ THE APP-VISIBLE SETTINGS ONLY. Not the field tunables —
            # gate thresholds, imprint strength and reindex cadence are
            # measured and load-bearing, not preferences.
            try:
                from . import config as _c
                st = _c._settings()
                return self._json({k: st.get(k) for k in _APP_SETTINGS})
            except Exception as e:                           # noqa: BLE001
                return self._json({"error": str(e)}, 500)
        if path == "/library":
            # ⚠️ A CATALOGUE, NEVER A READER. The spec in
            # By design: studied verbatim is Mati's private
            # reading and is not browsable — Mati may quote a line in a
            # reply, but a store of book text with a text browser attached is
            # an ebook reader carrying someone's library, which is a
            # licensing problem and a product problem at once.
            try:
                from . import library, study
                return self._json({
                    "works": library.catalogue(),
                    "reading": study.status(),
                })
            except Exception as e:                           # noqa: BLE001
                return self._json({"works": [], "error": str(e)})
        if path == "/encyclopedia":
            # ⚠️ NOT TERMINAL-ONLY, for the library's reason. Seeing which
            # subject references Mati has is managing its specialisation,
            # which is the library's job description — not an inspection of
            # the mechanism. It is also read-only and reaches nothing but a
            # socket on this machine.
            try:
                from . import encyclopedia as _enc
                st = _enc.status()
                return self._json({
                    "connected": st["connected"],
                    "serving": st["serving"],
                    "binary": bool(st["binary"]),
                    "zim_dir": st["zim_dir"],
                    "zims": [{"name": z["name"], "mb": round(z["bytes"]/1e6)}
                             for z in st["zims"]],
                    "mb": round(st["bytes"] / 1e6),
                    "says": _enc.describe(),
                })
            except Exception as e:                           # noqa: BLE001
                return self._json({"connected": False, "error": str(e)})
        if path == "/stations":
            # ⚠️ READ-ONLY, AND IT REPORTS THE HELD SEARCH — not a new one.
            # The user: *"I would like to see a list of the five stations under
            # the avatar. That way I can change stations as I see fit."*
            # `radio.remember_found` already keeps the last search so that
            # "next station" and "the third one" work by voice; this exposes
            # the same list to the page, so the buttons and the words are
            # looking at ONE thing. A separate list for the UI would drift
            # from what "next station" means.
            try:
                from . import media, radio
                # ⚠️ PRESETS TRAVEL WITH IT. The user: *"The presets can show
                # up there when the radio is playing."* Which is the car-radio
                # logic: a found list matters just after a search, and once
                # something is playing the useful buttons are your own. One
                # request returns both so the page never has to decide which
                # to fetch — and never shows a stale half.
                from . import presets as _pre
                return self._json({
                    "stations": radio.listing(limit=5),
                    "presets": [{"slot": p["slot"], "name": p["name"]}
                                for p in (_pre.all_presets() or [])],
                    "playing": media.playing_now(),
                    "remaining": radio.remaining(),
                })
            except Exception as e:                           # noqa: BLE001
                return self._json({"stations": [], "error": str(e)})
        if path == "/playing":
            try:
                from . import media
                return self._json({"playing": media.playing_now()})
            except Exception as e:                           # noqa: BLE001
                return self._json({"playing": None, "error": str(e)})
        if path == "/state":
            return self._json(svc.telemetry.snapshot())
        if path == "/episodes":
            return self._json({"episodes": svc.episodes()})
        if path == "/study":
            return self._json(study.status())
        if path == "/signal":
            snap = svc.telemetry.snapshot()
            return self._json({
                "signal": snap.get("signal"),
                "describe": sig.describe(),
            })
        if path == "/pending":
            return self._json({"pending": schedule.pending(include_held=True),
                               "consent": schedule.consent(),
                               "when": schedule.sense_line()})
        if path == "/sessions":
            return self._json({"sessions": sorted(svc.sessions)})
        if path.startswith("/sessions/"):
            sid = path[len("/sessions/"):]
            return self._json({"session": sid, "history": svc.history(sid)})
        if path == "/events":
            return self._events()
        return self._json({"error": "not found", "path": path}, 404)

    def do_DELETE(self):
        path = self.path.split("?")[0].rstrip("/")
        if path.startswith("/sessions/"):
            sid = path[len("/sessions/"):]
            return self._json({"cleared": self.service.clear_session(sid)})
        return self._json({"error": "not found", "path": path}, 404)

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/") or "/"
        body = self._body()
        if body is None:
            return self._json({"error": "invalid JSON"}, 400)
        svc = self.service

        if path == "/slow":
            # ⚠️ NEVER CALL `_body()` TWICE. `do_POST` has already read the
            # request body above; a second call blocks on `rfile.read(n)`
            # waiting for bytes that were consumed, and the route simply hangs
            # until the client gives up. This one did — `/sleep`, which is how
            # slow mode is reached, would sit there until it timed out.
            body = body or {}
            svc.slow = bool(body.get("on", True))
            activity.log(activity.REST if svc.slow else activity.STIR)
            if svc.slow:
                from . import rapport
                try:
                    rapport.build(verbose=svc.verbose)
                except Exception:
                    pass          # resting must not fail on housekeeping
            return self._json({"slow": svc.slow, "asleep": svc.asleep})
        # ⚠️ THE STOP BUTTON, AND IT MUST NOT GO THROUGH A TURN. The user had the
        # radio on with no way to silence it, and the chat route took 16.5s.
        # A stop that needs a sentence parsed, a field read and an episode
        # written is not a stop button. This does one thing.
        if path == "/settings":
            body = body or {}
            bad = [k for k in body if k not in _APP_SETTINGS]
            if bad:
                return self._json({"error": f"not a setting: {bad}"}, 400)
            try:
                from . import config as _c
                _c.save_settings(**body)
                # ⚠️ TURNING THE VOICE OFF MUST TAKE EFFECT NOW. The setting is
                # read at the START of a reply, so without this a person who
                # flips the switch mid-sentence keeps being talked at until the
                # sentence ends — and the radio stays ducked with it. `stop()`
                # restores the volume too.
                if "voice_on" in body and not body.get("voice_on"):
                    try:
                        from . import speech as _sp
                        # ⚠️ `drain()`, NOT `stop()`. `stop` silences what is
                        # speaking and leaves the QUEUE, so turning the voice
                        # off mid-reply would cut one sentence and then carry
                        # straight on with the next — which reads as the
                        # switch not working.
                        _sp.drain()
                    except Exception:
                        pass
                st = _c._settings()
                return self._json({k: st.get(k) for k in _APP_SETTINGS})
            except Exception as e:                           # noqa: BLE001
                return self._json({"error": str(e)}, 500)
        if path == "/library/read":
            # ⚠️ READS A PATH ON THE USER'S MACHINE. Loopback-only, but the
            # route still refuses what it cannot read as plain text rather
            # than handing an arbitrary path to `study.read_file`.
            body = body or {}
            src = (body.get("path") or "").strip()
            # ⚠️ TEXT OR A PATH, BECAUSE A BROWSER CANNOT GIVE A PATH. A file
            # input hands JavaScript the CONTENT, never a usable filesystem
            # location — so the app could not add a reading at all while this
            # route took only `path`. Accepting text is also the narrower
            # power: no filesystem access, nothing to traverse.
            raw = body.get("text")
            title = (body.get("title") or "").strip()
            if raw is not None:
                from . import study as _st
                # ⚠️ THE BROWSER PATH NEVER TOUCHES `read_file`, so it needs
                # the same cleaning. A file input hands JavaScript the raw
                # bytes — an RTF arrives here as `{\rtf1\ansi…` with no
                # suffix to notice, and pasted text can carry zero-width
                # characters from wherever it was copied.
                text = _st.clean_text(str(raw))
                if not text.strip():
                    return self._json({"error": "nothing to read"}, 400)
                # ⚠️ REFUSE MARKUP RATHER THAN STUDY IT. The browser cannot
                # run `textutil`, so an RTF pasted or dropped here cannot be
                # converted — and studying it would succeed, queue thousands
                # of chunks, and imprint font tables for hours. Saying what
                # to do is more use than a generic failure.
                head = text.lstrip()[:16]
                if head.startswith("{\\rtf"):
                    return self._json({"error":
                        "that is RTF, not text. Convert it first:  "
                        "textutil -convert txt <file>"}, 400)
                origin = title or "pasted text"
                try:
                    from . import study
                    with svc.field():
                        q = study.enqueue(text, origin, svc.tok)
                    q["minutes"] = round(q["chunks"] * 30 / 60.0)
                    q["origin"] = origin
                    return self._json(q)
                except Exception as e:                       # noqa: BLE001
                    return self._json({"error": str(e)}, 400)
            if not src or not os.path.exists(os.path.expanduser(src)):
                return self._json({"error": f"no file at {src!r}"}, 400)
            try:
                from . import study
                text = study.read_file(os.path.expanduser(src))
                with svc.field():
                    q = study.enqueue(text, os.path.basename(src), svc.tok)
                # ⚠️ THE COST, BEFORE ANY OF IT RUNS. ~30s a chunk; a book is
                # hours. A person adding one deserves the number first.
                q["minutes"] = round(q["chunks"] * 30 / 60.0)
                return self._json(q)
            except Exception as e:                           # noqa: BLE001
                return self._json({"error": str(e)}, 400)
        if path == "/library/reshelf":
            # ⚠️ SHELVING HAD NO INVERSE, which made "archive instead of
            # delete" a half-promise. The decision was that an archived work
            # stays in the house — but with no way back it was off the shelf
            # permanently, which is a delete that merely keeps the rows.
            #
            # ⚠️ `archived = NULL` IS THE WHOLE OPERATION, and it is why
            # archiving was the right choice over deleting: nothing was lost,
            # so nothing has to be rebuilt. The passages return to the
            # working set and `rebalance_tiers` re-addresses them in its own
            # time.
            body = body or {}
            title = (body.get("title") or "").strip()
            if not title:
                return self._json({"error": "which work?"}, 400)
            try:
                import sqlite3 as _sq
                from . import bridge
                conn = _sq.connect(bridge.NOTES_DB_PATH)
                try:
                    bridge._ensure_schema(conn)
                    n = conn.execute(
                        "UPDATE notes SET shelved = NULL "
                        "WHERE shelved IS NOT NULL AND source LIKE ?",
                        (f"studied:%{title}%",)).rowcount
                    conn.commit()
                finally:
                    conn.close()
                return self._json({"title": title, "restored": n})
            except Exception as e:                           # noqa: BLE001
                return self._json({"error": str(e)}, 500)
        if path == "/library/deshelve":
            # ⚠️ THIS ONE REALLY DELETES, and it is the second real delete in
            # Mati — `forget()` is the other. Archiving exists precisely so
            # that taking a book off the shelf is NOT this; deshelving is for
            # when the person wants the text gone, not put away.
            #
            # ⚠️ IT COUNTS FIRST AND REFUSES WITHOUT CONFIRMATION. A delete
            # that reports what it did AFTER doing it is no use to someone who
            # mistyped a title — `studied:%X%` is a LIKE, so a short title can
            # match more works than intended. The count comes back, the person
            # sends it again with `confirm`, and only then does anything go.
            body = body or {}
            title = (body.get("title") or "").strip()
            if not title:
                return self._json({"error": "which work?"}, 400)
            try:
                import sqlite3 as _sq
                from . import bridge
                conn = _sq.connect(bridge.NOTES_DB_PATH)
                try:
                    bridge._ensure_schema(conn)
                    like = f"studied:%{title}%"
                    n = conn.execute(
                        "SELECT COUNT(*) FROM notes WHERE source LIKE ?",
                        (like,)).fetchone()[0]
                    works = [r[0] for r in conn.execute(
                        "SELECT DISTINCT source FROM notes WHERE source LIKE ?",
                        (like,))]
                    if not n:
                        return self._json({"error": f"nothing matches {title!r}"},
                                          404)
                    if not body.get("confirm"):
                        return self._json({
                            "title": title, "would_delete": n,
                            "works": works, "confirm_required": True,
                            "detail": (f"{n} passages across {len(works)} "
                                       f"work(s) would be deleted for good. "
                                       f"Send again with confirm to do it.")})
                    conn.execute("DELETE FROM notes WHERE source LIKE ?", (like,))
                    conn.commit()
                finally:
                    conn.close()
                return self._json({"title": title, "deleted": n, "works": works})
            except Exception as e:                           # noqa: BLE001
                return self._json({"error": str(e)}, 500)
        if path == "/library/archive":
            # ⚠️ ARCHIVES, DOES NOT DELETE, and is not called `forget` for
            # that reason — `forget()` is Mati's one real delete and the
            # name is load-bearing. An archived work is still findable by its
            # words; this takes it off the shelf, not out of the house.
            body = body or {}
            title = (body.get("title") or "").strip()
            if not title:
                return self._json({"error": "which work?"}, 400)
            try:
                import sqlite3 as _sq
                from . import bridge, library
                conn = _sq.connect(bridge.NOTES_DB_PATH)
                try:
                    bridge._ensure_schema(conn)
                    n = conn.execute(
                        "UPDATE notes SET shelved = ? WHERE shelved IS NULL "
                        "AND source LIKE ?",
                        (time.time(), f"studied:%{title}%")).rowcount
                    conn.commit()
                finally:
                    conn.close()
                return self._json({"title": title, "archived": n})
            except Exception as e:                           # noqa: BLE001
                return self._json({"error": str(e)}, 500)
        if path == "/station":
            # Switch to the nth of the held list. ⚠️ GOES THROUGH THE SAME
            # `nth_found` THAT "THE THIRD ONE" USES, so a button press and a
            # spoken request are the same operation — not two paths that can
            # disagree about which station is third.
            body = body or {}
            try:
                n = int(body.get("n") or 0)
            except (TypeError, ValueError):
                n = 0
            if n < 1:
                return self._json({"error": "which one? n starts at 1"}, 400)
            # ⚠️ A PRESET AND A FOUND-LIST ROW ARE DIFFERENT THINGS with the
            # same shaped request, so the caller says which. Both go through
            # `media.answer`, the path the spoken forms use.
            if (body.get("kind") or "found") == "preset":
                try:
                    from . import media
                    # ⚠️ `slot`, NOT a verb. In `media.answer` a `slot` key
                    # outranks everything and goes straight to
                    # `_play_preset` — "play 3" is a button, not a search for
                    # a station called 3. A `verb: "preset"` would be ignored
                    # and reads as if it were doing the work.
                    # ⚠️ `verb: "play"` IS REQUIRED, and omitting it returned
                    # None silently. `media.answer` has `if verb != "play":
                    # return None` BEFORE it reads `slot`, so a bare
                    # {"slot": n} fell straight through — the button reported
                    # success with an empty `said` and nothing played. A
                    # preset press IS a play that names a button.
                    return self._json(
                        {"said": media.answer({"verb": "play", "slot": n}),
                         "playing": media.playing_now()})
                except Exception as e:                       # noqa: BLE001
                    return self._json({"error": str(e)}, 500)
            try:
                # ⚠️ `media.act`, NOT `radio.nth_found`. `nth_found` only
                # RETURNS the station — it does not play it, and a first cut
                # here called it and reported success while nothing started.
                # Routing the button through the same `answer({"verb": "nth"})`
                # the spoken "the third one" uses means one code path, one
                # set of failure messages, and no way for a tap and a
                # sentence to disagree.
                from . import media
                said = media.answer({"verb": "nth", "n": n})
                return self._json({"said": said,
                                   "playing": media.playing_now()})
            except Exception as e:                           # noqa: BLE001
                return self._json({"error": str(e)}, 500)
        if path == "/silence":
            try:
                from . import media
                return self._json({"said": media.stop_all()})
            except Exception as e:                           # noqa: BLE001
                return self._json({"error": str(e)}, 500)
        if path == "/interrupt":
            # The person typed over the reply. `said` is what they had actually
            # been shown — the client knows it, the server does not, because a
            # stream that was abandoned mid-flight may have got further here
            # than it did on screen.
            body = body or {}
            return self._json({"interrupted": svc.interrupt(
                body.get("session") or "default", body.get("said") or "")})
        if path == "/reflect":
            # Thinking about a day, on demand. Normally this happens inside
            # `consolidate()` at sleep; the route exists so a night can be
            # exercised without waiting for one, and so a day that was missed
            # can be thought about later.
            day = (body or {}).get("day")
            try:
                from . import reflect as _R
                with svc.field():
                    wrote = _R.reflect(svc.lm, svc.tok, svc.chat_llm,
                                       day=day, verbose=svc.verbose)
                if wrote:
                    activity.log(activity.REFLECT, f"{len(wrote)} rounds")
                return self._json({"day": day, "rounds": len(wrote),
                                   "wrote": wrote})
            except Exception as e:                    # noqa: BLE001
                return self._json({"error": repr(e)}, 500)
        if path == "/rest":
            # The night: consolidate, keep the field, go on dreaming.
            return self._json(svc.rest())
        if path == "/release":
            return self._json(svc.release())
        if path == "/sleep":
            # ⚠️ The old name for the handoff, kept so an older terminal build
            # still works. New callers use /release.
            return self._json(svc.release())
        if path in ("/pack", "/shutdown"):
            # ⚠️ `/shutdown` is kept as an ALIAS so nothing in flight breaks,
            # but it never stopped anything — it packs. `/quit` is the stop.
            out = svc.pack()
            if path == "/shutdown":
                out["note"] = ("this packed the install; it did NOT stop the "
                               "process — use /stop for that")
            return self._json(out)
        if path == "/stop":
            return self._json(svc.stop())
        if path == "/restart":
            return self._json(svc.restart())
        if path == "/wake":
            return self._json(svc.wake())

        # Everything below touches the field, which Mati does not hold while
        # asleep — the pro terminal may have it.
        if path in ("/chat", "/imprint", "/dream", "/reindex", "/study"):
            try:
                svc._require_awake()
            except svc.Asleep as e:
                return self._json({"error": "asleep", "detail": str(e)}, 503)

        if path == "/chat":
            return self._chat(body)
        if path == "/imprint":
            text = (body.get("text") or "").strip()
            if not text:
                return self._json({"error": "text required"}, 400)
            try:
                with svc.field():
                    bridge.imprint_text(svc.lm, svc.tok, text,
                                        passes=body.get("passes"),
                                        strength=body.get("strength"),
                                        steps=body.get("steps"))
                    bridge.save_notes_field(svc.lm)
            except bridge.TooShortToImprint as e:
                return self._json({"error": str(e)}, 400)
            svc.last_activity = time.time()
            return self._json({"imprinted": True, "text": text})
        if path == "/dream":
            cycles = int(body.get("cycles", 1))
            # Announce it. Dreaming was the one operation that held the field
            # in silence, which is backwards: it is also the only one that
            # measurably deforms θ₀ (30 cycles erode a fresh imprint by
            # −0.0499 against −0.0245 for an untouched control). The most
            # destructive thing Mati does should be the most visible.
            svc.telemetry.publish(stage="dreaming", cycles=cycles)
            try:
                with svc.field():
                    out = bridge.dream(svc.lm, svc.tok, cycles=cycles,
                                       verbose=svc.verbose)
            finally:
                # Count it and reset the clock, so a dream you asked for and
                # a dream it took itself are the same event. Without this
                # `/dream` left `dreams` at 0 and the hourly timer untouched,
                # so the worker could dream again moments later.
                svc.stats["dreams"] = svc.stats.get("dreams", 0) + 1
                svc.last_dream = time.time()
                svc.telemetry.publish(stage="idle", dreamed=cycles)
            svc.last_activity = time.time()
            return self._json({"dreamed": cycles, "result": out})
        if path == "/study":
            # Fast: chunk, dedupe, queue, record ONE episode saying it read the
            # thing. The imprinting happens on the worker when Mati is idle —
            # a document is not something you absorb while someone waits.
            text = body.get("text")
            origin = body.get("origin") or body.get("name")
            if not text and body.get("path"):
                try:
                    text = study.read_file(body["path"])
                    origin = origin or os.path.basename(body["path"])
                except OSError as e:
                    return self._json({"error": f"cannot read: {e}"}, 400)
            if not text or not text.strip():
                return self._json({"error": "text or path required"}, 400)
            summary = study.enqueue(text, origin or "an untitled document",
                                    svc.tok)
            svc.work.put(("reading episode", lambda s=summary: bridge.remember(
                svc.lm, svc.tok,
                study.reading_episode(s["origin"], s), source="episode")))
            svc.last_activity = time.time()
            return self._json({**summary,
                               "pending": study.status()["pending"]})
        if path == "/consent":
            for k, v in (body or {}).items():
                schedule.set_consent(str(k), str(v))
            return self._json(schedule.consent())
        if path == "/remind":
            text = (body.get("text") or "").strip()
            when = (schedule.read_time(body["when"]) if body.get("when")
                    else schedule.read_time(text))
            if not text:
                return self._json({"error": "text required"}, 400)
            if when is None:
                # Refused rather than approximated — see Mati/schedule.py.
                return self._json({"error": "could not read a time",
                                   "detail": "say when, e.g. 'in 2 hours' or "
                                             "'at 3pm'"}, 400)
            sid = schedule.say_later(text, kind=schedule.REMINDER, due_at=when)
            return self._json({"id": sid, "text": text, "due_at": when})
        if path == "/study_now":
            # Reading at full speed until the queue drains or the window ends,
            # whichever comes first. Bounded because the budget exists for a
            # measured reason: a queued book silenced every other background
            # function for a night.
            minutes = float(body.get("minutes") or C.STUDY_PRIORITY_MINUTES)
            minutes = max(1.0, min(minutes, 24 * 60))
            svc.study_priority_until = time.time() + minutes * 60
            svc._save_focus()
            pending = 0
            try:
                from . import study as _study
                pending = _study.status().get("pending", 0)
            except Exception:
                pass
            return self._json({"priority_minutes": minutes,
                               "until": svc.study_priority_until,
                               "pending": pending})
        if path == "/reindex":
            with svc.field():
                n = bridge.reindex(svc.lm, svc.tok, limit=body.get("limit"))
                bridge.save_notes_field(svc.lm)
            svc.last_activity = time.time()
            return self._json({"reindexed": n})
        return self._json({"error": "not found", "path": path}, 404)

    # ── streaming chat ───────────────────────────────────────────────────────

    def _chat(self, body):
        text = (body.get("text") or "").strip()
        if not text:
            return self._json({"error": "text required"}, 400)
        session = body.get("session") or "default"
        remember = bool(body.get("remember", True))

        if not body.get("stream", True):
            try:
                reply, info = self.service.chat(session, text, remember=remember)
            except Refusal as e:
                return self._json({"error": "refused", "category": e.category}, 502)
            except LLMUnavailable as e:
                return self._json({"error": "llm unavailable", "detail": str(e)}, 503)
            return self._json({"reply": reply, "info": info, "session": session})

        self._sse_open()
        try:
            reply, info = self.service.chat(
                session, text, on_chunk=lambda p: self._sse("chunk", {"text": p}),
                remember=remember)
            # The reply is complete here; the episode write is already queued and
            # runs after this response closes. That deferral is the whole point.
            self._sse("done", {"reply": reply, "info": info, "session": session})
        except Refusal as e:
            self._sse("error", {"error": "refused", "category": e.category,
                                "explanation": e.explanation})
        except LLMUnavailable as e:
            self._sse("error", {"error": "llm unavailable", "detail": str(e)})
        except BrokenPipeError:
            pass                                   # client hung up mid-reply
        except Exception as e:
            self._sse("error", {"error": f"{type(e).__name__}: {e}"})

    def _events(self):
        """Live telemetry for the monitoring app. Never touches the field."""
        q = self.service.telemetry.subscribe()
        self._sse_open()
        try:
            self._sse("state", self.service.telemetry.snapshot())
            while True:
                try:
                    self._sse("state", q.get(timeout=15.0))
                except queue.Empty:
                    self._sse("ping", {"t": time.time()})
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self.service.telemetry.unsubscribe(q)


# ─────────────────────────────────────────────────────────────────────────────

class _RunLoopServer(ThreadingHTTPServer):
    """The HTTP server, plus the macOS main run loop.

    ⚠️ THIS IS WHY RADIO HAD NO SOUND. AVFoundation does its asset loading
    against the **main thread's** run loop, and this process never ran one —
    `serve_forever` is a selector loop. Measured  on a 320kbps
    stream:

        no run loop pumped    player.status=0  item.status=0   forever, no error
        main loop pumped      player.status=1  item.status=1   ready at 5.5s

    Status 0 is `Unknown`. The player accepted `play()`, reported `rate=1.0`,
    and sat silent indefinitely — and because `radio.playing()` tested `rate`,
    which is the REQUESTED rate, Mati reported the station as playing. Silence
    with a confirmation on top.

    ⚠️ `service_actions` IS THE RIGHT HOOK, AND A SEPARATE THREAD IS THE WRONG
    FIX. My first attempt ran a dedicated audio thread with its own loop,
    reasoning that run loops are per-thread — true, and not the point. Every
    station still timed out. `serve_forever` calls `service_actions()` once per
    `poll_interval` pass, on the main thread, which is exactly the 0.5s cadence
    the measurement above used.

    ⚠️ NON-BLOCKING. `pump(0.0)` drains what is pending and returns, so this
    cannot delay a request. If AVFoundation is absent it reports so once and is
    never called again — a Linux build should not pay for this on every pass.
    """

    _pumping = True

    def service_actions(self):
        super().service_actions()
        if not self._pumping:
            return
        try:
            from . import radio
            self._pumping = radio.pump(0.0)
        except Exception:                                    # noqa: BLE001
            self._pumping = False


def make_server(service, host=None, port=None):
    """Bind an HTTP server to an existing service. Returns the httpd.

    ⚠️ `_RunLoopServer`, not `ThreadingHTTPServer` — see the class.
    """
    host = host or C.SERVER_HOST
    port = C.SERVER_PORT if port is None else port
    Handler.service = service
    httpd = _RunLoopServer((host, port), Handler)
    httpd.daemon_threads = True
    # So `quit()` can stop the loop it is being served from.
    service.httpd = httpd
    return httpd


def serve(host=None, port=None, provider=None, verbose=True):
    """Start the field owner. Blocks."""
    host = host or C.SERVER_HOST
    port = C.SERVER_PORT if port is None else port

    print("loading field...", file=sys.stderr)
    service = FieldService(provider=provider, verbose=verbose)
    httpd = make_server(service, host, port)

    print(f"  [mati] {service._llm_detail}", file=sys.stderr)
    print(f"  [mati] {service.n_episodes()} episodes", file=sys.stderr)
    print(f"  [mati] listening on http://{host}:{port}  "
          f"(loopback only)", file=sys.stderr)
    try:
        # ⚠️ 0.05s, NOT THE DEFAULT 0.5s, AND THE REASON IS AUDIO. `poll_interval`
        # is how often `service_actions` runs, and that is where the macOS run
        # loop gets serviced — so it is also the audio pipeline's clock.
        #
        # : a live radio stream reached ReadyToPlay in ~4s at
        # the 0.5s cadence, but a podcast EPISODE — a 24MB file behind a
        # five-hop redirect chain — timed out at 12s. Pumped properly in a tight
        # loop the same episode was ready in 2.5s, so the URL was never the
        # problem and neither was the timeout: the loop was being starved.
        #
        # ⚠️ IT DOES NOT BLOCK. `pump(0.0)` drains what is pending and returns;
        # what changes here is how OFTEN it gets the chance. The cost is a
        # `select()` wakeup 20×/s on an idle loopback server.
        httpd.serve_forever(poll_interval=0.05)
    except KeyboardInterrupt:
        print("\n  [mati] stopping", file=sys.stderr)
    finally:
        # Sleep BEFORE shutdown, and the order is not cosmetic. `sleep()`
        # drains the work queue and snapshots θ₀; `shutdown()` sets the flag
        # the worker loop exits on. Reversed, the worker dies with items still
        # queued and `work.join()` waits forever for a `task_done` that will
        # never come — a hang on exit.
        #
        # Without this, quitting silently dropped any queued-but-unwritten
        # episode: Mati had answered, the user had seen the reply, and the
        # memory of it was thrown away. Consolidation is deferred off the turn
        # to keep replies fast, which means exit is exactly when the debt is
        # outstanding.
        # ⚠️ BOUNDED, AND THIS IS THE BUG THAT CAUSED THAT HANG. `pack()`
        # was given timeouts and THIS was left unbounded, so a wedged field
        # let the process start exiting and never finish: the port closed,
        # `[mati] finishing up` was the last line, and it sat at 0% CPU
        # holding three days of unsaved θ₀ until it had to be SIGKILLed.
        #
        # A shutdown that cannot finish is worse than one that finishes badly
        # — at least the latter can be retried. So release gets its own
        # thread and its own deadline, and the process exits either way.
        done = threading.Event()
        err = {}

        def _finish():
            try:
                service.release()  # process exit: the field genuinely goes
            except Exception as e:                           # noqa: BLE001
                err["e"] = e
            finally:
                done.set()

        print("  [mati] finishing up", file=sys.stderr)
        threading.Thread(target=_finish, daemon=True).start()
        if not done.wait(RELEASE_TIMEOUT):
            print(f"  [mati] ⚠️ release did not finish in "
                  f"{RELEASE_TIMEOUT}s — exiting anyway. θ₀ may not have been "
                  f"saved; the notes database is committed per turn and is "
                  f"not affected.", file=sys.stderr)
        elif err:
            print(f"  [mati] could not finish cleanly: {err['e']!r}",
                  file=sys.stderr)
        service.shutdown()
        httpd.server_close()
    return 0


def main():
    ap = argparse.ArgumentParser(description="Mati — the persistent field owner")
    ap.add_argument("--host", default=C.SERVER_HOST,
                    help="loopback by default; do not expose this")
    ap.add_argument("--port", type=int, default=C.SERVER_PORT)
    ap.add_argument("--provider", default=None, help="local | anthropic")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        # Mati's pitch is that nothing leaves the machine. Binding elsewhere
        # would put an unauthenticated field-writing API on the network.
        print(f"[refused] --host {args.host}: Mati binds loopback only.",
              file=sys.stderr)
        return 2
    try:
        return serve(args.host, args.port, args.provider, verbose=not args.quiet)
    except LLMUnavailable as e:
        print(f"[llm unavailable] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
