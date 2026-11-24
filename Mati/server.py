"""
The one process that owns the field.

    python3 -m Mati.server                    # start it
    python3 -m Mati.server --port 8756
    curl -N localhost:8756/chat -d '{"text":"hello"}'

WHY A SERVER. θ₀ is mutable state backed by a file. Two processes holding it
would write over each other, so exactly one may own it — and Mati is two apps
(a chat app for people, a monitoring/training terminal for pro users) that both
need it at once. That makes the owner a server and both apps its clients.

WHAT IT ACTUALLY BUYS. Not the reload: loading the field takes a second or two,
a small part of a turn. Most of a turn's work is consolidation — writing the
episode — which the person has no reason to wait for. A persistent process can
move it off the critical path; a per-invocation script cannot. That is the
point of this file.

So a turn here streams its reply and hands the episode write to a worker.
Re-addressing stored notes happens once a night, after `rest()` promotes the
day's field into the address space — never per turn.

CONCURRENCY. One lock guards the field; every mutation serialises through it.
Monitoring does NOT take the lock — the turn publishes telemetry as it goes and
readers get the last snapshot, so watching Mati think never blocks it. Long
background work checks between steps whether a turn is waiting and yields, so
an arriving message waits for one step rather than a whole pass.

BINDING. 127.0.0.1 only, and not configurable to a public interface: "nothing
leaves your machine" is a property of the binding, not a promise in a README.
"""

import argparse
import inspect
import base64
import hashlib
import json
import re
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

# ⚠️ IS THE CODE ANSWERING THE CODE ON DISK? A long-lived process that silently
# serves old behaviour turns every later measurement into a measurement of the
# past, and nothing else in `/health` would say so: every filter, writer and
# gate passes an offline test, because the code on disk is correct — it simply
# has not been loaded. So `/health` reports `code_stale`.
#
# Module mtimes rather than the git hash on purpose: it catches an uncommitted
# edit too.

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
    # ⚠️ MATI'S OWN MODULES ONLY, `field/` included. A file Mati never imports
    # must not mark its code stale — this is the one field people check before
    # trusting a behaviour report — and a module it does import must not be
    # missed.
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
    that has loaded it.
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

        self.provider = provider or C.PROVIDER
        # ⚠️ ONLY AN EXPLICIT PROVIDER IS PASSED ON. `self.provider` defaults
        # to C.PROVIDER, which is "local", and handing that to `make_llm`
        # would pin every client to Ollama and silently ignore the model the
        # person chose (`config.model_choice`).
        self._provider_arg = provider
        # Why the chosen model is not the one speaking, or None. See `_open_llm`.
        self._model_fallback = None
        if llm is not None:
            self.llm = llm
            self._llm_detail = self.provider
            if hasattr(self.llm, "available"):
                ok, detail = self.llm.available()
                if not ok:
                    raise LLMUnavailable(detail)
                self._llm_detail = detail
        else:
            self.llm, self._llm_detail = self._open_llm()

        # TWO clients on purpose. `self.llm` carries the bare system prompt and
        # is what the study/dream/inquiry worker uses. `self.chat_llm` carries
        # the system prompt PLUS the waking block, and is the only one that
        # talks to a person.
        #
        # They are separate because the whole saving depends on the chat prefix
        # never changing: Ollama re-prefills only what changed, and a cached
        # token costs a tiny fraction of a new one. Giving the worker the same
        # client would not corrupt anything, it would just carry several hundred
        # tokens it has no use for on every chunk it reads. (The two caches do
        # not evict each other.)
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
        # ⚠️ RESTING IS NOT LEAVING. Resting keeps the field, keeps answering,
        # and simply does less: the idle loop stops starting new work and only
        # finishes what it owes. Handing the field over is `release()`.
        self.slow = False
        # Exchanges cut off mid-reply, per session. See interruption.py.
        self._pending = {}
        self.was_shutdown = False

        # Try to unlock BEFORE loading the field, so what the field reads first
        # is readable.
        # ⚠️ CLAIM θ₀ BEFORE OPENING IT, against ANY holder — a terminal holding
        # the field with `/load` as much as another server — or both would
        # write it back. Refusing to start is the right failure: the person can
        # see the terminal they left open.
        if not exclusive.take("Mati server"):
            cur = exclusive.held_by() or {}
            raise SystemExit(
                f"\n  {cur.get('holder', 'Something')} (pid {cur.get('pid')}) "
                f"is holding the field.\n"
                f"  Close it first — two processes writing θ₀ overwrite each "
                f"other.\n")

        # ⚠️⚠️ A LOCKED VAULT MUST NOT STOP THE SERVER FROM STARTING.
        # `vault.unlock()` RAISES on a passphrase-only vault — there is no
        # keychain key by design, so nothing can open it unprompted — and when
        # a fingerprint goes unanswered. Raised out of `__init__`, that would be
        # a completed setup and a product that cannot run.
        #
        # ⭐ IT DEGRADES INSTEAD, which is what every other reader of sealed
        # data already does: `daylog.rows()` returns [], `presets._open()`
        # returns None, `charter_chunks()` returns []. Mati comes up, answers,
        # and reads none of its sealed memories until it is unlocked — which
        # `POST /unlock` does without a restart.
        self.locked_at_start = False
        self._unlock_failures = []        # when wrong passphrases arrived
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
        # Worker liveness, reported by /health. A thread that dies leaves
        # nothing behind but a number that stops moving.
        self.worker_beat = time.time()
        self.last_worker_error = None
        # Start ready to dream rather than an hour out: a fresh process that
        # has been idle since boot has nothing owed and may as well rest.
        self.last_dream = 0.0
        # When the person has asked for reading to be the priority. Bounded:
        # full speed starves dreaming and deepening, which is the whole reason
        # the budget exists — so this is a request, not a new default.
        # ⚠️ PERSISTED, because a window kept only in memory is lost in silence
        # at a restart — and an overnight window is the case most likely to
        # span a restart and least likely to be watched.
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
        # ⚠️ DO NOT ADD A SIGUSR1 faulthandler HERE. Dumping all threads from a
        # signal handler while numpy holds the GIL in C is not safe in this
        # process: it can dump a frame or two and then kill the server
        # mid-write. If a stack is needed, use `sudo py-spy dump --pid`, which
        # is out-of-process.
        self._worker.start()
        self._pulse.start()
        self.check_addresses("startup")
        # ⚠️ ITS OWN DOCUMENTATION, KEPT CURRENT. Re-indexed only when the
        # documents have changed, and never while locked — `selfdoc.refresh`.
        # A locked start does it in `_after_unlock` instead.
        try:
            from . import selfdoc as _selfdoc
            _n = _selfdoc.refresh()
            if _n and self.verbose:
                print(f"  [mati] self-knowledge re-indexed ({_n} passages)",
                      file=sys.stderr)
        except Exception as e:                               # noqa: BLE001
            if self.verbose:
                print(f"  [mati] self-knowledge skipped: {e}", file=sys.stderr)
        # ⚠️ THE FIRST SEARCH OF A PROCESS IS SLOW AND EVERY ONE AFTER IT IS
        # FAST. `bridge.search` spawns openssl per body, so the first call
        # decrypts the whole store; the cache makes the rest free. Paying it
        # here means the person never does — otherwise they would pay it in
        # their first sentence after a restart.
        try:
            t_warm = time.time()
            bridge.search("warm the body cache", k=1)
            # ⚠️ ONE SEGMENT, NOT THE YEAR. The newest
            # `notes.VERBATIM_SEGMENT_DAYS` of exact words, bounded by
            # `warm_verbatim`'s budget to a few seconds, so Mati can be asked
            # about what was literally said this month from its first
            # sentence. The rest of the year arrives a segment per idle pass.
            #
            # ⚠️ AND NOT THE WHOLE YEAR HERE, DELIBERATELY. Decrypting every
            # verbatim inside the year at once makes a cold search take over a
            # minute, because in a young store "inside the year" is the entire
            # store. The year bounds the archive; it does not make this year
            # cheap.
            from . import notes as _nwarm
            _nwarm.warm_verbatim(segment=0)
            if self.verbose:
                print(f"  [mati] memory cache warm "
                      f"({time.time()-t_warm:.0f}s)", file=sys.stderr)
        except Exception as e:
            if self.verbose:
                print(f"  [mati] cache warm skipped: {e}", file=sys.stderr)
        # ⚠️ STARTING IS A KIND OF WAKING. The server comes up with
        # `asleep = False`, so it never passes through `wake()` — the one place
        # that builds the orientation. Without this, a freshly started Mati
        # would have `orientation = None` and every turn would pay the full
        # prefill this exists to remove: healthy from outside, simply slow.
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
            # holding it. Decrementing on exit instead of on acquire would make
            # a holder count itself, so `contended` would be True exactly while
            # the worker was working — and every
            # `should_stop=lambda: self.contended` would fire on its first
            # check, silently stopping all background work.
            with self.svc._waiting_lock:
                self.svc._waiting += 1
            got = (self.svc._lock.acquire() if self.timeout is None
                   else self.svc._lock.acquire(timeout=self.timeout))
            with self.svc._waiting_lock:
                self.svc._waiting -= 1      # no longer waiting, either way
            if not got:
                # ⚠️ RAISED, NOT RETURNED. A caller that got no field must not
                # run its body against `self.lm` and then claim it succeeded.
                # ⚠️ REPORTS WHAT IT OBSERVED, AND GUESSES NOTHING. The holder
                # can be an admin terminal or Mati's own study task; a guessed
                # cause stated confidently in an error message sends the reader
                # after the wrong thing.
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

    # ── which model speaks ───────────────────────────────────────────────────

    def _local_name(self):
        return C._settings().get("local_model") or C.LOCAL_MODEL

    def _client(self, extra=""):
        """A client for the model that speaks now, with `extra` after the
        system prompt (the waking block, for the one that talks to a person).

        ⚠️ ONE PLACE BUILDS THEM, so the plain client, the oriented one and a
        switch can never disagree about which model is speaking.
        """
        from . import llm as _L
        if self._model_fallback:
            return _L.LocalLLM(system_prompt(remote=False) + extra,
                               model=self._local_name())
        return make_llm(system_prompt() + extra, self._provider_arg)

    def _open_llm(self):
        """(client, detail) for the person's choice — or the local model, with
        the reason kept in `_model_fallback`, if that choice cannot be used.

        ⚠️ A REMOTE SERVICE BEING DOWN MUST NOT STOP MATI STARTING. Offline,
        or with the key's vault locked, the conversation goes on locally and
        `/health` says why. Only when the LOCAL model is unusable does
        starting fail, as it always did.
        """
        self._model_fallback = None
        why = None
        try:
            llm = self._client()
            ok, detail = (llm.available() if hasattr(llm, "available")
                          else (True, self.provider))
            if ok:
                return llm, detail
            why = detail
        except Exception as e:                               # noqa: BLE001
            why = str(e)
        if self._provider_arg or C.model_choice()["kind"] == "local":
            raise LLMUnavailable(why)
        self._model_fallback = why
        if self.verbose:
            print(f"  [mati] the chosen model cannot be used now ({why}) — "
                  f"speaking through {self._local_name()} instead",
                  file=sys.stderr)
        llm = self._client()
        ok, detail = llm.available()
        if not ok:
            raise LLMUnavailable(detail)
        return llm, detail + " — standing in"

    def switch_model(self):
        """Use whatever `config.model_choice` now says. From the next reply.

        ⚠️ NO RESTART. The model is the language faculty only: addresses come
        from the field's own lm and tokenizer, so nothing is reindexed. A turn
        in flight keeps the client it started with.
        ⚠️ THE OLD LOCAL MODEL IS UNLOADED. Ollama keeps a model resident for
        `keep_alive`; two 3B models side by side beside the field is more than
        8 GB holds, and that swaps rather than fails.
        """
        from . import llm as _L
        old = self.llm
        try:
            new, detail = self._open_llm()
        except Exception as e:                               # noqa: BLE001
            return {"switched": False, "error": str(e)}
        self.llm, self._llm_detail = new, detail
        if self.orientation is not None and not self._llm_injected:
            self._oriented_llm = self._client("\n\n" + self.orientation)
        if (isinstance(old, _L.LocalLLM)
                and getattr(new, "model", None) != old.model):
            try:
                old._post("/api/generate", {"model": old.model,
                                            "keep_alive": 0}, timeout=15)
            except Exception:
                pass
        if not _L.is_remote(self.chat_llm):
            threading.Thread(target=self._prewarm, daemon=True).start()
        return {"switched": not self._model_fallback, "llm": detail,
                "remote": _L.is_remote(new), "fallback": self._model_fallback}

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

        ⚠️ NOT PART OF THE HANDOFF. Consolidating and letting go are separate
        so that a prediction does not cost the field: `release()` consolidates
        on its way out, and `rest()` consolidates and stays.
        """
        # ⚠️ RAPPORT IS BUILT HERE, NOT DECIDED IN THE MOMENT. Rapport belongs
        # to the prediction: the day's settled exchanges are learned from the
        # record while nobody is waiting, and the turn path only ever looks
        # them up.
        #
        # Idempotent and day-stamped: called again on the same day it does
        # nothing, because a prediction something contradicts is held until the
        # end of the day — it may just be delayed.
        # ⚠️ CONSOLIDATION HAPPENS HERE, AND ONLY HERE. The prediction is made
        # while the person sleeps: sleep is the boundary of a day, and a
        # prediction OF THE DAY has to be made at one — not in a dream, which
        # can happen in any quiet hour.
        #
        # Nothing here runs on a turn or on idle. Noticing happens through the
        # day; deciding what tomorrow is about happens at the end of one.
        try:
            # ⚠️ THINKING ABOUT THE DAY COMES BEFORE PREDICTING IT. People spend
            # much of a day recomputing what they experienced in relation to
            # themselves and everyone around them, and that is what makes their
            # predictions faster and more likely. The reflection is that
            # recomputing, and it runs first so the day's thinking exists
            # before the prediction is made. ⚠️ `predict` does not read it
            # yet; the morning report is its reader.
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

        ⚠️ THIS IS WHAT A QUIET NIGHT DOES, AND IT IS NOT `release()`. Nothing
        here hands the field over, drops `self.lm` or touches the exclusive
        lock, so the terminal stays connected through the night and Mati can
        still be spoken to — it simply stops starting new work.

        Dreaming continues on purpose: `_maybe_dream` is not gated on `slow`,
        so the idle loop goes on turning over what is held once an hour. A
        released field could not dream, because there would be no field in
        memory to dream with.

        `self.slow` is its own idempotence guard: the idle pass only calls this
        when not already slow.
        """
        self.slow = True                      # keep the field, do less
        self.consolidate()
        # ⚠️ THE DAY'S LEARNING BECOMES THE ADDRESS SPACE HERE, AND NOWHERE
        # ELSE. Through the day Mati addresses against a frozen θ₀ and
        # imprints into the live one, so what it has learned today is not yet
        # in the space it recalls through. Promoting at night is what lets the
        # day in; it is deliberately once a night because promotion is the
        # only thing that makes an address stale, and the repair costs seconds
        # a note.
        #
        # ⚠️ AFTER `consolidate()`, NEVER BEFORE — the queue still holds
        # episodes to imprint, and promoting first would publish a space that
        # is missing the last of the day and then mark nothing stale for it.
        #
        # ⚠️ AND THE REINDEX GOES WITH IT, IN THE SAME BREATH. Nothing else
        # repairs addresses (`C.REINDEX_ON_IDLE` is off), so promoting with
        # nothing to repair it would invalidate the whole working set every
        # night and never fix it — strictly worse than not promoting at all,
        # and silent.
        #
        # ⚠️ IT CONVERGES. This is ONE bounded pass after ONE deliberate
        # invalidation, against a fingerprint that holds still until the next
        # `rest()`. It runs on the worker and yields to an arriving turn.
        #
        # ⚠️ AND NOT WHILE LOCKED. The repair needs each note's text; without
        # the key it cannot run, and promoting anyway leaves the whole working
        # set filed in a space that no longer exists. The day stays in the
        # live field and is promoted by the next rest that can finish the job.
        if _vault_state() is False:
            promoted = {"promoted": False, "reason": "memories are locked"}
        else:
            try:
                with self.field():
                    bridge.promote(self.lm)
                    promoted = {"promoted": True,
                                "address_gs": bridge.address_gs(self.lm)}
                self.work.put(("reindex after promote",
                               self._reindex_after_promote))
            except Exception as e:            # never let this block the night
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
            # ⚠️ SAY WHEN THE FLOOR COULD NOT BE HELD. A year is the floor and
            # capacity the ceiling; if a year of exact words does not fit in
            # this machine's budget, that is a fact about the machine and the
            # owner should see it rather than wonder why old conversations
            # only come back as summaries.
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

    # Wrong passphrases tolerated per window before `unlock` asks for a pause.
    # A guess costs the guesser nothing and costs Mati a key derivation.
    UNLOCK_TRIES, UNLOCK_WINDOW = 5, 60.0

    def unlock(self, passphrase=None):
        """Open the vault in THIS process, without restarting it.

        With a passphrase, that is what opens it. With none, the vault's own
        way is retried — the keychain, after a fingerprint where the vault
        calls for one — which is what recovers a start whose Touch ID prompt
        nobody was there to answer.

        ⚠️ THE KEY HAS TO ARRIVE IN THE PROCESS THAT ANSWERS. Unlocking
        anywhere else opens nothing here: a Mati that started locked reads
        none of its memories and stores none of what it is told, and would
        stay that way until it was restarted.

        Returns at once. What a locked start skipped is caught up on a thread
        of its own (`_after_unlock`), because warming the caches takes minutes
        and whoever asked is waiting on this reply.
        """
        if not vault.is_set_up():
            return {"unlocked": None, "reason": "there is no vault"}
        if vault.unlocked():
            return {"unlocked": True, "already": True}
        now = time.time()
        self._unlock_failures = [t for t in self._unlock_failures
                                 if now - t < self.UNLOCK_WINDOW]
        if passphrase and len(self._unlock_failures) >= self.UNLOCK_TRIES:
            wait = self.UNLOCK_WINDOW - (now - self._unlock_failures[0])
            return {"unlocked": False, "retry_after": round(max(1.0, wait)),
                    "error": "too many wrong passphrases — wait a minute "
                             "and try again"}
        try:
            vault.unlock(passphrase=passphrase or None,
                         reason="unlock Mati's memories", quiet=True)
        except Exception as e:                               # noqa: BLE001
            if passphrase:
                self._unlock_failures.append(now)
            return {"unlocked": False,
                    "error": " ".join(str(e).split()) or e.__class__.__name__}
        self.locked_at_start = False
        self._unlock_failures = []
        self.telemetry.publish(stage="unlocked", memories_locked=False)
        if self.verbose:
            print("  [vault] unlocked — catching up", file=sys.stderr)
        threading.Thread(target=self._after_unlock, daemon=True,
                         name="mati-unlock").start()
        return {"unlocked": True}

    def _after_unlock(self):
        """What a start does with the key, done late because the key came late.

        Each step stands alone: a failure in one must not cost the others,
        and none of them is worth leaving Mati locked for.
        """
        out = {}
        # Anything stored while locked went in unsealed rather than dropped.
        try:
            out["sealed"] = vault.migrate_tables(verbose=False)
        except Exception as e:                               # noqa: BLE001
            out["seal_error"] = repr(e)
        # Re-addressing needs each note's text, so nothing could be repaired
        # while locked. Check now, and repair what is owed.
        try:
            if (self.check_addresses("unlock") or {}).get("stale"):
                out["reindexed"] = (self._reindex_after_promote()
                                    or {}).get("reindexed")
        except Exception as e:                               # noqa: BLE001
            out["address_error"] = repr(e)
        try:
            bridge.search("warm the body cache", k=1)
            from . import notes as _nwarm
            _nwarm.warm_verbatim(segment=0)
        except Exception as e:                               # noqa: BLE001
            out["warm_error"] = repr(e)
        # Its own documentation is sealed, so a locked start could not
        # re-index it.
        try:
            from . import selfdoc as _selfdoc
            out["self_docs"] = _selfdoc.refresh()
        except Exception as e:                               # noqa: BLE001
            out["self_docs_error"] = repr(e)
        # The waking block was built without a single readable memory.
        try:
            self.build_orientation()
        except Exception as e:                               # noqa: BLE001
            out["orientation_error"] = repr(e)
        self.telemetry.publish(stage="idle", caught_up=out)
        if self.verbose:
            print(f"  [vault] caught up: {out}", file=sys.stderr)
        return out

    def release(self):
        """Finish up, persist, and LET GO OF THE FIELD. Idempotent.

        ⚠️ NOT THE NIGHT. "Sleep" is what a person does nightly and expects to
        keep their mind through; this drops `self.lm` and hands θ₀ to whoever
        takes the exclusive lock next. Nightly rest is `rest()`. This is for an
        explicit handoff — the pro terminal taking the field — and for process
        exit. Nothing calls it on a timer.
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

    # ⚠️ SLEEP IS THE NIGHT: `rest()`, which keeps the field. It is never the
    # handoff — that is `release()`, and only that.
    def sleep(self):
        return self.rest()

    def pack(self):
        """Make this install carryable. ⚠️ DOES NOT STOP THE PROCESS.

        The four verbs, each meaning one thing:

            rest()      nightly. KEEPS the field, still dreams, still answers.
            release()   hands the field over. Process keeps running.
            pack()      makes the directory copyable. Process keeps running.
            stop()      packs, then actually STOPS the process.

        ⚠️ **THIS IS NOT `release()`, AND THE DIFFERENCE MATTERS.** Release
        drains, snapshots and lets go of the field so the pro terminal can take
        it — the process stays up and is expected to wake. Packing leaves a
        **portable install**, because Mati is a few megabytes that can move to
        another machine on a drive.

        Two things release does not do, both of which produce a copy that looks
        fine and is not:

          - **A full reindex.** An install handed over with stale addresses
            carries recall that has quietly collapsed toward recency.
            Unbounded here; a transfer is worth the seconds.
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
        """Pack, then STOP THE PROCESS.

        ⚠️ `/pack` and its older name `/shutdown` keep serving; this is the
        one that ends the process, so a restart needs no manual `kill`.

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
        # ⚠️ NEITHER THE PORT NOR THE PID IS A RELIABLE SIGNAL. Waiting on the
        # PORT launches too early: the socket closes at `httpd.shutdown()`
        # while the FIELD is released later, in the cleanup after it, so the
        # successor would be refused with "Mati server is holding the field"
        # and exit — leaving nothing running. Waiting on the PID is worse: a
        # dead process stays visible to `os.kill(pid, 0)` as a zombie until it
        # is reaped, so the poll can hang on a process that has already exited.
        #
        # ⚠️ SO IT RETRIES INSTEAD OF PREDICTING. The successor tries to
        # start, checks whether it is answering, and tries again if not. That
        # needs no model of the shutdown's timing, and it recovers from a slow
        # release rather than dying on one.
        # ⚠️ `SERVER_PORT`, NOT `PORT`. `C.PORT` does not exist; reading it
        # would fall back to 8756 on every machine, which is right only by
        # coincidence — on any other port the waiter would poll the wrong
        # socket and the successor would die on the exclusive lock.
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
        # measure, and hold no memories until it is unlocked — and the person
        # asked for a restart, not for amnesia.
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

        Waking is precisely when θ₀ is most likely to have moved: the field is
        handed over so that the pro terminal can TRAIN it, so the field that
        comes back is routinely not the field that left.

        Addresses filed against the old θ₀ are then stale, and stale addresses
        do not fail loudly — retrieval quietly collapses toward recency. Mati
        would come up looking healthy and remembering the wrong things.

        This DOES NOT reindex inline. Reindexing takes a while and grows with
        the corpus; blocking a wake on it would trade a silent fault for a
        startup that looks hung. It measures and reports.
        """
        try:
            gs_now = bridge.address_gs(self.lm)   # the frozen space's figure
            conn = sqlite3.connect(bridge.NOTES_DB_PATH)
            try:
                bridge._ensure_schema(conn)
                # ⚠️ WORKING SET ONLY. `bridge.reindex` never touches archived
                # notes by design, so counting them would report a number of
                # stale addresses that no amount of re-addressing could bring
                # down — and anything acting on it would loop forever.
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
            # Ask for repair rather than performing it: the worker rotates by
            # staleness, so the worst rows go first, and it yields to a turn —
            # this fires right after startup or wake, which is exactly when
            # someone is likely to type.
            # ⚠️ MEASURED, NOT ACTED ON — see C.REINDEX_ON_IDLE. Re-addressing
            # whenever anything is stale is a loop that cannot converge while
            # the target keeps moving. The number is still reported so
            # `/addresses` tells the truth about the index; repair happens once
            # a night, after promotion.
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
            # ⚠️ SAY WHAT ACTUALLY HAPPENED. A log line asserting work that is
            # not being started is worse than none.
            tail = ("reindex queued" if out.get("queued")
                    else "not repaired — field reindexing is retired")
            print(f"  [mati] {stale}/{len(rows)} addresses stale after {why} "
                  f"(worst {worst:.3e}) — {tail}", file=sys.stderr)
        return out

    def wake(self):
        """Take the field back. Costs one load (~1.5s).

        Symmetric with `release()`: that one snapshots and lets go, this one
        picks up and then CHECKS what it picked up. See `check_addresses`.

        ⚠️ IT ALSO ENDS A SLEEP. Sleeping keeps the field, so there is nothing
        to load — waking from it only starts work again. Without this, Wake
        after Sleep would answer "already awake" and change nothing.
        """
        if not self.asleep:
            if self.slow:
                self.slow = False
                self.last_activity = time.time()
                activity.log(activity.STIR)
                self.telemetry.publish(stage="idle", slow=False)
                return {"asleep": False, "slow": False, "woke": True}
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

        Field state, self-knowledge and recalled memories rebuilt on every turn
        would be re-prefilled on every turn — tens of seconds before the model
        could say a word. Built once and held constant, the same material is
        cached after its first send and costs almost nothing.

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
            self._oriented_llm = self._client("\n\n" + block)
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
        # ⚠️ NEVER FOR A REMOTE MODEL. Warming is a generation, and a remote
        # generation is billed — Mati would spend the person's money on every
        # start for a cache the service keeps on its own terms.
        from . import llm as _L
        if _L.is_remote(self.chat_llm) or not (
                getattr(self.chat_llm, "settings", None) or {}).get(
                    "prewarm", True):
            return
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

        ⚠️ A RELEASED SERVER WAKES ITSELF FOR A TURN rather than turning it away
        until someone types /wake. Mati should be READY BEFORE THE APP OPENS;
        being told to knock twice is the opposite of that.
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

        # ⚠️ SPOKEN FIRST, BEFORE ANY FIELD WORK. Otherwise most of a turn is a
        # blank screen while the field works. This is a lookup and an echo, no
        # model call, so it appears in about a second.
        #
        # ⚠️ It is SPEECH and is therefore part of the record. Everything in it
        # is copied from the store or is affect about the act itself, so there
        # is nothing here that can be false.
        # ⚠️ AN ASKED-FOR ACTION IS ANSWERED IMMEDIATELY. "play some jazz" does
        # not need the field, the memory or the model — it needs a station and a
        # play command, and both take about two seconds. Routing it through the
        # ordinary turn would make the person wait minutes to hear music.
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
                # ⚠️ A MEDIA ANSWER IS CANNED AND NEVER REACHES `turn()`, so it
                # is spoken here. "Playing Blue Note Radio" is something Mati
                # DID, and it should say that aloud as readily as anything a
                # model generated.
                if said and isinstance(said, str):
                    try:
                        from . import speech as _sp
                        _sp.say(said)
                    except Exception:
                        pass
                if said:
                    chunk(said)
                    # ⚠️⚠️ THE PERSON'S WORDS STAY IN THE CONVERSATION. A
                    # message taken for a media request must still be appended
                    # to the history, or it vanishes from the thread: the model
                    # meets it only as a recalled "I was asked … I did it", and
                    # a misrouted message leaves nothing to go back to. In the
                    # history, a misroute is one visible exchange the next turn
                    # can see and recover from.
                    self.append_history(session_id, text, said)
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
            # ⚠️ THE OPENER IS SPOKEN TOO. This is the patience line — "Hm.",
            # "Let me sit with that a second." — and it is the single most
            # MATI thing in a reply: nothing generated it, Mati says it because
            # a person is waiting.
            #
            # ⚠️ ENQUEUED, NOT SAID. `say()` interrupts; the reply's own
            # sentences are queued right behind this one, and interrupting
            # would cut the opener off mid-word as the first sentence landed.
            try:
                from . import speech as _sp
                _sp.enqueue(opener)
            except Exception:
                pass

        # ⚠️ WHO IS TYPING. An episode records the speaker, or a third party's
        # words would be filed as the owner's and Mati could not say whether
        # the person in front of it was the owner.
        #
        # ⚠️ MATI DOES NOT ASK WHO IS THERE. Its assumption is that the peer is
        # the owner, and it is up to Mati to notice when that is not so.
        # Opening a session by asking for a name would greet the person who
        # built it as a stranger, and put the work of identification on them
        # when it belongs to Mati. An introduction is TAKEN when offered rather
        # than solicited.
        #
        # ⚠️ Attribution is not weakened by that. `named_in` records a name the
        # moment someone gives one, and `attribute` marks a memory as somebody
        # else's.
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
        live phases while θ₀ does not move at all (see `expression.py`). So
        this is motion that means "awake", not motion that means "changing".
        Between turns the panel's basin and coherence are the LAST TURN's
        numbers, frozen; animating those would be putting movement on stale
        values, which is the same lie as a progress bar that keeps going after
        the work stopped.

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
        # ⚠️ SCHEDULED, SO MATI IS READY BEFORE THE APP OPENS. Resting and
        # waking follow a time learned from the person's own days.
        #
        # ⚠️ INERT UNTIL THE PATTERN IS REAL. `rhythm.daily_schedule` returns
        # None while the first turn of the day varies by hours, and this then
        # does nothing, because waking at a median drawn from noise would put
        # Mati to sleep mid-conversation. When the days settle, it starts
        # working with no further change.
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
        # ⚠️ THE SHORT TIER, AND IT COMES FIRST: text handed over to be
        # imprinted later is settled within seconds, because the field being
        # behind the conversation is a different thing from the field being
        # idle. A turn's own imprint cannot be deferred (see `mati.turn`), so
        # nothing is normally waiting here.
        #
        # Everything else — studying, dreaming, questions, inquiry — waits for
        # the long tier below. Catching up beats acquiring.
        if self._drain_imprints():
            self.last_activity = time.time()
            return
        # SLEEP — left alone, not merely paused. The expensive work lives here
        # so a gap in conversation is never spent imprinting a document chunk
        # while the person waits to type again. Ordered: absorb what was
        # queued, finish what was written thin, only then acquire anything new.
        # ⚠️ RESTING STILL PERCOLATES. Mati is left running when the apps are
        # closed, for continuity, and so that something can percolate. The
        # division that matters is not cheap-versus-expensive, it is OUTWARD
        # versus INWARD:
        #
        #   outward — reading a new document, asking itself new questions.
        #             Acquisition. Stops while resting.
        #   inward  — dreaming. Turning over what is already held. Continues.
        #
        # A person resting is not learning a new subject and is very much still
        # digesting the day.
        # ⚠️ THE THRESHOLD IS THE PERSON'S, NOT A CONSTANT. `rhythm.idle_after`
        # reads their own p95 gap, floored at 45 minutes. Cached for a minute
        # so this stays a comparison rather than a query.
        # ⚠️ THE NIGHT IS `rest()`, NEVER THE HANDOFF. Resting consolidates the
        # day — the prediction and the day's rapport are rebuilt there and
        # nowhere else, so a machine simply left running needs this to rebuild
        # them at all — goes slow, keeps the field and goes on dreaming; a
        # quiet night must not disconnect the terminal.
        if (not self.asleep and not self.slow
                and time.time() - self.last_turn >= C.SLEEP_AFTER_SECONDS):
            self.rest()
            return True

        if time.time() - self.last_turn >= self._idle_after():
            # ⚠️ FIRST, AND THAT IS THE POINT. The verbatim is split into time
            # segments so nothing has to decrypt everything at once, and the
            # last year should be searchable by its exact words. One segment is
            # a few seconds and needs NO FIELD LOCK, so it is the cheapest
            # useful thing an idle pass can do — and behind the debts below it
            # would starve: `_maybe_study` and `_maybe_deepen` return early to
            # finish their own work, take minutes doing it, and a job behind
            # them simply never runs while there is anything to read. See
            # `notes.warm_verbatim`.
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
            # ⚠️ BEFORE INQUIRY, AFTER THE DEBTS. `_maybe_inquire` fires on
            # nearly every pass while topics stay open, so a dream placed after
            # it would never be reached — starved, not missing, which looks
            # identical from outside.
            #
            # It is safe here because it rate-limits itself to once an hour
            # (DREAM_EVERY_SECONDS), so it cannot starve inquiry in turn. The
            # order is: repay what is owed, rest, then acquire something new.
            # Dreaming turns over what is held. It does NOT rebuild the
            # prediction — that happens in `rest()`, at the end of a day.
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
        # IDLE — a gap in conversation. Only what is quick: re-addressing
        # anything still filed in an older space. Between promotions there is
        # nothing to do and it returns at once; it is bounded and yields per
        # note, so it fits in a gap.
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
                    # An episode write imprints for most of a minute; holding
                    # the field for all of it would make a turn arriving
                    # mid-write wait the whole time. Tasks that accept
                    # `should_stop` yield at a safe boundary; those that do not
                    # are simply called.
                    # Ask the signature rather than catching TypeError: a
                    # TypeError raised INSIDE the thunk would otherwise trigger
                    # the fallback and run the whole write a SECOND time —
                    # imprinting the same episode twice.
                    if _takes_should_stop(thunk):
                        info = thunk(should_stop=lambda: self.contended) or {}
                    else:
                        info = thunk() or {}
                # Tasks are plain callables and not all of them return a dict —
                # `bridge.remember` hands back (note_id, triplet) — so the
                # telemetry below must not assume one, or the log would claim a
                # failure after the work had already succeeded.
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
        the whole point of this shape. A reflection that raised on a
        normally-started server would otherwise leave no trace anywhere — no
        activity row, no `mine` note, no complaint — and "it ran and declined"
        must be distinguishable from "it never ran". An invisible failure in a
        nightly path nobody watches is the worst kind.
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

        Studying stores the content and labels it `studied`, so Mati can say
        "I read that" when a passage is handed to it — but asked what it has
        BEEN reading, it would have nothing, because studying produces text and
        no experience. This is the experience.
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
        if _vault_state() is False:
            return 0                  # the queued text is sealed
        if study.status().get("pending", 0) == 0:
            return 0
        # ⚠️ YIELD THE PASS. Study returns early from `_idle_pass`, so without
        # this a queued document starves everything below it — a large upload
        # would stop self-talk and dreaming for a whole night. Taking only 1
        # pass in STUDY_EVERY lets deepening, resolving, question-forming,
        # inquiry and dreaming keep their turns while the reading still gets
        # most of the wall clock, because a study pass is long and theirs are
        # seconds.
        self._study_pass = getattr(self, "_study_pass", 0) + 1
        # Asked for directly, reading takes every pass — on purpose, and with
        # an end. Outside that window it yields.
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

        AFTER THE DEBTS, deliberately. Study, deepening and resolving all repay
        debts; a dream deforms θ₀, and once the address space moves that can
        create one. Running it before the debts are paid means repairing twice.
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
                # ⚠️ THE FIRST TIME SOMETHING IS ACTUALLY DUE is the only honest
                # moment to ask about interruptions — before that it is a
                # hypothetical. Consent is off by default, so without this
                # question the feature would be unreachable. The item is NOT
                # lost either way; it waits for the next conversation.
                try:
                    from . import asks as _asks
                    _asks.ask_interrupt_if_needed()
                except Exception:
                    pass
                continue                      # waits for the user to speak
            self.telemetry.publish(stage="idle", announce={
                "id": item["id"], "kind": item["kind"], "body": item["body"]})
            # ⚠️ AND ACTUALLY DELIVER IT. Publishing alone reaches nobody —
            # `announce` is consumed by no client — so a due reminder would
            # arrive only when the person happened to start a conversation:
            # rent due on the 1st, told on the 3rd.
            #
            # ⚠️ MARKED DELIVERED ONLY IF IT WAS. If the notification could
            # not be shown the item stays queued and surfaces at the top of
            # the next conversation. A reminder marked delivered but never seen
            # is worse than a late one.
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
        # C.INQUIRE_MAX_SILENCE_H: a big enough document would otherwise turn
        # this subsystem off for months, which is not a priority, it is a mute.
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
        switch. The README lists every connection and says this one can be
        turned off; a promise in the README with no setting behind it would be
        an overclaim.
        `idle_work=False` already stops it with everything else — this is the
        narrower control for someone who wants the quiet work and not the
        feeds.

        ⚠️ ON by default, and that is defensible precisely because it sends
        NOTHING about the person: a GET to each feed the person added, with a
        fixed User-Agent, and nothing at all until they add one. The thing
        being defaulted on is a download, not a disclosure.
        """
        try:
            if C._settings().get("headlines") is False:
                return 0
        except Exception:
            pass
        return self._fetch_world_now()

    def _fetch_world_now(self):
        """The fetch itself. Split out so the gate above reads as a gate.

        Asked whether it has heard any interesting news, Mati should not make
        the person wait while it searches: it has already looked, overnight,
        unless nothing interesting was found. `expire()` runs inside
        `fetch_all`, so this is also what ages old items out.

        ⚠️ IT RUNS WHILE RESTING, UNLIKE THE OTHER OUTWARD WORK. Studying and
        inquiring stop overnight because they are acquisition that competes
        with the person. Reading a feed is seconds of network and no field
        work at all, and the whole point is that it has happened BEFORE
        morning: a prediction should cause PREPARATION, not an announcement.
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

        Cheap — SQL over timestamps, no field and no model.

        ⚠️ NOTHING CONSULTS THE RESULT YET. `predict.may_influence()` gates
        that, and it is False until enough predictions have resolved with a
        good enough hit rate. Until then this only builds the record that will
        decide whether Mati can predict anything at all; on a young store it
        predicts nothing, and that is the correct output.
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
                # ⚠️ RE-CHECK AFTER TAKING THE LOCK. `/release` sets `self.lm`
                # to None to hand the field over, and it can land between the
                # batch loop above and this save. The addresses are already
                # committed by `bridge.reindex`; only persisting θ₀ is skipped
                # here, and release persists it anyway.
                if self.asleep or self.lm is None:
                    return
                bridge.save_notes_field(self.lm)
            self.stats["reindexed"] += n_total
            self.stats["last_reindex"] = time.time()
            self.telemetry.publish(stage="idle", reindexed_total=n_total)
            # Republish the staleness figures, or `/addresses` and the
            # terminal's FIELD panel would keep showing a startup snapshot while
            # the worker repaired addresses behind them — a monitor that
            # disagrees with `/queue` and `/health`. It is a plain SELECT over
            # feat_gs, no field forwards, so it is cheap enough to run after
            # every pass.
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
            # ⚠️ REPORTED, BECAUSE A LOCKED VAULT IS A RUNNING STATE RATHER
            # THAN A CRASH. Mati starts and answers with its sealed memories
            # unreadable, which looks healthy on every other measure.
            # `memories_locked` is how someone finds out without guessing why
            # recall is empty.
            # ⚠️ `_vault_state()` NEVER PROMPTS — it is a status read on a
            # route that polls, so asking for Touch ID here would put a
            # fingerprint prompt on a timer. True = unlocked, False = locked,
            # None = no vault, so `is False` is the only correct test: `not
            # None` would report an unencrypted install as locked.
            "memories_locked": _vault_state() is False,
            "unlock_method": vault.unlock_method(),
            "vault_id": vault.vault_id(),
            "provider": self.provider,
            "llm": self._llm_detail,
            # Which model speaks, whether its words leave this machine's
            # network, and — if the chosen one could not be used — why not.
            "model": getattr(self.llm, "label", None)
                     or getattr(self.llm, "model", None),
            "model_remote": bool(getattr(self.llm, "remote", False)),
            "model_fallback": self._model_fallback,
            "uptime_seconds": round(time.time() - self.started, 1),
            # Stale means: the files changed after this process imported them,
            # so what is answering is not what is on disk. See `_code_stamp`.
            "code_stale": _code_stamp() > _CODE_AT_START + 1.0,
            # ⚠️ WITHOUT THIS, AN AMNESIAC SERVER LOOKS HEALTHY. Bodies are
            # sealed; a process that cannot unlock reads none of them, and
            # every other field here would still say it is fine. Touch ID needs
            # someone present, and a server started in the background — which
            # is what a restart does — cannot count on one.
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

# `/sleep`, `/wake` and `/rest` are not in this set: they are things a person
# does to their own Mati — "go to sleep now", "wake up" — not field mechanics.
# The gate answers "should a stranger reach this over HTTP", not "is this a
# user action".
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


# ── who may ask ──────────────────────────────────────────────────────────────
#
# ⚠️ LOOPBACK IS NOT THE SAME AS "ONLY THIS PERSON". Binding 127.0.0.1 keeps
# the network out, but every web page open in a browser on this Mac runs here
# too, and a page can send requests to 127.0.0.1. Two ways, both measured
# against a live server before this existed:
#
#   - a cross-site POST (`text/plain`, no preflight) was parsed and routed, so
#     any page could chat, set reminders, switch the terminal on through
#     `/settings`, stop the server, or hand `/library/read` a path to read;
#   - DNS rebinding — a page whose own name is re-pointed at 127.0.0.1 — was
#     answered in full with `Host: attacker.example`, which is everything this
#     server can show: conversations, memories, settings.
#
# So every request is checked before anything else runs. The Host must name
# this server, an Origin (sent by browsers) must be this server's own, and a
# body must be JSON — which a page on another origin cannot send without a
# preflight this server never answers. Nothing that already talks to it is
# affected: the app is served from this origin and sends JSON, the client and
# `curl` send no Origin, and a bodiless `curl -X POST` stays a valid request.
_LOCAL_NAMES = ("127.0.0.1", "localhost", "[::1]")
_BODY_METHODS = ("POST", "PUT", "PATCH", "DELETE")


def foreign_request(method, headers, port):
    """Why this request must be refused, or None if it may proceed.

    `headers` is anything with `.get` — the request's own headers.
    """
    own = {f"{h}:{port}" for h in _LOCAL_NAMES}
    host = (headers.get("Host") or "").strip().lower()
    # ⚠️ AN ABSENT HOST IS ALLOWED, a WRONG one is not. Every browser sends
    # one; an HTTP/1.0 tool may not, and has no page behind it.
    if host and host not in own:
        return f"this server answers to 127.0.0.1:{port}, not {host}"
    origin = headers.get("Origin")
    if origin is not None:
        # ⚠️ "null" IS A FOREIGN ORIGIN: sandboxed frames and file:// pages
        # send it, and neither is this app.
        if origin.strip().lower() not in {f"http://{h}" for h in own}:
            return "requests from other pages are refused"
    if method.upper() in _BODY_METHODS:
        ctype = (headers.get("Content-Type") or "").split(";")[0]
        try:
            n = int(headers.get("Content-Length") or 0)
        except ValueError:
            return "unreadable Content-Length"
        if ((n or headers.get("Transfer-Encoding"))
                and ctype.strip().lower() != "application/json"):
            return "a request body must be JSON"
    return None


# ⚠️ THE PAGE POLICY HASHES THE APP'S OWN INLINE SCRIPTS, and allows nothing
# else to run. The app is one inline <script>; the avatar frame has its own.
# A hash, unlike 'unsafe-inline', also forbids inline event handlers — which
# is how markup slipped into the page (an `<img onerror>` in a title) would
# run. Computed as each page is served, so editing the app cannot leave a
# stale hash behind. Films stream from archive.org into the page's <video>,
# so media is the one thing allowed from elsewhere.
_INLINE_SCRIPT = re.compile(
    rb"<script\b(?![^>]*\bsrc\s*=)[^>]*>(.*?)</script\s*>", re.S | re.I)


def page_policy(html):
    """The Content-Security-Policy for one served HTML page (bytes)."""
    hashes = []
    for m in _INLINE_SCRIPT.finditer(html):
        # Browsers hash the script with line endings normalised to LF.
        text = m.group(1).replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        digest = base64.b64encode(hashlib.sha256(text).digest()).decode()
        hashes.append(f"'sha256-{digest}'")
    return ("default-src 'self'; "
            f"script-src 'self' {' '.join(hashes)}".rstrip() + "; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "media-src 'self' https://archive.org https://*.archive.org; "
            "connect-src 'self'; "
            "frame-src 'self'; frame-ancestors 'self'; "
            "object-src 'none'; base-uri 'none'; form-action 'self'")


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
        self.send_header("X-Content-Type-Options", "nosniff")
        if ext == ".html":
            self.send_header("Content-Security-Policy", page_policy(body))
            self.send_header("X-Frame-Options", "SAMEORIGIN")
            self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(body)
        return True

    def _refuses_foreign(self):
        """Answer 403 and return True when the request is not this app's.

        ⚠️ CALLED FIRST BY EVERY METHOD, before a body is read or a route is
        matched. See `foreign_request`.
        """
        why = foreign_request(self.command, self.headers,
                              self.server.server_address[1])
        if not why:
            return False
        # ⚠️ AN UNREAD BODY WOULD BE TAKEN FOR THE NEXT REQUEST on a kept-alive
        # connection, so the connection ends with the refusal.
        self.close_connection = True
        self._json({"error": "refused", "why": why}, 403)
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
        if self._refuses_foreign():
            return
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
        # HERE RATHER THAN IN THE APP. The terminal is available only once the
        # person switches it on in the app's settings — and a toggle that only
        # hid a BUTTON would leave every route reachable with curl. Hiding a
        # door is not locking it, and these are the routes that reindex the
        # field, imprint text, and hand the field over. Loopback-only is not an
        # argument: anything running as this user could reach them.
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
                comfortable = round(max(1.0, (prof.get("ram_gb") or 8) * 0.3), 1)
                # ⚠️ ONE LIST FOR THE SELECTOR: the local models Ollama has,
                # then the connections the person added. Keys are never in it
                # — only whether one is stored, which needs no unlocking.
                from . import credentials as _cr, llm as _L
                from . import model_settings as _ms
                ch = _c.model_choice()
                # ⚠️ A LICENCE THAT MUST BE AGREED TO travels with each model,
                # so the window can ask EVERY time it is chosen (the user).
                local = [{"id": f"local:{n}", "name": n,
                          "gb": sizes.get(n),
                          "tight": bool(sizes.get(n) and sizes[n] > comfortable),
                          "licence": _ms.licence_for(n)}
                         for n in (names or [])]
                conns = [{"id": f"conn:{c['id']}", "name": c.get("label")
                          or c["model"], "model": c["model"], "kind": c["kind"],
                          "remote": c["kind"] == "anthropic"
                          or not _L.is_local_url(c.get("url")),
                          "has_key": _cr.have(_cr.MODEL_PREFIX + c["id"]),
                          "licence": _ms.licence_for(c["model"])}
                         for c in _c.model_connections()]
                return self._json({
                    "current": ch["choice"], "label": ch["label"],
                    "remote": any(x["id"] == ch["choice"] and x["remote"]
                                  for x in conns),
                    "fallback": svc._model_fallback,
                    # The model in use, if its licence is not yet agreed to
                    # in this run — the window asks before the first reply.
                    "needs_agreement": (None if _ms.agreed(ch["model"])
                                        else _ms.licence_for(ch["model"])),
                    "local": local, "connections": conns, "reachable": ok,
                    # The old shape, for anything still reading it.
                    "models": names, "sizes": sizes,
                    "ram_gb": prof.get("ram_gb"),
                    "comfortable_gb": comfortable})
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
            # `radio.remember_found` already keeps the last search so that
            # "next station" and "the third one" work by voice; this exposes
            # the same list to the page, so the buttons and the words are
            # looking at ONE thing. A separate list for the UI would drift
            # from what "next station" means.
            try:
                from . import media, radio
                # ⚠️ PRESETS TRAVEL WITH IT, which is the car-radio logic: a
                # found list matters just after a search, and once something is
                # playing the useful buttons are your own. One request returns
                # both so the page never has to decide which to fetch — and
                # never shows a stale half.
                from . import presets as _pre
                return self._json({
                    "stations": radio.listing(limit=5),
                    "presets": [{"slot": p["slot"], "name": p["name"]}
                                for p in (_pre.all_presets() or [])],
                    "playing": media.playing_now(),
                    "paused": media.paused_now(),
                    "remaining": radio.remaining(),
                })
            except Exception as e:                           # noqa: BLE001
                return self._json({"stations": [], "error": str(e)})
        if path == "/playing":
            try:
                from . import media
                return self._json({"playing": media.playing_now(),
                                   "paused": media.paused_now()})
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
        if self._refuses_foreign():
            return
        path = self.path.split("?")[0].rstrip("/")
        if path.startswith("/sessions/"):
            sid = path[len("/sessions/"):]
            return self._json({"cleared": self.service.clear_session(sid)})
        return self._json({"error": "not found", "path": path}, 404)

    def do_POST(self):
        if self._refuses_foreign():
            return
        path = self.path.split("?")[0].rstrip("/") or "/"
        body = self._body()
        if body is None:
            return self._json({"error": "invalid JSON"}, 400)
        svc = self.service

        if path == "/slow":
            # ⚠️ NEVER CALL `_body()` TWICE. `do_POST` has already read the
            # request body above; a second call blocks on `rfile.read(n)`
            # waiting for bytes that were consumed, and the route simply hangs
            # until the client gives up.
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
        # ⚠️ THE STOP BUTTON, AND IT MUST NOT GO THROUGH A TURN. A stop that
        # needs a sentence parsed, a field read and an episode written is not a
        # stop button. This does one thing.
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
        if path == "/model":
            # ⚠️ THE MODEL SELECTOR. Takes effect from the next reply, no
            # restart (`switch_model`). A choice that cannot be used is put
            # BACK — a bad pick must not leave Mati speaking through a model
            # the person never chose without saying so.
            from . import config as _c
            body = body or {}
            ch = str(body.get("choice") or "")
            if ch.startswith("local:"):
                from . import machine as _m
                _ok, names = _m.ollama_ready()
                if ch[len("local:"):] not in (names or []):
                    return self._json({"error": "that model is not in Ollama "
                                       "on this Mac"}, 400)
            elif ch.startswith("conn:"):
                if not _c.connection(ch[len("conn:"):]):
                    return self._json({"error": "no such connection"}, 400)
            else:
                return self._json({"error": "choose local:<name> or "
                                   "conn:<id>"}, 400)
            # ⚠️ A LICENCE IS AGREED TO EVERY TIME THE MODEL IS CHOSEN
            # (the user). The window shows it and sends `agree`; without it the
            # choice is refused, whatever was agreed before.
            from . import model_settings as _ms
            _name = (ch[len("local:"):] if ch.startswith("local:")
                     else _c.connection(ch[len("conn:"):])["model"])
            _lic = _ms.licence_for(_name)
            if _lic:
                if body.get("agree") is not True:
                    return self._json({"error": "its licence has to be "
                                       "agreed to", "licence": _lic}, 409)
                _ms.agree(_name)
            before = _c.model_choice()["choice"]
            _c.save_settings(model_choice=ch, **(
                {"local_model": ch[len("local:"):]}
                if ch.startswith("local:") else {}))
            res = svc.switch_model()
            if not res.get("switched"):
                _c.save_settings(model_choice=before)
                svc.switch_model()
                return self._json({"error": res.get("fallback")
                                   or res.get("error") or "it could not be "
                                   "used", "kept": before}, 409)
            return self._json(dict(res, choice=ch))
        if path == "/model/connect":
            # ⚠️ ADDING A SERVICE, with the key the person brings. Tested
            # BEFORE anything is saved, so a mistyped key is a sentence now
            # rather than a broken reply later. ⚠️ A REMOTE MODEL RECEIVES
            # EVERY TURN'S PROMPT — memories, profile, the day's conversation
            # — so the window shows that warning and this refuses without
            # `understood`. The key goes straight to the sealed store.
            import secrets as _sec
            from . import config as _c, credentials as _cr, llm as _L
            body = body or {}
            kind = body.get("kind")
            model = " ".join(str(body.get("model") or "").split())
            label = " ".join(str(body.get("label") or "").split())[:40]
            url = str(body.get("url") or "").strip().rstrip("/")
            key = " ".join(str(body.get("key") or "").split()) or None
            if kind not in _c.CONNECTION_KINDS or not model:
                return self._json({"error": "a kind and a model name are "
                                   "needed"}, 400)
            if kind == "anthropic":
                url = _c.ANTHROPIC_URL
                if not key:
                    return self._json({"error": "Claude needs an API key"},
                                      400)
            remote = kind == "anthropic" or not _L.is_local_url(url)
            if remote and body.get("understood") is not True:
                return self._json({"error": "the warning has to be accepted "
                                   "first"}, 400)
            try:
                test = (_L.ClaudeLLM("", model, key, label=label or "Claude")
                        if kind == "anthropic" else
                        _L.OpenAICompatLLM("", model, url, key=key,
                                           label=label or model))
                ok, detail = test.available()
            except Exception as e:                           # noqa: BLE001
                ok, detail = False, str(e)
            if not ok:
                return self._json({"error": detail}, 400)
            cid = (re.sub(r"[^a-z0-9]+", "-", (label or model).lower())
                   .strip("-")[:24] or "model") + "-" + _sec.token_hex(2)
            conns = _c.model_connections() + [{
                "id": cid, "kind": kind, "label": label or model,
                "url": url, "model": model}]
            _c.save_settings(model_connections=conns)
            if key:
                try:
                    _cr.set_key(_cr.MODEL_PREFIX + cid, key)
                except Exception as e:                       # noqa: BLE001
                    _c.save_settings(model_connections=[
                        c for c in conns if c["id"] != cid])
                    return self._json({"error": f"the key could not be "
                                       f"stored: {e}"}, 500)
            before = _c.model_choice()["choice"]
            _c.save_settings(model_choice=f"conn:{cid}")
            res = svc.switch_model()
            if not res.get("switched"):
                _c.save_settings(model_choice=before)
                svc.switch_model()
            return self._json(dict(res, id=f"conn:{cid}", added=True))
        if path == "/model/agree":
            # Agreeing to the licence of the model ALREADY in use, for this
            # run — asked by the window before the first reply after a start.
            from . import config as _c, model_settings as _ms
            model = _c.model_choice()["model"]
            if str((body or {}).get("model") or "") != model:
                return self._json({"error": "that is not the model in use"},
                                  400)
            _ms.agree(model)
            return self._json({"agreed": model})
        if path == "/model/forget":
            # Removes a connection AND its key. If it was speaking, the local
            # model speaks from the next reply.
            from . import config as _c, credentials as _cr
            cid = str((body or {}).get("id") or "")
            cid = cid[len("conn:"):] if cid.startswith("conn:") else cid
            if not _c.connection(cid):
                return self._json({"error": "no such connection"}, 400)
            _c.save_settings(model_connections=[
                c for c in _c.model_connections() if c["id"] != cid])
            try:
                _cr.forget(_cr.MODEL_PREFIX + cid)
            except Exception:
                pass
            if _c._settings().get("model_choice") == f"conn:{cid}":
                _c.save_settings(model_choice=f"local:{svc._local_name()}")
                svc.switch_model()
            return self._json({"forgotten": cid})
        if path == "/voice/sample":
            from . import speech as _sp
            name = str((body or {}).get("voice") or "")
            return self._json({"playing": _sp.sample(name)})
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
            # ⚠️ SHELVING HAS AN INVERSE, or "archive instead of delete" would
            # be a half-promise: an archived work stays in the house, and with
            # no way back it would be off the shelf permanently — a delete that
            # merely keeps the rows.
            #
            # ⚠️ `shelved = NULL` IS THE WHOLE OPERATION, and it is why
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
                    # ⚠️ `verb: "play"` IS REQUIRED. `media.answer` has
                    # `if verb != "play": return None` BEFORE it reads `slot`,
                    # so a bare {"slot": n} would fall straight through — the
                    # button would report success with an empty `said` and
                    # nothing playing. A preset press IS a play that names a
                    # button.
                    return self._json(
                        {"said": media.answer({"verb": "play", "slot": n}),
                         "playing": media.playing_now()})
                except Exception as e:                       # noqa: BLE001
                    return self._json({"error": str(e)}, 500)
            try:
                # ⚠️ `media.act`, NOT `radio.nth_found`. `nth_found` only
                # RETURNS the station — it does not play it. Routing the button
                # through the same `answer({"verb": "nth"})` the spoken "the
                # third one" uses means one code path, one set of failure
                # messages, and no way for a tap and a sentence to disagree.
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
        # The play/pause buttons under the now-playing label. Like `/silence`,
        # neither spends a turn.
        if path in ("/pause", "/resume"):
            try:
                from . import media
                said = (media.pause_all() if path == "/pause"
                        else media.resume_all())
                return self._json({"said": said,
                                   "playing": media.playing_now(),
                                   "paused": media.paused_now()})
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
            # ⚠️ SLEEP IS THE NIGHT, NEVER THE HANDOFF. It keeps the field and
            # rests, exactly as /rest does. Handing the field over is
            # /release, and only that.
            return self._json(svc.rest())
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
        if path == "/unlock":
            # ⚠️ THE BODY MAY CARRY THE PASSPHRASE. Only the outcome goes
            # back, and nothing about the request is written anywhere.
            out = svc.unlock((body or {}).get("passphrase"))
            status = 200
            if out.get("unlocked") is False:
                status = 429 if out.get("retry_after") else 403
            return self._json(out, status)

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
            # whichever comes first. Bounded because a queued book at full
            # speed would silence every other background function.
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

    ⚠️ WITHOUT THIS, RADIO HAS NO SOUND. AVFoundation does its asset loading
    against the **main thread's** run loop, and `serve_forever` is a selector
    loop that never runs one. Unpumped, the player accepts `play()`, reports
    `rate=1.0` — the REQUESTED rate — and sits silent indefinitely at status
    `Unknown`, with no error: silence with a confirmation on top.

    ⚠️ `service_actions` IS THE RIGHT HOOK, AND A SEPARATE THREAD IS THE WRONG
    FIX. Run loops are per-thread, but a dedicated audio thread with its own
    loop does not help: the loading needs the MAIN thread's. `serve_forever`
    calls `service_actions()` once per `poll_interval` pass, on the main
    thread.

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
        # loop gets serviced — so it is also the audio pipeline's clock. At
        # 0.5s a live stream gets ready, but a podcast episode behind a long
        # redirect chain times out: the loop is starved, not the URL.
        #
        # ⚠️ IT DOES NOT BLOCK. `pump(0.0)` drains what is pending and returns;
        # what changes here is how OFTEN it gets the chance. The cost is a
        # `select()` wakeup 20×/s on an idle loopback server.
        httpd.serve_forever(poll_interval=0.05)
    except KeyboardInterrupt:
        print("\n  [mati] stopping", file=sys.stderr)
    finally:
        # Release BEFORE shutdown, and the order is not cosmetic. `release()`
        # drains the work queue and snapshots θ₀; `shutdown()` sets the flag
        # the worker loop exits on. Reversed, the worker dies with items still
        # queued and `work.join()` waits forever for a `task_done` that will
        # never come — a hang on exit.
        #
        # Without this, quitting would drop any queued-but-unwritten episode:
        # Mati has answered, the person has seen the reply, and the memory of
        # it would be thrown away. Consolidation is deferred off the turn to
        # keep replies fast, which means exit is exactly when the debt is
        # outstanding.
        # ⚠️ BOUNDED. A wedged field must not let the process start exiting and
        # never finish — the port closed, still holding unsaved θ₀ — so release
        # gets its own thread and its own deadline, and the process exits
        # either way. A shutdown that cannot finish is worse than one that
        # finishes badly: the latter can be retried.
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
