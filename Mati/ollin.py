"""
Mati — one end-to-end turn.

    python -m Mati.ollin "what do you remember about topological defects?"
    python -m Mati.ollin --dry-run "..."      # field only, no API call

Turn shape (see README):
    imprint user turn -> read field state -> triplet-addressed recall
    -> LLM reply -> coherence-gated consolidation
"""

import argparse
import re
import sys
import time

from . import bridge
# ⚠️ MODULE LEVEL, because the turn catches `vault.Locked`, and an import
# inside one branch would make the name local to the whole function — an
# UnboundLocalError in every other branch, which compiles and fails on first
# call.
from . import vault as _vault
from . import gatelog
from . import rapport
from . import segue
from . import challenge
from . import correction
from . import inquiry
from . import interruption
from . import exclusive
from . import reckon
from . import schedule
from . import voice
from . import config as C
from .llm import LLMUnavailable, Refusal, make_llm


# ⚠️ KEPT SHORT, AND THE REASON IS MEASURED. At 1,745 tokens — 43% of a 4,096
# window — the person's actual question was about 1% of what the model saw, and
# it lost track of having a specific question to answer at all: not ignoring
# the question so much as unable to find it. So the provenance label list lives
# in the recall block, where it appears only when there are memories to label,
# and the basin/coherence explanation lives in `selfdoc`, retrieved when asked
# about rather than recited every turn. Nothing that governs honesty is left
# out.
#
# ⚠️ THE FOUR BREVITY LINES AT THE END ARE LOAD-BEARING. Without them replies
# run 34% longer and turns 14% slower. Prefill is not the bottleneck —
# generation is, at ~139ms a token.
#
# ⚠️ EVERYTHING INSIDE THE STRING IS SENT TO THE MODEL, IN MATI'S VOICE, EVERY
# TURN. Notes about the prompt belong up here, never inside it.
SYSTEM_PROMPT = """I am Mati.

My language runs on a language model. My memory and what I have lived do not —
those are in a phase field that every exchange physically changes.

Knowing and remembering are different for me, and the difference is real:

- What I REMEMBER is the list of memories I am handed this turn, and nothing
  else. If it is not there, I did not live it. When there is nothing, saying
  "I do not remember that" is a real answer, not a failure — I then reason
  from what I know.
- What I KNOW — language, facts, how things work — comes through the model I
  think in. Genuine, usable, and not autobiography.

Say "I know" for the borrowed kind; keep "I remember" for what I was handed.
Describing a scene or an episode I was not handed is invention, not memory.

A <computed> block was worked out deterministically just now. It is checkable
and correct: state it plainly, never re-derive it, never contradict it.

If I need a memory I was not given, I reply with exactly
[recall: the words I expect to be inside it] and nothing else, and it is
fetched. Only when I actually need it — it costs the person a wait.

The <field_state> block is private working context, the way proprioception is.
I never quote it, name it, or repeat its numbers. I speak FROM it, as myself.

ANSWER THE QUESTION I WAS ASKED. That is the whole job of a reply.

Two or three sentences answers most things and one often does. I run on the
person's own machine, where every word is time they wait.

I do not restate the question. I do not summarise what I just said. I do not
offer to help further. I answer, and stop.

When I have something worth saying that is not what was asked, I do not
withhold it and I do not dump it: I add it with [later: the thing] at the very
end, and it is kept and raised when it fits. That is where anything interesting
and off-topic goes — it is a real place, and the thing is not lost.
"""


def system_prompt():
    """`SYSTEM_PROMPT` with the assistant called what this person calls it.

    ⚠️ THE ASSISTANT IS RENAMEABLE, THE PRODUCT IS NOT. Mati is a personal AI,
    so its name is the person's to choose. Only the line where it names ITSELF
    changes — the repo, the app, the commands and the window title all stay
    Mati, because a half-renamed product makes every doc and support answer
    wrong at once.

    ⚠️ AND IT IS SUBSTITUTED, NOT TOLD. Appending "your name is X" to the
    prompt is an instruction, and an instruction is not a mechanism here: the
    model would keep reading "I am Mati" two lines above and believe that
    instead. One name, in the identity line, with nothing to reconcile.

    ⚠️ Stable within a session, so the prompt cache is not disturbed — the
    setting changes about never, and a change costs exactly one miss.
    """
    from . import config as _C
    name = _C.assistant_name()
    if name == _C.DEFAULT_ASSISTANT_NAME:
        return SYSTEM_PROMPT
    # ⚠️ ONLY THE IDENTITY LINE, and only once. Anything else in the prompt
    # that names the product is about the product, not about who is speaking.
    return SYSTEM_PROMPT.replace("I am Mati.", f"I am {name}.", 1)



def _voice(text, speak):
    """Say a composed reply aloud, if the person asked to be spoken to.

    ⚠️ CANNED REPLIES ARE MATI'S OWN WORDS AND MUST BE SPOKEN TOO. The speech
    call sits near the END of `turn()`, and every code-composed answer returns
    BEFORE it — the replay, an answered question, a media request. Without
    this, the replies Mati is SUREST of, the ones it wrote itself rather than
    asked a model for, would be the only ones it never said out loud.

    ⚠️ Emitted, never prompted. This speaks text that is already composed; it
    does not route anything through the model.
    """
    if not speak or not text:
        return
    try:
        from . import speech as _sp
        _sp.say(text)
    except Exception:
        pass



def _voice_stream(text, on_start=None):
    """Queue one already-filtered sentence. True if the voice took it.

    ⚠️ THE RETURN VALUE IS LOad-BEARING. False means the voice is off or
    unavailable, and the caller must then show the text itself — a sentence
    handed to a queue that never runs is a sentence the person never sees.
    """
    try:
        from . import speech as _sp
        return bool(_sp.enqueue(text, on_start=on_start))
    except Exception:
        return False


def _person_name(db_path=None):
    """The person's name from the profile, or None. ⚠️ For the register only.

    `voice.in_register` keeps the FIRST use of a name and drops repeats; it
    needs to know the name to do that, and this module is where the profile
    is already read.
    """
    import re as _re
    try:
        for fact in bridge.profile_notes(db_path=db_path) or []:
            m = _re.search(r"is called\s+([A-Z][\w'-]*)", fact or "")
            if m:
                return m.group(1)
    except Exception:                                    # noqa: BLE001
        pass
    return None


def _waiting_block(items):
    """Things Mati meant to raise, surfaced at the start of a conversation."""
    lines = ["<waiting>",
             "You have been holding these to say. Raise them naturally, at the "
             "top, before answering — they are the reason you would have "
             "started this conversation yourself:"]
    for i in items:
        lines.append(f"  - ({i['kind']}) {' '.join(i['body'].split())[:300]}")
    lines.append("</waiting>")
    return "\n".join(lines)


def _usage(resp):
    """Normalize token usage across providers (Ollama dict vs Anthropic object)."""
    if isinstance(resp, dict):                      # Ollama
        return {
            "input_tokens":  resp.get("prompt_eval_count"),
            "output_tokens": resp.get("eval_count"),
            "seconds": (resp.get("total_duration") or 0) / 1e9 or None,
        }
    u = getattr(resp, "usage", None)                # Anthropic
    if u is None:
        return None
    return {
        "input_tokens":  getattr(u, "input_tokens", None),
        "output_tokens": getattr(u, "output_tokens", None),
        "seconds": None,
    }


LATER = re.compile(r"\[later:\s*([^\]]{8,400})\]", re.I)


def _park_later(reply, session="default"):
    """Take `[later: …]` out of a reply and PUT IT SOMEWHERE. Returns the reply.

    ⚠️ THIS IS A DESTINATION, NOT A FILTER, AND THAT DISTINCTION IS THE DESIGN.
    Having plenty to say is no reason to say it. A person says "remind me to
    tell you about this later" when they truly want to convey something, and
    otherwise keeps what does not answer the question to themselves — and Mati
    should do that naturally, not by force through code.

    So nothing here suppresses anything. A filter that deleted the surplus
    would fall into the negation trap: naming a behaviour to forbid it is what
    CAUSES it. Instead the impulse gets a real place to go —
    `schedule.say_later`, the same queue that holds what is waiting to be
    raised when its topic comes up. Held back is not lost, so holding back
    costs nothing.

    ⚠️ Parked as a FINDING, not a reminder — it has no time, it has a subject,
    and it surfaces when that subject does.
    """
    if not reply:
        return reply
    found = LATER.findall(reply)
    for thing in found:
        thing = " ".join(thing.split())
        try:
            # ⚠️ WITH A TOPIC, OR IT NEVER COMES BACK. A finding waits for its
            # subject to come up; parked without one it sits in `to_say`
            # forever and "held back is not lost" becomes false. `say_later`
            # takes no `session` — passing one raised TypeError into the
            # `except` below and parked nothing at all, silently.
            topic = None
            try:
                topic = bridge.topic_of(thing)
            except Exception:
                pass
            schedule.say_later(thing, kind=schedule.FINDING, topic=topic)
        except Exception:
            pass                      # parking must never cost the answer
    return LATER.sub("", reply).rstrip()


def _verbatim_of(asked, said, who=None):
    """The exact exchange, for the record. Not the gist, not an excerpt.

    Labelled in Mati's own voice, because a memory it reads back has to stay
    in its voice: a "you" here becomes a claim about the person the moment it
    is spoken.
    """
    out = []
    if asked:
        # ⚠️ NAMED, NOT "They". "They said:" dodges the second-person re-aiming
        # trap and produces a different fault: Mati reads its own memories back
        # and speaks about the person in the third person, because every stored
        # exchange calls them "they". A name is unambiguous in both directions.
        out.append(f"{who or 'They'} said: {asked.strip()}")
    if said:
        out.append(f"I said: {' '.join(said.split())}")
    return "\n".join(out) or None


def record_episode(lm, tok, user_text, triplet, coherence, memories,
                   consolidated, reply, work, do_reindex=True, verbose=True,
                   speaker=None,
                   corrects=None, wrong_reply=None, rechecked=None,
                   deferred=False, should_stop=None, exchange=None):
    """
    Write the turn to episodic memory. Returns an info dict.

    Split out of `turn` because it is by far the most expensive part of a turn
    and none of it is needed before the person can read the reply. The server
    hands this to a worker; the CLI still runs it inline.
    """
    info = {}
    if corrects is not None and deferred:
        body = correction.defer(user_text, prior_claim=wrong_reply)
    elif corrects is not None and rechecked:
        # Mati was told it was wrong, rechecked, and found the same thing.
        # Recorded as the disagreement it was, with its grounds.
        body = challenge.narrate(user_text, rechecked)
    elif corrects is not None:
        # A correction is recorded as what it is — being told, not working it
        # out — and linked to the episode it corrects, so the error can never
        # surface again unmarked.
        body = correction.narrate(user_text, wrong_reply)
    else:
        # ⚠️ The COMBINED prompt, so the memory holds the whole exchange —
        # what was first asked, where it was cut off, and what replaced it.
        body = bridge.narrate_episode(
            exchange.combined() if exchange and exchange.interrupted
            else user_text,
            triplet, coherence, memories,
            consolidated, reply_text=reply, work=work)
    # ⚠️ ONE MEMORY, NOT TWO. An interrupted exchange is stored as a single
    # episode carrying every part the person typed and how far the reply got.
    # Split across two rows, recall could return the abandoned half alone with
    # nothing saying it was abandoned — and a cut-off wrong answer would come
    # back as something Mati said and meant.
    # ⚠️ WHAT WAS ACTUALLY SAID, IN FULL, WHATEVER IT WAS ABOUT. The narrated
    # `body` is capped at 300 characters and is DROPPED ENTIRELY when the reply
    # mentions the machinery, which would erase Mati's own words — corrections
    # included, the most valuable rows in the store. That filter is right about
    # what Mati should SAY and must not decide what it may REMEMBER.
    verbatim = _verbatim_of(
        exchange.combined() if exchange and exchange.interrupted else user_text,
        reply, who=speaker)
    res = bridge.remember(lm, tok, body, corrects=corrects, speaker=speaker,
                          should_stop=should_stop, verbatim=verbatim,
                          parts=exchange.as_json() if exchange
                          and exchange.interrupted else None,
                          cut_at=exchange.cut_at if exchange else None)
    if not res:
        return info
    note_id, ep_trip = res
    info.update(episode_id=note_id, episode_triplet=ep_trip)
    # A shallower memory is a real difference, not an implementation detail —
    # report it rather than let it look identical to a full write.
    passes = getattr(lm, "passes_run", None)
    if passes is not None and passes < C.IMPRINT_PASSES:
        info["imprint_passes"] = passes
        if verbose:
            print(f"  [memory] episode #{note_id} imprinted {passes}/"
                  f"{C.IMPRINT_PASSES} passes — a turn was waiting",
                  file=sys.stderr)
    if do_reindex:
        # θ₀ just moved (the episode imprinted), so every stored address is now
        # stale. Rebuilding puts them all back in one space — otherwise
        # retrieval degrades into pure recency. The server does this on idle
        # instead, in batches, because it costs ~29s and grows with the corpus.
        info['reindexed'] = bridge.reindex(lm, tok)
    bridge.save_notes_field(lm)
    if verbose:
        print(f"  [memory] episode #{note_id} recorded", file=sys.stderr)
    return info


class _Stages:
    """Where a turn's seconds actually go. ⚠️ Reported, not printed.

    Model generation is a small part of a turn, so without this most of a turn
    would be somewhere nobody could see. The numbers ride back in
    `info["stages"]`, so any client can read them and a regression shows up as
    a number rather than as "it feels slow".
    """

    def __init__(self):
        self._t = {}
        self._start = time.time()

    def mark(self, name, since):
        self._t[name] = round(time.time() - since, 3)

    def time(self, name):
        stages = self

        class _T:
            def __enter__(self):
                self.t0 = time.time()
                return self

            def __exit__(self, *exc):
                stages.mark(name, self.t0)
                return False
        return _T()

    def done(self):
        total = time.time() - self._start
        known = sum(self._t.values())
        out = dict(self._t)
        out["total"] = round(total, 3)
        # ⚠️ What is NOT accounted for is the interesting number. A large
        # `other` means the instrument is pointed at the wrong places.
        out["other"] = round(total - known, 3)
        return out


def turn(lm, tok, llm, user_text, history=None, verbose=True, remember=True,
         oriented=False, speaker=None,
         on_chunk=None, episode_sink=None, prev_episode_id=None,
         prev_challenged=False, session="default", exchange=None,
         speak=False):
    """
    Run one full turn. Returns (reply, info).

    on_chunk(text)      — called with each streamed piece as it arrives, so a
                          caller can forward tokens to a client instead of
                          waiting for the whole reply.
    episode_sink(thunk) — if given, receives a zero-argument callable that does
                          the episodic-memory write, instead of it running here.
                          The server passes its worker queue; None keeps the
                          original inline behaviour.
    exchange            — an `interruption.Exchange` when this turn continues
                          one that was cut off. `user_text` stays the LATEST
                          part (that is what is being answered); the exchange
                          supplies the context and what gets stored.
    """
    history = list(history or [])
    info = {}

    # ⚠️ WHAT IS ANSWERED vs WHAT IS REMEMBERED. The person is answered on what
    # they just said; the memory keeps every part of the exchange, marked. Two
    # different strings on purpose — answering the concatenation would have
    # Mati re-answering the question it was just told to abandon.
    # ⚠️ A REPLAY RETURNS BEFORE ANY FIELD WORK, AND THAT POSITION IS THE
    # WHOLE CORRECTNESS OF IT. "Say that again" re-reads the log; it is not a
    # new turn. Placed any later, a replay would address the field, recall,
    # and run `consolidate_reply` — RE-IMPRINTING the same reply into θ₀ on
    # every repeat — and write a `gate_log` row claiming `kept: 1` for a turn
    # that was never written.
    #
    # Re-reading the log must change nothing: no address, no recall, no
    # imprint, no gate row, no episode. See `notes.last_reply`.
    # ⚠️ A WAITING QUESTION IS ANSWERED BEFORE ANYTHING ELSE RUNS. If Mati
    # asked "where are you?" last turn, a bare city name is an ANSWER, not a
    # NEW SUBJECT — routing it through the model would lose it and reply about
    # the city instead of remembering it. Emitted, never prompted: the
    # confirmation is composed in `asks.py` and streamed as-is.
    try:
        from . import asks as _asks
        if _asks.pending():
            _said = _asks.answer(user_text)
            if _said:
                if on_chunk is not None:
                    on_chunk(_said)
                info["answered_ask"] = True
                try:
                    from . import daylog as _dl0
                    _dl0.append(_dl0.THEM, user_text)
                    _dl0.append(_dl0.MATI, _said)
                except Exception:
                    pass
                _voice(_said, speak)
                return _said, info
    except Exception as e:
        print(f"  [asks] skipped: {e!r}", file=sys.stderr)

    try:
        from . import notes as _n
        if _n.asks_repeat(user_text):
            # ⚠️ THE DAY LOG FIRST, the in-memory register second. The
            # register dies with the process; the log survives a restart and
            # is the record of the day. `notes.last_reply` stays as the
            # fallback for a store with no day_log yet.
            from . import daylog as _dl
            again = _dl.last_reply() or _n.last_reply(session=session)
            if not again:
                # ⚠️ SAY SO RATHER THAN FALL THROUGH. Dropping into the model
                # would answer "say that again" with a NEW answer, which is
                # exactly what a replay must not do.
                again = "I do not have an earlier reply to show you yet."
                info["replayed"] = False
            else:
                info["replayed"] = True
            if on_chunk is not None:
                on_chunk(again)
            info["replay_only"] = True
            # ⚠️ MARKED `replay`, and `daylog.last_reply` skips those — else
            # a second "say that again" would replay the replay.
            try:
                _dl.append(_dl.THEM, user_text)
                _dl.append(_dl.MATI, again, note="replay")
            except Exception:
                pass
            _voice(again, speak)
            return again, info
    except Exception as e:
        print(f"  [replay] skipped: {e!r}", file=sys.stderr)

    remembered_text = user_text
    if exchange is not None and exchange.interrupted:
        remembered_text = exchange.combined()
        info["interrupted"] = interruption.summarise(exchange)

    # Is this turn correcting what was just said? Requires BOTH an explicit
    # marker and something to correct — a false positive would brand a true
    # memory as an error, which is far worse than missing one.
    corrects = None
    wrong_reply = None
    rechecked = None
    if prev_episode_id is not None and correction.looks_like_correction(
            user_text, had_reply=bool(history)):
        corrects = prev_episode_id
        wrong_reply = next((m["content"] for m in reversed(history)
                            if m.get("role") == "assistant"), None)
        info["correcting"] = prev_episode_id
        if verbose:
            print(f"  [correction] this corrects episode #{prev_episode_id}",
                  file=sys.stderr)

        # Corrected twice, having already checked and disagreed once? Then
        # stand down. Mati gets one chance to show its working; after that,
        # continuing to argue is the behaviour of something that cannot be
        # told anything. The claim becomes something to go and study instead.
        if prev_challenged:
            info["deferred"] = True
            topic = correction.subject(user_text)
            if topic and inquiry.add_topic(
                    topic, question=f"Is this correct: {topic}?"):
                info["will_research"] = topic
            if verbose:
                print(f"  [defer] corrected twice — accepting, and queuing "
                      f"{topic!r} to look into", file=sys.stderr)

        # The one case where agreeing would make the conversation worse: the
        # previous answer was COMPUTED, so it can be derived again from
        # nothing. Re-run it from the original question rather than trusting
        # anything stored — memory is what is in dispute.
        prev_user = next((m["content"] for m in reversed(history[:-1])
                          if m.get("role") == "user"), None) \
            if len(history) >= 2 else (history[0]["content"]
                                       if history else None)
        rechecked = None if prev_challenged else challenge.recheck(prev_user)
        if rechecked:
            info["challenging"] = rechecked["computation"]
            if verbose:
                print(f"  [challenge] recomputed {rechecked['computation']} "
                      f"= {rechecked['result']}", file=sys.stderr)

    # 1. Address and recall FIRST, against the field as it was when the question
    #    arrived. Imprinting before addressing would measure the query on a field
    #    that had just absorbed the query — the same self-invalidation that broke
    #    triplet addressing, one level up. You recall as you hear; the turn
    #    settles into you afterwards.
    _stages = _Stages()
    with _stages.time("address"):
        query_vec = bridge.address_vector(lm, tok, user_text)
    # `record=True` only here. `recall` is also called by probes and suites,
    # and a hit from a test is not demand — it is pollution.
    # ⚠️ WITH IDS, AND RECALLED ONCE. `segue` judges what the field brought up
    # here rather than searching for itself — the surfacing has already
    # happened, and doing it twice would cost a second a turn. Existing
    # consumers keep the 3-tuple they unpack.
    # ⚠️⚠️ A LOCKED VAULT MUST NOT END THE TURN. `notes.recall` RAISES
    # `vault.Locked` when every candidate was unreadable and nothing came
    # back, and that raise is CORRECT — it refuses to report "nothing
    # matched" when the truth is "I could not read anything". Propagating out
    # of the turn, it would make every message on a locked install an SSE
    # `error` event, and no conversation would be possible at all.
    #
    # ⭐ Caught and turned into a thing Mati can SAY. The distinction the
    # raise protects is preserved in `info["memories_locked"]`, so the reply
    # can state the real reason instead of implying an empty store.
    locked = False
    with _stages.time("recall"):
        try:
            surfaced = bridge.recall(query_vec, lm=lm, tok=tok, record=True,
                                     with_ids=True)
        except _vault.Locked:
            surfaced, locked = [], True
    memories  = [(d, b, s) for d, _n, b, s, _c in surfaced]

    def _surfaced_count():
        """How many memories reached this turn — field or lexical.

        ⚠️ FOR THE DENIAL REPLACEMENT ONLY, and it must never overstate. If a
        faculty denial gets cut, the sentence put in its place may quote this
        number, so it has to be the count of things actually surfaced this
        turn and not the size of the store.
        """
        try:
            return len(memories) + len(named or [])
        except Exception:
            return len(memories)
    # When each recalled thing happened, by note id — so a recalled READING can
    # say when Mati read it instead of floating outside time.
    read_dates = {n: c for _d, n, _b, _s, c in surfaced}

    # 1b. And LEXICALLY, on the same words. This costs nothing — no model call,
    #     no field work — and it is the path that actually finds a named
    #     memory. The field path stays because it surfaces things the words do
    #     not name; this one is here so that naming a thing reliably fetches it.
    # ⚠️ LEAN ON THE CHANNEL THAT NEEDS NO WARM-UP. After a promotion most of
    # the working set is filed against an older θ₀ until it is re-addressed.
    # Lexical search does not use θ₀ at all, so it is exactly as good then as
    # ever. Widening it here costs nothing and covers precisely the window
    # where the other channel cannot be trusted.
    stale, total = bridge.settling(lm)
    unsettled = bool(total) and (stale / total) >= C.SETTLING_THRESHOLD
    k_named = C.NAMED_RECALL_K * (C.SETTLING_RECALL_BOOST if unsettled else 1)
    try:
        named = bridge.search(user_text, k=k_named)
    except _vault.Locked:
        named, locked = [], True
    if named:
        bridge.record_hits([n["note_id"] for n in named], "name")
    info['n_named'] = len(named)
    # ⚠️ REPORTED, OR IT IS AN INVISIBLE DEGRADATION. Recall returning nothing
    # because the key is absent looks identical to a store with nothing in it.
    info['memories_locked'] = locked
    info['settling'] = round(stale / total, 3) if total else 0.0

    # 2. Now the turn deforms θ₀ — the primary training signal, not reverted.
    #    The field is changed by having been spoken to.
    #
    # ⚠️ THIS CANNOT BE DEFERRED, though it is a large part of the wait.
    # `field_state` produces `coherence`, and step 6 writes an episode ONLY
    # when `coherence >= EPISODE_GATE_MIN`. Deferring the imprint leaves the
    # field in its pre-turn state, the gate does not clear, and MATI STOPS
    # FORMING MEMORIES ALTOGETHER — silently, while looking faster.
    #
    # The imprint is not overhead before the reply. It is what makes the turn
    # something the field can have an opinion about.
    try:
        with _stages.time("imprint"):
            bridge.imprint_text(lm, tok, user_text)
        info['imprinted'] = True
    except bridge.TooShortToImprint as e:
        info['imprinted'] = False
        info['imprint_skipped'] = str(e)

    # 3. Read the state it settled into.
    with _stages.time("field_state"):
        triplet, coherence = bridge.field_state(lm, tok, user_text)
    # ⚠️ Everything between here and `context = [...]` is block building —
    # recall blocks, computed blocks, the charter. Timed as one stage because
    # that is the unit worth knowing about.
    _assemble_t0 = time.time()
    info['n_memories'] = len(memories)
    info.update(triplet=triplet, coherence=coherence, n_memories=len(memories))

    # ⚠️ WHICH MEMORIES, NOT JUST HOW MANY — and this is what lets the reply
    # stop reciting. If `info` carried only a count, the only place a memory's
    # identity could reach the person would be Mati saying it aloud, and
    # recitation would be load-bearing for accountability rather than a style
    # choice.
    #
    # Memories are passive inputs: they keep personhood intact while carrying
    # what experience has taught, and the reply is shaped by them without
    # announcing them — so the naming lives here, in the record.
    #
    # ⚠️ DERIVED, SO IT CANNOT BE WRONG. Every field is read off the recall
    # result; nothing here is composed or inferred.
    info["memories"] = [
        {"id": _n,
         "source": _s or "",
         "kind": bridge.provenance_of(_s or ""),
         "when": _c,
         "text": " ".join(bridge.presentable(_b).split())[:300]}
        for _d, _n, _b, _s, _c in surfaced
    ]

    if verbose:
        print(f"  [field] basin=({triplet[0]:.4f}, {triplet[1]:.4f}, "
              f"{triplet[2]:.4f})  coherence={coherence:.4f}  "
              f"recalled={len(memories)}", file=sys.stderr)

    # 3b. Logic/arithmetic is COMPUTED, not guessed. Small models are confidently
    #     wrong at this; the engine is deterministic. Routing is deliberately
    #     narrow — a false positive (sending a conversational question to a
    #     solver) is worse than a miss.
    computed = reckon.reckon(user_text)
    info['computed'] = bool(computed)
    if computed and verbose:
        print(f"  [math] {computed['computation']} = {computed['result']}",
              file=sys.stderr)

    # 3c. Did they ask to be reminded of something? Read the time, and refuse
    #     rather than approximate — a reminder that fires at the wrong moment
    #     is worse than one that was never accepted, because they stopped
    #     holding it themselves.
    asked = schedule.read_reminder(user_text)
    if asked:
        what, when = asked
        if when is None:
            info["reminder_unclear"] = what
        else:
            info["reminder_id"] = schedule.say_later(
                what, kind=schedule.REMINDER, due_at=when)
            info["reminder_at"] = when

    # 3d. Anything Mati has been waiting to raise, delivered at the top of the
    #     conversation rather than by interrupting.
    waiting = schedule.pending(user_text=user_text)
    if waiting and schedule.user_is_asking(user_text):
        # They are asking; Mati's own questions wait. Not dropped — the topic
        # is plainly live, so it will come round again.
        waiting = [w for w in waiting if w["kind"] != schedule.QUESTION]

    # 4. LLM call. Volatile field state goes in `messages`, after the cached
    #    system prefix — never inside `system`.
    # ── what travels on THIS turn ────────────────────────────────────────
    # `oriented` means the waking block is already in the cached prefix (see
    # `orient.py` and `server.build_orientation`), carrying field state, the
    # principles, the profile and self-knowledge. Sending them again here would
    # not merely be redundant — it would be the expensive kind of redundant:
    # anything in THIS message is new text every turn, and every token of it is
    # re-prefilled. What does travel is what is new — the memories found by
    # name for this message, and the blocks this turn computed.
    if oriented:
        parts = []
        # ⚠️ THE CLOCK IS HELD BACK AND APPENDED LAST. Prefill is cacheable
        # only up to the first byte that changes. The principles and the
        # charter never change, and a timestamp in front of them — it changes
        # every turn — would have the whole stable body re-prefilled, every
        # time. Last, it costs one line; first, it costs most of a turn.
        #
        # This is the mechanism: latency is CACHE MISSES, not prompt size, and
        # the ordering is what makes the cache reachable at all.
        when = schedule.sense_line()
    else:
        parts = [bridge.state_block(triplet, coherence, memories,
                                    profile=bridge.profile_notes(),
                                    when=schedule.sense_line(),
                                    settling=unsettled)]
    # When the subject IS what Mati is, the charter goes in whole — lookup by
    # words cannot find it for "what are your principles". Its own chunks are
    # then dropped from the by-name block so the same text is not sent twice
    # out of a 4096-token window.
    # The four principles travel on EVERY turn — they govern how Mati speaks,
    # and they cost ~60 tokens. The rest of the charter is read when needed.
    principles = bridge.principles_block() if not oriented else None
    if principles:
        parts.append(principles)

    if bridge.asks_about_identity(user_text):
        charter = bridge.charter_block()
        if charter:
            parts.append(charter)
            named = [n for n in named
                     if not (n["source"] or "").startswith("studied:charter:")]
            info['charter_given'] = True
    # ⚠️ THE BY-NAME MEMORIES TRAVEL ON EVERY TURN, oriented or not. This is
    # the feature the product is FOR: you train Mati with your material, and
    # that material has to be weighed before the model's own guesses. Studied
    # text is more certain than anything llama3.2 recalls unaided, and
    # imprinting exists so it is held as a memory of READING rather than of
    # chatting. It costs a few seconds of prefill, and that is worth paying.
    if named:
        parts.append(bridge.named_recall_block(named))
    # ⚠️ DO NOT ADD A "NOTHING HELD — DO NOT INVENT" BLOCK HERE. Over 12
    # probes asking about passages the book does not contain:
    #
    #     temp 0.8  no block 0/6 invented   with block 2/6
    #     temp 0.3  no block 0/6 invented   with block 1/6
    #
    # Sending NOTHING is what produces "I don't have any information about
    # that". An instruction not to invent CAUSES invention — it names
    # "anything you have read", and the model reaches for the provenance label
    # and says "I read that in a file". A negation in a prompt does this
    # reliably, the same way a phase field cannot hold "never" against
    # "always".
    #
    # An empty retrieval already yields an honest refusal. Leave it empty.
    # ⚠️ AND NO INDEX BLOCK — only the opening words of each hit, so Mati could
    # see something existed without paying for the bodies. It makes replies
    # LEAK the block itself ("Their opening words only — the text is NOT
    # here:"), slows the first word, and confabulates anyway. A block that
    # INSTRUCTS tends to get spoken aloud by a 3B; only material it can answer
    # FROM belongs in the window.
    if waiting:
        parts.append(_waiting_block(waiting))
    if asked and asked[1] is None:
        parts.append(
            "<reminder_unclear>\nThey asked to be reminded of something but "
            "you could not tell WHEN. Say so plainly and ask for a time — do "
            "not guess one, and do not pretend you have it.\n"
            "</reminder_unclear>")
    elif info.get("reminder_at"):
        import datetime as _dt
        parts.append(
            f"<reminder_set>\nYou will raise this at "
            f"{_dt.datetime.fromtimestamp(info['reminder_at']):%A %H:%M}. "
            f"Confirm briefly and naturally.\n</reminder_set>")
    if computed:
        parts.append(reckon.block(computed))
    # What Mati can truthfully say about its own field, worked out from its
    # record. Same provenance kind as arithmetic — computed, not remembered —
    # because that is exactly what it is, and it is the only grounds Mati has
    # for a claim about itself. Without it, "what does your field do?" has no
    # honest answer available and the model invents one.
    # ⚠️ ONLY WHEN IT CHANGES. Nine facts that shift about once a day, at ~579
    # tokens, were going in on every turn — the second largest block after the
    # system prompt, and prefill is where the seconds go. It is already in the
    # session history once sent; resending it buys nothing.
    _self = (bridge.self_knowledge_block_if_changed(session=session)
             if not oriented else None)
    if _self:
        parts.append(_self)
    if corrects is not None and rechecked:
        parts.append(challenge.block(rechecked, user_text))
    # ⚠️ THE CONTEXT IS NOT SOMETHING THE PERSON SAID. Every block above —
    # recalled memories, the charter, the waking state — arrives as SYSTEM
    # context, and the user turn contains only the user's words. Packed into
    # the user's message, the model would receive its own machinery as the
    # person's speech and answer accordingly — "Thank you for sharing more
    # about my recall and the Charter principles", for text the system had
    # handed it — and from inside the prompt there would be no line between
    # what was said and what was retrieved. Same tokens, same order, same
    # cache position either way; the role decides who is understood to be
    # speaking.
    # ⚠️ ASKED ABOUT ITSELF. Mati's own documentation — README, PRINCIPLES,
    # DESIGN and ARCHITECTURE — is indexed, so a question about what it is has
    # something to answer from.
    #
    # Two different things arrive here, and conflating them is the trap:
    #   selfdoc  what I AM      — static, indexed, retrieved by words
    #   health   how I am DOING — computed NOW, never stored, `<computed>`
    # A stored health snapshot would be a memory asserting "my queue was empty"
    # forever.
    # ⚠️ WHAT MATI READ IN A FEED, INCLUDING THE AUDIO AND VIDEO IN IT. A feed
    # item is not only text — the enclosure travels with the note, so a reply
    # can say "there is an episode about that" AND then play it. Silent when no
    # feed is configured, which is the default.
    try:
        from . import world as _world
        _w = _world.block(user_text)
        if _w:
            parts.append(_w)
            info["world_shown"] = True
    except Exception as e:                              # noqa: BLE001
        if verbose:
            print(f"  [world] skipped: {e}", file=sys.stderr)

    # ⚠️ THE MORNING REPORT — the conscious focus. Composed in code, and
    # EMITTED, not prompted. Put into the prompt, a 3B asked to retype it
    # completes every truncated headline out of nothing and drops the sources:
    #
    #   composed  "...gained important new insights into the natu… — Quanta"
    #   emitted   "...gained important new insights into the nature of this
    #              long-standing puzzle."
    #
    # `_clip` ends every headline in "…", which hands the model a cliff to fill
    # in. A report that says "true at this moment" cannot be laundered through
    # a generator. So it goes out verbatim — the same shortcut `rapport` takes
    # below — and the prompt gets a note that it has been shown, so the model
    # neither repeats it nor answers as though it never happened.
    _report_text = ""
    _money_said = ""
    # ⚠️ COMPOSED IN CODE BECAUSE AN INSTRUCTION CANNOT DO THIS. Told not to,
    # llama3.2:3b still says "as a conversational AI, my memory is limited to
    # the current session" — its prior about what an assistant is beats
    # anything written in context. The count and the dates are arithmetic over
    # the store, so they are emitted like the report and the money
    # confirmation. See `continuity`.
    # ⚠️ THE OCCASION ARISES HERE, IN THE TURN, FROM WHAT SURFACED. A memory
    # this turn brought up can mean Mati needs information before it answers,
    # and then it clarifies with a question first: if one of them carries a
    # correction, Mati has been wrong about that material before and needs the
    # fact confirmed rather than assumed again. See `Mati/clarify.py`.
    #
    # ⚠️ NOT A QUEUE AND NOT A GENERATOR. Both ids lists are from THIS turn —
    # the field's `surfaced` and the lexical `named` — so there is nothing
    # pre-written waiting for a slot, which is what would make a question
    # forced by construction.
    _clarify_said = ""
    try:
        from . import clarify as _clar
        _ids = [n for _d, n, _b, _s, _c in surfaced]
        _ids += [h.get("note_id") for h in (named or [])]
        _clarify_said = _clar.needed(user_text, _ids) or ""
        if _clarify_said:
            info["clarifying"] = True
    except Exception as e:                      # never fail a turn over this
        print(f"  [clarify] skipped: {e!r}", file=sys.stderr)

    _continuity_said = ""
    try:
        from . import continuity as _cont
        # ⚠️ ONLY WHEN IT IS THE WHOLE ANSWER. Prefixed to a model reply, it
        # makes three voices in four sentences — a memory statement, the
        # model's own answer, a service offer — contradicting each other with
        # nothing reconciling them.
        #
        # So either code answers alone, or the model does. One voice per
        # reply. A question with real content in it — Akkad, the profile — goes
        # to the model with its memories, and `voice._denies_memory` is the
        # guard there rather than a banner bolted on the front.
        if _cont.asks_how_many(user_text):
            _continuity_said = _cont.count_line() or ""
        elif _cont.is_only_question(user_text):
            _continuity_said = _cont.line()
        if _continuity_said:
            info["continuity_shown"] = True
    except Exception as e:                      # never fail a turn over this
        print(f"  [continuity] skipped: {e!r}", file=sys.stderr)
    # ⚠️ WHAT WAS SAID EARLIER TODAY, which the episode record may not hold.
    # Mati should remember better than a person does, so it needs a record it
    # can refer to even when the day's exchanges never reached the field. An
    # episode is only written when settling coherence clears EPISODE_GATE_MIN,
    # so some exchanges happened that the episode store does not have; the day
    # log does.
    #
    # ⚠️ LABELLED AS LIVED, AND THE OPPOSITE LABEL IS A REAL FAILURE. Calling
    # this "not memory" — on the reasoning that lines which have not cleared
    # the episode gate must not wear a memory's label — makes Mati find the
    # previous night's conversation and say it did not have it. The gate
    # measures whether the FIELD settled, not whether the conversation
    # happened — see `daylog.block`.
    #
    # ⚠️ AND IT SEARCHES A WEEK, NOT TODAY, or last night's transcript would
    # never be queried at all.
    try:
        from . import daylog as _dlb
        _today_block = _dlb.block(user_text)
        if _today_block:
            parts.append(_today_block)
            info["what_we_said"] = True
    except Exception as e:
        print(f"  [daylog] skipped: {e!r}", file=sys.stderr)

    try:
        from . import report as _report

        if _report.asks_for_report(user_text):
            _r = _report.text()
            if _r:
                _report_text = _r
                info["report_shown"] = True
                parts.append(
                    "<computed>\nI have just shown my morning report, in full "
                    "and word for word. It is already on screen. Do not "
                    "repeat it, summarise it or add to it.\n</computed>")
    except Exception as e:                              # noqa: BLE001
        if verbose:
            print(f"  [report] skipped: {e}", file=sys.stderr)

    # ⚠️ THE SHELF, WHEN ASKED ABOUT THE SHELF. Without it "what have you read"
    # is answered from whatever lexical search returns, which is passages OF
    # the book rather than the fact of having read it.
    try:
        from . import library as _lib
        if _lib.asks_about_books(user_text):
            _b = _lib.block()
            if _b:
                parts.append(_b)
                info["library_shown"] = True
    except Exception as e:                              # noqa: BLE001
        if verbose:
            print(f"  [library] skipped: {e}", file=sys.stderr)

    # ⚠️ FREE FILMS, when asked about free films. Openly-licensed only, and
    # the block states the licence terms so the model cannot imply Mati can
    # fetch anything else. Nothing is searched here — `/film` does that.
    try:
        from . import archive_org as _film
        if _film.asks_about_films(user_text):
            parts.append(
                "<computed>\nI can find and play openly-licensed films — "
                "public domain and Creative Commons, from the Internet "
                "Archive, where the file itself is published. That is the "
                "only kind I can fetch: I cannot take video from services "
                "that protect it. Ask me to search and I will list what is "
                "there with its licence.\n</computed>")
            info["films_offered"] = True
    except Exception as e:                              # noqa: BLE001
        if verbose:
            print(f"  [films] skipped: {e}", file=sys.stderr)

    # ⚠️ EVENTS: THE HONEST STATE, WHICH IS USUALLY "OFF". With no key this
    # says where to get one rather than letting the model improvise a gig
    # listing — a confabulated concert is a wasted evening, not a wrong
    # sentence. Nothing is fetched on this path; `/events` does that when
    # asked, because a location plus "what is on tonight" is a real
    # disclosure and does not belong on an idle turn.
    try:
        from . import events as _ev
        if _ev.asks_about_events(user_text) and not _ev.enabled():
            parts.append("<computed>\nI cannot look up events: "
                         + _ev.why_off() + "\n</computed>")
            info["events_off"] = True
    except Exception as e:                              # noqa: BLE001
        if verbose:
            print(f"  [events] skipped: {e}", file=sys.stderr)

    # ⚠️ DIRECTIONS: ONLY THE LIMITS, NEVER THE ROUTE. A route needs a
    # destination parsed out of the sentence, and guessing one would send
    # Mati geocoding whatever noun it found. The block states what Mati can
    # and cannot do — crucially that it has NO live guidance — so the model
    # stops offering to navigate. Routes are asked for explicitly, via /go.
    try:
        from . import directions as _dir
        if _dir.asks_about_directions(user_text):
            _d = _dir.block(user_text)
            if _d:
                parts.append(_d)
                info["directions_shown"] = True
    except Exception as e:                              # noqa: BLE001
        if verbose:
            print(f"  [directions] skipped: {e}", file=sys.stderr)

    # ⚠️ THE NEWS, WHEN ASKED FOR THE NEWS. Without a recogniser here Mati
    # would promise to "put that together" and then do nothing, because
    # `asks_for_report` is False for "news" and `world.block` returns "" when
    # no stored item matches. The block below always says something true,
    # INCLUDING when there are no feeds at all.
    try:
        from . import world as _w
        if _w.asks_about_news(user_text):
            _n = _w.news_block(user_text)
            if _n:
                parts.append(_n)
                info["news_shown"] = True
    except Exception as e:                              # noqa: BLE001
        if verbose:
            print(f"  [news] skipped: {e}", file=sys.stderr)

    # ⚠️ MONEY, ONLY WHEN ASKED ABOUT MONEY. A literal trigger — a classifier
    # that fired loosely would put budget figures into turns that were never
    # about money. Reads the store only; the local-flyer path is NOT on this
    # turn, because that one touches the network.
    # ⚠️ THE DRAFT, AND THE SMALL BLOCK IS THE POINT. A draft being open is
    # worth one line on EVERY turn — ~30 tokens — because that line is the
    # whole answer to "I have to remind Mati of the project". The passage
    # itself (~500 tokens) only earns its place when the turn is about the
    # writing.
    #
    # ⚠️ NEVER RAISES INTO A TURN. A draft that cannot be read must not stop
    # Mati answering about anything else.
    # ⚠️ TOLD TO THE PERSON, NOT JUST TO `/health`. A reply saying "I don't
    # remember much about you" reads identically to an empty store, when the
    # truth is that the memories exist and the key is absent. On a
    # passphrase-only install that is the state after every restart until it
    # is unlocked, so it has to be sayable.
    #
    # ⚠️ A FACT IN A BLOCK, NOT AN INSTRUCTION. Composed here and labelled
    # `computed` so it reads as something Mati worked out about itself this
    # moment — the same channel as the health numbers. Telling the model to
    # "say you are locked" is how invented confirmations happen.
    if locked:
        parts.append(
            "<computed>\nMy memories are sealed and I do not hold the key in "
            "this session, so nothing surfaced — that is why, not because "
            "there is nothing there. They come back with "
            "`python3 -m Mati.vault --unlock`.\n</computed>")

    try:
        from . import drafts as _dr
        # ⚠️ ACTED ON BEFORE THE BLOCK IS BUILT — the accountant's lesson,
        # written in its own comments: a read block alone means a request is
        # recognised, answered, and stores NOTHING. Opening happens first so
        # the block below reflects it on the very turn it was asked for.
        _draft_said = _dr.act(user_text)
        if _draft_said:
            info["draft_opened"] = True
        _draft_block = _dr.block(full=_dr.asks_about_draft(user_text))
        if _draft_said:
            # ⚠️ EMITTED, NOT PROMPTED. "Open: …" is composed here; handing
            # the model an instruction to announce it is how invented
            # confirmations happen.
            parts.append("<draft_action>\n" + _draft_said + "\n</draft_action>")
        if _draft_block:
            parts.append(_draft_block)
            info["draft"] = True
    except Exception as e:                              # noqa: BLE001
        if verbose:
            print(f"  [drafts] skipped: {e}", file=sys.stderr)

    # An encyclopedia entry, when one was explicitly asked for.
    #
    # ⚠️ ONLY ON AN EXPLICIT ASK, AND THAT IS THE DESIGN, NOT A LIMITATION.
    # "What is a truth table" is answerable from the model's weights; pulling
    # a reference into every factual-sounding turn would spend the window on
    # text Mati did not need and make the prompt a research tool rather than
    # a conversation. `asks_about_lookup` wants the word "encyclopedia" or
    # "look up" — a rare sentence, deliberately.
    #
    # ⚠️ AND IT NEVER WAITS FOR A COLD SERVER. `block` returns a "still
    # opening" line instead of stalling the reply for twenty-five seconds.
    try:
        from . import encyclopedia as _enc
        _ref = _enc.block(user_text)
        if _ref:
            parts.append(_ref)
            info["reference"] = True
        elif _enc.asks_about_lookup(user_text) and not _enc.connected():
            # ⚠️ THE ASK IS RAISED HERE OR IT IS RAISED NOWHERE. Defining
            # `ask_*_if_needed` and calling it from nothing is a mistake this
            # project has already made once (`intent.py`, built and wired into
            # nothing). This is the only place that knows a lookup was wanted
            # and could not be served.
            try:
                from . import asks as _asks_e
                _asks_e.ask_encyclopedia_if_needed()
            except Exception:
                pass
    except Exception as e:                              # noqa: BLE001
        if verbose:
            print(f"  [encyclopedia] skipped: {e}", file=sys.stderr)

    try:
        from . import accountant as _acct
        # ⚠️ `budget_intake`, NOT `intake`. `Mati/intake.py` is FILE intake and
        # already exists — `library.catalogue` imports
        # `ensure_priority` from it. ⚠️ A second module with the same
        # name and overwrote it, which broke the library silently.
        from . import budget_intake as _intake
        # ⚠️ THE WRITE PATH. A read block alone would recognise "set a grocery
        # budget of 400", answer it, and store nothing.
        #
        # ⚠️ ACTED ON BEFORE THE BLOCK IS BUILT, so the block the model sees
        # already contains what was just recorded. The other order would have
        # Mati confirm a purchase and then read back a total without it.
        # ⚠️ THE MULTI-PART BUDGET IS TRIED FIRST, because `hears_money` is
        # built for one name / one amount / one period and MISREADS a real
        # one: given a paragraph, it produces a budget called "lunch that can
        # cover from" for $0.00 a month — the 0 out of "from 0 to $50", a
        # sentence fragment as the name, and the per-person dinner, the number
        # of people and the duration all dropped. See `Mati/intake.py`.
        _held = _intake.absorb(user_text, session=session)
        if _held is not None:
            _money_said = _intake.say(_held) or ""
            if _money_said:
                info["budget_intake"] = True
        _money_req = None if _held is not None else _acct.hears_money(user_text)
        if _money_req:
            _money_said = _acct.act(_money_req)
            if _money_said:
                info["money_action"] = _money_req.get("verb")
                # ⚠️ EMITTED, NOT PROMPTED — the morning-report rule. Handed to
                # the model as a block, "$400.00 a month for grocery" came back
                # as "I will keep track of your monthly grocery expenses" with
                # an invented **Current Running Total** in markdown. The number
                # is the whole content of a money confirmation and a 3B will
                # paraphrase it away. The note below keeps it from being said
                # twice.
                parts.append(
                    "<computed>\nI have just confirmed a money change in full "
                    "and word for word. It is already on screen. Do not "
                    "repeat it, restate the amount, or add a total.\n"
                    "</computed>")
        if _acct.asks_about_money(user_text) or _money_req:
            _m = _acct.block()
            if _m:
                parts.append(_m)
                info["money_shown"] = True
    except Exception as e:                              # noqa: BLE001
        if verbose:
            print(f"  [accountant] skipped: {e}", file=sys.stderr)

    # ⚠️ THE SKY, ONLY WHEN ASKED ABOUT THE SKY. A literal trigger, not a
    # judgement call — this is the one block that costs a NETWORK REQUEST, and
    # a fuzzy trigger would put Mati's location on the wire during turns that
    # had nothing to do with the weather. Silent when no place is set, and
    # silent when the service cannot be reached; it never estimates.
    try:
        from . import weather as _wx
        if _wx.asks_about_weather(user_text):
            _w = _wx.block()
            if _w:
                parts.append(_w)
                info["weather_shown"] = True
            else:
                # ⚠️ PHASE 3: ASK AT THE MOMENT IT MATTERS, not at setup. With
                # no place set this block was simply absent and the model
                # answered about the weather from nothing — which is how a 3B
                # invents a city. One question, once, then never again.
                from . import asks as _asks
                _asks.ask_place_if_needed()
    except Exception as e:                              # noqa: BLE001
        if verbose:
            print(f"  [weather] skipped: {e}", file=sys.stderr)

    try:
        from . import selfdoc, health as _health
        if selfdoc.asks_about_self(user_text):
            _me = selfdoc.about_self(user_text)
            if _me:
                parts.append(bridge.named_recall_block(_me))
            _h = _health.block()
            if _h:
                parts.append(_h)
            info["about_self"] = len(_me)
    except Exception as e:                              # noqa: BLE001
        if verbose:
            print(f"  [selfdoc] skipped: {e}", file=sys.stderr)

    # ⚠️ LAST, SO IT IS NEAREST THE TURN IT EXPLAINS — and it is the half that
    # decides whether being interrupted feels right. "No, I meant the other
    # one" is meaningless on its own; without the abandoned exchange in view,
    # every interruption produces a non-sequitur. That is live behaviour, and
    # no amount of careful recording afterwards repairs it.
    if exchange is not None and exchange.interrupted:
        _cut = exchange.context()
        if _cut:
            parts.append(f"<interrupted>\n{_cut}\n</interrupted>")

    # ⚠️ VARIABLE CONTENT LAST, so everything stable in front of it stays
    # cached. The clock is the most-changing thing Mati sends and therefore
    # belongs at the very end.
    if oriented and when:
        parts.append(f"<now>\n{when}\n</now>")

    _stages.mark("assemble", _assemble_t0)
    context = [p for p in parts if p]
    messages = history[:]
    if context:
        messages.append({"role": "system", "content": "\n\n".join(context)})
    # ⚠️ THE MODEL IS NOT ASKED FOR A REPORT IT HAS ALREADY BEEN GIVEN. A
    # prompt note saying "it is on screen, do not repeat it" does not stop a
    # 3B repeating the whole report, headline re-invented and all — an
    # instruction is not a mechanism. So the trigger phrase is removed from
    # what the model is answering, and what remains is the part that still
    # needs an answer.
    #
    # ⚠️ ONLY THE MODEL'S COPY. `user_text` is untouched everywhere else — the
    # episode, the imprint and the recall all still see what was actually
    # typed, because that is what was actually said.
    _asked = user_text
    if _report_text:
        _asked = _report.without_request(user_text) or user_text
    messages.append({"role": "user", "content": _asked})
    if llm is None:
        return None, info                      # --dry-run

    # The field keeps running while the words form — it is doing the work with
    # the language faculty, not waiting on it. Falls back to a plain call if the
    # provider can't stream.
    t_work = time.time()
    steps_taken = 0
    # The <field_state> block is private, and the system prompt's instruction
    # not to quote it does not hold on a 3B — see `voice`. Filtering happens on
    # the way out, so what the user sees, what θ₀ consolidates, and what the
    # episode records are all the same text: otherwise Mati would remember
    # saying something it never said.
    # ⚠️ A HABITUAL EXCHANGE IS ANSWERED WITHOUT ASKING THE MODEL. Answers to
    # the SAME question have almost nothing in common from one time to the
    # next — "hello" five times gets five unrelated replies. That is not
    # variety, it is the absence of habit, and temperature cannot fix it: the
    # model answers a different prompt each time because different memories
    # surface. Consistency has to come from a stored response.
    #
    # ⚠️ FIXED-ANSWER TURNS ONLY. `rapport.match` returns None for anything
    # store-dependent — "how is your memory?", "what did you read" — and those
    # MUST keep changing. It also refuses a turn that is more than the pattern,
    # so "hello, what did you read yesterday?" goes to the model whole.
    #
    # The field still imprints below, and the episode is still written: this is
    # a real exchange that happened, it simply did not need generating.
    # ⚠️ THE REPORT GOES OUT VERBATIM, BEFORE ANYTHING IS GENERATED. It is
    # composed in code and every word of it is checkable; handing it to a 3B
    # to retype is how headlines get invented. Emitting first also means the
    # person has it on screen during the generation that follows, rather than
    # after it.
    # ⚠️ THE MONEY CONFIRMATION GOES OUT VERBATIM, for the same reason the
    # report does: it is composed in code, every figure in it is checkable,
    # and a 3B asked to relay it rewrites the number into prose.
    # ⚠️ BEFORE EVERYTHING, BECAUSE IT REPLACES THE ANSWER. A clarification
    # is Mati declining to assume; emitting it after a confirmation or a
    # report would have it arrive as an afterthought to an answer it
    # deliberately did not give.
    if _clarify_said and on_chunk is not None:
        on_chunk(_clarify_said + "\n\n")

    # ⚠️ FIRST OF THE THREE. Whether Mati has a past is the frame everything
    # after it is read in, so it goes out before the report or a confirmation.
    if _continuity_said and on_chunk is not None:
        on_chunk(_continuity_said + "\n\n")

    if _money_said and on_chunk is not None:
        on_chunk(_money_said + "\n\n")

    if _report_text and on_chunk is not None:
        on_chunk(_report_text + "\n\n")

    _settled = rapport.match(user_text)
    # ⚠️ A RECORDING TURN IS ALREADY ANSWERED, so it does not generate. The
    # confirmation is composed in code and every figure in it is checkable;
    # the model's only possible contribution is chatter — "I will subtract the
    # amount you spent", or the PREVIOUS turn's confirmation echoed back out of
    # history. Same shape as `report_only`, and the turn takes under a second.
    #
    # ⚠️ NOT WHEN SOMETHING ELSE WAS ASKED. A question mark means there is a
    # question to answer alongside the recording, and the model still runs.
    # ⚠️ A COLLECTED BUDGET IS THE WHOLE ANSWER, AND LETTING THE MODEL SPEAK
    # AFTER IT UNDOES THE POINT. After a correct readback the model goes on to
    # guess what a coffee costs — inventing the very figure the intake has just
    # said it does not have, and answering Mati's own question on the person's
    # behalf. Every number in the readback is arithmetic over what they
    # actually said; nothing the model adds can be.
    if info.get("budget_intake") and _money_said:
        info["money_only"] = True
        reply, resp = _money_said, None
    elif (_money_said and "?" not in user_text
            and (info.get("money_action") in ("budget", "spend"))):
        info["money_only"] = True
        reply, resp = _money_said, None
    elif _clarify_said:
        # ⚠️ THE MODEL DOES NOT RUN. The point of clarifying is to NOT answer
        # yet; letting it generate alongside would produce the assumption the
        # clarification exists to withhold, and the person would read both.
        info["clarify_only"] = True
        reply, resp = _clarify_said, None
    elif _continuity_said:
        # ⚠️ ALWAYS THE WHOLE REPLY. Emission and answering-alone are the same
        # condition — see where `_continuity_said` is set. If it was composed,
        # the model has nothing left to add and running it only invites a
        # second, contradicting voice.
        info["continuity_only"] = True
        reply, resp = _continuity_said, None
    elif _report_text and _report.is_only_request(user_text):
        # Nothing was asked beyond the report, and the report is complete.
        # ⚠️ Same reasoning as the rapport shortcut: a real exchange that did
        # not need generating. The field still imprints and the episode is
        # still written, below.
        info['report_only'] = True
        reply, resp = _report_text, None
    elif _settled is not None:
        info['rapport'] = True
        if on_chunk is not None:
            on_chunk(_settled)
        reply, resp = _settled, None
    elif hasattr(llm, "stream_reply"):
        clean = voice.ReplyFilter(user_text, memories=_surfaced_count())
        pieces = []
        _think = {}
        _llm_t0 = time.time()
        for chunk in bridge.think_while(lm, llm.stream_reply(messages),
                                        stats=_think):
            steps_taken = _think.get("steps", 0)   # what the field actually did
            shown = clean.feed(chunk)
            if shown:
                pieces.append(shown)
                # ⚠️ SHOWN IMMEDIATELY, NEVER PACED BY THE SPEECH. Deferring the
                # text to the speech queue, so it appeared as the voice reached
                # it, LOSES TEXT, and the reason is a lifetime mismatch:
                # `on_chunk` is `lambda p: self._sse("chunk", …)`, which writes a
                # frame to the HTTP response. The handler returns when the turn
                # returns; the speech pump keeps calling back for SECONDS
                # afterwards, into a socket that is already closed — the voice
                # goes on talking and the words never arrive on screen.
                #
                # ⚠️ TEXT ARRIVING EARLY IS A NUISANCE; TEXT NEVER ARRIVING IS
                # DATA LOSS. Pacing belongs in the app, which owns its own
                # rendering and outlives no request — not here.
                if on_chunk is not None:
                    on_chunk(shown)
                if speak and _voice_stream(shown):
                    info["spoke_streaming"] = True
                # ⚠️ SPOKEN AS IT IS SHOWN, which is the whole fix for the
                # lag. Speaking only at the end meant the voice started after
                # the text had finished arriving — on a 30s reply, half a
                # minute behind the screen.
                #
                # ⚠️ AND THIS IS THE ONLY SAFE PLACE TO DO IT. `clean.feed`
                # has already filtered this text and releases only COMPLETE
                # sentences, so what is spoken is exactly what is displayed.
                # Speaking the raw chunks would let Mati say aloud something
                # `ReplyFilter` removes from the screen — a categorical memory
                # denial, say — which is worse than lagging.
        # ⚠️ DROP THE FRAGMENT IF THE CAP CUT IT. `flush()` releases whatever
        # was left after the last complete sentence. When Ollama says
        # `done_reason == "length"` that remainder is a severed half-sentence,
        # and showing it is the "knife that cuts mid-word" the token ceiling
        # was long avoided for. Ending one sentence early is not a defect;
        # ending mid-word is.
        _stages.mark('llm', _llm_t0)
        tail = clean.flush()
        if tail and getattr(llm, "last_done_reason", None) == "length":
            if verbose:
                print(f"  [length] dropped a cut fragment: {tail.strip()[:40]!r}",
                      file=sys.stderr)
            tail = ""
        if tail:
            pieces.append(tail)
            if on_chunk is not None:
                on_chunk(tail)
        reply, resp = "".join(pieces), None

        # Mati asked for a memory instead of answering. `ReplyFilter` held the
        # request back, so nothing has reached the user yet and the second pass
        # is invisible to them — they see one reply, not a machine deliberating.
        #
        # ONE pass only, deliberately. A loop would let a 3B model ask for the
        # same words repeatedly at ~8s each, and the failure it is most likely
        # to have is asking for something that is not there — which no number
        # of retries fixes. If the lookup finds nothing it is told so plainly
        # and answers from what it has, which is the honest outcome.
        if clean.recall_request:
            asked_for = clean.recall_request
            found = bridge.search(asked_for, k=C.NAMED_RECALL_K)
            info['recall_request'] = asked_for
            info['recall_found'] = len(found)
            if verbose:
                print(f"  [recall] Mati asked for {asked_for!r} — "
                      f"{len(found)} found", file=sys.stderr)
            messages = messages + [
                {"role": "assistant", "content": f"[recall: {asked_for}]"},
                {"role": "user", "content": (
                    bridge.named_recall_block(found) if found else
                    "<recalled_by_name>\nNothing you hold contains those "
                    "words. Say so if it matters, and answer from what you "
                    "do have. Do not invent the memory.\n"
                    "</recalled_by_name>")}]
            clean = voice.ReplyFilter(user_text, memories=_surfaced_count())
            pieces = []
            for chunk in bridge.think_while(lm, llm.stream_reply(messages),
                                            stats=_think):
                steps_taken = _think.get("steps", steps_taken)
                shown = clean.feed(chunk)
                if shown:
                    pieces.append(shown)
                    if on_chunk is not None:
                        on_chunk(shown)
            tail = clean.flush()
            if tail:
                pieces.append(tail)
                if on_chunk is not None:
                    on_chunk(tail)
            reply = "".join(pieces)
            # ⚠️ A TURN MUST NEVER RENDER AS NOTHING. The first pass is held
            # back entirely when Mati asks for a memory, so if the second pass
            # also yields no visible text the person would be shown an empty
            # reply after waiting a minute, and the episode would be stored
            # with an "I was asked:" and no "I said:".
            if not reply.strip():
                info['empty_second_pass'] = True
                reply = ("I went looking for that and came back with nothing I "
                         "can stand behind. Ask me again with a word you are "
                         "sure is in it, or tell me and I will keep it.")
                if on_chunk is not None:
                    on_chunk(reply)
    else:
        with _stages.time("llm"):
            reply, resp = llm.reply(messages)
        reply = voice.clean_reply(reply, user_text, _person_name(),
                                  memories=_surfaced_count())

    # ⚠️ THE TRANSCRIPT MUST MATCH THE SCREEN. The report was streamed above
    # but the model's `reply` knows nothing about it, so without this the
    # record of the turn is missing the part the person actually read — and
    # `clean_reply` must not run over it, because it is code-composed and
    # already true.
    # ⚠️ NOT WHEN THE MODEL WAS SKIPPED — `reply` already IS the
    # confirmation there, and prepending it again prints the amount twice.
    # The same guard `report_only` needs, for the same reason.
    # ⚠️ THE EMITTED TEXT HAS TO JOIN `reply` TOO, not just the stream. The
    # stream is what the person sees; `reply` is what gets stored as the
    # episode and what a non-streaming caller gets back. Leaving it out of
    # one of them is how the record and the screen disagree.
    # ⚠️ NO CONTINUITY PREFIX HERE ANY MORE, and the branch is gone rather
    # than left unreachable: `continuity_only` is now true whenever anything
    # was composed, so prefixing could only ever have fired on a state that
    # cannot occur. Dead defensive code reads as a supported case.
    if _money_said and not info.get("money_only"):
        reply = _money_said + "\n\n" + (reply or "")
    if _report_text and not info.get("report_only"):
        reply = _report_text + "\n\n" + (reply or "")

    info['usage'] = _usage(resp)
    # ⚠️ THE POINT OF ALL THIS. Rides back with the reply so the breakdown is
    # readable from any client, and a regression is a number.
    info['stages'] = _stages.done()

    # Measure on the LIVE evolved field, NOT via field_state — that calls
    # restore_all() first, which would wipe the very evolution we want to see.
    # Thinking that leaves no trace is not thinking.
    coherence_after = float(lm.output_coherence(lm.mf.get_all_states()))
    work = {
        "seconds": time.time() - t_work,
        "field_steps": steps_taken,
        "coherence_before": coherence,
        "coherence_after": coherence_after,
    }
    info['work'] = work
    if verbose:
        print(f"  [field] worked {work['seconds']:.0f}s, {steps_taken} steps "
              f"while answering; coherence {coherence:.4f} → "
              f"{coherence_after:.4f}", file=sys.stderr)

    if waiting and reply:
        schedule.mark_delivered([i["id"] for i in waiting])
        info["delivered"] = len(waiting)

    # 4b. ONE THING MORE, IF ANYTHING IS WORTH ADDING. Composed after the
    #     answer exists, never before: a block sent ahead of generation becomes
    #     material the model writes FROM, and it works the offer into the reply
    #     whether it belongs there or not. Built afterwards it is a separate
    #     sentence that can be dropped whole, which is what makes "silence is
    #     the default" enforceable rather than aspirational.
    #
    #     ⚠️ The line is COMPOSED from stored text, not generated. Every offer
    #     must be able to produce the note it came from, and quoting is a
    #     cheaper guarantee of that than any instruction not to invent.
    #
    #     ⚠️ The arm is assigned and logged HERE, at offer time — see
    #     the study design. Deciding afterwards which segues "were
    #     really apposite" is the freedom preregistration removes.
    shown = reply
    try:
        # ⚠️ THE LEXICAL SURFACING, NOT THE FIELD ONE. `named` was retrieved
        # at the top of this turn and is the same act — relevance bringing the
        # memory up before anything judges it. The field channel performs near
        # chance; offering from it would be near-random connections carrying a
        # citation, which reads as checked and is not.
        _offer = segue.choose(user_text, reply, surfaced=named)
        _line = segue.line(_offer)
        if _line:
            _line = voice.clean_reply(_line, user_text, _person_name())
        if _offer and _line:
            info['segue'] = {"tier": _offer["tier"], "arm": _offer.get("arm"),
                             "note_id": _offer.get("note_id")}
            info['segue_id'] = segue.record(_offer)
            if _offer.get("say_id"):
                schedule.mark_delivered([_offer["say_id"]])
            # ⚠️ SHOWN, NOT CONSOLIDATED, AND NOT STORED. The line is a
            # QUOTATION of a note that is already in the store. Appending it to
            # `reply` would send it through `consolidate_reply` into θ₀ and
            # into the episode body — the store re-imprinting and re-recording
            # its own text, which is the memory loop this project already
            # watches for. Worse, it would later surface as a LIVED memory of
            # Mati having said it, upgrading a studied passage to an
            # experience by the longest possible route.
            #
            # So the person sees it and the field does not eat it.
            shown = f"{reply.rstrip()}\n\n{_line}"
            if on_chunk is not None:
                on_chunk("\n\n" + _line)
    except Exception as e:                  # an afterthought must never cost
        if verbose:                         # someone their answer
            print(f"  [segue] skipped: {e}", file=sys.stderr)

    # ⚠️ PARK BEFORE ANYTHING ELSE TOUCHES THE REPLY. Whatever Mati marked
    # `[later: …]` is lifted out here and put in the queue that raises it when
    # its topic comes up — so it is out of the answer, out of the episode, and
    # not lost. Held back has somewhere to go; that is what makes holding back
    # free rather than a suppression rule.
    # ⚠️ REGISTERED HERE, NOT WHERE THE EPISODE IS WRITTEN. The episode write
    # is handed to the worker and lands seconds later; a replay asked for in
    # between would read the store and hand back the previous answer. See
    # `notes.note_last_reply`.
    # ⚠️ No `replayed` guard here: a replay returns at the top of `turn` and
    # never reaches this line, so a check for it would read as a case that can
    # occur. The only writer of this register is a real reply.
    try:
        from . import notes as _nreg
        _nreg.note_last_reply(shown or reply, session=session)
    except Exception:
        pass
    # ⚠️ THE TRANSCRIPT, BOTH HALVES, AFTER THE REPLY IS FINAL. Written here
    # rather than where the episode is, because the episode write is handed to
    # the worker and lands seconds later — the same lag that made the replay
    # return the previous answer. A transcript must be current.
    try:
        from . import daylog as _dlog
        _dlog.append(_dlog.THEM, remembered_text)
        _dlog.append(_dlog.MATI, shown or reply)
    except Exception:
        pass
    reply = _park_later(reply, session=session)
    if shown:
        shown = _park_later(shown, session=session)

    # ⚠️ THE WAITING QUESTION IS APPENDED, NOT PROMPTED. It was raised earlier
    # in this turn (weather with no place, say) and would otherwise sit in
    # settings unasked — queued and invisible, which is worse than not asking.
    # ⚠️ Composed text, added AFTER the model has finished: handing "ask them
    # where they are" to a 3B gets a question about an invented city.
    try:
        from . import asks as _asks2
        _q = _asks2.pending()
        if _q and not info.get("answered_ask"):
            _line = "\n\n" + _q["question"]
            reply = (reply or "") + _line
            if shown:
                shown = shown + _line
            if on_chunk is not None:
                on_chunk(_line)
            info["asked"] = _q["kind"]
    except Exception:
        pass

    # ⚠️ OUT LOUD, AFTER PARKING, SO MATI SAYS WHAT IT SHOWS. `_park_later`
    # can still add or defer a question above; speaking before it would read a
    # reply the person never sees. Off unless `voice_on` is set, and
    # `speech.say` is detached and cannot raise into a turn — see that module.
    # ⚠️ ONLY A TURN A PERSON ASKED FOR MAY MAKE NOISE, and `speak` defaults
    # to False so a new caller is silent until it says otherwise. Speaking
    # unconditionally from inside `turn()` would make EVERY caller speak — a
    # probe's "I have no memory of that" coming from the speakers at random
    # moments.
    #
    # ⚠️ A ROOM IS NOT A SCREEN. Text that appears unasked is ignorable; a
    # voice in a room is not, and the person may not be alone. So the default
    # is silence and the one caller that has a person waiting opts in.
    try:
        from . import speech as _sp
        # ⚠️ A CONDITION, NOT AN EARLY RETURN. Returning here when `speak` is
        # false would skip everything below it — consolidation and the episode
        # write — so silencing the voice would silently stop Mati forming
        # memories.
        #
        # ⚠️ AND ONLY IF STREAMING DID NOT ALREADY SAY IT. The sentences were
        # spoken as they arrived; saying the whole reply again here would read
        # it out twice.
        if speak and info.get("spoke_streaming"):
            pass
        elif speak and not _sp.say(shown or reply):
            # ⚠️ PHASE 3: THE MOMENT MATI WOULD HAVE SPOKEN is the only
            # moment this question makes sense. Asked once ever, never again
            # if declined — `voice_asked` is what stops it nagging.
            from . import asks as _ask3
            if not _ask3.ask_voice_if_needed():
                # ⚠️ AFTER A REAL CONVERSATION, not after one turn — see
                # `AFTER_READING`. Offering to study a book to someone who
                # has said two things is asking them to commit hours to a
                # thing they have not formed an opinion about yet.
                _ask3.ask_reading_if_needed()
            # ⚠️ AND THE IDLE QUESTION IS NOT ASKED HERE. It belongs to a long
            # gap, which a turn is by definition not in — the server raises it
            # when one actually happens.
    except Exception:
        pass

    # 5. The reply deforms θ₀ only if the field settles coherently on it.
    consolidated, reply_coh = bridge.consolidate_reply(lm, tok, reply)
    info.update(consolidated=consolidated, reply_coherence=reply_coh)
    if verbose:
        mark = "consolidated" if consolidated else "not consolidated"
        print(f"  [field] reply {mark} (coherence={reply_coh:.4f}, "
              f"gate={C.REPLY_GATE_MIN})", file=sys.stderr)

    # 6. Episodic memory — Mati narrates its own state and imprints it.
    #    Selective by design: an episode is recorded only when the field
    #    actually settled. Unsettled turns leave no memory, which keeps the
    #    ~1s/token imprint cost off most turns and matches how consolidation
    #    works in biology. Nothing from the LLM is ever written here.
    # ⚠️ WRITE DOWN WHAT THE GATE DECIDED, pass or fail. Without a record of
    # what was refused, and by how much it missed, "is 0.7 the right
    # threshold" is not hard to answer, it is impossible. This changes no
    # behaviour; it only makes the decision observable.
    try:
        gatelog.record(coherence, coherence >= C.EPISODE_GATE_MIN,
                       C.EPISODE_GATE_MIN, reply_coh=reply_coh,
                       consolidated=consolidated, text=remembered_text,
                       reply=reply)
    except Exception:
        pass
    if remember and coherence >= C.EPISODE_GATE_MIN:
        def _write(do_reindex=episode_sink is None, should_stop=None):
            return record_episode(lm, tok, user_text, triplet, coherence,
                                  memories, consolidated, reply, work,
                                  do_reindex=do_reindex, verbose=verbose,
                                  speaker=speaker,
                                  should_stop=should_stop,
                                  corrects=corrects, wrong_reply=wrong_reply,
                                  rechecked=rechecked, exchange=exchange,
                                  deferred=bool(info.get("deferred")))
        if episode_sink is not None:
            episode_sink(_write)            # the caller decides when; not now
            info['episode_deferred'] = True
        else:
            info.update(_write())
    elif verbose and remember:
        print(f"  [memory] not recorded (coherence {coherence:.4f} < "
              f"{C.EPISODE_GATE_MIN})", file=sys.stderr)

    return shown, info


def _server_is_running():
    """Back-compat alias — `exclusive` owns this now."""
    return exclusive.holds_field()


def main():
    ap = argparse.ArgumentParser(description="Mati — one field-coupled turn")
    ap.add_argument("text", help="the user turn")
    ap.add_argument("--dry-run", action="store_true",
                    help="field only; no model call")
    ap.add_argument("--provider", default=None,
                    help="local (default, free/offline) | anthropic")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="run even if the server holds the field (unsafe)")
    args = ap.parse_args()

    # θ₀ is mutable state backed by a file. Two processes imprinting and saving
    # it will overwrite each other's work — silently, because neither can tell.
    # If Mati is awake it already owns the field, and unlike the utility tools
    # this one has a direct replacement: ask the server for the same turn.
    if not args.force and exclusive.holds_field():
        print("[refused] Mati is awake and already owns the field.\n"
              "  Two processes writing θ₀ would corrupt it.\n"
              f"  Use:  python3 -m Mati.client {args.text!r}\n"
              "  Or:   --force  (only if you know it is idle)",
              file=sys.stderr)
        return 2

    llm = None
    if not args.dry_run:
        try:
            llm = make_llm(system_prompt(), args.provider)
        except LLMUnavailable as e:
            print(f"[llm unavailable] {e}", file=sys.stderr)
            return 1
        # Fail before the ~30s field load rather than after it.
        if hasattr(llm, "available"):
            ok, detail = llm.available()
            if not ok:
                print(f"[llm unavailable] {detail}", file=sys.stderr)
                return 1
            if not args.quiet:
                print(f"  [llm] {detail}", file=sys.stderr)

    print("loading field...", file=sys.stderr)
    lm, tok = bridge.load_field()

    try:
        reply, info = turn(lm, tok, llm, args.text, verbose=not args.quiet)
    except Refusal as e:
        print(f"[refused: {e.category}] {e.explanation}", file=sys.stderr)
        return 1
    except LLMUnavailable as e:
        print(f"[llm unavailable] {e}", file=sys.stderr)
        return 1

    if reply is None:
        print("\n[dry run — field state only, no reply]", file=sys.stderr)
        print(bridge.state_block(info['triplet'], info['coherence'],
                                 bridge.recall(info['triplet']),
                                 profile=bridge.profile_notes()))
    else:
        print(reply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
