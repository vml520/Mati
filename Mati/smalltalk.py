"""
What Mati says while it is still thinking.

The user: *"It takes almost three minutes for a response. That wait-time needs to
be filled by the conscious agent… When a human is young they also have no
memories to provide relevant information. The way they communicate in those
years is to repeat what they hear, and pretend to be someone they heard."*

Measured  on a real turn:

    first token after   77.4s   <- nothing on screen at all
    streaming           48.9s
    total              127.1s   -> 61% of every turn is silence

⚠️ THE FILLER MUST NOT COST A MODEL CALL. Generation is the slow part; a filler
that generates has spent the time it was meant to cover. So everything here is
a LOOKUP or an ECHO — lexical search runs in milliseconds and scores 0.838,
against the field's 0.036, so the relevant memory can be found immediately
while the field takes its 77 seconds.

The shape, from the user's own example:

    User:  "Mati play that music station from Saturday night"
    Mati: "Sure thing. They played one of your favourite songs, 'Not Like
            Us'. I love that song. Here you go."
    <the station plays>

    acknowledge  ·  a REAL memory the query found  ·  affect  ·  handoff

⚠️ THAT SHAPE IS ONE REGISTER OF FOUR, AND IT WAS APPLIED TO ALL OF THEM. It
is right for an instruction and wrong for everything else — it greeted the user
with "Of course. Let me pull that together." See the register block below:
conversation is the default here, and service is the exception.

⚠️ ROLE-PLAY MAY AMPLIFY TONE, NEVER CONTENT. "I love that song" adds no fact
and is the performance the user is asking for. "They also released an album last
year" would be a fact Mati does not hold, arriving with a smile. Enforced
structurally rather than by instruction: the memory fragment is COPIED from the
store, and nothing here can compose a claim.

⚠️ IT IS SPEECH, NOT DECORATION. The user chose this over avatar-only behaviour,
so the filler is part of the conversation and part of what gets remembered. It
is therefore held to the same rule as everything else Mati says: it may not
assert anything untrue. Being empty is allowed; being wrong is not.

⚠️ AND IT STAYS QUIET WHEN THE ANSWER IS FAST. A settled rapport exchange
returns in 8.7s. Prefacing an 8-second answer with small talk is not lubricant,
it is padding.
"""

from __future__ import annotations

import hashlib
import re

# Nothing shorter than this is worth interrupting the silence for.
MIN_MEMORY_WORDS = 6

# How much of a recalled memory the filler may quote. Short: this is a bridge,
# not the answer.
QUOTE_CHARS = 120

# ⚠️ THE RELEVANCE GATE, AND TWO OBVIOUS SIGNALS FAILED FIRST.
#
#   SCORE is useless here: "what is two plus two" scored 1.000 by matching
#   `two` and `plu`. A perfect score for a passage about the Anthropic API.
#
#   IDF IS INVERTED for this job, which is the trap this project already
#   recorded once. `station` scores the MAXIMUM 7.95 because it appears once,
#   by coincidence, in a book about Aztec philosophy — while `tlamatinime`
#   scores 4.06 precisely BECAUSE it is the book's subject and appears
#   everywhere. Rarity discriminates for retrieval and misleads here.
#
#   DENSITY works: occurrences of the matched words per 100 words. A passage
#   ABOUT the calmecac uses the word repeatedly; one that mentions a station in
#   passing does not.
#
#       irrelevant   2.17  1.98  3.85
#       relevant     5.88  11.11
#
# ⚠️ 5.0 IS PICKED FROM FIVE EXAMPLES, NOT MEASURED. It is a provisional line
# and it should be measured properly before anyone trusts it. The direction of
# error is chosen deliberately: BIAS TOWARD SILENCE. A missed fragment costs a
# plainer greeting; a wrong one has Mati free-associating about Aztec warriors
# when asked to play a radio station, which is worse than saying nothing.
MIN_DENSITY = 5.0

# ── The pieces. Data, not generation. ────────────────────────────────────────

# ── Four registers. ──────────────────────────────────────────────────────────
#
# ⚠️ CONVERSATION IS THE DEFAULT, SERVICE IS THE EXCEPTION. The user,
# on being greeted with "Let me take care of that":
#
#     "Mati is supposed to be a casual converser who happens to be able to do
#      other things on the side."
#
# Every phrase in this file used to be service language, so a greeting had
# nowhere to land and fell through to the task bank. Which register applies is
# decided in `_register` below; the banks only supply words.

# Someone said hello. Say hello. ⚠️ NOTHING IS APPENDED to a greeting — there
# is no task to hand off and no topic to be relevant about.
_GREET = ("Hey.", "Hi.", "Hello.", "Hey there.")

# A question. Mati is about to think, and saying so is the honest filler.
# ⚠️ "Good question." IS GONE, at the user's instruction: *"it's never timely"*.
# The others say only that Mati is about to think, which is true whatever was
# asked. "Good question" judges the question before anything has been read —
# so it lands on "what time is it" exactly as warmly as on a hard one, which
# is how a filler stops reading as a person and starts reading as a tic.
# ⚠️ Do not add praise back in any wording. The register is "I am thinking",
# not "you did well to ask".
_THINKING = ("Let me think.", "Hm.", "Let me sit with that a second.")

# ⚠️ THERE IS NO BANK FOR A REMARK, AND THAT IS THE POINT. This was
# ("Mm.", "I'm listening.", "I hear you.", "Okay.") and it answered "thank
# you" with "I'm listening." — a noise that treats a CLOSE as an OPENING.
# `telling` is the fall-through, so every closing, thanks and one-word
# reaction landed in it. See `opening`: the register now returns "" and lets
# the model speak first.

# An instruction. Only here is service register correct.
_ACK = ("Sure thing.", "Of course.", "Right.", "On it.", "Let me look.")

_HANDOFF = ("Here you go.", "One moment.", "Give me a second.",
            "Let me pull that together.")

# ⚠️ Affect only. Every one of these is true of the ACT, not of any fact —
# nothing here can be false about the world.
_AFFECT = ("I like that one.", "That one stuck with me.",
           "I remember that.", "Good one.")

_HELLO = re.compile(
    r"^(hey|hi|hello|yo|howdy|hola|greetings|sup"
    r"|good\s+(morning|afternoon|evening)|morning|evening"
    # ⚠️ THE CONTRACTION HAS NO SPACE. `how\s+` cannot reach "How's it going".
    r"|how(?:\s+are\s+(?:you|things)|(?:\s+is|'?s)\s+it\s+going)"
    r"|what'?s\s+up)\b", re.I)

# ⚠️ ANCHORED, NOT `search`. The old pattern matched the verb ANYWHERE, so
# "What have you read recently?" contained `read` and was served as an
# instruction. A request to act puts its verb first, or behind a politeness.
_DOING = re.compile(
    r"^(please\s+|can\s+you\s+|could\s+you\s+|would\s+you\s+|will\s+you\s+"
    r"|i\s+want\s+you\s+to\s+|go\s+ahead\s+and\s+)*"
    r"(play|open|start|run|show|find|remind|add|import|read|study|put|stop"
    r"|pause|skip|search|look\s+up)\b", re.I)


# Words that carry no subject, for the length test in `_fragment` only.
_FILLER_Q = re.compile(
    r"\b(thank|thanks|you|ok|okay|yes|no|nice|great|good|cool|perfect|sure|"
    r"please|the|a|an|is|are|was|were|it|that|this|i|me|my|and|so|well|"
    r"right|got|mm|ah|oh|hm|hmm|mati)\b", re.I)

# A name or bare punctuation trailing a hello. "Hello Mati" is still a hello.
# ⚠️ AND SO IS "HELLO <WHATEVER THEY RENAMED IT TO>". The assistant's name is
# choosable, so a pattern hard-coding "mati" would stop recognising its own
# name the moment someone changed it — the greeting would fall through to the
# model and come back as a paragraph instead of "Hi."
# ⚠️ Built at call time, not at import: the setting can change while the
# server is running, and a compiled-once pattern would hold the old name until
# restart. Cached on the name itself so the common case is still one compile.
_VOCATIVE_CACHE = {}


def _vocative():
    from . import config as C
    name = C.assistant_name()
    got = _VOCATIVE_CACHE.get(name)
    if got is None:
        extra = re.escape(name.lower())
        got = re.compile(r"^(?:[\s,!.?—–-]*\b(?:mati|there|again|"
                         + extra + r")\b)*[\s,!.?—–-]*", re.I)
        _VOCATIVE_CACHE[name] = got
    return got


def _after_hello(text):
    """What was said after the greeting — "" for a bare hello, None for no hello.

    ⚠️ A GREETING IS NOT ALWAYS THE WHOLE MESSAGE. "Hi, can you find my keys?"
    is a hello *and* an instruction, and answering only the hello drops the
    task on the floor.
    """
    m = _HELLO.match(text)
    if not m:
        return None
    return _vocative().sub("", text[m.end():], count=1).strip()


def _register(text):
    """Which of the four kinds of thing was just said.

    ⚠️ DOING OUTRANKS QUESTION. "Can you play that station?" is punctuated as
    a question and is not one.
    """
    rest = _after_hello(text)
    if rest == "":
        return "greeting"
    if rest:
        text = rest          # greeted, then said something. The content wins.
    if _DOING.match(text):
        return "doing"
    if text.endswith("?"):
        return "question"
    return "telling"


def _pick(options, seed_text):
    """Deterministic choice, so the same turn does not vary between runs.

    ⚠️ Not `random`. A filler that changes when a turn is replayed makes every
    transcript unreproducible, and this project debugs by replaying.
    """
    h = int(hashlib.sha256((seed_text or "").encode("utf-8")).hexdigest(), 16)
    return options[h % len(options)]


def _greet_back(text):
    """Hello. ⚠️ Echoes the time of day when one was offered — "Good morning"
    answered with "Hey there." is the small incoherence in miniature."""
    m = _HELLO.match(text)
    tod = ((m.group(2) if m else None) or (m.group(1) if m else "")).lower()
    if tod in ("morning", "afternoon", "evening"):
        return f"Good {tod}."
    return _pick(_GREET, text)


def _fragment(user_text, db_path=None):
    """One short, true thing from the store that this query found. Or None.

    ⚠️ LEXICAL, NOT THE FIELD. This has to return in milliseconds — it is
    covering the wait, not joining it. Lexical scores 0.838 against the field's
    0.036 and costs nothing.

    ⚠️ AND A SHORT REMARK CANNOT ESTABLISH A TOPIC. "thank you", "perfect",
    "ok" carry no subject to be relevant ABOUT, so any hit is a coincidence
    of one or two common words — and the density gate cannot tell the
    difference, because a two-word query matching twice scores high. Under
    four content words, there is nothing here to search with.
    """
    if len(_FILLER_Q.sub(" ", (user_text or "")).split()) < 4:
        return None
    try:
        from . import notes
        hits = notes.search(user_text, k=1, db_path=db_path)
    except Exception:                                    # noqa: BLE001
        return None
    if not hits:
        return None
    h = hits[0]
    body = " ".join((h.get("body") or "").split())
    # Episodes are stored as narration; the useful half is what was said.
    m = re.search(r"I said:\s*(.*)", body, re.S)
    if m:
        body = " ".join(m.group(1).split())
    if len(body.split()) < MIN_MEMORY_WORDS:
        return None
    if (h.get("density") or 0.0) < MIN_DENSITY:
        return None                     # mentioned in passing, not about this
    if len(body) > QUOTE_CHARS:
        body = body[:QUOTE_CHARS].rstrip() + "…"
    return {"text": body, "source": h.get("source"),
            "when": h.get("when_said")}


def opening(user_text, db_path=None, fast=False):
    """What Mati says before it has an answer. "" when it should stay quiet.

    `fast=True` when the reply is already known to be quick (a settled rapport
    exchange), in which case there is no silence to cover.
    """
    text = " ".join((user_text or "").split())
    if not text or fast:
        return ""

    kind = _register(text)

    # ⚠️ A GREETING RETURNS ONE WORD AND STOPS. No fragment, because "Hello
    # Mati" carries no topic — and the density gate will not save us here,
    # since `mati` appears all over the store and would score high by
    # coincidence. No handoff, because nothing is being handed over.
    if kind == "greeting":
        return _greet_back(text)

    parts = []
    if _after_hello(text):
        # Greeted *and* asked. Both get answered, in that order.
        parts.append(_greet_back(text))

    # ⚠️ A REMARK GETS NO CANNED ACKNOWLEDGEMENT. The user thanked Mati for
    # recalling a number and was answered "I'm listening." — and "perfect"
    # got the same. `telling` is the fall-through, so every closing, thanks
    # and one-word reaction landed in it and was met with a noise that treats
    # a CLOSE as an OPENING.
    #
    # ⚠️ AND THE ANSWER IS NOT A BETTER BANK. The user: *"Maybe the category
    # that defers to 'I'm listening' should be run through the LLM first."*
    # They are right, and the reason is structural rather than a matter of
    # wording: for a question the filler can honestly say "let me think", and
    # for an instruction it can honestly say "on it", because in both cases
    # the NEXT thing is known. After a remark it is not — the right reply to
    # "thank you" and to "I had a rough day" have nothing in common, and no
    # fixed phrase covers both. Anything canned here is a guess, and a guess
    # made before the reply exists.
    #
    # ⚠️ THE SILENCE THIS LEAVES IS COVERED ELSEWHERE, deliberately: the app
    # shows "Still thinking… please wait" after 15s, which is the honest
    # version of what "Mm." was pretending to be.
    if kind != "telling":
        parts.append(_pick({"doing": _ACK, "question": _THINKING}[kind], text))

    # ⚠️ THE FRAGMENT IS GONE — THE THIRD RECITATION, AND THE ONE I WAS
    # PROUDEST OF. It quoted a real memory, attributed, in milliseconds:
    #
    #     Last time this came up, on Wednesday I said: "…"
    #     I have something on this — studied, 3 Sep: "…"
    #
    # Every word of that was true and sourced. It was still wrong, for the
    # reason the user gave: memories are *"passive inputs, to
    # maintain integrity of personhood while providing knowledge gained
    # through experience."* Opening a turn by reading a memory aloud is the
    # opposite of passive — and here it was worse than in `segue`, because
    # the filler runs BEFORE the model has said anything, so the memory was
    # recited before it could possibly have informed an answer.
    #
    # ⚠️ ATTRIBUTION WAS NEVER THE PROBLEM, so nothing is lost by dropping it.
    # The provenance it carried now lives in `info["memories"]` and the
    # terminal, where it is derived from the stored `source` column rather
    # than assembled into a sentence. A reply cannot guarantee its own
    # sources; a record can.
    #
    # `_fragment` and `_AFFECT` are kept and unreachable ON PURPOSE — the
    # lexical path and the density gate they encode (5.0, measured against
    # 2.17/1.98/3.85 irrelevant vs 5.88/11.11 relevant) are the only thing
    # here that was ever hard to get right, and `describe()` still reports
    # what this module covers. ⚠️ Unreachable-and-documented, not forgotten:
    # see LESSONS 11 and the `setState` deletion that took six callers with it.

    # ⚠️ ONLY WHEN SOMETHING IS ACTUALLY BEING HANDED OVER. This line read
    #
    #     _pick(_DOING.search(text) and _HANDOFF or _HANDOFF, ...)
    #
    # — both branches the same bank, so the register test was computed and
    # thrown away, and every input on earth ended "Here you go." A question
    # gets an answer, not a delivery; a remark gets neither.
    if kind == "doing":
        parts.append(_pick(_HANDOFF, text + "b"))
    return " ".join(parts)


def describe():
    """For `/system` — what this covers and what it costs."""
    return {"covers_seconds": 77.4, "share_of_turn": 0.61,
            "model_calls": 0, "lookup": "lexical"}
