"""
When asked whether it remembers, Mati answers from the STORE, not the model.

⚠️ AN INSTRUCTION IS NOT A MECHANISM. With four memories in its prompt,
llama3.2:3b answers questions about its own past like this:

    "What do you know about me?"        -> "I don't have any prior knowledge"
    "What did we talk about yesterday?" -> "my memory is limited to the
                                            current session"
    "Do you have continuous memory?"    -> "No ... as if we're starting from
                                            a blank slate"

Retrieval is not the problem — self-retrieval scores 6/6 on a real store. Nor
is the wording of the waking block: with an explicit "never say you have no
memory of earlier conversations, that is false" in it, the model still says
*"As a conversational AI, my memory is limited to the current session."* A
3B's RLHF prior about what an assistant is beats anything written in context.

So the same answer as the morning report and the money confirmation:
**EMIT, NEVER PROMPT.** The facts here are arithmetic over the store — a
count, two dates, the span between them. Every one is checkable, none of them
needs a language model, and handing them to a 3B is how a true sentence
becomes a confident denial.

⚠️ THIS IS THE CONSCIOUS LAYER, AND IT IS CODE. The field is the
subconscious; whether Mati has a past is not a matter of opinion for the 3B
to weigh.
"""

from __future__ import annotations

import os
import re
import sqlite3
import time

# ⚠️ TWO CONDITIONS, BOTH REQUIRED: this has to be a question ABOUT Mati's
# memory, not a question that merely contains the word "remember". "Remind me
# to pay the bill" and "do you remember what I said about the calmecac" are
# different turns — the first is a reminder to file, the second is a claim
# about continuity being tested. Answering the first from here would talk
# about the store when the person wanted a task done.
_ABOUT_MEMORY = re.compile(
    r"\b("
    r"continuous memory|persistent memory|long[- ]term memory|"
    r"do you (?:even )?(?:have|keep|retain|store)\s+(?:a\s+|any\s+)?memor|"
    r"can you (?:remember|recall|retain)|"
    r"(?:do|will) you (?:remember|recall) (?:me|this|that|us|our)|"
    r"your memory|"
    r"blank slate|"
    r"start(?:ing)? (?:over|again) (?:every|each)|"
    r"between (?:our )?(?:conversations|sessions)|"
    r"(?:how|what) (?:much |do )?you (?:remember|recall)|"
    r"what do you know about me|"
    r"how long have (?:we|you)|"
    # ⚠️ "What did we talk about yesterday?" has no "memory" in it and is the
    # worst case: the model denies having memory and then recalls correctly in
    # the same breath — "my memory is limited to the current session. However,
    # I can recall that you mentioned earlier..." Asking what was said BEFORE
    # is a continuity question whether or not the word "memory" appears in it.
    r"what (?:did|'d) (?:we|you|i) (?:talk|speak|discuss)"
    r"|(?:talk|speak|discuss)(?:ed|ing)? about\s+"
    r"(?:yesterday|last night|last week|before|earlier|previously)"
    r"|(?:do|did) you (?:remember|recall)\s+"
    r"(?:what|when|that|anything|our|the last)"
    r")\b",
    re.I,
)

# ⚠️ AND A NEGATIVE GATE. Each of these makes the sentence about something
# else, and a match here means this module stays out of the way entirely.
_NOT_ABOUT_MEMORY = re.compile(
    r"\b(remind me|reminder|don'?t forget|remember to)\b", re.I)

# Asking for the SIZE of the memory. Only this gets a number — see
# `count_line`. Everything else about memory gets a plain answer, because a
# figure nobody asked for is an inventory read-out.
_HOW_MANY = re.compile(
    r"\b(?:how many|how much|how big|what size)\b[^.?!]{0,30}"
    r"\b(?:memor(?:y|ies)|remember|recall|notes?|episodes?|conversations?)\b"
    r"|\b(?:memor(?:y|ies)|notes?)\b[^.?!]{0,20}\b(?:do you have|are there)\b",
    re.I,
)

# Questions whose WHOLE subject is the faculty itself. Only these may answer
# without the model — see `is_only_question`. Everything else in
# `_ABOUT_MEMORY` gets the facts emitted and still generates a reply.
_PURE = re.compile(
    r"\b("
    r"continuous memory|persistent memory|long[- ]term memory|"
    r"do you (?:even )?(?:have|keep|retain|store)\s+(?:a\s+|any\s+)?memor|"
    r"can you (?:remember|recall|retain)\b|"
    r"your memory|blank slate|"
    r"between (?:our )?(?:conversations|sessions)"
    r")\b",
    re.I,
)


def asked(text_in):
    """Is this a question about whether Mati has a past?"""
    t = (text_in or "").strip()
    if not t:
        return False
    if _NOT_ABOUT_MEMORY.search(t):
        return False
    return bool(_ABOUT_MEMORY.search(t))


def is_only_question(text_in):
    """True when continuity is the whole of what was asked.

    ⚠️ SAME SHAPE AS `report.is_only_request`, AND FOR THE SAME REASON. The
    facts are emitted either way; this decides whether the MODEL still has
    something left to answer. Erring permissive costs a needless generation,
    which is the cheap direction — erring the other way would swallow a real
    question that happened to mention memory.
    """
    if not asked(text_in):
        return False
    # ⚠️ ONLY THE PURE CONTINUITY PHRASINGS MAY SUPPRESS THE MODEL. "What do
    # you know about me?" is about memory AND has a real answer the model has
    # to give from the profile — swallowing it would replace "you work on
    # research, you are interested in mesoamerica" with a bare note about how
    # many memories exist, which answers a question nobody asked. The broad
    # `_ABOUT_MEMORY` list would let it through to only=True.
    if not _PURE.search(text_in or ""):
        return False
    rest = _ABOUT_MEMORY.sub(" ", text_in or "")
    # What is left once the continuity question is removed. Anything with
    # content in it means there is more here than this.
    rest = re.sub(r"[^a-z]+", " ", rest.lower())
    leftover = [w for w in rest.split()
                if w not in {
                    # ⚠️ MEMORY WORDS COUNT AS FILLER *HERE*, and only because
                    # `_PURE` has already decided this is a faculty question.
                    # "Do you actually remember anything between our
                    # conversations?" left {remember, anything} behind and so
                    # read as "there is more here", which meant the model
                    # answered and could deny. They are safe to drop because
                    # "what do you know about me" never reaches this line —
                    # it is not in `_PURE`.
                    "remember", "recall", "memory", "memories", "anything",
                    "something", "between", "conversations", "conversation",
                    "sessions", "session", "keep", "retain", "store",
                    "do", "you", "have", "a", "any", "the", "is", "it", "s",
                    "and", "or", "so", "about", "me", "my", "i", "of", "that",
                    "this", "what", "how", "hey", "hi", "hello",
                    "please", "tell", "can", "does", "did", "your", "really",
                    "actually", "still", "yet", "also", "too", "right", "now",
                    "there", "then", "us", "our", "we", "to", "be", "are",
                } and w not in _name_words()]
    return not leftover


def _name_words():
    """The assistant's own name, as filler words — "do you remember, Mati?"

    ⚠️ The name this person gave it, read at call time, never the product's.
    """
    from . import config as C
    return set(re.sub(r"[^a-z]+", " ", C.assistant_name().lower()).split())


def _span(seconds):
    """Plain words for a duration. Never a bare number of seconds."""
    days = seconds / 86400.0
    if days < 1.5:
        hours = max(1, int(round(seconds / 3600.0)))
        return f"{hours} hour" + ("s" if hours != 1 else "")
    if days < 14:
        d = int(round(days))
        return f"{d} day" + ("s" if d != 1 else "")
    if days < 70:
        w = int(round(days / 7.0))
        return f"{w} week" + ("s" if w != 1 else "")
    m = int(round(days / 30.44))
    return f"{m} month" + ("s" if m != 1 else "")


def facts(db_path=None):
    """Counts and dates from the store. Arithmetic, no model, no opinion.

    Returns None when there is genuinely nothing — a first run. ⚠️ NOT a zeroed
    dict: "0 memories kept since never" is the confident-denial failure wearing
    numbers, and the caller must be able to tell "nothing yet" from "nothing
    found".
    """
    from . import bridge
    db_path = bridge.NOTES_DB_PATH if db_path is None else db_path
    if not db_path or not os.path.exists(db_path):
        return None
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            # ⚠️ EVERY NOTE, ARCHIVED INCLUDED. The working set is a RETRIEVAL
            # bound (`config.WORKING_SET` notes stay field-addressed), not the
            # extent of what is kept — archived memories are still there and
            # still findable by their words. Counting only the working set
            # would under-report the store by thousands and make an honest
            # answer into a false modesty.
            "SELECT count(*), min(wall_created), max(wall_created) FROM notes"
        ).fetchone()
    except sqlite3.Error:
        return None
    finally:
        conn.close()
    n, first, last = row or (0, None, None)
    if not n:
        return None
    return {"kept": int(n), "first": first, "last": last}


def line(db_path=None, now=None):
    """The sentence Mati says about its own memory. Emitted, never prompted.

    ⚠️ FIRST PERSON AND PRESENT TENSE, because this is spoken BY Mati to the
    person. Nothing in here is addressed to Mati, so there is no "you" to be
    re-aimed at the person when it speaks.

    ⚠️ NO COUNT, AND NO AGE. "I keep memory between conversations, and I have
    4,299 of them right now, the oldest going back 2 months" still sounds like
    the LLM, not Mati, and the inventory is half of why.

    `bridge.state_block` tells Mati, of its own memories: *"Do not announce
    them, quote them, or say how many there are."* Code that announced and
    counted them on every memory-adjacent turn would break that rule on the
    model's behalf. Nobody asking "do you recall our conversation about
    Akkad?" wants an inventory read-out first — and the same sentence arriving
    every time is a canned response, whichever module it comes from.

    The count is not thrown away; it is in `count_line`, for the one question
    where a number IS the answer.
    """
    if facts(db_path) is None:
        # ⚠️ STILL NOT A DENIAL. An empty store means nothing has happened
        # yet, which is a different sentence from "I cannot remember".
        return ("This is our first conversation, so there is nothing in it "
                "yet — but what we say stays with me.")
    return "Yes. What we say stays with me between conversations."


def asks_how_many(text_in):
    """Did they ask for the SIZE of the memory, not whether it exists?

    ⚠️ A SEPARATE QUESTION, AND THE ONLY ONE A NUMBER ANSWERS. "Do you
    remember?" wants yes; "how much do you remember?" wants the figure. Mixing
    them would prefix the figure to turns that are about something else
    entirely.
    """
    return bool(_HOW_MANY.search(text_in or ""))


def count_line(db_path=None, now=None):
    """The figure, for when the figure was asked for. None if there is none."""
    f = facts(db_path)
    if f is None:
        return None
    age = _age_of(f["first"], time.time() if now is None else now)
    return (f"{f['kept']:,}"
            + (f", going back about {age}" if age else "") + ".")


def _age_of(stamp, now):
    """How long ago an ISO `wall_created` was, in words. None if unreadable.

    ⚠️ `wall_created` IS AN ISO-8601 STRING, NOT AN EPOCH FLOAT —
    `notes._now()` returns `datetime.isoformat(timespec='seconds')`.
    `float(stamp)` raises ValueError on "2026-07-30T06:43:45+00:00", and if a
    bare `except` swallowed it the whole "going back ..." clause would vanish
    from every answer while the line still read as a complete sentence. Hence
    the parse failure is printed, not hidden.
    """
    if not stamp:
        return None
    try:
        import datetime as dt
        when = dt.datetime.fromisoformat(str(stamp))
        if when.tzinfo is None:
            when = when.replace(tzinfo=dt.timezone.utc)
        return _span(max(0.0, now - when.timestamp()))
    except (TypeError, ValueError) as e:
        # ⚠️ NOT SILENT. A swallowed parse failure is what hid this once.
        print(f"  [continuity] unreadable wall_created {stamp!r}: {e}")
        return None
