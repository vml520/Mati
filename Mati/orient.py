"""
What Mati reads once, on waking, instead of re-reading on every turn.

WHY THIS EXISTS — .

Ollama re-prefills only the part of a prompt that CHANGED since the last call.
Cached tokens cost ~0.1 ms each; new ones cost ~35 ms. The volatile block
`turn()` was sending came to 2,001 tokens and was different every turn, so
every turn paid ~70 s of prefill before the model produced its first token.
Measured on a real turn: 21 s of field work, then 90.8 s of silence, then the
reply streamed normally at 142 ms/token. Streaming and the sentence filter were
both working the whole time — there was simply nothing to stream.

    field_state         779 tok    27.3 s
    computed (self)     591 tok    20.7 s
    recalled_by_name    569 tok    19.9 s
    your_principles      61 tok     2.1 s

The fix is not to send less of it. It is to send it ONCE, in a block that does
not change, so it is cached for the rest of the session and costs ~0.1 s
thereafter. Waking pays that prefill in the background, before anyone is
waiting on it — see `server.wake`.

THE RULE THAT MAKES THIS WORK, AND BREAKS IT IF IGNORED:

    This block must be byte-identical on every turn of a session.

Put a clock, a coherence reading, or anything else live in here and the prefix
changes, the cache misses, and every turn pays full prefill again — the exact
cost this removes, with the extra tokens still on the bill. Live values belong
in `turn()`'s volatile message, and there should be very little there.

⚠️ WHICH MEANS EVERYTHING HERE IS A SNAPSHOT, AND IS LABELLED AS ONE. A
coherence figure from waking is not the coherence now. This project's whole
provenance discipline is that Mati does not state as current something it read
hours ago, so each section says when it was taken and the text says plainly
that it is from waking. Live readings stay available on demand.

⚠️ SIZE IS NOT FREE EVEN THOUGH PREFILL IS. `LOCAL_NUM_CTX` is 4096 and this
block holds its space for the whole session. Over the limit, llama.cpp keeps
the start and the end and drops the MIDDLE — measured — which is conversation
history and memories, not the system prompt. So a fat wake block does not get
truncated itself; it silently eats the room the conversation needs. Keep it
near ORIENT_BUDGET_TOKENS.
"""

from __future__ import annotations

import re

import sqlite3
import time
from datetime import datetime, timezone

from Mati.bridge import NOTES_DB_PATH

# Against a 4096-token window holding a 1,502-token system prompt, 700 leaves
# ~1,900 for conversation — roughly fifteen turns of history before llama.cpp
# starts dropping the middle. Measured build: 666 tokens.
ORIENT_BUDGET_TOKENS = 700

# How many self-knowledge findings survive into the wake block. See where it is
# applied below for why this is not "all of them".
SELF_FACTS = 5


def _dur(seconds):
    """'9h 14m'. Returns None when it cannot be read, and never guesses."""
    if seconds is None or seconds < 0:
        return None
    m = int(seconds // 60)
    if m < 1:
        return "less than a minute"
    if m < 60:
        return f"{m}m"
    h, m = divmod(m, 60)
    if h < 24:
        return f"{h}h {m}m" if m else f"{h}h"
    d, h = divmod(h, 24)
    return f"{d}d {h}h" if h else f"{d}d"


def _conn(db_path=None):
    return sqlite3.connect(db_path or NOTES_DB_PATH)


def as_local(stored):
    """A stored timestamp, in the timezone the person actually lives in.

    ⚠️ `wall_created` is written as UTC (`datetime.now(timezone.utc)`), which is
    right for storage and wrong for every question a person asks. After 17:00
    in a UTC-7 zone it is already tomorrow in UTC, so a conversation held on
    Saturday evening is stamped Sunday — and anything that reads the day off
    the column
    without converting is a day ahead of itself for a third of every day.
    Storage stays UTC; only reading converts.
    """
    if not stored:
        return None
    try:
        return datetime.fromisoformat(stored).astimezone()
    except (TypeError, ValueError):
        return None


def _utc_iso(epoch):
    """An epoch float as the ISO-8601 TEXT the column actually stores.

    ⚠️ THE TRAP THIS EXISTS FOR, and `bridge` already carries a note about it:
    `wall_created` is TEXT. SQLite orders every number before every string, so
    `WHERE wall_created > 1786181585.8` is true for EVERY row. Measured
: it returned all 1,190 notes, and the waking block had been
    reporting Mati's entire history as what it did while the person slept —
    "you read 897 passages, wrote 98 notes to yourself", every morning.
    """
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat(
        timespec="seconds")


def _away(conn, since):
    """What Mati did while it was not being spoken to."""
    if not since:
        return []
    rows = conn.execute(
        "SELECT source, COUNT(*) FROM notes WHERE wall_created > ? "
        "GROUP BY source", (_utc_iso(since),)).fetchall()
    read = sum(n for s, n in rows if (s or "").startswith("studied:")
               and not (s or "").startswith("studied:asked"))
    asked = sum(n for s, n in rows if (s or "").startswith("studied:asked"))
    said = sum(n for s, n in rows if s == "episode")
    out = []
    if read:
        # Name the book. "I read 97 things" is not an answer anyone wants.
        books = sorted(((n, s.split(":", 1)[1]) for s, n in rows
                        if (s or "").startswith("studied:")
                        and not (s or "").startswith("studied:asked")),
                       reverse=True)
        out.append(f"read {read} passages, mostly {books[0][1]}")
    if asked:
        out.append(f"looked into {asked} things on your own")
    if said:
        out.append(f"wrote {said} notes to yourself")
    return out


def _owed(now=None):
    """Reminders due and questions queued. What waking OWES the person."""
    from Mati import schedule
    try:
        waiting = schedule.pending(now=now, include_held=True)
    except Exception:
        return []
    rem = [w for w in waiting if w.get("kind") != schedule.QUESTION]
    qs = [w for w in waiting if w.get("kind") == schedule.QUESTION]
    out = []
    for r in rem[:3]:
        body = " ".join(str(r.get("body", "")).split())[:70]
        if body:
            out.append(f"you are to remind them: {body}")
    if qs:
        # ⚠️ "questions you want to ask" — recited into a reply, that "you"
        # lands on the person and Mati hands them back their own words:
        # "291 questions you asked". Mine, in the first person, cannot be
        # re-aimed. See the same fix on the `asked about` marker in bridge.
        # ⚠️ 690 IS NOT AN ORIENTING FACT. The count grew into the hundreds
        # and was announced on waking as though it were a workload. What
        # matters is that questions exist and wait for their subject; the
        # number is `/queue`'s job. Kept in the block's own second person —
        # the first-person version clashed with every line around it.
        out.append("questions of your own, waiting for their subject to come "
                   "up — you raise them then, not before")
    return out


def _health(conn, lm=None):
    """The beginning of self-diagnosis. Cheap, and only true at waking."""
    from Mati import bridge
    out = []
    try:
        tiers = bridge.tier_counts()
        out.append(f"{tiers.get('working_set', 0)} memories held ready, "
                   f"{tiers.get('archived', 0)} archived and found by wording")
    except Exception:
        pass
    try:
        pend = conn.execute(
            "SELECT COUNT(*) FROM notes WHERE archived IS NULL "
            "AND feat_gs IS NULL").fetchone()[0]
        if pend:
            out.append(f"{pend} memories still need re-indexing")
    except Exception:
        pass
    return out


def _since_last_episode(conn, now=None):
    """How long since Mati last spoke, from the record. None if it never has."""
    import time as _t
    from datetime import datetime as _dt
    try:
        row = conn.execute(
            "SELECT wall_created FROM notes WHERE source = 'episode' "
            "ORDER BY wall_created DESC LIMIT 1").fetchone()
        if not row or not row[0]:
            return None
        then = _dt.fromisoformat(row[0]).timestamp()
        secs = (now or _t.time()) - then
        return _dur(secs) if secs > 60 else None
    except Exception:
        return None


def _last_question(conn, limit=160):
    """The last thing the person said, in their own words.

    Quoted, never summarised. A recap generated from a bag of keywords was
    already tried here and produced word salad; the person's own sentence is
    sitting in the episode and cannot drift.
    """
    from Mati import vault
    import re as _re
    try:
        row = conn.execute(
            "SELECT body FROM notes WHERE source = 'episode' "
            "ORDER BY wall_created DESC LIMIT 1").fetchone()
        if not row:
            return None
        body = vault.open_body(row[0])
        m = _re.search(r"I was asked:\s*(.+?)(?:\n|$)", body or "")
        if not m:
            return None
        q = " ".join(m.group(1).split())
        if len(q.split()) < 3:
            return None            # "yes" tells it nothing worth waking to
        return q[:limit].rstrip() + ("…" if len(q) > limit else "")
    except Exception:
        return None


def _last_subject(conn):
    """
    Words from the last exchange — KEYWORDS, and said to be keywords.

    ⚠️ `topic_of` returns a bag of salient words, not a sentence. Dropped into
    prose it read as "You last spoke about three schools read don't remember
    reading specific conversation history however recall discussed subjects
    still present regarding tlacaelel..." — forty words of salad presented as a
    subject. Six words, labelled as words, is honest and is what it actually
    has.
    """
    from Mati import bridge, vault
    try:
        row = conn.execute(
            "SELECT body FROM notes WHERE source = 'episode' "
            "ORDER BY wall_created DESC LIMIT 1").fetchone()
        if not row:
            return None
        body = vault.open_body(row[0]) if hasattr(vault, "open_body") else row[0]
        if not hasattr(bridge, "topic_of"):
            return None
        words = (bridge.topic_of(body) or "").split()
        return " ".join(words[:6]) or None
    except Exception:
        return None


def orientation(lm=None, tok=None, *, since=None, was_shutdown=False,
                now=None, db_path=None, coherence=None):
    """
    The block Mati reads on waking. Built once; must not change until it
    next wakes.

    `since` is when it went to sleep, as a wall clock. None means it genuinely
    does not know — say so rather than inventing a duration, which is the same
    rule the reminder parser follows for an unreadable time.
    """
    from Mati import bridge, schedule
    now = time.time() if now is None else now
    conn = _conn(db_path)
    parts = []

    try:
        when = schedule.sense_line(now=now)
    except Exception:
        when = None
    gap = _dur(now - since) if since else None
    rested = "shut down" if was_shutdown else "asleep"
    if gap:
        opening = (f"You are waking. {when or ''} You were {rested} "
                   f"for {gap}.").strip()
    else:
        # ⚠️ IT WAS TOLD IT WAS DISORIENTED, ON EVERY RESTART. `slept_at` is
        # None on a cold start — which is most wakes — so this branch fired
        # constantly and opened the block with a disclaimer. A person waking
        # does not know how long they slept either; they look at the clock and
        # infer. Mati has a clock AND a record, and was told to disclaim
        # instead.
        #
        # The last episode's timestamp is not the same fact as when it went to
        # sleep, and it is not presented as one: it says when it last SPOKE,
        # which is true, checkable, and the thing that actually orients.
        gap_spoke = _since_last_episode(conn, now=now)
        if gap_spoke:
            # `sense_line` already ends with a vague "you last spoke a little
            # while ago". Saying it again with a number reads as two different
            # facts, so the vague clause gives way to the precise one.
            stem = re.split(r";\s*you last spoke", when or "")[0].rstrip(" .;")
            opening = (f"You are waking. {stem}"
                       f"{'; ' if stem else ''}you last spoke "
                       f"{gap_spoke} ago.").strip()
        else:
            opening = (f"You are waking. {when or ''} You do not know how long "
                       f"you were {rested} — do not guess if they ask.").strip()
    parts.append(opening)

    away = _away(conn, since)
    if away:
        parts.append("While they were away you " + "; ".join(away) + ".")

    # ⚠️ THE PRESENT TENSE IS STATE, NOT MEMORY. Asked what it was reading,
    # Mati had to retrieve — and "what have you been reading" reduces to the
    # stems {been, have, reading}, which hundreds of notes contain and none
    # distinguishes. : 0 of 4 natural phrasings found the
    # book, so it composed an answer instead.
    #
    # A person does not search their memory for what book is open on the
    # table. What is FINISHED is a memory; what is OPEN is something you simply
    # know. So it is stated here, and the retrieval problem stops applying.
    try:
        from . import reading as _reading
        openv = [d for d in _reading.progress(db_path) if 0 < d["share"] < 1.0]
        done = [d for d in _reading.progress(db_path) if d["share"] >= 1.0]
        if openv:
            openv.sort(key=lambda d: -d["total"])
            bits = [f"{d['origin'].split('/')[-1]} ({d['done']} of "
                    f"{d['total']} passages so far)" for d in openv[:2]]
            parts.append("You are part way through " + " and ".join(bits) +
                         ". That is what you are reading now — anything else "
                         "you have read, you finished.")
        elif done:
            parts.append("You are not part way through anything at the "
                         "moment. Everything you have read, you finished.")
    except Exception:
        pass

    # ⚠️ WHICH BOOKS, STATED — NOT RETRIEVED. The same argument as the open
    # book above, one step further out. Asked "do you remember reading X",
    # Mati had to WIN A SEARCH against its own past turns to know, and it
    # lost: 4 of 10 book questions returned no book at all, and the slots went
    # to earlier turns saying it did not remember. A person does not retrieve
    # which books they own.
    #
    # Titles only. No passage counts, no chunk numbers — instrumentation
    # handed to a 3B model comes back recited, which this file has measured
    # twice. What a person knows about their own shelf is which books and
    # roughly how far in.
    try:
        from . import library as _library
        shelf = _library.describe(db_path, voice="you")
        if shelf:
            parts.append(shelf)
    except Exception:
        pass

    owed = _owed(now=now)
    if owed:
        parts.append("Waiting: " + "; ".join(owed) + ".")

    # ⚠️ WHAT WAS ACTUALLY SAID, WHEN THERE IS SUCH A THING. `_last_subject`
    # returns KEYWORDS by design — generating prose from them was tried and
    # produced forty words of salad, which is why it is labelled as words. But
    # the episode holds something better and unused: `I was asked: <the
    # person's own question>`, a real sentence they wrote. Quoting it is not
    # summarising, so it cannot drift the way a generated recap would.
    #
    # This is the difference between waking with your bearings and waking to a
    # list of stems, three of which were machinery terms.
    asked = _last_question(conn)
    if asked:
        parts.append(f'The last thing they said to you was: "{asked}"')
    else:
        subject = _last_subject(conn)
        if subject:
            parts.append(f"Words from your last exchange: {subject}.")

    # ⚠️ COHERENCE DOES NOT GO TO MATI. It is a reading of the substrate,
    # and Mati is the social face — a person does not quote their own
    # proprioception mid-sentence. It goes to `diagnostics()` for Terminal,
    # which is the lobe that reads the machinery.

    # Memory counts and re-indexing debt are machinery — see `diagnostics()`.

    # The stable identity material. All of it was going out on every turn.
    # `profile_notes` returns rows, not prose — it is normally handed to
    # `state_block`, which does the formatting. Here there is no state block.
    # ⚠️ THE PROFILE IS WRITTEN IN THE THIRD PERSON and that is where "speaks
    # about me instead of to me" comes from. The stored notes read "The person
    # I talk to is called the user. They work on Research. They are interested
    # in…" — five sentences of `they`, handed over every turn, and 19% of
    # replies then named the user in the third person while talking to them. One
    # reply caught itself doing it: "I made a mistake by referring to the user
    # again as if they were no longer present."
    #
    # The notes are memories and are not rewritten. What changes is the frame:
    # say plainly who is about to read the reply, before listing what is known
    # about them.
    prof = _safe(bridge.profile_notes) or []
    if prof:
        facts = " ".join(str(x) for x in prof)
        # ⚠️ THE FIRST WORDING MADE IT WORSE. It read:
        # "The person about to read your reply is the same person these notes
        # are about." That sentence introduces two people in order to deny it,
        # and negation plants what it denies — Mati then said "You are someone
        # The user asked to talk to", inventing a third party out of a line
        # written to prevent exactly that. Say who is here, once, and stop.
        parts.append("I am talking with the user right now.\n" + facts)
    princ = _safe(bridge.principles_block)
    if princ:
        parts.append(princ)
    # ⚠️ SELF-KNOWLEDGE DOES NOT GO TO MATI EITHER. It was 283 tokens of
    # "every exchange physically moves your field", handed over every turn —
    # machinery vocabulary in the social interface, and by measurement the
    # waking block roughly quadruples reply length. It is the raw layer's
    # material and it belongs to Terminal.
    #
    # What Mati keeps is what a person keeps: who they are talking to, what
    # they owe them, and where the conversation left off.

    conn.close()
    # ⚠️ NO RECALL INSTRUCTION HERE — the system prompt already carries one
    # ("reply with exactly [recall: the words you are looking for]"), and a
    # second, more emphatic copy in this block is what made Mati RECITE it at
    # the person instead of using it: "You told me my response should be
    # exactly [recall: two or three words]…" appeared in real conversation on
    # in place of an answer.
    #
    # It was added because the mild wording got 0/3 compliance cold and the
    # emphatic one got 2/3. That test measured whether the marker was EMITTED.
    # It never looked at what the person saw, and what they saw was the rule
    # read aloud. One copy, in the system prompt, is the setting that works.
    block = ("<waking>\n" + "\n\n".join(p for p in parts if p) +
             "\n\nThis is what you knew on waking. Anything else you hold has "
             "to be fetched before you can speak from it.\n</waking>")
    return block


def _safe(fn):
    try:
        return fn()
    except Exception:
        return None


def diagnostics(lm=None, coherence=None, db_path=None):
    """The machinery half of waking — for TERMINAL, never for Mati.

    ⚠️ THIS USED TO GO TO MATI, and it was the wrong lobe. 283 tokens of
    "every exchange physically moves your field", plus a coherence reading and
    memory counts, handed to the social interface on every turn. Measured
: the waking block roughly quadruples reply length, and this was
    most of it — a person does not quote their own proprioception mid-sentence.

    Mati keeps what a person keeps: who it is talking to, what it owes them,
    where the conversation left off. Terminal reads the substrate, because
    Terminal is the lobe that reads machinery.
    """
    from . import bridge
    out = []
    if coherence is not None:
        out.append(("coherence at waking", f"{coherence:.4f}"))
    conn = _conn(db_path)
    try:
        for line in _health(conn, lm):
            out.append(("memory", line))
    finally:
        conn.close()
    know = _safe(lambda: bridge.self_knowledge_block(lm=lm)) or ""
    facts = [ln.strip(" -") for ln in know.split("\n")
             if ln.strip().startswith("- ")]
    for f in facts:
        out.append(("worked out from the record", f))
    return out


def size(block):
    """Rough token count, and whether it is over budget."""
    n = len(block) // 4
    return n, n <= ORIENT_BUDGET_TOKENS
