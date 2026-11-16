"""
Taking a document in: a summary now, the reading later.

A document gets a summary for immediate use, unless Mati is specifically told
to read it — and then the answer depends on the file's size. Either way, once
received, the file goes into the personal library, marked to read and given
an appropriate priority.

Three things happen on arrival and they are deliberately separate:

  SUMMARISE  one bounded LLM pass over a sample, available immediately. This is
             what the person can talk about in the next minute.
  SHELVE     an entry in the library, status `wanted`, with a priority. Mati
             knows it has the thing before it has read a word of it.
  ESTIMATE   what reading it would actually cost, in the numbers this machine
             produces rather than in an encouraging guess.

⚠️ READING IS NOT SUMMARISING AND THE DIFFERENCE IS HOURS. Studying imprints
each passage into θ₀ — measured: **~48 seconds of work per passage**, and
because studying only runs on idle, **one passage per ~444 seconds of wall
clock**. A 1,744-passage book is not a progress bar, it is a fortnight, and a
person asking "read this" deserves the real figure before it starts rather
than after.

⚠️ AND A SUMMARY IS NOT A MEMORY. What comes back from `summarise` is the
model's account of a sample, held for the conversation. It is not imprinted, it
is not `studied:`, and it must never be presented as something Mati read — the
`learning.py` docstring is the long version of why that distinction is the whole
design.

Plain text only, per `study.read_file`, and deliberately.
"""

from __future__ import annotations

import sqlite3

from . import bridge

# Measured, not assumed. `WORK_S` is the time a passage costs while it is being
# studied; `WALL_S` is how long one takes to come round, because studying waits
# for an idle machine. The second is what a person actually experiences and is
# the one quoted.
WORK_S_PER_PASSAGE = 48
WALL_S_PER_PASSAGE = 444

# `study.chunk_text` targets 64 tokens; ~0.75 words per token.
WORDS_PER_PASSAGE = 48

# How much of a document the summary is allowed to see. A bounded sample, so
# arrival costs one short call whatever the file weighs.
SAMPLE_WORDS = 700

_PROMPT = (
    "Below is part of a document that was just handed to you. Say what it is "
    "about, in two or three plain sentences, as you would tell someone who "
    "asked. Name the subject and the kind of document. Do not pretend to have "
    "read all of it, and do not list what you were given.")


def size_of(text):
    """What this document is, and what reading it would cost."""
    words = len((text or "").split())
    passages = max(1, round(words / WORDS_PER_PASSAGE))
    return {
        "words": words,
        "passages": passages,
        "work_hours": round(passages * WORK_S_PER_PASSAGE / 3600, 1),
        "wall_hours": round(passages * WALL_S_PER_PASSAGE / 3600, 1),
    }


def describe_size(text):
    """The honest sentence to say when someone asks Mati to read something."""
    s = size_of(text)
    if s["passages"] <= 3:
        return (f"That is short — {s['words']} words. I can take it in now.")
    wall = s["wall_hours"]
    if wall < 24:
        when = f"about {wall:.0f} hours"
    else:
        when = f"about {wall / 24:.1f} days"
    return (f"That is {s['words']:,} words, roughly {s['passages']:,} passages. "
            f"Reading it properly takes {when} of quiet time, because I take "
            f"each passage in separately and only while nothing else is "
            f"happening. I can summarise it now and read it in the background.")


def _sample(text, words=SAMPLE_WORDS):
    """The beginning, the middle and the end — not just the first page.

    ⚠️ A summary from the opening alone describes the preface. Documents put
    their subject where they like.
    """
    w = (text or "").split()
    if len(w) <= words:
        return " ".join(w)
    third = words // 3
    mid = len(w) // 2
    return " ".join(w[:third] + ["…"] + w[mid:mid + third] + ["…"] + w[-third:])


def summarise(text, llm=None, verbose=False):
    """What this document is about. One bounded pass, or None.

    ⚠️ Returns None rather than a fallback. A document that produces nothing
    usable should be shelved without a description; inventing one is exactly
    the failure that `learning.learn_from` refuses for the same reason.
    """
    sample = _sample(text)
    if len(sample.split()) < 20:
        return None
    try:
        if llm is None:
            from . import llm as L
            llm = L.make_llm(_PROMPT)
        said, _ = llm.reply([{"role": "user", "content": sample[:4000]}],
                            max_tokens=160)
    except Exception as e:
        if verbose:
            print(f"  [intake] {e}")
        return None
    said = " ".join((said or "").split())
    return said if len(said.split()) >= 5 else None


def shelve(title, text, priority=None, db_path=None):
    """Put it in the library, marked to read, with a place in the order.

    ⚠️ Priority is SIZE-ORDERED by default, shortest first, and that is a
    choice worth naming: a 40-page paper finishing this week is worth more than
    a 400-page book finishing next month, and a queue ordered by arrival buries
    everything behind whatever was handed over first.
    """
    from . import library
    library.ensure(db_path)
    ensure_priority(db_path)
    s = size_of(text)
    if priority is None:
        priority = s["passages"]          # shortest first
    library.add(title, status=library.WANTED, db_path=db_path)
    conn = sqlite3.connect(db_path or bridge.NOTES_DB_PATH, timeout=30)
    try:
        conn.execute(f"UPDATE {library.TABLE} SET priority=? WHERE title=?",
                     (int(priority), title.strip()))
        conn.commit()
    except sqlite3.Error:
        pass
    finally:
        conn.close()
    return priority


def ensure_priority(db_path=None):
    from . import library
    conn = sqlite3.connect(db_path or bridge.NOTES_DB_PATH, timeout=30)
    try:
        cols = {r[1] for r in conn.execute(
            f"PRAGMA table_info({library.TABLE})")}
        if "priority" not in cols:
            conn.execute(
                f"ALTER TABLE {library.TABLE} ADD COLUMN priority INTEGER")
            conn.commit()
        return True
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def receive(text, title, llm=None, db_path=None, verbose=False):
    """A document arrives. Summarise, shelve, and report what reading costs."""
    s = size_of(text)
    summary = summarise(text, llm=llm, verbose=verbose)
    priority = shelve(title, text, db_path=db_path)
    try:
        from . import activity
        activity.log("shelved", f"{title} — {s['passages']} passages")
    except Exception:
        pass
    return {"title": title, "summary": summary, "size": s,
            "priority": priority, "reading": describe_size(text)}
