"""
Mati's library: what it has read, what was suggested, what it wants to read.

⚠️ THE GAP THIS FILLS IS A CATEGORY, NOT A TABLE. `provenance_of` knew five
kinds — lived, charter, studied, self-asked, and `about-you`. There was a
category for THE PERSON and none for MATI. Anything Mati held about itself
that was not an episode had nowhere to go, so it went nowhere: the reading list
existed only as a `source` string on 1,744 scattered passages, and "what have
you read" had to be answered by pattern-matching note sources at query time.

That absence had a measured cost. 1 of 400 book notes contains the words "Aztec
Thought and Culture" — the title lived in `source` and nothing resolved it — so
"what does the book say about X" could not match the book BY CONSTRUCTION, and
Mati filled the silence by inventing a passage and attributing it to a book it
could not name. The missing catalogue is what made the confabulation possible.

So two things, and the first is the point:

  A KIND. `mati:` sources are Mati's own — provenance "mine". Distinct from
  `about-you` (the person), from `lived` (what happened), and from `studied`
  (what was read). It is where anything Mati holds ABOUT ITSELF belongs.

  A CATALOGUE. One row per work, with a status. Read, reading, suggested by
  someone, or wanted. Derived from what is already true where it can be, so
  the library is not a second copy of reality that can drift from the first.

⚠️ WHAT IS DERIVED AND WHAT IS STORED. Progress is DERIVED, every time, from
`study_queue` — the number of absorbed passages is a fact about the store and
duplicating it here would create two answers to one question. Only what cannot
be derived is stored: that someone suggested a book, who, and that Mati wants
to read it. Nothing in the store records an intention.

⚠️ THIS IS NOT A READING SCHEDULER. It records what is true about a work; it
does not decide what to study next. `study.py` owns that, and a catalogue that
quietly became a queue would be two systems disagreeing about the same list.
"""

from __future__ import annotations

import os
import re
import sqlite3

from . import bridge

TABLE = "library"

# Mati's own. Everything under this prefix is provenance "mine".
MINE_PREFIX = "mati:"
LIBRARY_SOURCE = MINE_PREFIX + "library"

READ = "read"
READING = "reading"
SUGGESTED = "suggested"
WANTED = "wanted"

# Sources that are documents Mati read. `studied:asked about X` is excluded:
# those are Mati's own questions to itself, not works — filing them as books
# would put 1,000 invented titles in a library meant to be checkable.
# ⚠️ The CHARTER is excluded too. It is a document and it was studied, but it
# is Mati's principles, not something it read and might disagree with — it
# already has its own provenance kind, and listing it as a book invites "what
# did you think of it".
_NOT_A_WORK = ("studied:asked about", "studied:charter:")


def ensure(db_path=None):
    """The catalogue table. Idempotent."""
    conn = sqlite3.connect(db_path or bridge.NOTES_DB_PATH, timeout=10)
    try:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {TABLE} (
                    title        TEXT PRIMARY KEY,
                    origin       TEXT,
                    status       TEXT,
                    suggested_by TEXT,
                    author       TEXT,
                    published    TEXT,
                    added_at     TEXT,
                    finished_at  TEXT,
                    note         TEXT
                )""")
        conn.commit()
        return True
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def ensure_metadata(db_path=None):
    """`author` and `published`, added after the table existed.

    ⚠️ WHY A WORK NEEDS ITS PROVENANCE. Asked who wrote Aztec Thought and
    Culture, Mati named three authors, none of them the real one. It was not
    being careless: the text itself carries NO author line — checked, none in
    the first 400 passages — so there was nothing to recall and it composed
    something plausible.

    And the user's second reason is the better one: a publication YEAR tells
    Mati how far to trust what it read. A 1963 study of Nahuatl thought is a
    fine source and is also sixty years old, and those are both true at once.
    """
    conn = sqlite3.connect(db_path or bridge.NOTES_DB_PATH, timeout=10)
    try:
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info({TABLE})")}
        for name in ("author", "published"):
            if name not in cols:
                conn.execute(f"ALTER TABLE {TABLE} ADD COLUMN {name} TEXT")
        conn.commit()
        return True
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def describe_work(title, author=None, published=None, db_path=None):
    """Record who wrote a work and when it was published."""
    ensure(db_path); ensure_metadata(db_path)
    conn = sqlite3.connect(db_path or bridge.NOTES_DB_PATH, timeout=10)
    try:
        cur = conn.execute(
            f"UPDATE {TABLE} SET author=COALESCE(?, author), "
            f"published=COALESCE(?, published) WHERE title=?",
            (author, published, title.strip()))
        if cur.rowcount == 0:
            conn.execute(
                f"INSERT INTO {TABLE} (title, status, author, published, "
                f"added_at) VALUES (?,?,?,?,?)",
                (title.strip(), READ, author, published, bridge._now()))
        conn.commit()
        return True
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def title_of(origin):
    """A work's title from the source string it was stored under.

    `studied:Aztec Thought and Culture.txt` -> `Aztec Thought and Culture`.
    The same normalisation search uses to make a title matchable at all.
    """
    t = re.sub(r"^studied:(charter:)?", "", origin or "")
    t = re.sub(r"\.(txt|md|pdf|epub|rtf|html?)$", "", t, flags=re.I)
    t = t.replace("_", " ").strip()
    return t or (origin or "")


def _read_works(db_path=None):
    """Works Mati has actually read, with progress derived from the queue."""
    conn = sqlite3.connect(f"file:{db_path or bridge.NOTES_DB_PATH}?mode=ro",
                           uri=True)
    try:
        rows = conn.execute(
            # ⚠️ ARCHIVED PASSAGES ARE COUNTED APART, or shelving a work
            # changes nothing a person can see. This counted every
            # `studied:%` row with no regard to `archived`, so a book you had
            # just taken off the shelf looked exactly like one you had not —
            # and with no way to tell, there was no way to put it back.
            "SELECT source, COUNT(*), "
            "       SUM(CASE WHEN shelved IS NULL THEN 1 ELSE 0 END) "
            "FROM notes WHERE source LIKE 'studied:%' "
            "GROUP BY source").fetchall()
    except sqlite3.Error:
        return {}
    finally:
        conn.close()

    try:
        from . import reading
        prog = {d["origin"]: d for d in reading.progress(db_path)}
    except Exception:
        prog = {}

    out = {}
    for source, n, live in rows:
        if any(source.startswith(p) for p in _NOT_A_WORK):
            continue
        title = title_of(source)
        # `study_queue` keys on the bare origin, `notes` on the `studied:`
        # form. Try both rather than assume; a missing match means progress is
        # unknown, which is reported as unknown and never as zero.
        p = prog.get(source) or prog.get(re.sub(r"^studied:", "", source))
        # ⚠️ `live` IS WHAT IS STILL ON THE SHELF. `passages` stays the full
        # count — an archived work has not lost anything, which is the whole
        # point of archiving rather than deleting — and `shelved` says how
        # much of it is currently put away.
        out[title] = {
            "title": title, "origin": source, "passages": n,
            "live": live, "shelved": n - (live or 0),
            "archived": (live or 0) == 0,
            "done": p["done"] if p else None,
            "total": p["total"] if p else None,
            "share": p["share"] if p else None,
        }
    return out


def add(title, status=WANTED, suggested_by=None, note=None, db_path=None):
    """Record an intention or a suggestion — the part nothing else stores."""
    ensure(db_path)
    conn = sqlite3.connect(db_path or bridge.NOTES_DB_PATH, timeout=10)
    try:
        conn.execute(
            f"INSERT INTO {TABLE} (title, origin, status, suggested_by, "
            f"added_at, note) VALUES (?, NULL, ?, ?, ?, ?) "
            f"ON CONFLICT(title) DO UPDATE SET status=excluded.status, "
            f"suggested_by=COALESCE(excluded.suggested_by, "
            f"{TABLE}.suggested_by), note=COALESCE(excluded.note, "
            f"{TABLE}.note)",
            (title.strip(), status, suggested_by, bridge._now(), note))
        conn.commit()
        return True
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def forget(title, db_path=None):
    """Drop an entry. Only ever removes an intention, never a read work —
    those are derived from the store and reappear, correctly, on the next
    call."""
    conn = sqlite3.connect(db_path or bridge.NOTES_DB_PATH, timeout=10)
    try:
        conn.execute(f"DELETE FROM {TABLE} WHERE title=?", (title.strip(),))
        conn.commit()
        return True
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def catalogue(db_path=None):
    """The whole library: read and in progress, then suggested and wanted.

    Read works come from the STORE, not from this table, so the catalogue
    cannot claim a book Mati has not absorbed a single passage of.
    """
    # ⚠️ THE PRIORITY COLUMN MUST EXIST BEFORE THE SELECT THAT READS IT.
    # Without this the query raised, `except sqlite3.Error` swallowed it, and
    # `catalogue()` returned an EMPTY LIST — every suggestion and every wanted
    # book silently gone. ⚠️ Only `test_terminal` catches this; nothing in
    # the normal path notices an empty catalogue.
    ensure(db_path); ensure_metadata(db_path)
    from .intake import ensure_priority
    ensure_priority(db_path)
    works = _read_works(db_path)
    for w in works.values():
        share = w.get("share")
        w["status"] = READING if (share is not None and share < 1.0) else READ

    conn = sqlite3.connect(f"file:{db_path or bridge.NOTES_DB_PATH}?mode=ro",
                           uri=True)
    try:
        rows = conn.execute(
            f"SELECT title, status, suggested_by, added_at, note, author, "
            f"published, priority FROM {TABLE}").fetchall()
    except sqlite3.Error:
        rows = []
    finally:
        conn.close()

    for title, status, by, added, note, author, published, pri in rows:
        if title in works:
            works[title]["author"] = author
            works[title]["published"] = published
            works[title]["priority"] = pri
            # It has since been read. The store wins — an intention that came
            # true is not still an intention.
            works[title]["suggested_by"] = by
            continue
        works[title] = {"title": title, "origin": None, "passages": 0,
                        "done": None, "total": None, "share": None,
                        "status": status or WANTED, "suggested_by": by,
                        "added_at": added, "note": note,
                        "author": author, "published": published,
                        "priority": pri}

    # ⚠️ WANTED IS ORDERED BY PRIORITY, WHICH IS WHAT MAKES IT A QUEUE. Without
    # this the number was decorative: a 40,000-word book at priority 833 listed
    # above a paper at 4, so "given appropriate priority" changed nothing about
    # what Mati would reach for next. Lower is sooner, and `intake.shelve`
    # sets it to the passage count — shortest first, because a short paper
    # finishing this week beats a long book finishing next month.
    order = {READING: 0, READ: 1, SUGGESTED: 2, WANTED: 3}

    def _key(w):
        pri = w.get("priority")
        if w["status"] in (WANTED, SUGGESTED) and pri is not None:
            return (order.get(w["status"], 9), pri, w["title"])
        return (order.get(w["status"], 9), -(w.get("passages") or 0),
                w["title"])

    return sorted(works.values(), key=_key)


def describe(db_path=None, limit=8, voice="i"):
    """The library in Mati's own words, for the waking block.

    ⚠️ NO COUNTS OF PASSAGES OR CHUNKS. That is instrumentation, and this
    project has measured twice what happens when instrumentation reaches a 3B
    model: it recites it. What a person knows about their own reading is which
    books, and roughly how far in — so that is what this says.
    """
    items = catalogue(db_path)
    if not items:
        return None
    reading = [w for w in items if w["status"] == READING]
    read = [w for w in items if w["status"] == READ]
    want = [w for w in items if w["status"] in (SUGGESTED, WANTED)]

    # ⚠️ THE WAKING BLOCK ADDRESSES MATI AS "YOU". Written first in the first
    # person, this line sat directly under "You are not part way through
    # anything" and flipped person mid-paragraph. "The internal instructions
    # are confusing the agent as to who is who" is a defect this project has
    # already paid for once; a block that changes speaker between adjacent
    # sentences is how it comes back.
    you = voice == "you"
    have = "You have" if you else "I have"
    am = "You are" if you else "I am"

    parts = []
    def _cite(w):
        """Title, who wrote it, and when — so Mati never has to guess.

        ⚠️ Asked who wrote Aztec Thought and Culture, Mati named three
        authors and none was right. The text carries no author line, so there
        was nothing to recall. And the YEAR is not decoration: it is how Mati
        knows that a good source can also be sixty years old.
        """
        bits = w["title"]
        if w.get("author"):
            bits += f", by {w['author']}"
        if w.get("published"):
            bits += f" ({w['published']})"
        return bits

    if read:
        names = [_cite(w) for w in read[:limit]]
        more = len(read) - len(names)
        parts.append(f"{have} read " + _join(names) +
                     (f", and {more} more" if more > 0 else "") + ".")
    if reading:
        bits = []
        for w in reading[:2]:
            if w.get("share") is not None:
                bits.append(f"{w['title']} (about {round(w['share']*100)}% "
                            f"of the way through)")
            else:
                bits.append(w["title"])
        parts.append(f"{am} part way through " + _join(bits) + ".")
    if want:
        bits = [f"{w['title']}" + (f", suggested by {w['suggested_by']}"
                                   if w.get("suggested_by") else "")
                for w in want[:limit]]
        parts.append(f"{have} not read " + _join(bits) + " yet.")
    return " ".join(parts) or None


def _join(items):
    items = list(items)
    if len(items) <= 1:
        return items[0] if items else ""
    return ", ".join(items[:-1]) + " and " + items[-1]


def find(query, db_path=None):
    """The work a question is about, if it names one.

    ⚠️ This is what "the book" had no way to resolve to. A title match here is
    what lets `search_sources` be scoped to one work instead of the whole
    store — the memory acting as the key to the verbatim lookup.
    """
    q = (query or "").lower()
    best = None
    for w in catalogue(db_path):
        t = w["title"].lower()
        if t and t in q:
            if best is None or len(t) > len(best["title"]):
                best = w
    if best:
        return best
    # ⚠️ "THE BOOK" IS HOW PEOPLE ACTUALLY ASK, and it names nothing. Resolved
    # only when there is exactly ONE book it could mean — with two, the honest
    # answer is that the question is ambiguous, and guessing is how a passage
    # gets attributed to the wrong work. That failure is the reason this
    # module exists; it must not be reintroduced here for convenience.
    if re.search(r"\bthe (book|text|document)\b", q):
        books = [w for w in catalogue(db_path)
                 if w["status"] in (READ, READING) and w.get("origin")]
        if len(books) == 1:
            return books[0]
    # Failing a title, a distinctive word from one. Two letters is not a work.
    for w in catalogue(db_path):
        words = [x for x in re.findall(r"[a-z]{4,}", w["title"].lower())
                 if x not in ("book", "the", "and", "thought")]
        if words and all(x in q for x in words[:2]) and len(words) >= 2:
            return w
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Reaching the turn
# ─────────────────────────────────────────────────────────────────────────────

# ⚠️ THE CATALOGUE WAS BUILT, CORRECT, AND NEVER SHOWN. Checked :
# `library.` appeared nowhere in `mati.py`, so "what have you read" was
# answered from whatever lexical search happened to return — usually passages
# OF the book rather than the fact of having read it. Same shape as `activity`
# and the telemetry: the data existed and had no reader.
_ASKS_ABOUT_BOOKS = (
    "what have you read", "what did you read", "have you read",
    "your library", "what books", "which books", "any books",
    "reading list", "what are you reading", "what do you want to read",
    "in your library", "books you", "read anything",
)


def asks_about_books(text):
    t = " ".join((text or "").lower().split())
    return any(p in t for p in _ASKS_ABOUT_BOOKS)


def block(db_path=None):
    """The shelf, in Mati's own voice. Derived, so it is `<computed>`.

    ⚠️ First person and NEVER "you" — a block addressed to Mati that says
    "you have read" comes back out as a claim about the person.
    """
    try:
        cat = catalogue(db_path=db_path)
    except Exception:
        return ""
    if not cat:
        return ""
    read = [b for b in cat if b.get("status") == "read"]
    reading = [b for b in cat if b.get("status") == "reading"]
    want = [b for b in cat if b.get("status") in ("wanted", "suggested")]
    L = ["<computed>",
         "My library, worked out from my own record just now — not remembered:"]

    def name(b):
        bits = b["title"]
        if b.get("author"):
            bits += f" by {b['author']}"
        if b.get("published"):
            bits += f" ({b['published']})"
        return bits

    for b in read:
        L.append(f"  - I have READ {name(b)}, all {b.get('total', 0):,} "
                 f"passages of it. I can quote it if I am asked about "
                 f"something in it.")
    for b in reading:
        L.append(f"  - I am part way through {name(b)} — "
                 f"{b.get('done', 0):,} of {b.get('total', 0):,} passages.")
    for b in want:
        L.append(f"  - I mean to read {name(b)}, and have not started.")
    L.append("</computed>")
    return "\n".join(L)
