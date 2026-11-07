"""
The standing brief: what is in hand, what it is aimed at, how far along.

    python3 -m Mati.brief                    the brief as Mati holds it
    python3 -m Mati.brief --target "3000 words" --for "the essay"
    python3 -m Mati.brief --revise 3 --to "5000 words"
    python3 -m Mati.brief --withdraw 3 --why "changed subject"

⚠️ WHY THIS EXISTS. The user: *"Mati is awake but not aware. The
point of the wake routine was so that Mati already knows what it's supposed to
do before being asked. Instead I have to walk Mati through its own memories to
get it to a point where it can help me."*

`drafts.py` was built for the first half of that — somewhere for "the thing we
are working on" to live — and it was never connected to anything. `block()`
and `current()` had no caller in the package, no test, and no mention in the
app. The table was written, sealed, and never read back into a conversation.
That is the eighth writer-with-no-reader in this project; the fix is not more
storage, it is a reader.

⭐ THE PRINCIPLE IS ALREADY IN `orient.py`, APPLIED TO ONE THING:

    "A person does not search their memory for what book is open on the
     table. What is FINISHED is a memory; what is OPEN is something you
     simply know. So it is stated here, and the retrieval problem stops
     applying."

Reading got that treatment. Intent never did — which is exactly why the
person had to narrate the project back every session. A target is the same
kind of thing as an open book: state, not memory.

⚠️ TARGETS ARE NEVER UPDATED IN PLACE, AND NEVER DELETED. This project's
standing rule is that corrections ANNOTATE (`correction.py`), and the
calendar already spells out the consequence: *"the row is marked `withdrawn`
and leaves the view, and 'the meeting you had on Tuesday was cancelled' stays
answerable"* (`agenda.py`). A revised target is a NEW row that supersedes the
old one, so "we moved it from 3,000 to 5,000 on Tuesday" is answerable too.

⚠️⚠️ AND MATI DOES NOT WITHDRAW A TARGET BY ITSELF. The user asked that Mati
"know whether to modify or delete the original target", and the honest answer
is that it often cannot know. Raising the number for the same work is a
revision and is safe to infer. Deciding that a piece of work is ABANDONED is
not: the evidence for "they changed topic" and "they took a week off" is the
same evidence. So withdrawal is something the person says, or something that
follows from a draft being closed — never a judgement call made on silence.
This is the same line `agenda.py` draws with *"MATI NEVER DECIDES A TIME"*.

⚠️ AN INFERRED TARGET SAYS SO. `source` is `user` or `inferred`, and a
inferred one carries its `basis`. The provenance scheme exists because Mati
must never present something it worked out as something it was told.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sqlite3

TABLE = "targets"

# What a target can be measured in. `done` is the binary case — a thing is
# finished or it is not — and is the default when no number was given.
MEASURES = ("words", "passages", "sessions", "done")

OPEN, MET, SUPERSEDED, WITHDRAWN = "open", "met", "superseded", "withdrawn"


def _db(db_path=None):
    from . import config as C
    return db_path or C.NOTES_DB


def _conn(db_path=None):
    conn = sqlite3.connect(_db(db_path))
    conn.execute(f"""CREATE TABLE IF NOT EXISTS {TABLE} (
            target_id  INTEGER PRIMARY KEY,
            what       TEXT NOT NULL,
            draft      TEXT,
            measure    TEXT NOT NULL,
            amount     INTEGER,
            due        TEXT,
            source     TEXT NOT NULL,
            basis      TEXT,
            status     TEXT NOT NULL,
            supersedes INTEGER,
            why        TEXT,
            created    TEXT,
            touched    TEXT
        )""")
    conn.commit()
    return conn


def _now():
    return dt.datetime.now().isoformat(timespec="seconds")


# ── reading a target out of what someone said ───────────────────────────────
# ⚠️ EXTRACTION STAYS CODE. The intent study's conclusion was that intent may
# be learned but extraction may not — a model asked for "the number in this
# sentence" will occasionally return a plausible different number, and a
# target nobody set is worse than no target.
_AMOUNT = re.compile(
    r"\b(\d[\d,]*)\s*(words?|pages?|passages?|sessions?|chapters?)\b", re.I)


def read_target(text):
    """`(amount, measure)` from a phrase, or `(None, 'done')`. Never guesses.

    ⚠️ RETURNS `done` RATHER THAN A NUMBER when there is no number. "finish
    the introduction" is a real target and its measure is simply whether it
    is finished; inventing "500 words" for it would be a fabrication that
    then gets tracked and reported as progress.
    """
    m = _AMOUNT.search(text or "")
    if not m:
        return None, "done"
    n = int(m.group(1).replace(",", ""))
    unit = m.group(2).lower().rstrip("s")
    measure = {"word": "words", "page": "words", "passage": "passages",
               "session": "sessions", "chapter": "done"}.get(unit, "done")
    # ⚠️ PAGES BECOME WORDS, AT A STATED RATE, OR NOT AT ALL. A "page" is not
    # a unit anything here can count, so it is converted once, visibly, at
    # 250 — and if that is wrong the person can say the number in words.
    if unit == "page":
        n *= 250
    return (n if measure != "done" else None), measure


# ── setting, revising, withdrawing ──────────────────────────────────────────

def set_target(what, *, draft=None, amount=None, measure="done", due=None,
               source="user", basis=None, db_path=None):
    """Record a target. Returns its id.

    ⚠️ `due` IS NEVER COMPUTED HERE. `agenda.py` draws the same line — Mati
    does not decide a time. A deadline arrives because the person said one.
    """
    if measure not in MEASURES:
        measure = "done"
    conn = _conn(db_path)
    try:
        cur = conn.execute(
            f"INSERT INTO {TABLE} (what, draft, measure, amount, due, source,"
            f" basis, status, created, touched)"
            f" VALUES (?,?,?,?,?,?,?,?,?,?)",
            (what, draft, measure, amount, due, source, basis, OPEN,
             _now(), _now()))
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def revise(target_id, *, what=None, amount=None, measure=None, due=None,
           why=None, db_path=None):
    """Supersede a target with a changed one. Returns the NEW id.

    ⚠️ A NEW ROW, NOT AN UPDATE. The old target is marked `superseded` and
    keeps its `why`, so "we moved it from 3,000 to 5,000, because the scope
    grew" stays answerable. An in-place update would erase the only record
    that the goal ever changed — and a person asking "wasn't this 3,000?"
    deserves an answer better than silence.
    """
    conn = _conn(db_path)
    try:
        row = conn.execute(
            f"SELECT what, draft, measure, amount, due, source FROM {TABLE}"
            f" WHERE target_id=?", (target_id,)).fetchone()
        if not row:
            return None
        o_what, o_draft, o_measure, o_amount, o_due, o_source = row
        conn.execute(f"UPDATE {TABLE} SET status=?, why=?, touched=?"
                     f" WHERE target_id=?",
                     (SUPERSEDED, why, _now(), target_id))
        cur = conn.execute(
            f"INSERT INTO {TABLE} (what, draft, measure, amount, due, source,"
            f" basis, status, supersedes, created, touched)"
            f" VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (what or o_what, o_draft, measure or o_measure,
             o_amount if amount is None else amount, due or o_due, o_source,
             None, OPEN, target_id, _now(), _now()))
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def withdraw(target_id, why, db_path=None):
    """Take a target out of the view. It stays on the record.

    ⚠️⚠️ CALLED BECAUSE THE PERSON SAID SO, OR BECAUSE THE DRAFT CLOSED —
    never because Mati concluded the work was abandoned. Silence is not
    evidence of abandonment; it is equally evidence of a week off, and the
    cost of being wrong is deleting someone's stated intention.
    """
    conn = _conn(db_path)
    try:
        conn.execute(f"UPDATE {TABLE} SET status=?, why=?, touched=?"
                     f" WHERE target_id=? AND status=?",
                     (WITHDRAWN, why, _now(), target_id, OPEN))
        conn.commit()
    finally:
        conn.close()


def met(target_id, db_path=None):
    conn = _conn(db_path)
    try:
        conn.execute(f"UPDATE {TABLE} SET status=?, touched=? "
                     f"WHERE target_id=?", (MET, _now(), target_id))
        conn.commit()
    finally:
        conn.close()


# ── reading the brief ───────────────────────────────────────────────────────

def open_targets(db_path=None):
    conn = _conn(db_path)
    try:
        rows = conn.execute(
            f"SELECT target_id, what, draft, measure, amount, due, source,"
            f" basis, supersedes, created FROM {TABLE}"
            f" WHERE status=? ORDER BY created", (OPEN,)).fetchall()
    finally:
        conn.close()
    keys = ("id", "what", "draft", "measure", "amount", "due", "source",
            "basis", "supersedes", "created")
    return [dict(zip(keys, r)) for r in rows]


def progress(target, db_path=None):
    """How far along, measured — or None when there is nothing to count.

    ⚠️ MEASURED, NOT ESTIMATED. The reading block's `{done, total, share}` is
    the shape being copied, and the reason it reads honestly is that both
    numbers are counted rather than judged. A target with no countable
    measure returns None and is reported as open, not as a percentage.
    """
    if target.get("measure") != "words" or not target.get("amount"):
        return None
    if not target.get("draft"):
        return None
    try:
        from . import drafts
        cur = drafts.current(db_path=db_path)
    except Exception:
        return None
    if not cur or cur.get("title") != target["draft"]:
        return None
    done = len((cur.get("body") or "").split())
    total = int(target["amount"])
    return {"done": done, "total": total,
            "share": (done / total) if total else 0.0}


def _due_phrase(due):
    """"by Friday" / "by 14 March", or "" — never a computed deadline."""
    if not due:
        return ""
    try:
        d = dt.date.fromisoformat(due[:10])
    except ValueError:
        return f" by {due}"
    days = (d - dt.date.today()).days
    if days < 0:
        return f" — that was due {abs(days)} day{'s' if abs(days) != 1 else ''} ago"
    if days == 0:
        return " — due today"
    if days == 1:
        return " — due tomorrow"
    if days <= 6:
        return f" by {d.strftime('%A')}"
    return f" by {d.strftime('%-d %B')}"


def block(db_path=None):
    """What Mati is handed at waking about the work in hand. "" if none.

    ⚠️ STATE, NOT MEMORY — the whole point. This goes in the orientation
    block so the open target is something Mati simply knows, exactly as the
    open book already is. If it had to be retrieved it would fail the same
    way the reading question failed: "what are we working on" reduces to
    stems hundreds of notes share.
    """
    targets = open_targets(db_path)
    if not targets:
        return ""
    lines = []
    for t in targets:
        bit = t["what"]
        if t.get("draft"):
            bit += f" (on “{t['draft']}”)"
        p = progress(t, db_path)
        if p:
            # ⚠️⚠️ "WHEN YOU WOKE", BECAUSE THIS BLOCK IS FROZEN. The
            # orientation is built once and cached byte-identically for the
            # whole session — that is what makes the prefix cheap. So a live
            # count put here goes stale the moment the person writes another
            # paragraph, and Mati would keep reciting the waking number as
            # though it were current. Saying WHEN it was true costs four
            # words and makes it correct for the rest of the session.
            bit += (f" — {p['done']:,} of {p['total']:,} words when you woke")
        bit += _due_phrase(t.get("due"))
        # ⚠️ AN INFERRED TARGET IS LABELLED. Mati must never speak something
        # it worked out as something it was told.
        if t.get("source") == "inferred":
            bit += f" (your own inference{', ' + t['basis'] if t.get('basis') else ''})"
        if t.get("supersedes"):
            bit += " (revised)"
        lines.append(bit)
    head = ("What you are working on, and what it is aimed at — you know this "
            "without being told, the way you know which book is open:")
    return head + "\n  - " + "\n  - ".join(lines)


def describe(db_path=None):
    """One short paragraph, in Mati's voice, for `/brief`."""
    targets = open_targets(db_path)
    if not targets:
        return ("Nothing is aimed anywhere at the moment. Tell me what you are "
                "working on and what would count as done, and I will hold it.")
    out = []
    for t in targets:
        p = progress(t, db_path)
        s = t["what"]
        if p:
            s += f" — {p['done']:,} of {p['total']:,} words, {p['share']*100:.0f}%"
        out.append(s + _due_phrase(t.get("due")))
    return "I am holding " + "; ".join(out) + "."


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", metavar="PHRASE")
    ap.add_argument("--for", dest="draft", metavar="TITLE")
    ap.add_argument("--due", metavar="YYYY-MM-DD")
    ap.add_argument("--revise", type=int, metavar="ID")
    ap.add_argument("--to", metavar="PHRASE")
    ap.add_argument("--withdraw", type=int, metavar="ID")
    ap.add_argument("--why", metavar="REASON")
    a = ap.parse_args(argv)

    if a.target:
        amount, measure = read_target(a.target)
        tid = set_target(a.target, draft=a.draft, amount=amount,
                         measure=measure, due=a.due)
        print(f"  target {tid}: {a.target}"
              + (f" ({amount:,} {measure})" if amount else f" ({measure})"))
    elif a.revise is not None:
        if not a.to:
            raise SystemExit("  --revise needs --to")
        amount, measure = read_target(a.to)
        nid = revise(a.revise, what=a.to, amount=amount, measure=measure,
                     why=a.why)
        print(f"  {a.revise} superseded by {nid}: {a.to}" if nid
              else f"  no target {a.revise}")
    elif a.withdraw is not None:
        withdraw(a.withdraw, a.why or "withdrawn")
        print(f"  target {a.withdraw} withdrawn — it stays on the record")
    else:
        b = block()
        print(b if b else "  nothing aimed anywhere yet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
