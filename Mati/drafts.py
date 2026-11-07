"""
A piece of writing Mati is helping with, held between turns.

    python3 -m Mati.drafts                 what is open
    python3 -m Mati.drafts --open "title"  start or resume one
    python3 -m Mati.drafts --close

⚠️ WHY THIS EXISTS. The user: *"Mati should act more like an
editor than a ghostwriter, but at the same time hold the document in memory
until completed. Right now I have to remind Mati of the project before they
can continue helping me."*

That last sentence is the whole defect, and it is not about memory in the
field sense: every turn simply started cold about the draft. There was
nowhere for "the thing we are working on" to live.

⚠️ MODELLED ON THE ACCOUNTANT, DELIBERATELY. `budgets` is the same shape — a
persistent, user-owned record that outlives a turn — and its integration
carries a lesson written in its own comments: the write path must be ACTED ON
BEFORE the read block is built, or a request is recognised, answered, and
stores nothing. `budgets: []` after weeks of being told about budgets.

⚠️ THE ONE THING THE ACCOUNTANT DOES NOT HAVE IS A POINTER. Budgets are all
equal; there is no "the one in hand". A draft needs exactly that, because
"I have to remind Mati of the project" IS the absence of it.

⚠️⚠️ THE TEXT DOES NOT GO IN THE PROMPT, AND THAT IS NOT A COMPROMISE.
Measured: a 4,096-token window, ~696 for the system prompt, 160 for the
reply, leaving ~3,240 for memories, charter, day log and world. A
1,500-word draft is ~2,025 tokens and would crowd out the things that make
Mati itself; a 3,000-word draft does not fit at all.

⭐ An EDITOR works on a passage with the shape of the whole in mind. A
GHOSTWRITER needs the entire text because it is generating the next
sentence. What the person asked for and what the hardware allows are the
same thing.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sqlite3
import sys

TABLE = "drafts"
OPEN, DONE = "open", "done"


def _db(db_path=None):
    from . import bridge
    return bridge.NOTES_DB_PATH if db_path is None else db_path


def _conn(db_path=None):
    conn = sqlite3.connect(_db(db_path))
    conn.execute(f"""CREATE TABLE IF NOT EXISTS {TABLE} (
            draft_id  INTEGER PRIMARY KEY,
            title     TEXT NOT NULL,
            body      TEXT NOT NULL,
            status    TEXT NOT NULL,
            created   TEXT,
            touched   TEXT
        )""")
    conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS {TABLE}_title "
                 f"ON {TABLE}(title)")
    return conn


def _now():
    return dt.datetime.now().isoformat(timespec="seconds")


def open_draft(title, body="", db_path=None):
    """Start one, or resume it if the title is already known.

    ⚠️ ONE OPEN AT A TIME, and opening a second closes the first. The pointer
    is the whole feature; two pointers is no pointer. Resuming does not touch
    the body — a draft is reopened to continue, not to be blanked.
    """
    from . import vault
    title = " ".join((title or "").split())[:200]
    if not title:
        return None
    conn = _conn(db_path)
    try:
        conn.execute(f"UPDATE {TABLE} SET status=? WHERE status=?",
                     (DONE, OPEN))
        row = conn.execute(f"SELECT draft_id FROM {TABLE} WHERE title=?",
                           (title,)).fetchone()
        if row:
            conn.execute(f"UPDATE {TABLE} SET status=?, touched=? "
                         f"WHERE draft_id=?", (OPEN, _now(), row[0]))
        else:
            conn.execute(
                f"INSERT INTO {TABLE} (title, body, status, created, touched) "
                "VALUES (?,?,?,?,?)",
                # ⚠️ SEALED. A draft is the person's own writing — at least as
                # private as an episode, and `shelved`/`personal_only` set the
                # precedent that what is theirs stays sealed at rest.
                (title, vault.seal(body or ""), OPEN, _now(), _now()))
        conn.commit()
    finally:
        conn.close()
    return title


def current(db_path=None):
    """The draft in hand, or None. ⚠️ This is the pointer."""
    from . import vault
    try:
        conn = sqlite3.connect(f"file:{_db(db_path)}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    try:
        row = conn.execute(
            f"SELECT title, body, created, touched FROM {TABLE} "
            f"WHERE status=? ORDER BY touched DESC LIMIT 1", (OPEN,)).fetchone()
    except sqlite3.Error:
        return None
    finally:
        conn.close()
    if not row:
        return None
    title, sealed, created, touched = row
    try:
        body = vault.open_body(sealed)
    except Exception:
        return {"title": title, "body": None, "locked": True,
                "created": created, "touched": touched}
    return {"title": title, "body": body, "locked": False,
            "created": created, "touched": touched,
            "words": len(body.split())}


def revise(body, db_path=None):
    """Replace the open draft's text. Returns the word count, or None.

    ⚠️ MATI DOES NOT CALL THIS ON ITS OWN. The person's writing is theirs;
    an editor marks a passage, it does not rewrite the manuscript while the
    author is out. Every caller is an explicit instruction.
    """
    from . import vault
    cur = current(db_path)
    if not cur:
        return None
    conn = _conn(db_path)
    try:
        conn.execute(f"UPDATE {TABLE} SET body=?, touched=? "
                     f"WHERE title=? AND status=?",
                     (vault.seal(body or ""), _now(), cur["title"], OPEN))
        conn.commit()
    finally:
        conn.close()
    return len((body or "").split())


def close(db_path=None):
    """Finish the open draft. Returns its title, or None."""
    cur = current(db_path)
    if not cur:
        return None
    conn = _conn(db_path)
    try:
        conn.execute(f"UPDATE {TABLE} SET status=?, touched=? WHERE title=?",
                     (DONE, _now(), cur["title"]))
        conn.commit()
    finally:
        conn.close()
    return cur["title"]


N_TABLE = "draft_notes"


def _notes_conn(db_path=None):
    conn = _conn(db_path)
    conn.execute(f"""CREATE TABLE IF NOT EXISTS {N_TABLE} (
            note_id   INTEGER PRIMARY KEY,
            title     TEXT NOT NULL,
            anchor    TEXT NOT NULL,
            note      TEXT NOT NULL,
            created   TEXT
        )""")
    return conn


def annotate(anchor_text, note, db_path=None):
    """Attach a note to a passage. Returns the anchor kept, or None.

    ⚠️ ANCHORED BY ITS WORDS, NOT BY AN OFFSET. A character position is
    invalidated by every edit above it, silently — the note would still
    resolve and point at the wrong sentence, which is worse than losing it.
    A quoted fragment can be looked for, and when it is gone it is VISIBLY
    gone.

    ⚠️ AND THAT IS THE LESSON FROM THIS PROJECT'S OWN NOTES. The audit found
    a `MEMORY.md` line that had been wrong for six weeks and sent a diagnosis
    the wrong way — an annotation whose subject had moved and which went on
    asserting. A stale anchor must SAY it is stale.
    """
    from . import vault
    cur = current(db_path)
    if not cur or not (anchor_text or "").strip() or not (note or "").strip():
        return None
    # ⚠️ A SHORT ANCHOR MATCHES EVERYWHERE. Enough words to be unique in a
    # draft, few enough to survive the sentence being reworded around it.
    keep = " ".join(anchor_text.split())[:120]
    conn = _notes_conn(db_path)
    try:
        conn.execute(
            f"INSERT INTO {N_TABLE} (title, anchor, note, created) "
            "VALUES (?,?,?,?)",
            (cur["title"], vault.seal(keep), vault.seal(note.strip()), _now()))
        conn.commit()
    finally:
        conn.close()
    return keep


def annotations(db_path=None, body=None):
    """Notes on the open draft, each marked attached or orphaned.

    ⚠️ ORPHANS ARE REPORTED, NOT DROPPED. A note whose passage was rewritten
    is still something Mati thought about this piece — and silently binning
    it would lose the observation while looking tidy. It is shown as adrift so
    the person can re-place it or let it go.
    """
    from . import vault
    cur = current(db_path)
    if not cur:
        return []
    text = body if body is not None else (cur.get("body") or "")
    try:
        conn = _notes_conn(db_path)
        rows = conn.execute(
            f"SELECT anchor, note, created FROM {N_TABLE} WHERE title=? "
            "ORDER BY note_id", (cur["title"],)).fetchall()
        conn.close()
    except sqlite3.Error:
        return []
    out = []
    for a, n, c in rows:
        try:
            anchor, note = vault.open_body(a), vault.open_body(n)
        except Exception:
            continue
        out.append({"anchor": anchor, "note": note, "created": c,
                    "attached": anchor in text})
    return out


# ⚠️ A CEILING IN WORDS, because the whole design rests on the draft NOT
# filling the window. ~3,240 tokens are left after the system prompt and the
# reply ceiling, and that space belongs to memories, the charter, the day log
# and the world. 260 words is ~350 tokens — a passage, not a manuscript.
TAIL_WORDS = 260
HEADING_MAX = 8


def shape(body):
    """Headings and length — the map, not the territory."""
    import re
    lines = [l.strip() for l in (body or "").splitlines() if l.strip()]
    heads = [l.lstrip("#").strip() for l in lines
             if l.startswith("#") or (len(l) < 60 and l.endswith(":"))]
    return heads[:HEADING_MAX]


def block(db_path=None, full=False):
    """What Mati is handed about the open draft. "" if none.

    ⚠️ TWO SIZES, AND THE SMALL ONE IS THE POINT. A draft being open is worth
    one line on EVERY turn — that line is the entire answer to "I have to
    remind Mati of the project". The passage is only worth its tokens when
    the turn is actually about the writing.

    ⚠️ THE TAIL, NOT THE HEAD. Without a cursor the best guess at "where we
    are" is where the writing stops. A draft is continued at its end far more
    often than at its beginning.

    ⚠️ AND IT SAYS EDITOR, IN THE BLOCK, NOT ONLY IN THE SYSTEM PROMPT. The
    standing finding here is that an instruction far from the material does
    not bind — the role has to arrive attached to the thing it governs.
    """
    cur = current(db_path)
    if not cur:
        return ""
    if cur.get("locked"):
        return ("<draft>\nA draft called “%s” is open, but the vault is "
                "locked so I cannot read it.\n</draft>" % cur["title"])
    words = cur.get("words") or 0
    if not full:
        # ⚠️ ONE LINE, EVERY TURN. Cheap enough to always be true.
        return ("<draft>\nWe have a piece of writing open: “%s”, %d words so "
                "far. I have not been shown the text this turn.\n</draft>"
                % (cur["title"], words))
    heads = shape(cur["body"])
    tail = " ".join((cur["body"] or "").split()[-TAIL_WORDS:])
    out = ["<draft>",
           "We are working on “%s” — %d words." % (cur["title"], words)]
    if heads:
        out.append("Its sections: " + "; ".join(heads))
    out.append("")
    out.append("The end of it as it stands, which is where we are:")
    out.append(tail)
    # ⚠️ ONLY THE NOTES ABOUT WHAT IS ON SCREEN, plus a count of the rest.
    # Every annotation in the prompt would cost more than the passage does,
    # and a note about a paragraph Mati cannot see is not actionable.
    notes = annotations(db_path, body=cur["body"])
    here = [n for n in notes if n["anchor"] in tail]
    adrift = [n for n in notes if not n["attached"]]
    elsewhere = len(notes) - len(here) - len(adrift)
    if here:
        out.append("")
        out.append("What I have noted about this part:")
        for n in here:
            out.append(f"  on “{n['anchor'][:52]}” — {n['note']}")
    if adrift:
        # ⚠️ SAID, NOT HIDDEN. The passage moved out from under these; the
        # observation still stands even though its anchor does not.
        out.append("")
        out.append("Notes whose passage has since been rewritten, so I cannot "
                   "place them any more:")
        for n in adrift[:3]:
            out.append(f"  (adrift) {n['note']}")
    if elsewhere:
        out.append("")
        out.append(f"I have {elsewhere} more note(s) on parts of this I am "
                   f"not looking at right now.")
    out.append("")
    out.append("⚠️ I am the EDITOR here, not the writer. The words are theirs. "
               "I ask about what is unclear, name what is not working and why, "
               "and offer an alternative only when asked for one. I do not "
               "hand back a rewritten passage as though it were theirs, and I "
               "do not claim to have read more of it than I was shown.")
    out.append("</draft>")
    return "\n".join(out)


def asks_about_draft(text_in):
    """Is this turn about the writing? A literal trigger, not a judgement."""
    import re
    return bool(re.search(
        r"\b(draft|essay|chapter|paragraph|wrote|writing|chapter|"
        r"this piece|the piece|chapter|edit|rewrite|reword|phrasing|"
        r"sentence|wording|prose)\b", text_in or "", re.I))


_OPEN_RE = re.compile(
    r"\b(?:let'?s |lets |please )?(?:start|begin|open|work on|working on|"
    r"pick up|continue|resume)\b[^.?!]{0,12}?\b(?:a |an |the |my |our )?"
    r"(?:draft|essay|piece|chapter|article|letter|story|paper)\b"
    r"(?:\s+(?:called|titled|named|on|about)\s+(?P<title>[^.?!\n]{2,80}))?",
    re.I)
_CLOSE_RE = re.compile(
    r"\b(?:that'?s|that is|we'?re|we are|i'?m|i am)\s+(?:it|done|finished)\b"
    r"|\b(?:close|finish|put away|set aside)\s+(?:the\s+)?"
    r"(?:draft|essay|piece|chapter|article)\b", re.I)


_NOTE_RE = re.compile(
    r"\b(?:make|leave|add|jot)\s+(?:a\s+)?note\b[^\w]*(?:that|about|saying|:)?\s*"
    r"(?P<note>.+)$"
    r"|^note(?:\s+that|\s*:)\s*(?P<note2>.+)$"
    r"|\bremember\s+(?:that\s+)?(?P<note3>[^.?!]{6,160})\s+about\s+(?:this|that)\b",
    re.I)
# A fragment the person quoted — the anchor they meant, said in their words.
_QUOTED = re.compile(r"[\"“]([^\"”]{6,120})[\"”]")


def note_from(text_in, db_path=None):
    """Make a note because the person asked for one. A sentence, or None.

    ⚠️ THE ANCHOR IS THE HARD PART AND IT IS GUESSED, SO IT IS STATED. "Make
    a note that this opener buries the claim" does not say WHICH passage. If
    they quoted one, that is the anchor and there is nothing to guess. If they
    did not, it anchors to where the writing stops — the same assumption the
    block already makes about "where we are" — and the confirmation SAYS what
    it attached to, so a wrong guess is visible and correctable rather than
    silently wrong.

    ⚠️ This is the shape `act` already follows: a literal trigger, an explicit
    confirmation, and no judgement about whether the observation is any good.
    """
    cur = current(db_path)
    if not cur or cur.get("locked"):
        return None
    m = _NOTE_RE.search(" ".join((text_in or "").split()))
    if not m:
        return None
    note = (m.group("note") or m.group("note2") or m.group("note3") or "").strip()
    note = note.strip(" .\"'“”")
    if len(note) < 4:
        return None
    q = _QUOTED.search(text_in or "")
    if q:
        anchor = q.group(1)
        where = f"“{anchor[:48]}”"
        # ⚠️ THE QUOTE IS THE ANCHOR, NOT THE NOTE. Left in, every quoted note
        # reads back as the passage followed by the remark, and the remark —
        # the only part worth keeping — gets truncated out of the 120 chars.
        note = note.replace(anchor, "")
        note = note.strip(" .,;:—-\"'“”") or "—"
    else:
        # ⚠️ The last sentence of the draft, not the last 260 words — an
        # anchor has to be short enough to survive the rest being reworded.
        tail_words = (cur.get("body") or "").split()
        anchor = " ".join(tail_words[-14:]) if tail_words else ""
        where = "where the writing stops" if anchor else "the piece as a whole"
    if not anchor:
        return None
    annotate(anchor, note, db_path)
    return f"Noted, against {where}: {note}"


def act(text_in, db_path=None):
    """Open or close a draft because the person said so. A sentence, or None.

    ⚠️ ACTED ON BEFORE THE BLOCK IS BUILT, which is the accountant's lesson
    written in its own comments: a read block alone means a request is
    recognised, answered, and stores NOTHING. `budgets: []` after weeks of
    being told about budgets. The same shape here would be Mati agreeing to
    work on an essay and forgetting it by the next turn — the exact complaint
    this module exists for.

    ⚠️ A LITERAL TRIGGER, NOT A JUDGEMENT. Opening a draft is a small,
    reversible act, but a false positive would hijack a conversation into
    being about writing. The patterns require a writing NOUN, not just a verb.

    ⚠️ AND IT NEVER WRITES THE PERSON'S PROSE. `act` only opens and closes.
    `revise` exists but nothing calls it automatically — an editor marks a
    passage, it does not rewrite the manuscript while the author is out.
    """
    t = " ".join((text_in or "").split())
    if not t:
        return None
    # ⚠️ NOTES FIRST. "make a note that we should open with the claim"
    # contains "open" and would otherwise be read as a request to open a
    # draft — the note trigger is the more specific reading and wins.
    said = note_from(text_in, db_path)
    if said:
        return said
    if _CLOSE_RE.search(t) and current(db_path):
        title = close(db_path)
        return f"Closed “{title}”." if title else None
    m = _OPEN_RE.search(t)
    if not m:
        return None
    title = (m.group("title") or "").strip(" \"'“”")
    cur = current(db_path)
    if not title:
        # ⚠️ "let's work on the essay" with one already open means CONTINUE,
        # not start a nameless second one.
        return None if cur else "Which piece should I open?"
    if cur and cur["title"].lower() == title.lower():
        return None                      # already in hand; say nothing
    open_draft(title, db_path=db_path)
    return f"Open: “{title}”. I will keep it until you say we are done."


def all_drafts(db_path=None, limit=20):
    try:
        conn = sqlite3.connect(f"file:{_db(db_path)}?mode=ro", uri=True)
    except sqlite3.Error:
        return []
    try:
        return [{"title": t, "status": s, "touched": u}
                for t, s, u in conn.execute(
                    f"SELECT title, status, touched FROM {TABLE} "
                    f"ORDER BY touched DESC LIMIT {int(limit)}")]
    except sqlite3.Error:
        return []
    finally:
        conn.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--open", metavar="TITLE")
    ap.add_argument("--close", action="store_true")
    a = ap.parse_args(argv)
    from . import vault
    if not vault.unlocked():
        vault.unlock(reason="read your drafts")
    if a.open:
        print(f"  open: {open_draft(a.open)}")
    elif a.close:
        print(f"  closed: {close() or '(nothing open)'}")
    cur = current()
    if cur:
        print(f"  in hand: {cur['title']}"
              f"  ({cur.get('words', '?')} words, touched {cur['touched']})")
    else:
        print("  nothing open")
    for d in all_drafts(limit=6):
        print(f"    {d['status']:<5} {d['title'][:48]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
