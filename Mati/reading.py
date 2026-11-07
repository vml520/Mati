"""
A memory of having read something.

⚠️ THE GAP THIS FILLS. Mati absorbed 844 passages of Aztec Thought and Culture
and held **no memory of the reading**. `provenance_of('studied:…')` returns
`studied` — the content is stored and labelled, so it can say "I read that"
when a passage is handed to it. But nothing recorded the ACT. Asked what it had
been reading, it had nothing to answer from, because studying produced text and
no experience.

That is not how anyone studies. You read to put something in reach, and you
also remember having read it — which book, roughly how far in, and the parts
that stayed with you.

So: one memory per document, written as a LIVED memory rather than a studied
one, because the reading is something Mati did rather than something it was
told. It names the passages it has actually returned to — `recall_hits`, not a
guess — because those are what "stayed with me" means for a thing whose
returning is recorded.

Written at each quarter of a document rather than only at the end: a 1,759
chunk book is weeks of reading, and Mati should be able to speak about it
while it is still going.
"""

from __future__ import annotations

import sqlite3

from . import bridge, vault

# ⚠️ ONE MEMORY PER DOCUMENT. It was four — a quarter, a half, three quarters
# and the end — described to the user as "one per document", and it was not.
# Eleven documents produced FORTY-FOUR near-identical memories: "I have been
# reading asked about genetics. I have taken in 16 passages of 18." four times
# over, for documents of eighteen chunks that never needed quarter-marks.
#
# They are all archived and all compete in the same lexical search someone uses
# to ask what Mati has been reading, so the book became one voice in a crowd
# I had made. In a store whose episodes already sit at 98.4% mutual similarity,
# manufacturing forty-four variations on one sentence is the worst thing to add.
#
# Written once, when the document is FINISHED. Progress while it is still going
# is visible in the terminal's TRAINING panel and in the waking block, which is
# where in-flight state belongs; a memory is for what happened.
MILESTONES = (1.0,)
KEEP_PASSAGES = 3


def _prefix():
    """The source prefix study writes under — owned by `study`, not guessed."""
    from .study import STUDY_PREFIX
    return STUDY_PREFIX


def _conn(db_path=None):
    return sqlite3.connect(db_path or bridge.NOTES_DB_PATH)


def progress(db_path=None):
    """Per document: how much is absorbed, out of how much."""
    conn = _conn(db_path)
    try:
        rows = conn.execute(
            "SELECT origin, COUNT(*), "
            "SUM(status IN ('learned','not_stuck','refused','too_short')) "
            "FROM study_queue GROUP BY origin").fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    return [{"origin": o, "total": n, "done": d or 0,
             "share": (d or 0) / n if n else 0.0} for o, n, d in rows]


def stayed_with_me(origin, k=KEEP_PASSAGES, db_path=None):
    """The passages of this document Mati has actually returned to.

    Measured, not chosen: `recall_hits` records every time a note was needed.
    A passage returned to eleven times is a different thing from one absorbed
    and never wanted again, and only one of them belongs in a memory of
    reading.
    """
    conn = _conn(db_path)
    try:
        rows = conn.execute(
            "SELECT n.body, COUNT(*) c FROM recall_hits h "
            "JOIN notes n ON n.note_id = h.note_id "
            "WHERE n.source = ? GROUP BY n.note_id "
            "ORDER BY c DESC LIMIT ?",
            (f"{_prefix()}{origin}", k)).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    out = []
    for body, _c in rows:
        try:
            t = " ".join((vault.open_body(body) or "").split())
        except Exception:
            continue
        if t:
            out.append(t[:180])
    return out


def already_recorded(origin, share, db_path=None):
    """Has this document been written already? Idempotent by design.

    ⚠️ Matches on the ORIGIN, not on origin-plus-milestone. The old marker
    included the percentage, so each quarter looked like a different memory and
    four were written per document.

    ⚠️ AND THE ORIGIN MAY CONTAIN COLONS — "charter:MATI_MISSION_AND_
    PRINCIPLES.txt" does. A parse that splits on the first colon reads the
    document as "charter" and silently fails to match, which is exactly how the
    charter kept four copies through a cleanup that removed thirty. Substring
    match on the whole origin, so the separator never has to be found.
    """
    conn = _conn(db_path)
    try:
        marker = f"[reading:{origin}:"
        rows = conn.execute(
            "SELECT body FROM notes WHERE source = 'episode'").fetchall()
    except sqlite3.Error:
        return True                       # do not risk a duplicate
    finally:
        conn.close()
    for (b,) in rows:
        try:
            if marker in (vault.open_body(b) or ""):
                return True
        except Exception:
            continue
    return False


def compose(origin, done, total, milestone, passages):
    """What the memory says, in Mati's own voice.

    Plain language, because this is a lived memory and Mati is the one who
    lived it. The counts are here rather than in a separate diagnostic record:
    knowing you are two thirds through a book is part of having read it, not
    machinery.
    """
    name = origin.split("/")[-1]
    # ⚠️ THE SHARE IN THE SENTENCE IS THE REAL ONE, not the milestone that
    # triggered the writing. Using the milestone produced "I have taken in
    # 1464 passages of 1759. I am about 50% of the way through." — arithmetic
    # a person can check and find wrong, in a memory that is meant to be more
    # certain than a guess. The milestone decides WHEN; the counts decide what
    # it says.
    share = (done / total) if total else 0.0
    where = ("I finished it." if share >= 0.999
             else f"I am about {int(share*100)}% of the way through.")
    lines = [f"I have been reading {name}. I have taken in {done} passages "
             f"of {total}. {where}"]
    if passages:
        lines.append("The parts I keep coming back to:")
        for p in passages:
            lines.append(f"  — {p}")
    # A marker so the same milestone is never written twice. It is stripped
    # from anything shown; see `voice`.
    lines.append(f"[reading:{origin}:{int(milestone*100)}]")
    return "\n".join(lines)


def record_due(lm, tok, db_path=None, verbose=False):
    """Write any reading memory that is now due. Returns what it wrote."""
    written = []
    for doc in progress(db_path):
        origin, share = doc["origin"], doc["share"]
        for m in MILESTONES:
            if share + 1e-9 < m:
                continue
            if already_recorded(origin, m, db_path):
                continue
            body = compose(origin, doc["done"], doc["total"], m,
                           stayed_with_me(origin, db_path=db_path))
            try:
                # EPISODE_SOURCE, deliberately: this is something Mati did,
                # not something it was handed. `provenance_of` will call it
                # "lived", which is the whole point.
                bridge.remember(lm, tok, body, db_path=db_path)
                written.append((origin, m))
                if verbose:
                    print(f"  [reading] recorded {origin} at {int(m*100)}%")
            except Exception as e:
                if verbose:
                    print(f"  [reading] could not record {origin}: {e}")
            break                       # one milestone per pass, oldest first
    return written
