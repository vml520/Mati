"""
Condensing a long memory so it FITS, instead of cutting it off.

Rather than cut out possibly relevant information, a long verbatim is
summarised to fit the limit — accuracy and clarity first.

⚠️ TRIMMING THROWS AWAY THE END, AND THE END IS OFTEN THE POINT. A transcript
cut at 1200 characters loses whatever was concluded — which in a correction is
the correction, and in an answer is the answer. `…[trimmed]` is honest about
having done it and that is not the same as not doing it.

⚠️ COMPUTED OFF THE TURN, NEVER DURING ONE. Summarising costs an LLM call
(~8s), and the focused block is assembled mid-turn while the person waits. So
the digest is made when nothing is happening and STORED; the turn only ever
looks one up. Same shape as `rapport`, and for the same reason.

⚠️ A DIGEST IS NOT THE VERBATIM AND IS NEVER PRESENTED AS IT. It is Mati's own
condensation of its own record — accurate, but not quotable as exact words. It
carries its own label into the prompt so a paraphrase can never be spoken as a
quotation. This is the same distinction `learning.py` draws between what Mati
says and what it can be held to, one level down.

⚠️ MOST MEMORIES NEVER NEED ONE. An ordinary exchange is well under the budget
and is shown in full. This only touches the long tail — book passages and long
replies — so the queue is short and the cost is bounded.
"""

from __future__ import annotations

import sqlite3

from . import bridge

# Only bother when the verbatim genuinely will not fit the largest slot a
# focused memory can get. Below this it is shown in full and there is nothing
# to decide.
NEEDS_DIGEST_OVER = 1200

# How much room the digest itself may take. Slightly under the slot so the
# label and the provenance fit beside it.
DIGEST_CHARS = 1000

# How many to make in one idle pass. Bounded so digesting cannot starve
# reindexing, which recall actually depends on.
PER_PASS = 2

_PROMPT = (
    "You are condensing one of your own memories so it fits in a smaller "
    "space. Keep it ACCURATE above all: every claim, name, number and "
    "correction in the original must survive, and you must not add anything "
    "that is not there. Keep it CLEAR second: plain sentences, no preamble. "
    "Do not comment on the task. Write only the condensed memory.")


def ensure_column(db_path=None):
    conn = sqlite3.connect(db_path or bridge.NOTES_DB_PATH, timeout=30)
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(notes)")}
        if "digest" not in cols:
            conn.execute("ALTER TABLE notes ADD COLUMN digest TEXT")
            conn.commit()
        return True
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def pending(limit=PER_PASS, db_path=None):
    """Notes whose verbatim is too long to show and have no digest yet.

    Newest first: a memory made today is likelier to be asked about than one
    from a book chapter read in July.
    """
    ensure_column(db_path)
    conn = sqlite3.connect(
        f"file:{db_path or bridge.NOTES_DB_PATH}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT note_id, verbatim FROM notes "
            "WHERE verbatim IS NOT NULL AND digest IS NULL "
            "ORDER BY note_id DESC LIMIT ?", (int(limit) * 8,)).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    from . import vault
    out = []
    for note_id, raw in rows:
        try:
            text = vault.open_body(raw)
        except Exception:
            continue                      # locked; try again another time
        if text and len(text) > NEEDS_DIGEST_OVER:
            out.append((note_id, text))
        if len(out) >= limit:
            break
    return out


def make(note_id, text, llm=None, db_path=None, verbose=False):
    """Condense one memory and store it. Returns the digest, or None.

    ⚠️ Returns None rather than a fallback. A digest that could not be made is
    absent, and the focused block falls back to showing the head of the
    verbatim and SAYING it is cut. Storing a bad condensation would be worse
    than storing none, because nothing downstream could tell.
    """
    from . import vault
    try:
        if llm is None:
            from . import llm as L
            llm = L.make_llm(_PROMPT)
        said, _ = llm.reply([{"role": "user", "content": text[:6000]}],
                            max_tokens=320)
    except Exception as e:
        if verbose:
            print(f"  [digest] #{note_id}: {e}")
        return None
    said = " ".join((said or "").split())
    if len(said.split()) < 8 or len(said) >= len(text):
        return None                       # no shorter, or nothing usable
    said = said[:DIGEST_CHARS].rstrip()
    conn = sqlite3.connect(db_path or bridge.NOTES_DB_PATH, timeout=30)
    try:
        conn.execute("UPDATE notes SET digest = ? WHERE note_id = ?",
                     (vault.seal(said), int(note_id)))
        conn.commit()
    except sqlite3.Error:
        return None
    finally:
        conn.close()
    if verbose:
        print(f"  [digest] #{note_id}: {len(text)} -> {len(said)} chars")
    return said


def step(llm=None, db_path=None, verbose=False):
    """One idle pass. True if anything was condensed."""
    todo = pending(db_path=db_path)
    if not todo:
        return False
    done = 0
    for note_id, text in todo:
        if make(note_id, text, llm=llm, db_path=db_path, verbose=verbose):
            done += 1
    return done > 0


def status(db_path=None):
    ensure_column(db_path)
    conn = sqlite3.connect(
        f"file:{db_path or bridge.NOTES_DB_PATH}?mode=ro", uri=True)
    try:
        have = conn.execute(
            "SELECT COUNT(*) FROM notes WHERE digest IS NOT NULL").fetchone()[0]
        # ⚠️ LENGTH() MEASURES THE CIPHERTEXT, NOT THE MEMORY. A sealed value
        # is the prefix plus base64 plus a MAC — roughly 1.4x the plaintext —
        # so counting on it overstates the backlog several times over (13 long
        # verbatims where there are 2), and has someone hunting a stalled
        # queue that is not stalled.
        #
        # The sealed length is still a SOUND PRE-FILTER: ciphertext is never
        # shorter than plaintext, so anything under the threshold sealed is
        # certainly under it open. Only the candidates are decrypted, which
        # keeps this from becoming a full-store decryption — 101s on a large
        # store.
        rows = conn.execute(
            "SELECT verbatim FROM notes WHERE verbatim IS NOT NULL "
            "AND LENGTH(verbatim) > ?", (NEEDS_DIGEST_OVER,)).fetchall()
    except sqlite3.Error:
        return {}
    finally:
        conn.close()
    from . import vault
    long_ones = 0
    unreadable = 0
    for (raw,) in rows:
        try:
            text = vault.open_body(raw)
        except Exception:
            unreadable += 1              # locked: cannot say, so do not
            continue
        if text and len(text) > NEEDS_DIGEST_OVER:
            long_ones += 1
    out = {"digests": have, "long_verbatims": long_ones,
           "candidates_by_size": len(rows)}
    if unreadable:
        # ⚠️ Say so rather than undercount. A locked vault must not look like
        # an empty queue.
        out["unreadable"] = unreadable
    return out
