"""
Reading turns into knowledge, not into quotations.

⚠️ WHY A QUOTATION CANNOT BECOME KNOWLEDGE. A studied note that holds the
author's sentence:

    "Aztec Thought and Culture by Miguel León-Portilla"

sits beside an episode that holds Mati's own:

    "I was asked: … It brought back 4 earlier moments…"

You can QUOTE a passage or paraphrase it. You cannot SPEAK FROM it. Studied
material stored as quotation does not transfer into a reply the way memory
does — it arrives as a foreign object, so Mati holds it up and describes it.
The narration is not a prompt defect; it is what anyone does with a
quotation.

Stored that way, studied material is almost never used: of 2,670 studied
notes, ~67 hours of imprinting, **202 were ever recalled — 7.6%**. Everything
that retrieves is lexical search over the author's words, matched against a
question in the person's words, which is why paraphrase retrieval sits at
8–13 of 25.

So: read the passage, work out what is now known, and store THAT — in Mati's
own sentences, as the same kind of object as a memory. The source text is kept
beside it, unchanged, so a claim can always be checked against what was
actually written.

Three things fall out, and only the first is the goal:

  1. Studied material transfers naturally, because it is Mati's own language.
  2. Retrieval improves sideways: a question in the person's words matches a
     sentence in Mati's words better than either matches the author's prose.
  3. SELF-CORRECTION becomes possible. Two of Mati's own statements can
     contradict each other, and a statement can be checked against the
     verbatim source it came from. Two quotations cannot disagree; they just
     sit there.
"""

from __future__ import annotations

import sqlite3

from . import bridge

# Deliberately plain. The instruction is not to summarise — a summary is still
# about the text. It is to say what is now KNOWN, which is a different act and
# produces a different sentence.
_PROMPT = (
    "You have just read the passage below. Say what you now know from it, in "
    "one or two plain sentences of your own — as a person would if asked "
    "later, not as a summary of a document. Keep every name, number and claim "
    "exactly. Do not mention the passage, the text, or reading. No preamble.")

_LLM = None


def _client():
    global _LLM
    if _LLM is None:
        from . import llm as L
        _LLM = L.make_llm(_PROMPT)
    return _LLM


def ensure_column(db_path=None):
    """`verbatim` holds the source text a learned sentence came from.

    Separate from `body` on purpose: `body` is what Mati says and what search
    matches, `verbatim` is what it can be held to. Keeping both is the whole
    point — the memory is the gist, the document settles arguments.
    """
    conn = sqlite3.connect(db_path or bridge.NOTES_DB_PATH, timeout=10)
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(notes)")}
        if "verbatim" not in cols:
            conn.execute("ALTER TABLE notes ADD COLUMN verbatim TEXT")
            conn.commit()
            return True
        return False
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def learn_from(passage, verbose=False):
    """One passage in, Mati's own sentence out. None if it cannot.

    ⚠️ Returns None rather than a fallback. A passage that produces nothing
    usable should stay a quotation; inventing a sentence to stand for it would
    be exactly the failure this exists to prevent.
    """
    text = " ".join((passage or "").split())
    if len(text.split()) < 20:
        return None                     # too short to have learned anything
    try:
        said, _ = _client().reply([{"role": "user", "content": text[:1200]}],
                                  max_tokens=110)
    except Exception as e:
        if verbose:
            print(f"  [learning] {e}")
        return None
    said = " ".join((said or "").split())
    if not said or len(said.split()) < 5:
        return None
    # ⚠️ A learned sentence that merely copies the passage has learned nothing,
    # and would reintroduce the quotation problem under a new name.
    import re
    a = set(re.findall(r"[a-z]{4,}", text.lower()))
    b = set(re.findall(r"[a-z]{4,}", said.lower()))
    if a and len(a & b) / max(len(b), 1) > 0.92:
        return None
    return said


def store(lm, tok, learned, verbatim, source, db_path=None):
    """Write the learned sentence as the memory, keeping the source beside it."""
    res = bridge.remember(lm, tok, learned, source=source, db_path=db_path)
    if not res:
        return None
    note_id = res[0]
    conn = sqlite3.connect(db_path or bridge.NOTES_DB_PATH, timeout=10)
    try:
        from . import vault
        conn.execute("UPDATE notes SET verbatim=? WHERE note_id=?",
                     (vault.seal(verbatim), note_id))
        conn.commit()
    except sqlite3.Error:
        pass
    finally:
        conn.close()
    return note_id


def verbatim_of(note_id, db_path=None):
    """The source a learned sentence came from — for checking, not for speech."""
    conn = sqlite3.connect(f"file:{db_path or bridge.NOTES_DB_PATH}?mode=ro",
                           uri=True)
    try:
        row = conn.execute("SELECT verbatim FROM notes WHERE note_id=?",
                           (note_id,)).fetchone()
    except sqlite3.Error:
        return None
    finally:
        conn.close()
    if not row or not row[0]:
        return None
    from . import vault
    try:
        return vault.open_body(row[0])
    except Exception:
        return None


def pending(limit=None, db_path=None):
    """Studied notes still held as quotations rather than as knowledge."""
    conn = sqlite3.connect(f"file:{db_path or bridge.NOTES_DB_PATH}?mode=ro",
                           uri=True)
    try:
        # ⚠️ THE CHARTER IS EXCLUDED, and finding out why cost nine corrupted
        # notes on a twelve-note slice. It is ALREADY in Mati's voice — first
        # person, its own principles — so "say what you now know from this"
        # turns it into a third party describing someone else:
        #
        #   written : "The language I speak comes from a language model. The
        #              memory I hold comes from the field."
        #   learned : "I'm able to generate human-like responses based on
        #              patterns learned from a large dataset..."
        #
        # The second contradicts the first. Converting text that is already a
        # first-person statement does not produce knowledge; it produces a
        # paraphrase of an identity by something that does not hold it.
        #
        # `studied:asked about...` is excluded for the same reason: those are
        # Mati's OWN answers to its own questions, already its own sentences.
        sql = ("SELECT note_id, body FROM notes "
               "WHERE source LIKE 'studied:%' "
               "AND source NOT LIKE 'studied:charter:%' "
               "AND source NOT LIKE 'studied:asked about%' "
               "AND verbatim IS NULL "
               "ORDER BY note_id")
        if limit:
            sql += f" LIMIT {int(limit)}"
        return conn.execute(sql).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()
