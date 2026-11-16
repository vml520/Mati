"""
Asking instead of answering, when a surfaced memory is one Mati got wrong.

There is no question generator and no queue. The occasion arises in the turn:

    a memory surfaces
      -> it carries a correction, so Mati has been wrong here before
        -> it needs the fact before it can use it
          -> so it asks, instead of assuming again

Self-limiting by design: below `CLARIFY_AFTER_CORRECTIONS` corrections in the
store, no clarifying question is ever asked. The behaviour is earned by a
track record of being wrong, not switched on by default.

The field does not author anything. Its only channel to the reply is four
floats; what it contributes is WHICH memory surfaced. Whether that memory was
corrected is a database lookup, and the sentence is composed in code. This is
the field occasioning a question, not asking one.

Corrections tend to be about inventing rather than asking ("You made that
up"), which is exactly the failure this removes: the second time, Mati asks.

⚠️ KNOWN LIMIT. A correction can be a durable fact ("the correct number is
69") or a passing preference ("not right now, I'd like a video on physics").
Checking the second ("is that still right?") reads as pedantry. Telling them
apart needs meaning, which word lists do not capture reliably. If it becomes a
nuisance, the fix is to mark a correction as factual or situational when it is
made, from the person's own turn, rather than inferring it afterwards.
"""

from __future__ import annotations

import re
import sqlite3

from . import config as C

# Only a question gets clarified. A statement does not need a fact before it
# can be answered: "I'm tired" met with "before I answer, you corrected me
# once" is machinery, not care.
_ASKING = re.compile(r"\?|^\s*(?:what|who|when|where|which|how|why|did|do|does|"
                     r"is|are|was|were|can|could|would|should|tell me)\b", re.I)


def total_corrections(db_path=None):
    """How many times Mati has been corrected. The track record."""
    from . import bridge
    db_path = bridge.NOTES_DB_PATH if db_path is None else db_path
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return 0
    try:
        return conn.execute(
            "SELECT count(*) FROM notes WHERE corrects IS NOT NULL"
        ).fetchone()[0] or 0
    except sqlite3.Error:
        return 0
    finally:
        conn.close()


def corrections_on(note_ids, db_path=None):
    """{corrected note_id: the correction's text}, for the notes given.

    Queried, not read off the surfaced rows. Only the lexical path sets a
    `corrected` flag (`notes.py`); the field path annotates the body text
    instead, so trusting the flag would miss every field-surfaced memory.
    """
    from . import bridge, vault
    ids = [int(i) for i in note_ids if i is not None]
    if not ids:
        return {}
    db_path = bridge.NOTES_DB_PATH if db_path is None else db_path
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return {}
    try:
        marks = ",".join("?" * len(ids))
        rows = conn.execute(
            f"SELECT corrects, body FROM notes WHERE corrects IN ({marks}) "
            "ORDER BY note_id DESC", ids).fetchall()
    except sqlite3.Error:
        return {}
    finally:
        conn.close()
    out = {}
    for target, body in rows:
        if target in out:
            continue                # newest correction wins
        try:
            out[int(target)] = vault.open_body(body)
        except Exception:
            continue                # a sealed correction is not a reason to ask
    return out


def _said(correction):
    """Just the person's own words out of a stored correction.

    ⚠️ LINE-BASED. A correction is stored as three lines:

        I was corrected: <what the person said>
        What I had said was: <Mati's wrong reply>
        I was wrong, and I know it because I was told so, ...

    Only the first line is the person's. Flattening whitespace before
    splitting would merge in Mati's own wrong reply and quote it back as
    though the person had said it.
    """
    first = (correction or "").split("\n")[0]
    text = re.sub(r"^\s*I was corrected:\s*", "", first, flags=re.I)
    # Lead with the substance rather than the refusal: "No." is the reaction,
    # what follows is the information.
    #
    # Only a complete refusal sentence is removed. A loose prefix list would
    # match "Not right" inside "Not right now. I'd like..." and mangle the
    # person's words.
    text = re.sub(r"^(?:no|nope|wrong|incorrect|that is not correct|"
                  r"that'?s not correct)[.!,;:]+\s*", "", text, flags=re.I)
    return " ".join(text.split()).strip()


def needed(user_text, surfaced_ids, db_path=None):
    """The question to ask instead of answering, or None.

    `surfaced_ids` are the note_ids the field and the lexical index brought up
    THIS TURN — the occasion. Returns a sentence composed here, never prompted.
    """
    if not C.CLARIFY_ON_CORRECTED:
        return None
    if not _ASKING.search(user_text or ""):
        return None
    # The track-record gate. Below the threshold Mati has not been wrong
    # often enough to have learned to check, and clarifying would be a
    # mannerism rather than caution.
    if total_corrections(db_path) < C.CLARIFY_AFTER_CORRECTIONS:
        return None
    fixes = corrections_on(surfaced_ids, db_path)
    if not fixes:
        return None
    # The most recently corrected of what surfaced.
    target = max(fixes)
    said = _said(fixes[target])
    if len(said) < 8:
        return None                 # nothing informative to check against
    if len(said) > 180:
        said = said[:177].rsplit(" ", 1)[0] + "…"
    # Quoting the person, not a memory. `bridge.state_block` forbids Mati
    # reciting its recalled memories as substance; this is the person's own
    # correction read back so they can confirm it, which is what makes the
    # question checkable rather than vague.
    return (f"Before I answer — I've had this wrong before, and what you told "
            f"me was: “{said}”  Is that still right?")
