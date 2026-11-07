"""
Asking instead of answering, when the field surfaces something Mati got wrong.

⚠️ THIS IS THE REPLACEMENT FOR THE QUESTION GENERATOR, AND IT IS A DIFFERENT
KIND OF THING. The user:

    "The field generates a memory that causes the agent to require information
     for the response. So, instead of answering it clarifies with a question
     first. This should happen after many memories of assuming incorrectly."

The old design manufactured questions offline from passages and parked them
until their topic came up — forced by construction, because a question written
days earlier has no relation to the moment it lands in. 863 of them were
purged.

This one has no generator and no queue. The occasion arises IN the turn:

    the field surfaces a memory
      -> that memory carries a correction, so Mati has been wrong here before
        -> it needs the fact before it can use it
          -> so it asks, instead of assuming again

⚠️ "AFTER MANY MEMORIES OF ASSUMING INCORRECTLY" IS THE DIAL, AND IT IS WHAT
MAKES THIS SELF-LIMITING. A store with no corrections produces no clarifying
questions at all — the behaviour is earned by a track record of being wrong,
not switched on by default. That is the opposite of the generator, which
produced questions at a fixed rate whether or not there was anything to ask.

⚠️ AND THE FIELD IS NOT AUTHORING ANYTHING. Measured: the only channel from
field to reply is four floats. What the
field contributes is WHICH memory surfaced; the fact that it was corrected is
a database lookup, and the sentence is composed in code. Do not describe this
as the field asking a question — it is the field occasioning one.

WHAT THE CORRECTIONS ACTUALLY SAY, measured  on 11 of 4,372 notes:

    "No. You made that up. That goes against your principles."
    "Not only is it wrong, but your answer appears to be completely invented"
    "That is not correct. the correct number to remember for me is between 68"

They cluster on INVENTING RATHER THAN ASKING, which is exactly the failure
this removes: the second time, Mati asks.

⚠️ KNOWN ROUGH EDGE, RECORDED RATHER THAN PATCHED. A correction can be a
durable fact ("the correct number is 69", "you are confusing your own
questions with mine") or a passing preference ("Not right now, I'd like an
educational video on physics"). Clarifying against the second reads oddly —
"is that still right?" about what someone wanted twenty minutes ago is not
caution, it is pedantry.

Distinguishing them needs meaning, and the last three attempts in this project
to separate meaning with a word list all leaked (`inquiry._DANGLING`,
`voice._denies_memory`, `askable`'s grounding). So it is left visible instead:
if this becomes a nuisance the fix is to mark corrections as factual or
situational AT THE POINT THEY ARE MADE, where the person's own turn says
which, rather than inferred afterwards from the text.
"""

from __future__ import annotations

import re
import sqlite3

from . import config as C

# Only a question gets clarified. ⚠️ A statement does not need a fact before
# it can be answered — "I'm tired" met with "before I answer, you corrected me
# once" is not care, it is machinery.
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

    ⚠️ QUERIED, NOT READ OFF THE SURFACED ROWS. Only the LEXICAL path sets a
    `corrected` flag (`notes.py`); the field path annotates the body text
    instead and returns no flag, so a consumer that trusted the flag would
    never clarify on a field-surfaced memory — which is the whole case this
    exists for.
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

    ⚠️ LINE-BASED, AND FLATTENING IT FIRST WAS A REAL BUG. A correction is
    stored as three lines:

        I was corrected: <what the person said>
        What I had said was: <Mati's wrong reply>
        I was wrong, and I know it because I was told so, ...

    The first cut here did `" ".join(text.split())` before anything else,
    which destroyed the line boundaries — so the quote handed back to the
    person contained Mati's own wrong answer and its self-reproach, reading
    "...educational video on physics What I had said was: Let me try again,
    the user...". Quoting Mati's mistake back as though it were the person's
    correction is worse than not asking.
    """
    first = (correction or "").split("\n")[0]
    text = re.sub(r"^\s*I was corrected:\s*", "", first, flags=re.I)
    # Lead with the substance rather than the refusal: "No." is the reaction,
    # what follows is the information.
    #
    # ⚠️ ONLY A COMPLETE REFUSAL SENTENCE. Written as a loose prefix list this
    # matched "not right" inside "Not right now. I'd like to see an
    # educational video" and left the quote starting at "now." — mangling the
    # person's words to tidy them.
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
    # ⚠️ THE TRACK RECORD GATE. Below the threshold Mati has not been wrong
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
    # ⚠️ QUOTING THE PERSON, NOT THE MEMORY, AND THE DIFFERENCE IS THE POINT.
    # `bridge.state_block` forbids Mati quoting its recalled memories — that
    # rule is about reciting its own past as substance. This is the person's
    # own correction read back to them so they can confirm it, which is the
    # only way an asked clarification can be checkable rather than vague.
    return (f"Before I answer — I've had this wrong before, and what you told "
            f"me was: “{said}”  Is that still right?")
