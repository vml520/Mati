"""
Who is Mati talking to?

⚠️ THE GAP THIS FILLS. An episode records a body, a source and a timestamp, and
**nothing about who spoke**. So every memory reads as though the owner said it.
A few episodes that mention "my AI assistant" are enough for Mati, asked who it
is talking to, to answer *"You are someone the user asked to talk to."*

That is not a confabulation — it is correct recall of a real third party,
attached to a conversation that had no way to say which of them was typing.
Both facts are true and Mati cannot tell them apart, because the store gives
it nothing to tell them apart with.

Two things, and only two:

  TAKE a name the moment it is offered — never by asking for one (see below).
  RECORD, on every episode, who it was told it was speaking with.

⚠️ THIS IS NOT AUTHENTICATION AND MUST NOT BE READ AS ANY. Mati cannot verify
a name — anyone can type one. What it gains is the ability to say "someone told
me they were Anna" rather than silently filing Anna's words as the owner's.
That distinction is the whole value: an unverified attribution is still
enormously better than a wrong one asserted as fact.
"""

from __future__ import annotations

import re
import sqlite3

from . import bridge

# What counts as being told a name. Deliberately narrow: a false positive here
# writes the wrong person onto a memory, which is worse than asking twice.
# ⚠️ THE LEAD-IN IS CASE-INSENSITIVE, THE NAME IS NOT. "I am Anna" and "this
# is Maria" must match; "I'm just checking" must not. Requiring a capitalised
# name is what separates them, so re.I belongs on the phrase and the capital
# stays on the group.
_TELLS = [
    re.compile(r"(?i:\b(?:i'?m|i am|this is|it'?s)\s+)([A-Z][a-z]{1,20})\b"),
    re.compile(r"(?i:\bmy name(?:'?s| is)\s+)([A-Z][a-z]{1,20})\b"),
    re.compile(r"(?i:\byou(?:'?re| are) (?:talking|speaking) (?:to|with)\s+)"
               r"([A-Z][a-z]{1,20})\b"),
    re.compile(r"(?i:\bspeaking with\s+)([A-Z][a-z]{1,20})\b"),
]

# Words that look like names to the patterns above but are not.
_NOT_A_NAME = {"Sorry", "Okay", "Yes", "No", "Just", "Still", "Here", "Back",
               "Fine", "Good", "Sure", "Right", "Well", "Also", "Actually"}


def ensure_column(db_path=None):
    """Add `speaker` if it is not there. NULL means nobody said."""
    conn = sqlite3.connect(db_path or bridge.NOTES_DB_PATH)
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(notes)")}
        if "speaker" not in cols:
            conn.execute("ALTER TABLE notes ADD COLUMN speaker TEXT")
            conn.commit()
            return True
        return False
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def named_in(text):
    """A name if the person just gave one, else None.

    ⚠️ Returns None far more often than it guesses. Writing the wrong speaker
    onto a memory is the failure this module exists to prevent, so the bar is
    an explicit self-introduction and nothing looser.
    """
    for pat in _TELLS:
        m = pat.search(text or "")
        if m:
            name = m.group(1).strip()
            if name and name not in _NOT_A_NAME:
                return name
    return None


def owner(db_path=None):
    """The name in the profile — who Mati talks to unless told otherwise."""
    try:
        for note in bridge.profile_notes(db_path=db_path) or []:
            m = re.search(r"called\s+([A-Z][a-z]{1,20})", str(note))
            if m:
                return m.group(1)
    except Exception:
        pass
    return None


# ⚠️ MATI DOES NOT ASK WHO IS THERE. Its assumption is that the person speaking
# to it is the owner, and it is up to Mati to work out on its own whether that
# is truly the case.
#
# Opening a session by asking for a name greets the person who set it up as a
# stranger, and it hands the work of identification to the human when the work
# belongs here.
#
# What there is: `named_in` takes a name the moment it is offered, and
# `attribute` marks a memory as somebody else's. The harder problem — noticing
# unprompted that the person is NOT the owner — is unsolved, and it is not
# solved by asking.


def attribute(body, speaker, known_owner=None):
    """Mark a memory's body with who said it, when it was not the owner.

    The owner is the unmarked case on purpose: episodes carry no speaker by
    default, and re-labelling the existing ones would be a claim the store
    cannot support. What is worth saying in the text is the EXCEPTION — that
    this one came from somebody else.
    """
    if not speaker or (known_owner and speaker == known_owner):
        return body
    return f"{body}\nI was speaking with {speaker} at the time, not the person whose field this is."
