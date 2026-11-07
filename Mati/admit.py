"""
What is allowed into memory, and why.

Every piece of content that becomes a memory — a lived episode, a fact you gave
at setup, a chunk of a file, an answer Mati asked for — passes through
`bridge.remember()`. This module is the check that sits there. One choke point,
one set of named rules, each tied to a principle in `PRINCIPLES.md`.

    from Mati import admit
    verdict = admit.check(body, source)
    if not verdict.ok:
        ...                       # verdict.rule says which invariant, and why

THE RULE THAT MOTIVATED THIS. Mati distinguishes what it lived from what it
read, and the distinction is carried by the `source` column. But the source is
a label on the outside of the text, and the text has a voice of its own. Upload
your own notes — an ordinary thing to do — and this is what recall shows:

    [I READ THIS IN my-notes.md — I studied it, I did not live it]
    I remember when we talked about the pyramid last spring. You told me...

The label says studied. The sentence says *I remember when we talked*. A small
model resolves that contradiction by taking the sentence's word for it, and
narrates a conversation that never happened — sourced from the user's own file.
That is a fabricated shared past, which the README calls worse than no memory
at all, arriving through a feature designed to be helpful.

It is not fixed by refusing the content: the notes contain real information the
user wanted Mati to have. It is fixed by **attributing** it. A quotation does
not transfer its voice. So first-person claims in non-lived content are wrapped
as what they are — something a document said — rather than something Mati did.

WHAT THIS IS NOT. It is not a truth filter. It cannot tell whether a studied
claim is correct, and it does not try; that would need semantics the field does
not have and the 3B model cannot be trusted for. It checks only what is
checkable: does admitting this violate an invariant Mati actually enforces.

REFUSALS ARE RECORDED. A silent drop is indistinguishable from a bug, and this
sits on the path where memories are made. Every refusal is written to a
`refusals` table with its rule and the text, so it can be audited and argued
with.
"""

import re
import sqlite3
import sys
import os
from collections import namedtuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .field import notes as N
from . import vault

Verdict = namedtuple("Verdict", "ok rule why body")

_DDL = """
CREATE TABLE IF NOT EXISTS refusals (
    refusal_id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule       TEXT NOT NULL,
    why        TEXT,
    source     TEXT,
    body       TEXT,
    at         TEXT
);
"""

# First-person experiential claims — sentences that assert a shared past.
# Deliberately narrow: "I think", "I know", "I saw a photograph" are all fine
# in studied material. What is not fine is claiming a conversation.
_EXPERIENTIAL = re.compile(
    r"\b("
    r"I remember(?:ed)?\b"
    r"|we (?:talked|discussed|spoke|agreed|said)\b"
    r"|you (?:told|asked|showed|mentioned|said to) me\b"
    r"|(?:last|the other) (?:time|day|week) (?:we|you)\b"
    r"|earlier you\b|as (?:we|you) (?:discussed|said)\b"
    r"|our (?:conversation|chat|discussion)\b"
    r"|when we (?:met|talked|spoke)\b"
    r")", re.I)

# Sources whose content IS lived experience, where the above is not a conflict.
LIVED = ("episode",)
# Mati thinking to itself (Mati/reflect.py). Its own voice — never a
# document, never attributed, and never "lived" either.
# ⚠️ A TUPLE, LIKE `LIVED`, AND MEMBERSHIP IS EXACT. `self-check` was added
#  after the first shipped artefact came out of `make_ship_field`
# with this in a body:
#
#   self-check says: "I checked how much I remember. I hold 0 lived
#                     conversations and 24 passages of my own charter…"
#
# `_EXPERIENTIAL` had matched the substring "I remember" inside "I checked how
# much I remember" — not a claim of shared experience at all — and because
# `self-check` was in neither LIVED nor OWN_VOICE, the row was wrapped as a
# quotation from a source called "self-check". Mati would then read its own
# measurement as something it had been told.
#
# ⭐ IT BELONGS HERE BY THIS MODULE'S OWN REASONING. These rows are composed
# in code by `make_ship_field.self_check()` out of counts queried from the
# artefact being built — no model output, no user text, nothing that could
# claim an experience it did not have. That is Mati's own voice in the same
# sense `mine` is.
#
# ⚠️ NOT added to LIVED, for the reason the comment below already gives: lived
# means an event shared WITH the person. A self-check is Mati alone, and a
# measurement is not even an event.
OWN_VOICE = ("mine", "self-check")

# Talking about the machinery. NARROWER than the reply filter's rule, and it
# has to be: `basin`, `coherence` and `resonance` are ordinary English words,
# and a bare-word list refused "Teotihuacan was built around 100 BCE in the
# Basin of Mexico" — core material for this user, blocked as internals.
#
# The real risk is reciting block VALUES, not using the words. So: internal
# identifiers are always refused, and the ambiguous words only when they arrive
# with a number attached, which is what recitation looks like.
_MACHINERY = re.compile(
    r"\b(field[_ ]state|memories[_ ]recalled)\b|</?field_state>?", re.I)
_RECITED = re.compile(
    r"\b(basin|coherence|resonance|triplet)\b[^.!?]{0,20}?\d*\.\d+", re.I)

MIN_WORDS = 3


def _conn(db_path=None):
    from . import bridge
    conn = sqlite3.connect(db_path or bridge.NOTES_DB_PATH)
    conn.executescript(_DDL)
    conn.commit()
    return conn


def attribute(body, source):
    """Wrap first-person claims as a quotation from wherever they came from.

    Keeps the information, removes the voice. `studied:my-notes.md` becomes
    `my-notes.md says: "..."`, and a quotation cannot be mistaken for
    autobiography by a model reading it.
    """
    origin = source.split(":", 1)[1] if ":" in source else source
    if origin.startswith("asked about "):
        speaker = "the answer I got when I asked about " + origin[12:]
    else:
        speaker = origin
    return f'{speaker} says: "{" ".join(body.split())}"'


def check(body, source, db_path=None, record=True):
    """Should this become a memory? Returns a Verdict; `body` may be rewritten.

    The body is returned UNCHANGED unless a rule rewrites it. An earlier
    version normalised whitespace on the way through, which silently flattened
    every episode: `narrate_episode` writes several lines on purpose, and
    collapsing them meant the stored text no longer matched the text it was
    generated from — so a note stopped retrieving itself. Matching is done on a
    normalised copy; the stored body keeps its shape.
    """
    original = body or ""
    text = " ".join(original.split())

    if len(text.split()) < MIN_WORDS:
        return _refuse("too-short", f"{len(text.split())} words", text, source,
                       db_path, record)

    # NEVER DECEIVE YOURSELF — the store is replayed, so machinery in it is
    # repeated back at the user forever.
    hit = _MACHINERY.search(text) or _RECITED.search(text)
    if hit:
        return _refuse("machinery", f"mentions {hit.group(0)!r}", text, source,
                       db_path, record)

    # NEVER DECEIVE YOURSELF — a claim of shared experience, in content that
    # was not experienced. Attributed rather than refused: the information is
    # wanted, the voice is not.
    # ⚠️ SELF-TALK IS MATI'S OWN VOICE AND MUST NOT BE ATTRIBUTED. Reflection
    # writes with source='mine', which is not LIVED — so a remark containing
    # "today's conversation" was wrapped as `mine says: "..."`, turning Mati's
    # own thinking into a quotation from a source called "mine". Found in the
    # morning report , where two of four entries read that way.
    #
    # It is NOT added to LIVED, deliberately: lived means an event shared WITH
    # the person, and self-talk is Mati alone. Own voice and shared experience
    # are different things, and collapsing them is the confusion the whole
    # provenance scheme exists to prevent.
    if (source not in LIVED and not source.startswith(LIVED)
            and source not in OWN_VOICE):
        claim = _EXPERIENTIAL.search(text)
        if claim:
            return Verdict(True, "attributed",
                           f"claims experience ({claim.group(0)!r}) but source "
                           f"is {source!r}", attribute(original, source))

    return Verdict(True, None, None, original)


def _refuse(rule, why, body, source, db_path, record):
    if record:
        try:
            conn = _conn(db_path)
            try:
                conn.execute(
                    "INSERT INTO refusals (rule, why, source, body, at) "
                    # ⚠️ SEALED. A refused passage is still a passage from
                    # the person's document; declining to learn it is not a
                    # reason to store it in the clear.
                    "VALUES (?,?,?,?,?)",
                    (rule, why, source, vault.seal_soft(body[:2000]), N._now()))
                conn.commit()
            finally:
                conn.close()
        except sqlite3.Error:
            pass                       # never let bookkeeping break a turn
    return Verdict(False, rule, why, body)


def refusals(limit=50, db_path=None):
    """What Mati declined to remember, and why."""
    conn = _conn(db_path)
    try:
        rows = conn.execute(
            "SELECT refusal_id, rule, why, source, body, at FROM refusals "
            "ORDER BY refusal_id DESC LIMIT ?", (limit,)).fetchall()
    finally:
        conn.close()
    return [{"id": r[0], "rule": r[1], "why": r[2], "source": r[3],
             "body": r[4], "at": r[5]} for r in rows]


def main():
    import argparse
    ap = argparse.ArgumentParser(description="what Mati refused to remember")
    ap.add_argument("--limit", type=int, default=25)
    args = ap.parse_args()
    rows = refusals(args.limit)
    if not rows:
        print("  nothing refused")
        return 0
    for r in rows:
        print(f"  [{r['rule']}] {r['why']}")
        print(f"      source: {r['source']}")
        print(f"      {r['body'][:100]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
