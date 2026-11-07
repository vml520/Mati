"""
The exchanges that should land the same way every time.

⚠️ MEASURED ABSENT. The user: *"we are creatures of habit, we repeat things like
catch phrases, quotes, greetings, and we expect the same response to those
repeated things."* Across 237 turns with a stored reply, the agreement between
answers to the SAME question was **0.06** — where 1.00 is identical wording and
0.00 is no shared content words. "hello", asked five times, got five answers
with almost nothing in common.

That is not variety. It is the absence of habit.

⚠️ AND TEMPERATURE CANNOT FIX IT. `LOCAL_TEMP` is already 0.3. The answers
diverge because the CONTEXT differs every time — different memories surface, a
different waking block, a different hour — so the model is answering a genuinely
different prompt each occasion and answering it reasonably. Consistency cannot
come from the model. It has to come from a stored response.

WHAT THIS MAY AND MAY NOT COVER, from the same measurement:

  FIXED — greetings, "who are you", "what is my name", "this is the user".
      Answered here. About 7.5% of all turns.
  STORE-DEPENDENT — "how is your memory?", "what is the number I asked you to
      remember?", "what did you read about the tlamatinime?". NEVER answered
      here. Those scored 0.04 agreement and that is CORRECT: the honest answer
      differs from day to day, and freezing it would be the failure.
  TIME-DEPENDENT — "what day is today?". Not answered here either; it is cheap
      to compute and wrong to cache.

⚠️ THE ANSWERS ARE TEMPLATES FILLED FROM AUTHORITATIVE SOURCES, NEVER LITERALS.
"Your name is the user" hardcoded would become a lie the day the profile changes.
Everything below reads the profile, the clock or the charter at the moment of
answering, so a rapport reply is as current as the thing it reports.

⚠️ AND IDENTICAL IS ALSO WRONG. A person does not repeat themselves word for
word; 1.00 would read as a recording. Greetings vary with the hour, because
that is a real reason to answer differently and the person shares it. Identity
answers do not vary, because they are facts.

⚠️ BUILT IN SLEEP, NOT DECIDED IN THE MOMENT. The user: *"the rapport isn't
something that happens in the moment. It should be part of the prediction."*
The first version matched hand-written patterns at runtime — which meant I had
decided what Mati says and to whom. `build()` now runs at sleep and writes the
day's table FROM THE RECORD: the exchanges this person actually repeats, not
the ones I guessed they would. `match()` is a lookup and nothing else.

**Rapport is therefore earned, and a fresh install has none.** That is correct
rather than a gap — you do not have rapport with a stranger, and a settled
answer to a question nobody has asked twice would be an invention.

⚠️ HELD FOR THE WHOLE DAY. The user: *"if something happens against the
prediction, the prediction is held until the end of the day, because it may
just be delayed."* The table is stamped with the day it was built for and is
not rebuilt or revised while that day runs. An exchange that has not come up
yet has not failed to come up.
"""

from __future__ import annotations

import re
import time
from datetime import datetime

# ⚠️ THE TURN MUST BE THE WHOLE PATTERN. "hello" is a greeting; "hello, what did
# you read yesterday?" is a question with a greeting attached, and answering it
# from here would shortcut the thing actually being asked. So every pattern is
# anchored at both ends and the turn is length-capped.
MAX_WORDS = 6

_GREETING = re.compile(
    r"^\s*(?:hi|hey|hello|good\s+(?:morning|afternoon|evening)|"
    r"morning|evening)\b[\s,!.]*(?:mati)?[\s,!.]*$", re.I)

_WHO_ARE_YOU = re.compile(
    r"^\s*(?:so\s+)?who\s+(?:are|r)\s+you\s*\??\s*$", re.I)

_WHO_AM_I = re.compile(
    r"^\s*(?:who\s+am\s+i|what(?:'s| is)\s+my\s+name)\s*\??\s*$", re.I)

# ⚠️ THE LEAD-IN IS CASE-INSENSITIVE, THE NAME IS NOT — the same split
# `speaker._TELLS` makes, and for the same reason. Without `(?i:)` on the
# phrase, "This is the user" failed while "this is the user" matched, which is the
# capitalisation a person actually types. Requiring the capital on the NAME is
# what keeps "i am tired" from being read as an introduction.
_ITS_ME = re.compile(
    r"(?i:^\s*(?:it'?s|this\s+is|i\s+am|i'?m)\s+)"
    r"([A-Z][a-z]{1,20})\s*[.!]?\s*$")


def _part_of_day(now=None):
    h = datetime.fromtimestamp(now or time.time()).hour
    if h < 12:
        return "morning"
    if h < 18:
        return "afternoon"
    return "evening"


def _owner(db_path=None):
    from . import speaker
    try:
        return speaker.owner(db_path=db_path)
    except Exception:
        return None


TABLE = "rapport"


def ensure(db_path=None):
    """The day's settled exchanges. One row per question Mati has learned."""
    import sqlite3
    from . import bridge
    from . import vault
    conn = sqlite3.connect(db_path or bridge.NOTES_DB_PATH, timeout=30)
    try:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {TABLE} (
                    question TEXT PRIMARY KEY,
                    kind     TEXT,
                    seen     INTEGER,
                    for_day  TEXT,
                    made_at  TEXT
                )""")
        # ⚠️ THE QUESTION IS THE PERSON'S OWN WORDS AND MUST NOT SIT IN THE
        # CLEAR — but it was also the PRIMARY KEY, looked up by equality, and
        # `vault.seal` is SALTED, so a sealed value can never match itself.
        # Sealing it alone silently breaks every lookup: `match` finds nothing,
        # returns None, and the model answers instead — which looks exactly
        # like "not learned yet" and nothing reports a fault.
        #
        # So the lookup moves to a hash and the text is sealed beside it. Same
        # pattern as `gate_log.text_hash`, and for the same reason.
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info({TABLE})")}
        if "qhash" not in cols:
            conn.execute(f"ALTER TABLE {TABLE} ADD COLUMN qhash TEXT")
            conn.execute(f"CREATE INDEX IF NOT EXISTS {TABLE}_qhash "
                         f"ON {TABLE}(qhash)")
        # ⚠️ AND BACKFILL IT, OR EVERY EXISTING ROW IS UNREACHABLE. Adding the
        # column leaves `qhash` NULL on rows written before it, so `match`
        # finds nothing and falls through to the model — which is
        # indistinguishable from "not learned yet" and reports no fault. The
        # column and its contents are one migration, not two.
        #
        # Only rows that actually OPEN are filled. With the vault locked the
        # question is still ciphertext, and hashing that would write a key
        # matching nothing and never be retried.
        for _rowid, _q in conn.execute(
                f"SELECT rowid, question FROM {TABLE} "
                f"WHERE qhash IS NULL AND question IS NOT NULL").fetchall():
            plain = vault.open_body(_q)
            if not plain or vault.is_encrypted(plain):
                continue                  # locked; leave it for a later run
            conn.execute(f"UPDATE {TABLE} SET qhash = ? WHERE rowid = ?",
                         (_qhash(plain), _rowid))
        conn.commit()
        return True
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def _normalise(t):
    return " ".join((t or "").split()).lower().strip(" .!?")


def _qhash(t):
    """Lookup key for a settled question. One-way, and of the NORMALISED text
    so that wording differences that `_normalise` already ignores keep
    matching."""
    import hashlib
    return hashlib.sha256(_normalise(t).encode("utf-8")).hexdigest()[:32]


def _kind(t):
    """Which settled answer this exchange takes, or None if it takes none.

    ⚠️ The CLASSIFIER stays here and the SET does not. Deciding that a greeting
    has a stable answer is a safety rule; deciding that this person says
    "hello" is an observation, and observations belong to the record.
    """
    if _GREETING.match(t):
        return "greeting"
    if _WHO_ARE_YOU.match(t):
        return "who_are_you"
    if _WHO_AM_I.match(t):
        return "who_am_i"
    if _ITS_ME.match(t):
        return "introduction"
    return None


def build(db_path=None, now=None, min_seen=2, verbose=False):
    """Learn the day's rapport from the record. Run once, in sleep.

    ⚠️ IDEMPOTENT AND DAY-STAMPED. Called again on the same day it does
    nothing: the prediction is held until the day ends, so an exchange that has
    not come up yet has not failed to come up.
    """
    import collections, re as _re, sqlite3
    from datetime import datetime as _dt
    from . import bridge, vault
    ensure(db_path)
    day = _dt.fromtimestamp(now or time.time()).strftime("%Y-%m-%d")
    conn = sqlite3.connect(db_path or bridge.NOTES_DB_PATH, timeout=30)
    try:
        held = conn.execute(
            f"SELECT COUNT(*) FROM {TABLE} WHERE for_day = ?", (day,)).fetchone()
        if held and held[0]:
            return 0                      # already built for today; held
        seen = collections.Counter()
        first_seen = {}
        for (b,) in conn.execute(
                "SELECT body FROM notes WHERE source='episode' "
                "ORDER BY wall_created DESC LIMIT 600"):
            try:
                t = vault.open_body(b) or ""
            except Exception:
                continue
            m = _re.search(r"I was asked:\s*(.+?)(?:\n|$)", t)
            if m:
                raw = " ".join(m.group(1).split())
                # ⚠️ CLASSIFY THE ORIGINAL, KEY ON THE NORMALISED FORM. `_kind`
                # requires a CAPITAL on a name — that is what separates "I am
                # The user" from "i am tired" — and normalising first lowercases
                # it away, so introductions could never be learned at all.
                seen[_normalise(raw)] += 1
                first_seen.setdefault(_normalise(raw), raw)
        rows = [(q, _kind(first_seen.get(q, q)), n)
                for q, n in seen.items() if n >= min_seen]
        rows = [(q, k, n) for q, k, n in rows if k]
        conn.execute(f"DELETE FROM {TABLE}")
        conn.executemany(
            f"INSERT OR REPLACE INTO {TABLE} (question, qhash, kind, seen, "
            f"for_day, made_at) VALUES (?,?,?,?,?,?)",
            [(vault.seal(q), _qhash(q), k, n, day, bridge._now())
             for q, k, n in rows])
        conn.commit()
        if verbose:
            print(f"  [rapport] {len(rows)} settled exchanges for {day}")
        return len(rows)
    except sqlite3.Error:
        return 0
    finally:
        conn.close()


def _answer(kind, db_path=None, now=None, said=None):
    """The settled answer for a kind, composed from sources at answer time."""
    if kind == "greeting":
        who = _owner(db_path)
        return (f"Good {_part_of_day(now)}"
                + (f", {who}." if who else ".") + " I am here.")
    if kind == "who_are_you":
        return ("I am Mati. I run entirely on this computer, I make no "
                "outbound network connections by default, and what I hold is "
                "stored here and changes as we talk.")
    if kind == "who_am_i":
        who = _owner(db_path)
        return (f"You are {who}." if who else
                "I do not have your name yet — tell me and I will keep it.")
    if kind == "introduction":
        who = _owner(db_path)
        if who and said and said.lower() == who.lower():
            return f"Hello, {who}. I am here."
        # ⚠️ `said` CAN BE None AND THE ANSWER USED TO INTERPOLATE IT — Mati
        # greeted the person with "Hello, None." The kind is matched on the
        # NORMALISED (lowercased) question, but `_ITS_ME` requires a capital
        # letter, so anyone typing "i am <name>" without the capital landed
        # here with no name at all. Common, and it read as a malfunction.
        if not said:
            return ("Hello. I did not catch the name — say it again and I will "
                    "keep it.")
        return (f"Hello, {said}. I will file this under your name; I have no "
                f"way to check it, so I am taking your word.")
    return None


def match(user_text, db_path=None, now=None):
    """A settled answer for a habitual exchange, or None.

    ⚠️ A LOOKUP, NOT A DECISION. The question must be in the day's table, which
    `build()` wrote in sleep from exchanges this person actually repeats. An
    exchange Mati has not learned goes to the model, as it should.
    """
    import sqlite3
    from . import bridge
    t = " ".join((user_text or "").split())
    if not t or len(t.split()) > MAX_WORDS:
        return None
    norm = _normalise(t)
    try:
        conn = sqlite3.connect(
            f"file:{db_path or bridge.NOTES_DB_PATH}?mode=ro", uri=True)
        try:
            row = conn.execute(
                f"SELECT kind FROM {TABLE} WHERE qhash = ?",
                (_qhash(norm),)).fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return None
    if not row:
        return None                       # not learned; the model answers
    kind = row[0]
    said = None
    if kind == "introduction":
        m = _ITS_ME.match(t)
        said = m.group(1) if m else None
    return _answer(kind, db_path=db_path, now=now, said=said)


def is_fixed(user_text):
    """Would `match` answer this? For callers that only need the question."""
    return match(user_text) is not None


def status(db_path=None):
    """What Mati has settled on, and for which day. For `/queue`."""
    import sqlite3
    from . import bridge, vault
    try:
        conn = sqlite3.connect(
            f"file:{db_path or bridge.NOTES_DB_PATH}?mode=ro", uri=True)
        try:
            rows = conn.execute(
                f"SELECT question, kind, seen, for_day FROM {TABLE} "
                f"ORDER BY seen DESC").fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return {"exchanges": [], "for_day": None}
    return {"for_day": rows[0][3] if rows else None,
            "exchanges": [{"question": vault.open_body(q), "kind": k,
                           "seen": n}
                          for q, k, n, _d in rows]}
