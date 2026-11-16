"""
Things Mati means to say, and when it is allowed to say them.

    python3 -m Mati.schedule --list
    python3 -m Mati.schedule --consent interrupt=on quiet=22:00-08:00
    python3 -m Mati.schedule --remind "call the dentist" --in "2 hours"

Everything else Mati does is reactive: you speak, it answers. This is the one
place it can start something, so it is also the one place it can become a
nuisance. The defaults are built accordingly.

TWO KINDS OF THING TO SAY, and they do not deserve the same permission:

    reminder   you asked for it, and it is bound to a time. Mati is holding
               it on your behalf, and failing to deliver it on time is the
               failure that matters.
    finding    Mati went and looked something up — usually after you corrected
               it twice and it deferred. It waits for the TOPIC TO COME UP.

WHY A FINDING WAITS FOR THE TOPIC. Delivering it at the top
of the next conversation is still an interruption, just a rescheduled one: you
say hello and Mati leads with something you were not thinking about. Held
until you raise the subject yourself, the same sentence arrives as an answer
rather than an intrusion.

This is not a separate rule. It is "always choose harmony" — the existing
principle — applied to timing, which is why it does not get an entry of its own
in PRINCIPLES.md. A principle that has to be restated for every case was not
doing any work.

THE CONSENT MODEL. **Nothing interrupts unless you said it could**, per kind.
The default for both is off: queued items wait until you next speak, and are
delivered at the top of that conversation. Turning interruption on for
reminders is the case that makes sense; leaving it off for findings is the
default because "I read something you might like" is never worth a buzz.

Quiet hours apply to interruption regardless of kind, and they apply to the
clock rather than to activity — Mati does not get to decide that you are
probably awake.

PRACTICAL TIME, NOT FUNCTIONAL TIME. `sense()` is what goes into a turn: not a
timestamp but the human shape of it — that it is late, that you have not spoken
in four days, that you asked about this yesterday morning. That is what lets
Mati say "it's late, this can wait" rather than reporting a clock reading.

REFUSING TO GUESS. A reminder whose time cannot be read confidently is
REFUSED, not approximated. A reminder that fires at the wrong time is worse
than one that was never accepted, because you stopped holding it yourself.
"""

import argparse
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from . import vault

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


REMINDER = "reminder"
FINDING = "finding"

# A QUESTION is something Mati wants to know and cannot find out on its own.
#
# `inquiry` was built to research a topic by asking its OWN language model, and
# stores the answers as `studied:asked about …` — which the README calls "the
# least reliable thing it holds, because it is the small model talking rather
# than anything Mati lived or checked". On a device with no corpus to search,
# that loop can only ever produce more of the weakest kind of content.
#
# The person is right there and actually knows. So Mati asks THEM, and their
# answer arrives as a lived episode — the primary training signal — instead of
# a 3B model answering its own question unattended.
#
# Delivered like a FINDING, never like a reminder: it waits until the subject
# comes up. A question that ambushes you at the top of a conversation is an
# interruption; the same question when you raised the topic is just interest.
QUESTION = "question"

_DDL = """
CREATE TABLE IF NOT EXISTS to_say (
    say_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    kind       TEXT NOT NULL,
    topic      TEXT,
    body       TEXT NOT NULL,
    due_at     REAL,
    made_at    REAL NOT NULL,
    delivered  REAL,
    session    TEXT,
    detail     TEXT
);
CREATE TABLE IF NOT EXISTS consent (
    key   TEXT PRIMARY KEY,
    value TEXT
);
CREATE INDEX IF NOT EXISTS to_say_pending ON to_say(delivered, due_at);
"""

# Nothing interrupts unless it was turned on. Both default off; a reminder is
# the one people usually want on, and it is still their call to make.
DEFAULT_CONSENT = {
    "interrupt.reminder": "off",
    "interrupt.finding": "off",
    # There is deliberately NO "interrupt.question" here. A question has one
    # route out — `pending(user_text=...)` matching its topic against what the
    # person just said — and `_maybe_announce` passes no user_text, so a
    # question cannot be raised unprompted no matter what any consent says.
    # A key for it would change nothing when turned ON, and a setting that
    # cannot take effect is worse than absent, because someone will one day
    # set it and believe it did something.
    # Whether Mati may form questions for the person at all. ON by default —
    # asking is how it learns from the one source on the device that actually
    # knows anything — but the person can say stop and it stops.
    "ask.questions": "on",
    "quiet": "22:00-08:00",
}


def _conn(db_path=None):
    from . import bridge
    # ⚠️ A BUSY TIMEOUT, BECAUSE THIS WRITES WHILE OTHERS READ. With sqlite's
    # 5s default, `say_later` from the idle worker can fail with "database is
    # locked" and lose what Mati meant to raise. Every other writer in this
    # codebase asks for 10-30s.
    conn = sqlite3.connect(db_path or bridge.NOTES_DB_PATH, timeout=30)
    conn.executescript(_DDL)
    # CREATE TABLE IF NOT EXISTS does not add columns to a table that already
    # exists, so a database written before `topic` was introduced keeps the old
    # shape and every insert fails. Same migration `bridge._ensure_schema`
    # does for `notes.corrects`.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(to_say)")}
    if "topic" not in cols:
        conn.execute("ALTER TABLE to_say ADD COLUMN topic TEXT")
    conn.commit()
    return conn


# ─────────────────────────────────────────────────────────────────────────────
# Reading a time out of what someone said
# ─────────────────────────────────────────────────────────────────────────────

_IN = re.compile(r"\bin\s+(?:(a|an|one|two|three|four|five|half\s+an?)\s+)?"
                 r"(\d+)?\s*(minute|min|hour|hr|day|week)s?\b", re.I)
_AT = re.compile(r"\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", re.I)
_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
_UNITS = {"minute": 60, "min": 60, "hour": 3600, "hr": 3600,
          "day": 86400, "week": 604800}
# Named times of day, resolved against the user's clock rather than guessed.
_NAMED = {"tonight": 20, "this evening": 20, "the evening": 20,
          "the morning": 9, "tomorrow morning": 9, "midday": 12, "noon": 12,
          "the afternoon": 14, "lunchtime": 12}


def read_time(text, now=None):
    """When does this want to happen? Returns a unix time, or None.

    None means "I could not read it", and the caller must refuse rather than
    approximate — see the module docstring.
    """
    now = datetime.fromtimestamp(now or time.time())
    low = " ".join((text or "").lower().split())

    m = _IN.search(low)
    if m:
        word, digits, unit = m.group(1), m.group(2), m.group(3).lower()
        if word and "half" in word:
            n = 0.5
        elif digits:
            n = float(digits)
        elif word:
            n = _WORDS.get(word.strip(), 1)
        else:
            n = 1
        return time.mktime(now.timetuple()) + n * _UNITS[unit]

    tomorrow = "tomorrow" in low
    m = _AT.search(low)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        ampm = (m.group(3) or "").lower()
        if ampm == "pm" and hour < 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0
        elif not ampm and hour < 8:
            hour += 12                  # "at 3" means the afternoon
        when = now.replace(hour=hour % 24, minute=minute, second=0,
                           microsecond=0)
        if tomorrow or when <= now:
            when += timedelta(days=1)
        return time.mktime(when.timetuple())

    for phrase, hour in sorted(_NAMED.items(), key=lambda kv: -len(kv[0])):
        if phrase in low:
            when = now.replace(hour=hour, minute=0, second=0, microsecond=0)
            if tomorrow or when <= now:
                when += timedelta(days=1)
            return time.mktime(when.timetuple())

    if tomorrow:                        # "tomorrow" with no hour -> 9am
        when = (now + timedelta(days=1)).replace(hour=9, minute=0, second=0,
                                                 microsecond=0)
        return time.mktime(when.timetuple())

    # ⚠️ A WEEKDAY AND A DAY OF THE MONTH, the two commonest ways to schedule a
    # bill:
    #
    #   "remind me to pay a bill on Friday"
    #   "remind me to pay the rent on the 1st"
    #
    # Unreadable, each would come back with no time, and the caller then
    # refuses and stores NOTHING — the reminder would never exist to fire. The
    # refusal is correct (approximating a time is worse); the parser has to be
    # able to read ordinary English.
    hour = _NAMED.get(next((p for p in _NAMED if p in low), ""), 9)

    m = _WEEKDAY.search(low)
    if m:
        want = _DAYS[m.group(1).lower()]
        # ⚠️ "on Friday" said ON a Friday means NEXT Friday, not today. Zero
        # days ahead would fire a reminder in the past or seconds from now.
        ahead = (want - now.weekday()) % 7
        if ahead == 0:
            ahead = 7
        if "next" in low and ahead < 7:
            ahead += 7
        when = (now + timedelta(days=ahead)).replace(
            hour=hour, minute=0, second=0, microsecond=0)
        return time.mktime(when.timetuple())

    m = _DAY_OF_MONTH.search(low)
    if m:
        day = int(m.group(1))
        if 1 <= day <= 31:
            when = _on_day(now, day, hour)
            if when:
                return time.mktime(when.timetuple())

    if "next week" in low:
        when = (now + timedelta(days=7)).replace(hour=hour, minute=0,
                                                 second=0, microsecond=0)
        return time.mktime(when.timetuple())
    return None


_DAYS = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
         "friday": 4, "saturday": 5, "sunday": 6}

_WEEKDAY = re.compile(
    r"\b(?:on|next|this)?\s*"
    r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I)

# ⚠️ REQUIRES "the" OR AN ORDINAL SUFFIX, so a bare number in the task text —
# "pay invoice 3" — is not read as the third of the month.
_DAY_OF_MONTH = re.compile(
    r"\bon the (\d{1,2})(?:st|nd|rd|th)?\b"
    r"|\bon (\d{1,2})(?:st|nd|rd|th)\b", re.I)


def _on_day(now, day, hour):
    """The next occurrence of day-of-month `day`. None if no month has it."""
    for ahead in range(0, 3):
        month = now.month + ahead
        year = now.year + (month - 1) // 12
        month = (month - 1) % 12 + 1
        try:
            when = now.replace(year=year, month=month, day=day, hour=hour,
                               minute=0, second=0, microsecond=0)
        except ValueError:
            continue                    # e.g. the 31st of a 30-day month
        if when > now:
            return when
    return None


_REMIND = re.compile(
    r"\b(?:remind|remember to tell|nudge|ping)\s+me\b(?:\s+to\b)?\s*(.*)",
    re.I)


def read_reminder(text, now=None):
    """(what, when) if this is a reminder request that can be honoured.

    Returns None when there is no reminder, and ("...", None) when there is one
    whose time could not be read — the caller should say so rather than
    silently picking a time.
    """
    m = _REMIND.search(text or "")
    if not m:
        return None
    what = m.group(1).strip(" .!?,")
    when = read_time(text, now=now)
    # Strip the time phrase out of the thing to be reminded of.
    # Longest first: stripping "tomorrow" before "tomorrow morning" leaves a
    # stray "morning" in the thing to be reminded of.
    what = re.sub(
        r"\b(?:"
        r"(?:tomorrow|this|in the|the)\s+(?:morning|afternoon|evening)"
        r"|tomorrow|tonight|midday|noon|lunchtime"
        r"|in\s+(?:a|an|one|two|three|four|five|half\s+an?|\d+)\s*"
        r"(?:minute|min|hour|hr|day|week)s?"
        r"|at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?"
        # ⚠️ ADDED WITH THE PARSER THAT READS THEM. Every form `read_time`
        # learns has to be stripped here too, or the task keeps its own date
        # — "pay a bill on Friday", delivered on Friday. The existing note
        # above says longest-first for exactly this reason, and "next
        # tuesday" must go before the bare weekday for the same one.
        r"|(?:on|next|this)\s+(?:monday|tuesday|wednesday|thursday|friday"
        r"|saturday|sunday)"
        r"|next\s+week"
        r"|on\s+the\s+\d{1,2}(?:st|nd|rd|th)?"
        r"|on\s+\d{1,2}(?:st|nd|rd|th)"
        r")\b", "", what, flags=re.I)
    what = re.sub(r"^\s*(?:about|to)\s+", "", what.strip(), flags=re.I)
    return " ".join(what.split()).strip(" .,") or "that thing", when


# ─────────────────────────────────────────────────────────────────────────────
# The queue
# ─────────────────────────────────────────────────────────────────────────────

def say_later(body, kind=FINDING, due_at=None, detail=None, db_path=None,
              topic=None):
    """Queue something for Mati to raise. Returns its id.

    `topic` makes a finding wait until the user raises that subject, rather
    than being delivered at the next opportunity.
    """
    conn = _conn(db_path)
    try:
        cur = conn.execute(
            "INSERT INTO to_say (kind, topic, body, due_at, made_at, detail) "
            "VALUES (?,?,?,?,?,?)",
            # ⚠️ SEALED. `to_say` is what Mati means to SAY — reminders it
            # was asked to keep, findings, questions it means to raise. Derived
            # from the person, and it was in the clear. `kind` stays plain: the
            # scheduler filters on it, and it is a category, not content.
            (kind, vault.seal_soft(topic) if topic else topic,
             vault.seal_soft(body) if body else body,
             due_at, time.time(), detail))
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


# Words too common to identify a subject.
_STOP = {"is", "was", "are", "the", "a", "an", "of", "to", "in", "on", "it",
         "and", "or", "that", "this", "for", "with", "as", "at", "by", "be",
         "not", "you", "i", "we", "my", "your", "correct", "about"}


def _stem(w):
    """Crude, and deliberately so: plural 's' is the difference between
    "primes" and "prime", and missing it made a natural mention fail to raise
    the topic. Anything cleverer needs a stemmer Mati does not ship."""
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def _keywords(topic):
    return {_stem(w) for w in re.findall(r"[a-z0-9']+", (topic or "").lower())
            if len(w) > 1 and w not in _STOP}


def on_topic(topic, user_text):
    """Has the user raised this subject?

    Lexical overlap, and appropriately so: "are they talking about primes" is a
    question about word surface, not meaning, which is the one thing this
    retrieval stack is actually good at. Conservative on purpose — requires two
    distinctive words where the topic has them. A finding that waits too long
    costs nothing; one that surfaces unbidden is the thing being avoided.
    """
    want = _keywords(topic)
    if not want:
        return False
    have = _keywords(user_text)
    hits = want & have
    return len(hits) >= (2 if len(want) >= 2 else 1)


def pending(now=None, db_path=None, user_text=None, include_held=False):
    """What Mati may say right now. Oldest first.

    Reminders come due by the clock. Findings with a topic wait until the user
    raises it — pass `user_text` to let them through. `include_held=True`
    returns everything undelivered regardless, for `/pending` and the CLI.
    """
    now = now or time.time()
    conn = _conn(db_path)
    try:
        rows = conn.execute(
            "SELECT say_id, kind, topic, body, due_at, made_at, detail "
            "FROM to_say WHERE delivered IS NULL "
            "AND (due_at IS NULL OR due_at <= ?) ORDER BY say_id",
            (now,)).fetchall()
    finally:
        conn.close()
    # ⚠️ DECRYPT ONLY WHAT IS NEEDED. Each sealed value is its own openssl
    # subprocess, so opening both columns of every candidate row before
    # deciding which ones are wanted would take tens of seconds with a few
    # hundred rows due — far past a client's timeout.
    #
    # `notes._open_cached` exists for exactly this, and is reused rather than
    # reinvented.
    #
    # And the BODY is opened only for rows that survive the topic gate: a
    # finding waiting on a subject that has not come up is never read aloud, so
    # decrypting it would be work done to throw away.
    from .notes import _open_cached

    def _open(v):
        return _open_cached(v) if v else v

    out = []
    for r in rows:
        kind, topic_raw = r[1], r[2]
        if not include_held and topic_raw and kind != REMINDER:
            if not on_topic(_open(topic_raw), user_text):
                continue                  # waits for the subject to come up
        out.append({"id": r[0], "kind": kind, "topic": _open(topic_raw),
                    "body": _open(r[3]), "due_at": r[4], "made_at": r[5],
                    "detail": r[6]})
    return out



_ASKING_OPENERS = (
    "who ", "what ", "what's", "whats ", "when ", "where ", "why ", "how ",
    "which ", "is ", "are ", "was ", "were ", "do ", "does ", "did ", "can ",
    "could ", "should ", "would ", "will ", "am i", "tell me", "explain ",
)


def user_is_asking(user_text):
    """Is this turn the person asking Mati something?

    Used to hold Mati's OWN questions back. Answering a question with a
    question is the thing that makes an assistant exhausting to talk to, and
    the queued question loses nothing by waiting — the topic is clearly live,
    so it will come up again.

    Deliberately loose in the direction of silence: a false positive costs one
    delayed question, a false negative costs an interruption at the worst
    possible moment.
    """
    t = " ".join((user_text or "").lower().split())
    if not t:
        return False
    if "?" in t:
        return True
    return any(t.startswith(o) for o in _ASKING_OPENERS)

def mark_delivered(ids, session=None, db_path=None):
    if not ids:
        return 0
    conn = _conn(db_path)
    try:
        conn.executemany(
            "UPDATE to_say SET delivered=?, session=? WHERE say_id=?",
            [(time.time(), session, i) for i in ids])
        conn.commit()
        return len(ids)
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Consent
# ─────────────────────────────────────────────────────────────────────────────


# Turning the questions off, and back on, by saying so.
#
# Matched against a small set of plain phrasings, deliberately — the same
# discipline `read_reminder` uses for times. A 3B model asked "did they mean
# stop?" will confabulate an answer, and getting this wrong in either direction
# is bad: ignoring "stop asking me things" is rude, and silencing Mati because
# someone said the word "question" is a fault they cannot see. Anything not on
# this list is left alone rather than guessed at.
_STOP_ASKING = (
    "stop asking questions", "stop asking me questions", "no more questions",
    "don't ask me questions", "dont ask me questions", "stop asking me things",
    "quit asking questions", "stop with the questions",
)
_START_ASKING = (
    "you can ask me questions", "ask me questions again", "you can ask again",
    "start asking questions", "questions are fine", "you may ask questions",
)


def reads_ask_control(user_text, db_path=None):
    """Did the person just turn Mati's questions off, or back on?

    Returns "off", "on", or None when they said neither. Applies the change.
    """
    t = " ".join((user_text or "").lower().split())
    if not t:
        return None
    for phrase in _START_ASKING:              # checked first: "ask again"
        if phrase in t:                       # contains "ask", so order matters
            set_consent("ask.questions", "on", db_path=db_path)
            return "on"
    for phrase in _STOP_ASKING:
        if phrase in t:
            set_consent("ask.questions", "off", db_path=db_path)
            return "off"
    return None


def may_ask(db_path=None):
    """Is Mati allowed to form new questions for the person?"""
    return consent(db_path=db_path).get("ask.questions", "on") == "on"

def consent(db_path=None):
    conn = _conn(db_path)
    try:
        stored = dict(conn.execute("SELECT key, value FROM consent").fetchall())
    finally:
        conn.close()
    return {**DEFAULT_CONSENT, **stored}


def set_consent(key, value, db_path=None):
    conn = _conn(db_path)
    try:
        conn.execute("INSERT INTO consent (key, value) VALUES (?,?) "
                     "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                     (key, value))
        conn.commit()
    finally:
        conn.close()


def in_quiet_hours(now=None, db_path=None, quiet=None):
    """Is it a time Mati has been told not to start anything?"""
    quiet = quiet if quiet is not None else consent(db_path)["quiet"]
    if not quiet or quiet.lower() in ("off", "none"):
        return False
    try:
        start, end = [x.strip() for x in quiet.split("-", 1)]
        sh, sm = [int(x) for x in start.split(":")]
        eh, em = [int(x) for x in end.split(":")]
    except (ValueError, AttributeError):
        # ⚠️ FAIL CLOSED. This returned False — "not quiet hours" — so a typo
        # in the setting silently licensed an interruption at 3am. The person
        # set a quiet range; that they set one is the instruction, and an
        # unreadable range means STAY QUIET rather than assume permission.
        # An unwanted silence costs a delayed reminder; an unwanted 3am
        # notification costs their trust in the whole feature.
        return True
    t = datetime.fromtimestamp(now or time.time())
    mins = t.hour * 60 + t.minute
    s, e = sh * 60 + sm, eh * 60 + em
    return (s <= mins or mins < e) if s > e else (s <= mins < e)


def may_interrupt(kind, now=None, db_path=None):
    """Is Mati allowed to start a conversation with this, right now?

    Two separate permissions and both must hold: the kind must be allowed at
    all, and it must not be quiet hours. Refusing does not lose the item — it
    waits for the user to speak.
    """
    c = consent(db_path)
    if c.get(f"interrupt.{kind}", "off").lower() not in ("on", "yes", "true"):
        return False, f"interruption for {kind}s is off"
    if in_quiet_hours(now, db_path, c.get("quiet")):
        return False, f"quiet hours ({c.get('quiet')})"
    return True, None


# ─────────────────────────────────────────────────────────────────────────────
# Practical time
# ─────────────────────────────────────────────────────────────────────────────

def _last_spoke(db_path=None):
    from . import bridge
    try:
        conn = sqlite3.connect(db_path or bridge.NOTES_DB_PATH,
                               timeout=30)
        try:
            row = conn.execute(
                "SELECT wall_created FROM notes WHERE source='episode' "
                "ORDER BY note_id DESC LIMIT 1").fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return None
    if not row or not row[0]:
        return None
    try:
        return datetime.fromisoformat(row[0].replace("Z", "+00:00")).timestamp()
    except (ValueError, AttributeError):
        return None


def _gap(seconds):
    if seconds < 3600:
        return "a little while ago"
    if seconds < 6 * 3600:
        return "earlier today"
    if seconds < 86400:
        return "yesterday" if seconds > 12 * 3600 else "earlier today"
    days = int(seconds // 86400)
    if days == 1:
        return "yesterday"
    if days < 7:
        return f"{days} days ago"
    if days < 14:
        return "about a week ago"
    if days < 60:
        return f"about {days // 7} weeks ago"
    return "a long time ago"


def sense(now=None, db_path=None):
    """The human shape of the current time — for the turn, not for a log.

    Deliberately not a timestamp. Mati should be able to say "it's late, this
    can wait", which needs a sense of lateness, not a clock reading.
    """
    t = datetime.fromtimestamp(now or time.time())
    hour = t.hour
    part = ("the middle of the night" if hour < 5 else
            "early morning" if hour < 8 else
            "morning" if hour < 12 else
            "afternoon" if hour < 17 else
            "evening" if hour < 21 else "late evening")
    out = {"part_of_day": part, "weekday": t.strftime("%A"),
           "clock": t.strftime("%H:%M"), "quiet": in_quiet_hours(now, db_path)}
    last = _last_spoke(db_path)
    if last:
        out["since_last"] = _gap((now or time.time()) - last)
    return out


def sense_line(now=None, db_path=None):
    """One line for the state block."""
    s = sense(now, db_path)
    line = f"It is {s['part_of_day']} on {s['weekday']}"
    if s.get("since_last"):
        line += f"; you last spoke {s['since_last']}"
    return line + "."


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="what Mati means to say, and when")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--remind", metavar="TEXT")
    ap.add_argument("--in", dest="in_", metavar="WHEN",
                    help='e.g. "2 hours", "30 minutes"')
    ap.add_argument("--consent", nargs="*", metavar="K=V",
                    help="interrupt.reminder=on  quiet=22:00-08:00")
    args = ap.parse_args()

    if args.consent is not None:
        for pair in args.consent:
            if "=" in pair:
                k, v = pair.split("=", 1)
                set_consent(k.strip(), v.strip())
        for k, v in sorted(consent().items()):
            print(f"  {k:<22} {v}")
        return 0

    if args.remind:
        when = read_time(f"in {args.in_}" if args.in_ else args.remind)
        if when is None:
            print("  I could not read a time from that, so I have not taken "
                  "it on.\n  Try: --in \"2 hours\"")
            return 1
        say_later(args.remind, kind=REMINDER, due_at=when)
        print(f"  will remind you: {args.remind}")
        print(f"  at {datetime.fromtimestamp(when):%a %H:%M}")
        return 0

    print(f"  {sense_line()}")
    c = consent()
    print(f"  interrupt: reminders {c['interrupt.reminder']}, "
          f"findings {c['interrupt.finding']}; quiet {c['quiet']}")
    items = pending()
    if not items:
        print("  nothing waiting")
        return 0
    for i in items:
        due = (f"{datetime.fromtimestamp(i['due_at']):%a %H:%M}"
               if i["due_at"] else "next time we talk")
        print(f"  [{i['kind']}] {due} — {i['body'][:60]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
