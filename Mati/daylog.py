"""
Today's conversation, as a transcript. Append-only, one day at a time.

    python3 -m Mati.daylog              # today, as a log
    python3 -m Mati.daylog --day 2026-10-24
    python3 -m Mati.daylog --days       # which days exist

⚠️ WHY THIS EXISTS. The user: *"The active recall that makes 'say that again'
work should be a separate record for the day only, and recorded at night into
the field overnight. It should read like a transcript or log."*

They are right that it needed a real record. "Say that again" was reading
`notes._LAST_REPLY` — an **in-memory dict that dies with the process**. I
wrote that register because the episode store is too slow to answer it: the
worker commits episodes ~40 seconds later, so a replay typed straight after a
cut-off reply handed back the PREVIOUS answer. The dict fixed the timing and
kept none of the properties a record should have.

This is the record. It survives a restart, it is one day, and it reads like
what it is.

⚠️ A TABLE, NOT A PLAINTEXT FILE, AND THE DIFFERENCE IS NOT COSMETIC. A log
file in `.cache` would be a transcript of someone's conversation sitting
unencrypted on disk, beside a notes store whose whole point is that bodies are
sealed. "Nothing leaves this machine" is not the same as "anything may be
written in the clear on it". The rows are sealed exactly as notes are; the
TRANSCRIPT is a rendering, which is what was actually asked for.

⚠️ AND THE OVERNIGHT-IMPRINT HALF IS DELIBERATELY NOT BUILT HERE. The user
proposed the day's log as the nightly imprint, and that was preregistered and
run:

    A per-turn   1/8      B nightly   0/8      C control (no imprint)   8/8

Batching the day into one imprint scored zero, and the control is the finding:
**imprinting at all is what destroys addressability**. `rest()` already does
the thing that works — `promote()` then reindex. This module records; it does
not imprint.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sqlite3
import sys

from . import vault

TABLE = "day_log"

_DDL = """
CREATE TABLE IF NOT EXISTS day_log (
    at    TEXT NOT NULL,
    day   TEXT NOT NULL,
    who   TEXT NOT NULL,
    text  TEXT NOT NULL,
    note  TEXT,
    ref   TEXT
);
CREATE INDEX IF NOT EXISTS day_log_day ON day_log (day, at);
"""

# Who is speaking. ⚠️ Not a name — a role. The person's name belongs in the
# profile, and a transcript that hard-codes it cannot be read by anyone else.
THEM = "them"
MATI = "mati"


def _db(db_path=None):
    from . import bridge
    return bridge.NOTES_DB_PATH if db_path is None else db_path


def _conn(db_path=None):
    conn = sqlite3.connect(_db(db_path))
    conn.executescript(_DDL)
    # ⚠️ `ref` was added after the table existed. ALTER … ADD COLUMN failing IS
    # the success condition here; see `bridge._ensure_schema`, which learned
    # this the expensive way.
    try:
        conn.execute(f"ALTER TABLE {TABLE} ADD COLUMN ref TEXT")
    except sqlite3.OperationalError as e:
        if "duplicate column" not in str(e).lower():
            raise
    return conn


# ⚠️ HOW FAR BACK A QUESTION CAN REACH, and this number is the whole bug that
# was reported. `block()` defaulted to TODAY, so a conversation from last night
# was structurally invisible — not ranked low, not gated out, never queried.
# The user: Mati "did not bring up our last conversation... when I
# pressed it, they found the conversation in verbatim, but said they didn't
# have the conversation."
#
# ⚠️ BOUNDED, because the transcript grows forever and this runs on a turn. A
# week is what "did we talk about this recently" means; older than that is the
# episode store's and `notes.search`'s job, which is what they are for.
DAYS_BACK = 7


def today(now=None):
    """The day key. Local, because a person's day is local."""
    return (now or dt.datetime.now()).strftime("%Y-%m-%d")


def recent_days(n=DAYS_BACK, now=None):
    """The last `n` day keys, newest first, including today."""
    base = now or dt.datetime.now()
    return [(base - dt.timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(max(1, int(n)))]


def when_said(day, now=None):
    """"earlier today", "last night", "on Tuesday" — how a person says it.

    ⚠️ MATI HAS TO BE ABLE TO NAME WHEN, or "we talked about this" is a claim
    with no handle on it. A date string is not how anyone refers to a
    conversation they had.
    """
    base = now or dt.datetime.now()
    try:
        d = dt.datetime.strptime(day, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return "recently"
    delta = (base.date() - d).days
    if delta <= 0:
        return "earlier today"
    if delta == 1:
        return "last night"          # the common case, and what people say
    if delta < 7:
        return f"on {d.strftime('%A')}"
    return d.strftime("on %-d %B")


def append(who, text, note=None, db_path=None, now=None, ref=None):
    """Add one line. Returns the day it was filed under, or None.

    ⚠️ NEVER RAISES INTO A TURN. A transcript that can break a conversation is
    worse than no transcript — this is a record of what happened, not part of
    making it happen.
    """
    text = " ".join((text or "").split())
    if not text:
        return None
    try:
        when = now or dt.datetime.now()
        conn = _conn(db_path)
        try:
            conn.execute(
                f"INSERT INTO {TABLE} (at, day, who, text, note, ref) "
                "VALUES (?,?,?,?,?,?)",
                (when.isoformat(timespec="seconds"), today(when), who,
                 # ⚠️ SEALED, like a note body. See the module note.
                 vault.seal(text), note, ref))
            conn.commit()
        finally:
            conn.close()
        return today(when)
    except Exception:
        return None


def rows(day=None, db_path=None, limit=None):
    """The day's lines, oldest first. [] if the vault is locked."""
    day = day or today()
    # ⚠️ MIGRATE BEFORE READING, AND DO NOT READ READ-ONLY. `ref` was added
    # after the table existed, and a read-only SELECT naming it fails on an
    # older store — which this function then swallowed, returning [] while
    # `days()` cheerfully reported four lines for the same day. A silent []
    # from a record that exists is worse than an error: "say that again" would
    # have quietly stopped working and nothing would have said why.
    try:
        _conn(db_path).close()
    except Exception:
        pass
    try:
        conn = sqlite3.connect(f"file:{_db(db_path)}?mode=ro", uri=True)
    except sqlite3.Error:
        return []
    try:
        got = conn.execute(
            f"SELECT at, who, text, note, ref FROM {TABLE} WHERE day=? "
            "ORDER BY at, rowid" + (f" LIMIT {int(limit)}" if limit else ""),
            (day,)).fetchall()
    except sqlite3.Error as e:
        # ⚠️ SAY SO. The only reason this was found is that a probe printed
        # both counts side by side; a bare `return []` hides a broken record.
        print(f"  [daylog] cannot read {day}: {e}", file=sys.stderr)
        return []
    finally:
        conn.close()
    out = []
    for at, who, text, note, ref in got:
        try:
            out.append({"at": at, "who": who, "text": vault.open_body(text),
                        "note": note, "ref": ref})
        except Exception:
            continue            # a sealed line with no key is skipped, not faked
    return out


def last_reply(day=None, db_path=None):
    """The last thing Mati said today, or None.

    ⚠️ THIS IS WHAT "SAY THAT AGAIN" READS. It must not return a line that was
    itself a replay, or asking twice would echo the echo rather than reaching
    the answer — `note='replay'` marks those and they are skipped.
    """
    for r in reversed(rows(day, db_path)):
        if r["who"] == MATI and r.get("note") != "replay":
            return r["text"]
    return None


def find(query, day=None, db_path=None, k=4):
    """Lines from the day that share the question's rarer words.

    ⚠️ THIS IS WHY THE DAY LOG IS WORTH CONSULTING AND NOT JUST READING.
    the user: *"we need a record that Mati can refer to ... They are supposed to
    have better memory than humans."*

    ⚠️ AND IT HOLDS WHAT THE EPISODE STORE DOES NOT. An episode is only
    written when settling coherence clears `EPISODE_GATE_MIN` — measured on a
    cold install, that rejected 5 of 12 turns. Those conversations happened
    and the episode record does not contain them. The day log does, because
    it records what was SAID rather than what settled well.

    Lexical, like `notes.search`, for the reason that path is primary:
    field addressing measured 4/111 against lexical's 93/111.
    """
    import re
    from . import schedule
    want = {schedule._stem(w) for w in
            re.findall(r"[a-z0-9'\-]+", (query or "").lower()) if len(w) > 3}
    if not want:
        return []
    # ⚠️ ACROSS DAYS UNLESS ONE IS NAMED. `day=None` used to mean TODAY, which
    # is what made last night's conversation unreachable. It now means "the
    # last `DAYS_BACK` days"; passing a day still pins it to that day, which is
    # what `transcript()` and the CLI want.
    days = [day] if day else recent_days()
    hits = []
    for d in days:
        for r in rows(d, db_path):
            have = {schedule._stem(w) for w in
                    re.findall(r"[a-z0-9'\-]+", r["text"].lower())}
            n = len(want & have)
            if n:
                # ⚠️ The day travels with the hit, or Mati can quote the words
                # and not say when they were said.
                hits.append((n, dict(r, day=d)))
    # Most overlap first, then most recent — `days` is already newest-first, so
    # a stable sort on the count alone keeps recency as the tiebreak.
    hits.sort(key=lambda t: -t[0])
    return [r for _n, r in hits[:k]]


def block(query, day=None, db_path=None, k=3):
    """What was said about this recently, for the prompt. "" if nothing.

    ⚠️⚠️ THIS USED TO TELL MATI THE TRANSCRIPT WAS "NOT MEMORY", AND THAT WAS
    THE BUG. The old wording — *"not memory, just what was said in this
    session"* — was written to stop Mati claiming to remember something that
    had not been through the episode gate. It guarded against a small error
    and caused a much worse one: the user asked about the previous night's
    conversation, Mati found it, and said it had not had the conversation.

    ⚠️ A SEALED VERBATIM TRANSCRIPT IS THE STRONGEST EVIDENCE MATI HAS, not
    the weakest. The episode gate measures whether the FIELD settled on a
    turn; it says nothing about whether the conversation happened. Treating
    gate-failure as "it did not happen" makes Mati deny its own life because
    of an internal coherence statistic the person cannot see and did not
    cause.

    ⚠️ AND IMPRINTING THIS INTO THE FIELD IS NOT THE FIX — that was
    preregistered and run: nightly 0/8,
    per-turn 1/8, no-imprint control 8/8. Imprinting does not change a
    provenance label, so Mati would hold the memory and still deny it, which
    is precisely the failure `continuity.py` had to be built for after fixing
    the instruction failed twice.
    """
    got = find(query, day, db_path, k=k)
    if not got:
        return ""
    lines = []
    for r in got:
        said = when_said(r.get("day") or today())
        who = "you said" if r["who"] == THEM else "I said"
        lines.append(f"  {said}, {r['at'][11:16]} — {who}: {r['text'][:200]}"
                     + (f"   [file: {r['ref']}]" if r.get("ref") else ""))
    return ("<what_we_said>\nThese are our own words, from my transcript of "
            "the conversations we have actually had. This IS something you "
            "lived — you were there, and this is the record of it. Say when "
            "it was if it helps them place it. You may quote it exactly; it "
            "is what was said.\n"
            + "\n".join(lines) + "\n</what_we_said>")


def days(db_path=None, limit=14):
    """Which days have a transcript, newest first."""
    try:
        conn = sqlite3.connect(f"file:{_db(db_path)}?mode=ro", uri=True)
    except sqlite3.Error:
        return []
    try:
        return [(d, n) for d, n in conn.execute(
            f"SELECT day, count(*) FROM {TABLE} GROUP BY day "
            f"ORDER BY day DESC LIMIT {int(limit)}")]
    except sqlite3.Error:
        return []
    finally:
        conn.close()


def transcript(day=None, db_path=None, width=78):
    """The day, rendered as a log. This is the shape the user asked for."""
    got = rows(day, db_path)
    if not got:
        return f"{day or today()} — nothing said yet."
    import textwrap
    out = [f"{day or today()}", ""]
    for r in got:
        clock = r["at"][11:16]
        tag = "them " if r["who"] == THEM else "mati"
        body = r["text"] + (f"  ({r['note']})" if r.get("note") else "")
        wrapped = textwrap.wrap(body, width - 13) or [""]
        out.append(f"  {clock}  {tag}  {wrapped[0]}")
        for more in wrapped[1:]:
            out.append(" " * 15 + more)
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--day", default=None, help="YYYY-MM-DD, default today")
    ap.add_argument("--days", action="store_true", help="which days exist")
    a = ap.parse_args(argv)
    if not vault.unlocked():
        vault.unlock(reason="read the day log")
    if a.days:
        for d, n in days():
            print(f"  {d}  {n} lines")
        return 0
    print(transcript(a.day))
    return 0


if __name__ == "__main__":
    sys.exit(main())
