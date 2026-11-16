"""
What Mati did, and when. A log, not a gauge.

⚠️ A GAUGE CANNOT ANSWER A QUESTION ABOUT YESTERDAY. Seeing when Mati sleeps
and wakes, and when it studies, reads and dreams, needs a record. `Telemetry`
holds the LAST KNOWN state and fans it out to live subscribers — a gauge,
which is exactly right for a dial and useless for a question about yesterday.
Every transition is visible for the instant it happens and then gone.

And a monitor that cannot answer "when did this start" cannot tell you that it
started eleven days ago and never stopped.

⚠️ SOME OF IT IS RECOVERABLE AND IS RECOVERED. `study_queue.studied`, the
reading memories and the episodes all carry timestamps, so the history does not
begin empty — `backfill()` reads them once. Sleeping, waking and dreaming are
written down nowhere else, so those begin when the log does.

Deliberately small: an event name, a moment, and a short detail. Not a second
copy of what happened — `notes` already holds that, encrypted — just the shape
of the day.
"""

from __future__ import annotations

import datetime as _dt
import sqlite3

from . import bridge

TABLE = "activity"


def local(at):
    """A stored stamp as a LOCAL-time datetime, or None if it will not parse.

    ⚠️ STORED IN UTC, READ BY A PERSON IN A TIMEZONE. Times go into the DB as
    `bridge._now()`, which is UTC with an explicit `+00:00` — correct, and
    unambiguous, and not what anybody lives in. The history display sliced the
    raw string (`at[:10]` for the day, `at[11:16]` for the clock), so an event
    at 23:53 on the 31st was shown as 06:53 on the 1st: wrong hour, wrong day,
    and wrong day is the one that misleads, because it moves things across the
    date heading they are grouped under.

    Storage is right and stays untouched — a UTC column with an offset can
    always be rendered; a local column can never be un-rendered. Only the
    reading changes.

    ⚠️ Naive stamps are treated as UTC, not as local. Older rows predate the
    offset, and guessing local for those would silently shift them by the
    offset while looking perfectly reasonable.
    """
    if not at:
        return None
    try:
        t = _dt.datetime.fromisoformat(str(at))
    except (ValueError, TypeError):
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=_dt.timezone.utc)
    return t.astimezone()


def local_parts(at):
    """(YYYY-MM-DD, HH:MM) in local time — what a history line needs.

    ⚠️ An unparseable stamp says so rather than being sliced. The raw slices
    turned "not a timestamp" into the time "tamp", which is the failure this
    whole fix is about in miniature: a display that would rather show something
    wrong than admit it does not know.
    """
    t = local(at)
    if t is None:
        return ("unknown", "--:--")
    return (t.strftime("%Y-%m-%d"), t.strftime("%H:%M"))

# The events worth a line. Anything not here is noise on a page meant to be
# read at a glance.
WAKE, SLEEP, REST, STIR = "wake", "sleep", "rest", "stir"
DREAM, STUDY, READ = "dream", "study", "read"
PREDICT, RAPPORT = "predict", "rapport"
# Cut off mid-reply by the person typing. A conversational event, not a
# fault — see interruption.py on why the word count is not a failure score.
INTERRUPTED = "interrupted"
# Mati thinking to itself about the day, during sleep. See reflect.py.
REFLECT = "reflect"


def ensure(db_path=None):
    conn = sqlite3.connect(db_path or bridge.NOTES_DB_PATH, timeout=30)
    try:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {TABLE} (
                    at     TEXT,
                    event  TEXT,
                    detail TEXT
                )""")
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS {TABLE}_at ON {TABLE}(at)")
        conn.commit()
        return True
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def log(event, detail=None, db_path=None, at=None):
    """One line. Never raises — a diary must not be able to stop the day."""
    try:
        ensure(db_path)
        conn = sqlite3.connect(db_path or bridge.NOTES_DB_PATH, timeout=30)
        try:
            conn.execute(f"INSERT INTO {TABLE} (at, event, detail) "
                         f"VALUES (?,?,?)",
                         (at or bridge._now(), event,
                          str(detail)[:120] if detail else None))
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass


def backfill(db_path=None):
    """Recover the history that other tables already carry. Idempotent.

    ⚠️ Only what is genuinely recorded elsewhere. Sleeping, waking and dreaming
    were never written down by anything, so they are absent before today and
    the log says nothing about them rather than inferring them from gaps.
    """
    ensure(db_path)
    conn = sqlite3.connect(db_path or bridge.NOTES_DB_PATH, timeout=30)
    added = 0
    try:
        have = {r[0] for r in conn.execute(
            f"SELECT at FROM {TABLE} WHERE event IN (?,?)", (STUDY, READ))}
        rows = conn.execute(
            "SELECT studied, origin, COUNT(*) FROM study_queue "
            "WHERE studied IS NOT NULL GROUP BY substr(studied,1,13), origin"
        ).fetchall()
        for at, origin, n in rows:
            if at in have:
                continue
            conn.execute(f"INSERT INTO {TABLE} (at, event, detail) VALUES (?,?,?)",
                         (at, STUDY, f"{n} passages of {origin}"))
            added += 1
        conn.commit()
    except sqlite3.Error:
        pass
    finally:
        conn.close()
    return added


def recent(limit=60, hours=None, db_path=None):
    """The last things Mati did, newest first."""
    ensure(db_path)
    conn = sqlite3.connect(
        f"file:{db_path or bridge.NOTES_DB_PATH}?mode=ro", uri=True)
    try:
        sql = f"SELECT at, event, detail FROM {TABLE}"
        args = ()
        if hours:
            import datetime as _dt
            since = (_dt.datetime.now(_dt.timezone.utc)
                     - _dt.timedelta(hours=hours)).isoformat()
            sql += " WHERE at >= ?"
            args = (since,)
        sql += " ORDER BY at DESC LIMIT ?"
        return conn.execute(sql, args + (int(limit),)).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()


def summary(days=7, db_path=None):
    """How often each thing happened lately."""
    ensure(db_path)
    import datetime as _dt
    since = (_dt.datetime.now(_dt.timezone.utc)
             - _dt.timedelta(days=days)).isoformat()
    conn = sqlite3.connect(
        f"file:{db_path or bridge.NOTES_DB_PATH}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            f"SELECT event, COUNT(*), MAX(at) FROM {TABLE} WHERE at >= ? "
            f"GROUP BY event ORDER BY 2 DESC", (since,)).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    return [{"event": e, "times": n, "last": last} for e, n, last in rows]
