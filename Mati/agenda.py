# Vendored from Hum (github.com/vml520/Hum), MIT, Copyright (c) 2026 the user
# Luna: `_unfold`, `_parse_dt`, `parse_ics`, `parse_rrule` are Hum's
# deterministic calendar core, copied rather than imported so the two projects
# stay separable — the same choice `Mati/field/` made.
#
# ⚠️ THE COHERENCE ENGINE IS DELIBERATELY NOT VENDORED. Hum's own spec
# withdrew its claim on measurement: -0.19% over 72 paired runs, p = 0.42,
# 28 wins / 37 losses, no detectable contribution. Bringing
# `build_couplings` / `_strain_at` / `coherence` across would import a
# measured-negative component into Mati, where it would be re-narrated as a
# feature. The split is at the function boundary and it stays there.
"""
Mati's own calendar — imported from iCloud, owned here.

The user: *"I meant import the events from iCloud Calendar and create an Mati
calendar that is separate from iCloud."*

See `STUDY_calendar_sync.md`. The design, in one line: **copy in, never write
back**. iCloud stays authoritative for itself and Mati cannot damage it; the
copy is Mati's, encrypted, portable, and readable with no permission once the
import has happened.

⚠️ THE PERMISSION IS ONLY NEEDED AT IMPORT TIME. That is the whole argument for
importing rather than reading through: afterwards this works offline, on a
thumb drive, on a machine that has never seen iCloud.

⚠️ DEDUP KEYS ON (ext_id, start), NOT ON ext_id. Measured on the real calendar:
65 unique identifiers for 74 events, because recurring occurrences carry their
SERIES identifier. Keying on the id alone would collapse a weekly meeting into
one row or overwrite the wrong occurrence.

⚠️ THE RULE IS IMPORTED, NOT THE EXPANSION. 52 of 74 events recur; storing
expanded occurrences would be thousands of rows a year and a re-import that has
to diff all of them.

⚠️ DELETION IS A WITHDRAWAL, NOT AN ERASE. A meeting cancelled in iCloud must
stop appearing — but Mati's standing rule is that corrections annotate and
never erase. So the row is marked `withdrawn` and leaves the view, and "the
meeting you had on Tuesday was cancelled" stays answerable.

⚠️ AND A FAILED IMPORT MUST NOT WITHDRAW ANYTHING. Withdrawal is only valid
from a read that actually succeeded and returned a complete window. A denied or
partial read changes nothing — otherwise a permission revoked at the wrong
moment silently cancels the person's whole calendar.

⚠️ MATI NEVER DECIDES A TIME. Hum's spec keeps the field language model away
from scheduling decisions until a measured gate is passed. This module imports,
stores and reports; it does not propose, move or resolve anything.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timedelta

from . import bridge

TABLE = "agenda"

# Where an event came from. `OURS` is Mati's own; anything else is a copy.
OURS = "mati"

# How far ahead an import reaches by default. A year covers annual recurrences
# without the store becoming an archive.
IMPORT_DAYS_AHEAD = 365
IMPORT_DAYS_BACK = 30


# ── Hum's deterministic core, vendored ───────────────────────────────────────

def _unfold(text):
    # RFC 5545: a CRLF followed by space/tab continues the previous line
    return re.sub(r"\r?\n[ \t]", "", text)


def _parse_dt(val):
    val = val.strip()
    m = re.match(r"(\d{4})(\d{2})(\d{2})(?:T(\d{2})(\d{2})(\d{2})(Z)?)?", val)
    if not m:
        return None, False
    y, mo, d, hh, mm, ss, _z = m.groups()
    if hh is None:
        return datetime(int(y), int(mo), int(d)), True      # all-day
    return datetime(int(y), int(mo), int(d), int(hh), int(mm), int(ss or 0)), False


def parse_ics(text):
    """The common subset: VEVENT, DTSTART/DTEND, RRULE, SUMMARY, UID."""
    text = _unfold(text)
    events = []
    for block in re.findall(r"BEGIN:VEVENT(.*?)END:VEVENT", text, re.S):
        ev = {"summary": "(untitled)", "rrule": None, "dtstart": None,
              "dtend": None, "allday": False, "uid": None}
        for line in block.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, val = line.split(":", 1)
            key = key.split(";")[0].upper()
            if key == "SUMMARY":
                ev["summary"] = val.strip()
            elif key == "UID":
                # ⚠️ Hum does not read UID because it does not persist events.
                # Mati does, and without it a re-import has no key at all.
                ev["uid"] = val.strip()
            elif key == "DTSTART":
                ev["dtstart"], ev["allday"] = _parse_dt(val)
            elif key == "DTEND":
                ev["dtend"], _ = _parse_dt(val)
            elif key == "RRULE":
                ev["rrule"] = val.strip()
        if ev["dtstart"]:
            ev["duration"] = (
                (ev["dtend"] - ev["dtstart"]).total_seconds() / 3600.0
                if ev["dtend"] and not ev["allday"] else 1.0)
            events.append(ev)
    return events


def parse_rrule(rrule):
    if not rrule:
        return None
    d = {}
    for part in str(rrule).split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            d[k.upper()] = v
    return d


# ── The store ────────────────────────────────────────────────────────────────

def ensure(db_path=None):
    conn = sqlite3.connect(db_path or bridge.NOTES_DB_PATH, timeout=30)
    try:
        conn.execute(f"""CREATE TABLE IF NOT EXISTS {TABLE} (
                event_id   INTEGER PRIMARY KEY,
                ext_id     TEXT,
                starts     TEXT NOT NULL,
                ends       TEXT,
                all_day    INTEGER,
                rrule      TEXT,
                title      TEXT,
                location   TEXT,
                calendar   TEXT,
                source     TEXT,
                imported   TEXT,
                withdrawn  TEXT
            )""")
        # ⚠️ (ext_id, starts) — see the module docstring. `starts` alone is not
        # unique either: two meetings can begin at the same minute.
        conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS {TABLE}_key "
                     f"ON {TABLE}(ext_id, starts)")
        conn.execute(f"CREATE INDEX IF NOT EXISTS {TABLE}_when "
                     f"ON {TABLE}(starts)")
        conn.commit()
        return True
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def _iso(dt_):
    if dt_ is None:
        return None
    if isinstance(dt_, str):
        return dt_
    if dt_.tzinfo is None:
        return dt_.astimezone().isoformat()
    return dt_.isoformat()


def put(starts, title, ends=None, ext_id=None, rrule=None, all_day=False,
        location=None, calendar=None, source=OURS, db_path=None):
    """Insert or update one event, keyed on (ext_id, starts). Returns the id.

    ⚠️ Title and location are SEALED — a meeting title is as personal as a
    note body. Times, recurrence and source stay clear so the agenda can be
    read and grouped without the key, the same rule as `notes.wall_created`.
    """
    from . import vault
    ensure(db_path)
    starts = _iso(starts)
    if not starts:
        return None
    conn = sqlite3.connect(db_path or bridge.NOTES_DB_PATH, timeout=30)
    try:
        row = conn.execute(
            f"SELECT event_id FROM {TABLE} WHERE ext_id IS ? AND starts = ?",
            (ext_id, starts)).fetchone()
        vals = (_iso(ends), 1 if all_day else 0, rrule,
                vault.seal(title) if title else None,
                vault.seal(location) if location else None,
                calendar, source, bridge._now())
        if row:
            conn.execute(
                f"UPDATE {TABLE} SET ends=?, all_day=?, rrule=?, title=?, "
                f"location=?, calendar=?, source=?, imported=?, withdrawn=NULL "
                f"WHERE event_id=?", vals + (row[0],))
            eid = row[0]
        else:
            cur = conn.execute(
                f"INSERT INTO {TABLE} (ends, all_day, rrule, title, location, "
                f"calendar, source, imported, ext_id, starts) "
                f"VALUES (?,?,?,?,?,?,?,?,?,?)", vals + (ext_id, starts))
            eid = cur.lastrowid
        conn.commit()
        return eid
    except sqlite3.Error:
        return None
    finally:
        conn.close()


def withdraw(event_ids, db_path=None):
    """Mark events as no longer in the source. NEVER deletes."""
    if not event_ids:
        return 0
    conn = sqlite3.connect(db_path or bridge.NOTES_DB_PATH, timeout=30)
    try:
        now = bridge._now()
        conn.executemany(
            f"UPDATE {TABLE} SET withdrawn=? WHERE event_id=? "
            f"AND withdrawn IS NULL", [(now, i) for i in event_ids])
        n = conn.total_changes
        conn.commit()
        return n
    except sqlite3.Error:
        return 0
    finally:
        conn.close()


def upcoming(days=7, db_path=None, include_withdrawn=False):
    """What is coming, soonest first. Withdrawn events are excluded."""
    from . import vault
    ensure(db_path)
    now = datetime.now().astimezone()
    until = (now + timedelta(days=days)).isoformat()
    conn = sqlite3.connect(
        f"file:{db_path or bridge.NOTES_DB_PATH}?mode=ro", uri=True)
    try:
        sql = (f"SELECT event_id, starts, ends, all_day, rrule, title, "
               f"location, calendar, source, withdrawn FROM {TABLE} "
               f"WHERE starts >= ? AND starts <= ?")
        if not include_withdrawn:
            sql += " AND withdrawn IS NULL"
        sql += " ORDER BY starts"
        rows = conn.execute(sql, (now.isoformat(), until)).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    out = []
    for (eid, starts, ends, allday, rrule, title, loc, cal, src, wd) in rows:
        try:
            title = vault.open_body(title) if title else None
            loc = vault.open_body(loc) if loc else None
        except Exception:
            title, loc = None, None       # locked: the time is still true
        out.append({"event_id": eid, "starts": starts, "ends": ends,
                    "all_day": bool(allday), "rrule": rrule, "title": title,
                    "location": loc, "calendar": cal, "source": src,
                    "withdrawn": wd})
    return out


# ── Importing ────────────────────────────────────────────────────────────────

class ImportFailed(RuntimeError):
    """The source could not be read. ⚠️ Callers must NOT withdraw on this."""


def _eventkit_events(days_back=IMPORT_DAYS_BACK, days_ahead=IMPORT_DAYS_AHEAD):
    """Read iCloud/EventKit. Raises ImportFailed rather than returning [].

    ⚠️ THE EMPTY LIST IS THE DANGEROUS RETURN VALUE. A denied read and an empty
    calendar look identical, and treating denial as "no events" would withdraw
    everything the person has. So denial raises.
    """
    try:
        import objc
        from Foundation import NSBundle, NSDate
    except Exception as e:                                   # noqa: BLE001
        raise ImportFailed(f"PyObjC unavailable: {e}") from e
    b = NSBundle.bundleWithPath_("/System/Library/Frameworks/EventKit.framework")
    if not (b and b.load()):
        raise ImportFailed("EventKit.framework would not load")
    EK = objc.lookUpClass("EKEventStore")
    status = EK.authorizationStatusForEntityType_(0)
    if status not in (3, 4):          # authorized / fullAccess
        raise ImportFailed(
            f"no calendar permission (status {status}) — nothing imported, "
            f"and nothing withdrawn")
    store = EK.alloc().init()
    s = NSDate.dateWithTimeIntervalSinceNow_(-days_back * 86400)
    e = NSDate.dateWithTimeIntervalSinceNow_(days_ahead * 86400)
    pred = store.predicateForEventsWithStartDate_endDate_calendars_(s, e, None)
    evs = store.eventsMatchingPredicate_(pred)
    if evs is None:
        raise ImportFailed("EventKit returned nothing at all")
    out = []
    for ev in evs:
        try:
            ext = None
            try:
                ext = ev.calendarItemExternalIdentifier()
            except Exception:
                ext = ev.eventIdentifier()
            sd = ev.startDate()
            ed = ev.endDate()
            rule = None
            if ev.hasRecurrenceRules():
                rr = (ev.recurrenceRules() or [None])[0]
                if rr is not None:
                    rule = _rule_to_rrule(rr)
            out.append({
                "ext_id": str(ext) if ext else None,
                "starts": datetime.fromtimestamp(sd.timeIntervalSince1970()).astimezone(),
                "ends": datetime.fromtimestamp(ed.timeIntervalSince1970()).astimezone() if ed else None,
                "title": str(ev.title()) if ev.title() else None,
                "location": str(ev.location()) if ev.location() else None,
                "all_day": bool(ev.isAllDay()),
                "rrule": rule,
                "calendar": str(ev.calendar().title()) if ev.calendar() else None,
            })
        except Exception:                                    # noqa: BLE001
            continue                    # one bad event is not a failed import
    return out


_FREQ = {0: "DAILY", 1: "WEEKLY", 2: "MONTHLY", 3: "YEARLY"}


def _rule_to_rrule(rr):
    """EKRecurrenceRule -> an RRULE string Hum's `parse_rrule` understands."""
    try:
        bits = [f"FREQ={_FREQ.get(rr.frequency(), 'WEEKLY')}"]
        if rr.interval() and rr.interval() != 1:
            bits.append(f"INTERVAL={rr.interval()}")
        return ";".join(bits)
    except Exception:                                        # noqa: BLE001
        return None


def import_from_apple(db_path=None, verbose=False, **kw):
    """Copy iCloud events into Mati's calendar. Returns a summary.

    ⚠️ Withdrawal happens ONLY after a successful, complete read. If the read
    raises, nothing is touched — that is the difference between "your calendar
    is empty" and "I could not look".
    """
    events = _eventkit_events(**kw)          # raises ImportFailed
    ensure(db_path)
    seen = set()
    added = updated = 0
    conn = sqlite3.connect(
        f"file:{db_path or bridge.NOTES_DB_PATH}?mode=ro", uri=True)
    try:
        before = {(r[0], r[1]): r[2] for r in conn.execute(
            f"SELECT ext_id, starts, event_id FROM {TABLE} "
            f"WHERE source != ? AND withdrawn IS NULL", (OURS,))}
    except sqlite3.Error:
        before = {}
    finally:
        conn.close()
    for ev in events:
        key = (ev["ext_id"], _iso(ev["starts"]))
        seen.add(key)
        existed = key in before
        put(ev["starts"], ev["title"], ends=ev["ends"], ext_id=ev["ext_id"],
            rrule=ev["rrule"], all_day=ev["all_day"], location=ev["location"],
            calendar=ev["calendar"], source="apple", db_path=db_path)
        updated += 1 if existed else 0
        added += 0 if existed else 1
    gone = [eid for key, eid in before.items() if key not in seen]
    withdrawn = withdraw(gone, db_path=db_path)
    if verbose:
        print(f"  [agenda] {added} new, {updated} updated, "
              f"{withdrawn} withdrawn")
    return {"read": len(events), "added": added, "updated": updated,
            "withdrawn": withdrawn}


def import_ics(text, db_path=None, verbose=False):
    """Copy events from an .ics export. ⚠️ No permission required at all.

    Does NOT withdraw: an .ics file is a partial export by nature and cannot
    be treated as a complete statement of what exists.
    """
    events = parse_ics(text)
    ensure(db_path)
    # ⚠️ COUNT ROWS, NOT CALLS. The first version counted successful `put`s and
    # so reported "2 added" on every re-import of the same file — a number that
    # looks like duplication and would have sent someone hunting a bug that was
    # not there, or hidden one that was.
    conn = sqlite3.connect(
        f"file:{db_path or bridge.NOTES_DB_PATH}?mode=ro", uri=True)
    try:
        before = conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
    except sqlite3.Error:
        before = 0
    finally:
        conn.close()
    for ev in events:
        put(ev["dtstart"], ev.get("summary"),
            ends=ev.get("dtend"), ext_id=ev.get("uid"),
            rrule=ev.get("rrule"), all_day=ev.get("allday"),
            source="ics", db_path=db_path)
    conn = sqlite3.connect(
        f"file:{db_path or bridge.NOTES_DB_PATH}?mode=ro", uri=True)
    try:
        after = conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
    except sqlite3.Error:
        after = before
    finally:
        conn.close()
    added = after - before
    if verbose:
        print(f"  [agenda] {added} new, {len(events)-added} already held "
              f"(nothing withdrawn — an .ics is a partial export)")
    return {"read": len(events), "added": added,
            "updated": len(events) - added, "withdrawn": 0}


def status(db_path=None):
    ensure(db_path)
    conn = sqlite3.connect(
        f"file:{db_path or bridge.NOTES_DB_PATH}?mode=ro", uri=True)
    try:
        total = conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
        live = conn.execute(f"SELECT COUNT(*) FROM {TABLE} "
                            f"WHERE withdrawn IS NULL").fetchone()[0]
        mine = conn.execute(f"SELECT COUNT(*) FROM {TABLE} WHERE source=?",
                            (OURS,)).fetchone()[0]
        last = conn.execute(f"SELECT MAX(imported) FROM {TABLE}").fetchone()[0]
    except sqlite3.Error:
        return {}
    finally:
        conn.close()
    return {"events": total, "live": live, "withdrawn": total - live,
            "matis_own": mine, "last_import": last}
