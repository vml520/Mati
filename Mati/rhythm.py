"""
When this person is actually here, learned from the record.

⚠️ IDLE MEANS GONE, NOT PAUSED. Mati is not really idle unless it has not been
spoken to in more than 45 minutes at least, depending on the person. Measured
over 333 real gaps between consecutive turns:

    median  3.0 min    p90  10.6 min    p95  32.8 min    p99  110 min

A 45-SECOND threshold is shorter than 96% of the gaps in a normal
conversation, so idle work would start after almost every turn; a 10-minute
one sits inside p90, so the expensive path would fire during ordinary pauses.
Either way idle would be on almost permanently.

⚠️ AND "DEPENDING ON THE PERSON" IS THE REAL POINT. A fixed number is wrong for
everybody except whoever it was tuned for. The pause that means "gone" belongs
to the person, and it is in their own record — so it is measured rather than
chosen, with 45 minutes as the floor.

⚠️ WHAT CANNOT BE LEARNED YET, AND IS NOT PRETENDED OTHERWISE. A daily schedule
needs a daily pattern, and a real record may not have one: the first turn of
the day can have a **standard deviation of 8.5 hours**, ranging from just
after midnight to late evening. Waking at the median would be wrong most days.
So the schedule is learned only when the spread is tight enough to mean
something, and until then the default stands and says so.
"""

from __future__ import annotations

import collections
import sqlite3
import statistics
from datetime import datetime

# The floor. Below this, a pause is a pause.
FLOOR_IDLE_S = 45 * 60

# Where in the person's own distribution "they have gone" sits. p95 by
# construction: the pause longer than nineteen out of twenty of their pauses.
IDLE_QUANTILE = 0.95

# A learned wake time is adopted only if the day actually starts at a
# consistent hour. 2.0 is generous — it still means two thirds of days begin
# within four hours of each other.
MAX_SPREAD_H = 2.0

# Days with fewer turns than this are not days, they are a passing look.
MIN_TURNS_PER_DAY = 5
MIN_DAYS = 10


def _turn_times(db_path=None):
    from . import bridge
    conn = sqlite3.connect(
        f"file:{db_path or bridge.NOTES_DB_PATH}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT wall_created FROM notes WHERE source='episode' "
            "AND wall_created IS NOT NULL ORDER BY wall_created").fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    out = []
    for (a,) in rows:
        try:
            out.append(datetime.fromisoformat(a).astimezone())
        except (ValueError, TypeError):
            continue
    return out


def idle_after(db_path=None):
    """How long a silence means this person has gone. Seconds.

    Never below `FLOOR_IDLE_S`. A quick correspondent does not get a system
    that decides they have left because they paused to think.
    """
    ts = [t.timestamp() for t in _turn_times(db_path)]
    gaps = sorted(b - a for a, b in zip(ts, ts[1:]) if 0 < b - a < 6 * 3600)
    if len(gaps) < 30:
        return FLOOR_IDLE_S            # not enough to learn from
    q = gaps[min(int(IDLE_QUANTILE * len(gaps)), len(gaps) - 1)]
    return max(FLOOR_IDLE_S, q)


def daily_schedule(db_path=None):
    """When the day usually starts and ends, or None if it does not.

    ⚠️ Returns None rather than a median when the spread is wide. A schedule
    built from an 8.5-hour standard deviation is a number with a decimal point
    and no meaning, and acting on it would put Mati to sleep mid-conversation.
    """
    byday = collections.defaultdict(list)
    for t in _turn_times(db_path):
        byday[t.strftime("%Y-%m-%d")].append(t.hour + t.minute / 60.0)
    days = {d: v for d, v in byday.items() if len(v) >= MIN_TURNS_PER_DAY}
    if len(days) < MIN_DAYS:
        return None
    firsts = [min(v) for v in days.values()]
    lasts = [max(v) for v in days.values()]
    spread = statistics.pstdev(firsts) if len(firsts) > 1 else 99.0
    if spread > MAX_SPREAD_H:
        return None                    # no pattern to honour
    return {"wake_hour": statistics.median(firsts),
            "sleep_hour": statistics.median(lasts),
            "spread_hours": spread, "days": len(days)}


def status(db_path=None):
    """What has been learned, and what has not. For `/system`."""
    ts = _turn_times(db_path)
    gaps = sorted(b.timestamp() - a.timestamp()
                  for a, b in zip(ts, ts[1:])
                  if 0 < b.timestamp() - a.timestamp() < 6 * 3600)
    sched = daily_schedule(db_path)
    out = {
        "turns": len(ts),
        "idle_after_minutes": round(idle_after(db_path) / 60, 1),
        "learned_from_gaps": len(gaps) >= 30,
        "schedule": sched,
    }
    if gaps:
        out["median_gap_minutes"] = round(gaps[len(gaps) // 2] / 60, 1)
    if sched is None:
        out["why_no_schedule"] = (
            "the day does not start at a consistent hour yet — a schedule "
            "built from that would be wrong more often than right")
    return out
