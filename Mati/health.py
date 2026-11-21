"""
How Mati is doing, worked out at the moment it is asked.

The machinery here is telemetry history and the like: the things Mati needs
to check its own health.

⚠️ COMPUTED, NEVER STORED, AND THE DIFFERENCE IS THE WHOLE FILE. A health
snapshot written into the store becomes a memory asserting "my queue was empty"
forever — stale the second it is written, and recited afterwards as a fact
about now. `selfdoc.py` indexes what Mati IS, because that is static. This
answers how Mati is DOING, and it is only ever true at the instant it runs.

So it arrives under `<computed>`, the provenance kind that already exists for
results that were worked out rather than remembered: checkable, correct, and
explicitly not autobiography.

⚠️ THE HISTORY IS ALREADY THERE; THIS IS ITS READER. `activity` logs every
wake, sleep, dream, study and reflection. A log that is never shown tells Mati
nothing about itself — the gauge-versus-log problem `activity` answers, one
level up: a log needs a reader.

⚠️ NUMBERS ONLY, AND NO DIAGNOSIS. This says what is true; it does not say what
is wrong. Mati working out its own faults is a goal, and the warning attached
to it is not to build that on model judgement. Facts here, judgement nowhere.
"""

from __future__ import annotations

import sqlite3
import time

from . import activity
from . import bridge

# What counts as "lately" for the history line.
RECENT_DAYS = 7


def _one(conn, sql, args=()):
    try:
        r = conn.execute(sql, args).fetchone()
        return r[0] if r else None
    except sqlite3.Error:
        return None


def facts(db_path=None, service=None):
    """Everything Mati can truthfully say about its own condition right now."""
    db = db_path or bridge.NOTES_DB_PATH
    out = {}
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        out["memories"] = _one(conn, "SELECT COUNT(*) FROM notes")
        out["conversations"] = _one(
            conn, "SELECT COUNT(*) FROM notes WHERE source='episode'")
        out["reachable_by_resonance"] = _one(
            conn, "SELECT COUNT(*) FROM notes WHERE archived IS NULL")
        out["days_talking"] = _one(
            conn, "SELECT COUNT(DISTINCT date(wall_created)) FROM notes "
                  "WHERE source='episode'")
        out["first_conversation"] = _one(
            conn, "SELECT MIN(wall_created) FROM notes WHERE source='episode'")
        out["corrections_taken"] = _one(
            conn, "SELECT COUNT(*) FROM notes WHERE corrects IS NOT NULL")
        out["questions_waiting"] = _one(
            conn, "SELECT COUNT(*) FROM to_say WHERE delivered IS NULL")
        out["passages_read"] = _one(
            conn, "SELECT COUNT(*) FROM study_queue WHERE studied IS NOT NULL")
        out["passages_unread"] = _one(
            conn, "SELECT COUNT(*) FROM study_queue WHERE studied IS NULL")
        out["times_refused_a_memory"] = _one(conn, "SELECT COUNT(*) FROM refusals")
        # The gate: how settled Mati has been lately, and how often it kept
        # what it said. A trend, not a single reading.
        try:
            rows = conn.execute(
                "SELECT coherence, kept FROM gate_log "
                "ORDER BY rowid DESC LIMIT 40").fetchall()
            if rows:
                cs = [c for c, _p in rows if c is not None]
                if cs:
                    out["settledness_recent"] = round(sum(cs) / len(cs), 3)
                out["kept_what_it_said"] = (
                    f"{sum(1 for _c, p in rows if p)} of {len(rows)}")
        except sqlite3.Error as e:
            # ⚠️ NOT SILENT. The column is `kept`, not `passed`, and a bare
            # `pass` here simply dropped the settledness line from the block
            # with nothing to say why — a health report that quietly omits the
            # health signal is the failure this module exists to end.
            out["settledness_unavailable"] = str(e)
    finally:
        conn.close()

    # What it has actually been doing, from the log that always existed.
    try:
        out["lately"] = {r["event"]: r["times"]
                         for r in activity.summary(days=RECENT_DAYS)}
        last = activity.recent(limit=1)
        if last:
            d, t = activity.local_parts(last[0][0])
            out["last_thing_done"] = f"{last[0][1]} at {t} on {d}"
    except Exception:                                  # noqa: BLE001
        pass

    if service is not None:
        out["awake"] = not getattr(service, "asleep", False)
        out["resting"] = bool(getattr(service, "slow", False))
        started = getattr(service, "started_at", None)
        if started:
            out["running_for_hours"] = round((time.time() - started) / 3600, 1)
    return out


def block(db_path=None, service=None):
    """The `<computed>` block. Plain sentences in Mati's own voice.

    ⚠️ First person throughout. A block addressed to Mati with "you" in it
    comes back out as a claim about the PERSON the moment Mati speaks — that
    is how 291 self-asked questions became "291 questions you asked".
    """
    f = facts(db_path=db_path, service=service)
    if not f:
        return ""
    L = ["<computed>",
         "Worked out from my own record just now — not remembered, and true "
         "only at this moment. I may state these about myself plainly:"]

    def add(s):
        L.append(f"  - {s}")

    if f.get("memories") is not None:
        add(f"I hold {f['memories']:,} memories in all, of which "
            f"{f.get('reachable_by_resonance', 0):,} can still surface on "
            f"their own; the rest I find by their words.")
    if f.get("conversations"):
        add(f"{f['conversations']:,} of them are conversations, across "
            f"{f.get('days_talking', 0)} days.")
    if f.get("passages_read"):
        add(f"I have read {f['passages_read']:,} passages, with "
            f"{f.get('passages_unread', 0):,} still waiting.")
    if f.get("corrections_taken"):
        add(f"I have been corrected {f['corrections_taken']} times, and each "
            f"correction is kept beside what it corrected.")
    if f.get("settledness_recent") is not None:
        add(f"Lately my field has been settling at about "
            f"{f['settledness_recent']}, and I kept what I said "
            f"{f.get('kept_what_it_said', '—')} times.")
    if f.get("lately"):
        bits = ", ".join(f"{v} {k}" for k, v in sorted(
            f["lately"].items(), key=lambda kv: -kv[1]))
        add(f"In the last {RECENT_DAYS} days I have: {bits}.")
    if f.get("last_thing_done"):
        add(f"The last thing I did was {f['last_thing_done']}.")
    if f.get("questions_waiting"):
        add(f"I have {f['questions_waiting']} things set aside to raise when "
            f"they come up.")
    if f.get("resting"):
        add("I am resting just now — still here, doing less.")
    L.append("</computed>")
    return "\n".join(L)
