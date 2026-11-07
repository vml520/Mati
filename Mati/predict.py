"""
Guessing what you will do, and keeping score.

    python3 -m Mati.predict --status      # what it can support, and its record
    python3 -m Mati.predict --propose     # what it would predict right now

Everything else Mati holds is a record of something: lived, read, asked,
computed. A prediction is the first thing it holds that is **not yet true**, and
that makes it the only thing it can be *scored* on. That is the point of this
module. Not to make Mati seem perceptive — to find out, measurably, whether it
is.

⚠️ **RIGHT NOW IT CAN PREDICT NOTHING, AND THAT IS THE CORRECT OUTPUT.**
: the store holds 14 episodes spanning **1 hour 41 minutes**,
one day, one weekday. Any claim about when you work or what you return to would
be invention dressed as observation. `propose()` returns an empty list, and it
will keep returning one until the evidence gates below are actually cleared.

FOUR RULES, AND THE FIRST IS THE WHOLE THING.

**1. A PREDICTION MUST BE FALSIFIABLE, OR IT IS NOT STORED.** Every prediction
carries a resolution — a time by which it either happened or did not — and
`resolve()` checks it against the episode store rather than against anybody's
impression. "You seem to like the evenings" is not a prediction. "You will start
a conversation before midnight tonight" is.

**2. NOTHING IS PREDICTED FROM THIN EVIDENCE.** A pattern needs
`EVIDENCE_MIN` observations across `DAYS_MIN` distinct days. One session is not
a habit, and the failure mode here is specific and bad: a brand-new user being
told "you always work late" on day one. That is not a charming quirk, it is
Mati fabricating a model of a person it has just met.

**3. IT DOES NOT ACT ON PREDICTIONS UNTIL IT HAS EARNED THE RIGHT.**
`may_influence()` is False until enough predictions have RESOLVED and the hit
rate clears `HIT_RATE_MIN`. Until then predictions are recorded and scored and
change nothing — no reminder timing, no interruption decisions. A system that
acts on an unmeasured model of you is worse than one that has no model.

**4. A CORRECTION ALWAYS WINS, AND IS NEVER CHALLENGED.** `challenge.py` lets
Mati push back on a correction *only where it can recompute the answer*. A
prediction about your behaviour can never be recomputed — you are the only
source of truth about how you work. So a corrected prediction is retired
immediately, with no recheck and no argument. The authority ordering that
already governs provenance gives this for free.

**PROVENANCE: `predicted` IS THE WEAKEST THING MATI HOLDS.** Below `asked`,
which is at least the model saying something. A prediction is Mati's own guess
about a person, and it must never be presented as remembered — if Mati voices
one, it voices it as a guess and invites the correction.

WHAT IT WATCHES. Two kinds, both checkable from data Mati already stores:

    timing        when you tend to be here            resolves against episodes
    recurrence    a subject you come back to          resolves lexically

Deliberately not "what you will ask next". That is not falsifiable without
reading your mind, and a wrong guess about it is the most intrusive thing on the
list.
"""

import argparse
import os
import sqlite3
import sys
import time
from collections import Counter
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from . import bridge
from . import vault

TIMING = "timing"
RECURRENCE = "recurrence"
# ⚠️ `observe()` HAS COMPUTED `weekdays` ALL ALONG AND NOTHING READ IT —
# measured on the real store: Sunday 252 of 807 episodes across 54 days, a
# far stronger signal than the busiest hour, sitting unused. Another writer
# with no reader, smaller than the drafts one and the same shape.
WEEKDAY = "weekday"
KINDS = (TIMING, RECURRENCE)

OPEN, HIT, MISS, CORRECTED = "open", "hit", "miss", "corrected"

# ── the gates ───────────────────────────────────────────────────────────────
#
# ⚠️ THESE ARE JUDGEMENT, NOT MEASUREMENT, and unlike most numbers in this
# codebase they cannot be measured yet — there is no history to calibrate them
# against. They are set where a reasonable person would stop calling something a
# coincidence, and they should be revisited once real predictions have resolved.
#
# What IS measured is that the current store clears none of them.
EVIDENCE_MIN = 5          # observations before a pattern is a pattern
DAYS_MIN = 3              # spread over at least this many distinct days
RESOLVED_MIN = 10         # resolved predictions before it may influence anything
HIT_RATE_MIN = 0.60       # and a record better than this

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    pred_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL,
    claim       TEXT NOT NULL,
    basis       TEXT,
    evidence_n  INTEGER,
    made_at     REAL NOT NULL,
    resolves_at REAL NOT NULL,
    state       TEXT NOT NULL,
    resolved_at REAL,
    detail      TEXT,
    correction  TEXT
);
CREATE INDEX IF NOT EXISTS predictions_open ON predictions(state, resolves_at);
"""


def _connect(db_path=None):
    conn = sqlite3.connect(db_path or bridge.NOTES_DB_PATH)
    conn.executescript(SCHEMA)
    # `CREATE TABLE IF NOT EXISTS` does NOT add columns to a table that already
    # exists. This has bitten twice in this codebase (notes.corrects,
    # to_say.topic), so new columns go here explicitly.
    have = {r[1] for r in conn.execute("PRAGMA table_info(predictions)")}
    for col, decl in (("correction", "TEXT"), ("detail", "TEXT")):
        if col not in have:
            conn.execute(f"ALTER TABLE predictions ADD COLUMN {col} {decl}")
    return conn


# ── what the record actually supports ───────────────────────────────────────

# ⚠️ EVIDENCE ABOUT THE PERSON — NOT ABOUT MATI'S OWN DAY. This module reads
# timestamps to answer "when do they usually do this", and it read EVERY note
# that had one. :
#
#     all notes           n=3090   busiest hour 16:00
#     conversation only   n= 324   busiest hour 22:00
#
# Conversation was 10.5% of the evidence. 16:00 is when the reading loop
# happened to write batches of Aztec Thought and Culture — so the one number
# prediction produced was Mati's own study schedule, offered as a habit of
# The user's, and wrong by six hours.
#
# ⚠️ IT FAILED ON ONE HALF ONLY, WHICH IS WHY IT SURVIVED. RECURRENCE was
# accidentally safe: `_asked()` returns "" without an `I was asked:` marker, so
# a book passage contributed nothing. TIMING has no such guard — it needs only
# a timestamp, and every note has one. Same table, same query, one half
# protected by a coincidence.
#
# WHAT BELONGS HERE is anything whose timestamp records something THE PERSON
# did. A calendar entry and a budget entry will qualify and are the reason this
# is a list rather than `source = 'episode'`: adding them is one line here, and
# doing it will silently widen what Mati claims about someone's habits, so it
# is a deliberate act and not a query edit somewhere else.
PERSON_SOURCES = ("episode",)
PERSON_PREFIXES = ()          # future: ("calendar:", "budget:")


def _person_where(alias=""):
    """SQL restricting a query to records of what the PERSON did."""
    col = f"{alias}source" if alias else "source"
    parts = [f"{col} IN ({','.join('?' * len(PERSON_SOURCES))})"]
    args = list(PERSON_SOURCES)
    for pre in PERSON_PREFIXES:
        parts.append(f"{col} LIKE ?")
        args.append(pre + "%")
    return "(" + " OR ".join(parts) + ")", args


def _episodes(db_path=None):
    conn = sqlite3.connect(db_path or bridge.NOTES_DB_PATH)
    try:
        where, args = _person_where()
        rows = conn.execute(
            "SELECT note_id, body, wall_created FROM notes "
            f"WHERE wall_created IS NOT NULL AND {where} "
            "ORDER BY wall_created", args).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()
    # ⚠️ BODIES ARE SEALED ON WRITE. `bridge` stores them through
    # `vault.seal()` and reads them back through `vault.open_body()`. Reading
    # the column raw would hand this module ciphertext, `_asked()` would find no
    # marker and return "", and RECURRENCE would silently see nothing while
    # timing predictions carried on working — a partial failure with no symptom.
    # Timestamps are plaintext by design, which is exactly why the two halves
    # would have failed differently.
    locked = 0
    out = []
    for nid, body, when in rows:
        try:
            # ⚠️ CONVERT TO LOCAL. `wall_created` is stored UTC, and everything
            # downstream reads a HUMAN unit off this — `.hour` for "when do
            # they usually talk", `.date()` for how many days the record spans,
            # `.strftime("%a")` for which weekday. Left in UTC, a 17:00 local
            # conversation is bucketed at hour 0 the NEXT DAY: measured
            # , the busiest hour read 0 with 247 episodes, which is
            # 17:00 local and not a habit anyone has.
            at = datetime.fromisoformat(when).astimezone()
        except (ValueError, TypeError):
            continue
        try:
            body = vault.open_body(body or "")
        except vault.Locked:
            # No key: count it rather than treating an unreadable episode as an
            # empty one. `observe()` reports this instead of claiming no pattern.
            locked += 1
            body = ""
        except Exception:
            body = ""
        out.append((nid, body, at))
    _episodes.locked = locked
    return out


_episodes.locked = 0


ASKED = "I was asked:"
SAID = "I said:"


def _asked(body):
    """Just the user's words out of an episode body.

    Everything after `I said:` is Mati talking, and Mati's own vocabulary must
    never become evidence about the user. Returns "" rather than the whole body
    when the marker is missing — a body Mati cannot attribute is not evidence.
    """
    if ASKED not in body:
        return ""
    rest = body.split(ASKED, 1)[1]
    return rest.split(SAID, 1)[0].strip()


def observe(db_path=None, now=None):
    """What the store supports, with the evidence counts that gate it.

    Returns the raw observation, NOT a prediction. Deliberately separate: this
    is allowed to describe a thin record, `propose()` is not allowed to predict
    from one.
    """
    eps = _episodes(db_path)
    days = {e[2].date() for e in eps}
    hours = Counter(e[2].hour for e in eps)
    weekdays = Counter(e[2].strftime("%a") for e in eps)

    words = Counter()
    for _, body, _ in eps:
        # ⚠️ ONLY THE USER'S HALF. An episode is
        #     I was asked: <what they said>
        #     I said: <the reply>
        #     It brought back N earlier moments.
        # Counting the whole body made the top recurring terms 'earlier',
        # 'brought', 'formed', 'turning', 'seconds', 'moments', 'steadier' —
        # Mati's own narration boilerplate. It would have predicted "you will
        # come back to 'steadier'", which is not a fact about the user at all.
        # A prediction about a person has to be built from what that person said.
        words.update({w.strip(".,:;!?\"'()").lower() for w in _asked(body).split()
                      if len(w) > 6})
    recurring = [(w, n) for w, n in words.most_common(20) if n >= EVIDENCE_MIN]

    return {
        "episodes": len(eps),
        # Sealed episodes Mati could not read. Recurrence needs the text;
        # timing does not, so a locked vault degrades one and not the other and
        # must say which.
        "unreadable": _episodes.locked,
        "days": len(days),
        "span_hours": (round((eps[-1][2] - eps[0][2]).total_seconds() / 3600, 2)
                       if len(eps) > 1 else 0.0),
        "hours": dict(hours),
        "weekdays": dict(weekdays),
        "busiest_hour": (hours.most_common(1)[0] if hours else None),
        "recurring": recurring,
        "enough": len(eps) >= EVIDENCE_MIN and len(days) >= DAYS_MIN,
        "why_not": _why_not(len(eps), len(days)),
        "note": (f"{_episodes.locked} episode(s) sealed — recurrence cannot be "
                 f"observed without the vault key; timing still can"
                 if _episodes.locked else None),
    }


def _why_not(n_eps, n_days):
    if n_eps < EVIDENCE_MIN and n_days < DAYS_MIN:
        return (f"{n_eps} episodes over {n_days} day(s) — needs "
                f"{EVIDENCE_MIN} over {DAYS_MIN}")
    if n_eps < EVIDENCE_MIN:
        return f"{n_eps} episodes — needs {EVIDENCE_MIN}"
    if n_days < DAYS_MIN:
        return (f"{n_days} distinct day(s) — needs {DAYS_MIN}. One long "
                f"session is not a habit")
    return None


def propose(db_path=None, now=None):
    """Predictions the record justifies. Empty is the common and correct answer.

    Each carries its own resolution time, so recording one commits Mati to
    being checked.
    """
    now = now or time.time()
    obs = observe(db_path, now)
    if not obs["enough"]:
        return []

    out = []
    hour, n = obs["busiest_hour"]
    if n >= EVIDENCE_MIN:
        # Resolve at the end of the day: did a conversation start in the window?
        end = datetime.fromtimestamp(now).replace(hour=23, minute=59, second=59)
        out.append({
            "kind": TIMING,
            "claim": f"you will start a conversation between "
                     f"{hour:02d}:00 and {hour+1:02d}:00",
            "basis": f"{n} of {obs['episodes']} episodes in that hour, "
                     f"across {obs['days']} days",
            "evidence_n": n,
            "resolves_at": end.timestamp(),
            "window": [hour, hour + 1],
        })

    # ── the week ────────────────────────────────────────────────────────
    wd = obs.get("weekdays") or {}
    if wd:
        day, n_day = max(wd.items(), key=lambda kv: kv[1])
        if n_day >= EVIDENCE_MIN:
            # Resolves at the end of the next occurrence of that weekday.
            today = datetime.fromtimestamp(now)
            names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            ahead = (names.index(day) - today.weekday()) % 7 or 7
            end = (today + timedelta(days=ahead)).replace(
                hour=23, minute=59, second=59)
            out.append({
                "kind": WEEKDAY,
                "claim": f"you are most likely to come back on a {day}day"
                         if day in ("Sun", "Mon") else
                         f"you are most likely to come back on a {day}",
                "basis": f"{n_day} of {obs['episodes']} episodes fell on "
                         f"{day}, across {obs['days']} days",
                "evidence_n": n_day,
                "resolves_at": end.timestamp(),
                "weekday": day,
            })

    for word, count in obs["recurring"][:2]:
        out.append({
            "kind": RECURRENCE,
            "claim": f"you will come back to {word!r} within a week",
            "basis": f"mentioned in {count} episodes",
            "evidence_n": count,
            "resolves_at": now + 7 * 86400,
            "term": word,
        })
    return out


# ── recording, resolving, scoring ───────────────────────────────────────────

def record(p, db_path=None, now=None):
    """Store an open prediction. Returns its id."""
    if p["kind"] not in KINDS:
        raise ValueError(f"unknown kind {p['kind']!r} — have {KINDS}")
    if not p.get("resolves_at"):
        raise ValueError("a prediction without a resolution is not a "
                         "prediction — it cannot be scored")
    now = now or time.time()
    detail = p.get("term") or (",".join(map(str, p["window"]))
                               if p.get("window") else None)
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            "INSERT INTO predictions (kind, claim, basis, evidence_n, made_at, "
            "resolves_at, state, detail) VALUES (?,?,?,?,?,?,?,?)",
            (p["kind"], p["claim"], p.get("basis"), p.get("evidence_n"),
             now, p["resolves_at"], OPEN, detail))
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def resolve(db_path=None, now=None):
    """Check every prediction whose time has come. Returns what it decided.

    Resolution reads the episode store, not anybody's recollection. A prediction
    Mati cannot check is left open rather than counted as a hit.
    """
    now = now or time.time()
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT pred_id, kind, made_at, resolves_at, detail FROM predictions "
            "WHERE state = ? AND resolves_at <= ?", (OPEN, now)).fetchall()
        if not rows:
            return []
        eps = _episodes(db_path)
        decided = []
        for pid, kind, made_at, resolves_at, detail in rows:
            after = [e for e in eps if e[2].timestamp() > made_at]
            if kind == TIMING and detail:
                lo, hi = (int(x) for x in detail.split(","))
                got = any(lo <= e[2].hour < hi for e in after)
            elif kind == RECURRENCE and detail:
                # Resolved against the USER's words too, for the same reason
                # the prediction was made from them. Mati echoing a term in a
                # reply must not score its own prediction a hit.
                got = any(detail in _asked(e[1]).lower() for e in after)
            else:
                continue                      # unrecognised: leave it open
            state = HIT if got else MISS
            conn.execute("UPDATE predictions SET state = ?, resolved_at = ? "
                         "WHERE pred_id = ?", (state, now, pid))
            decided.append({"pred_id": pid, "kind": kind, "state": state})
        conn.commit()
        return decided
    finally:
        conn.close()


def correct(pred_id, correction_text, db_path=None, now=None):
    """The user says that is not how they work. Retired, not argued with.

    No recheck, ever. `challenge.py` earns the right to push back by being able
    to recompute an answer from nothing; a claim about how a person behaves can
    never be recomputed, and the person is the only source of truth about it.
    """
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            "UPDATE predictions SET state = ?, resolved_at = ?, correction = ? "
            "WHERE pred_id = ? AND state = ?",
            (CORRECTED, now or time.time(), correction_text, pred_id, OPEN))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def score(db_path=None):
    """The record. The only basis on which Mati may claim to predict anything."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT kind, state, count(*) FROM predictions "
            "GROUP BY kind, state").fetchall()
    finally:
        conn.close()

    by_kind = {}
    for kind, state, n in rows:
        by_kind.setdefault(kind, {})[state] = n
    total = {}
    for kind, states in by_kind.items():
        hit, miss = states.get(HIT, 0), states.get(MISS, 0)
        judged = hit + miss
        total[kind] = {
            "open": states.get(OPEN, 0),
            "hit": hit, "miss": miss,
            "corrected": states.get(CORRECTED, 0),
            # A corrected prediction is NOT a miss. It was withdrawn, not
            # tested — counting it as a miss would punish Mati for being
            # told something, which is the opposite of the incentive wanted.
            "hit_rate": (round(hit / judged, 3) if judged else None),
            "resolved": judged,
        }
    all_hit = sum(v["hit"] for v in total.values())
    all_judged = sum(v["resolved"] for v in total.values())
    return {
        "by_kind": total,
        "resolved": all_judged,
        "hit_rate": (round(all_hit / all_judged, 3) if all_judged else None),
        "corrected": sum(v["corrected"] for v in total.values()),
    }


# ── the three horizons, for the waking block ────────────────────────────────
# ⚠️⚠️ AWARENESS IS NOT ACTION, AND THE GATE STAYS ON THE ACTION.
# `may_influence()` asks whether Mati has earned the right to ACT on its model
# of someone — to time a reminder, to decide an interruption — and at a 0.307
# hit rate the answer is correctly no. Holding an expectation in mind, LABELLED
# AND WITH ITS RECORD ATTACHED, is a different thing: it is what the prediction
# design was for. A prediction nobody is allowed to see cannot be checked by
# the one person able to tell Mati it was wrong.
#
# ⭐ SO THESE ARE STATED AS EXPECTATIONS AND CARRY THE HIT RATE. "I expect X;
# I have been right about this kind of thing 31% of the time" is honest at any
# accuracy. "X will happen" is not honest at 31%.
#
# ⚠️ A YEAR IS NOT PREDICTABLE FROM FOUR MONTHS, and saying so is the useful
# answer rather than a failure. The record spans Jul–Oct; a seasonal claim
# needs a year of it. `_why_not` already sets the precedent of explaining the
# absence instead of returning nothing.
YEAR_MONTHS_MIN = 12


def horizons(db_path=None, now=None):
    """What Mati expects today, this week, and this year — or why it cannot.

    Returns a dict of horizon -> {"claim", "basis", "known"}. `known` False
    means there is no basis yet and `basis` says what is missing.
    """
    now = now or time.time()
    obs = observe(db_path, now)
    props = propose(db_path, now)
    by_kind = {}
    for p in props:
        by_kind.setdefault(p["kind"], p)

    out = {}
    day = by_kind.get(TIMING)
    out["day"] = ({"claim": day["claim"], "basis": day["basis"], "known": True}
                  if day else
                  {"claim": None, "known": False,
                   "basis": _why_not(obs["episodes"], obs["days"])
                            or "no hour stands out yet"})

    week = by_kind.get(WEEKDAY) or by_kind.get(RECURRENCE)
    out["week"] = ({"claim": week["claim"], "basis": week["basis"],
                    "known": True} if week else
                   {"claim": None, "known": False,
                    "basis": _why_not(obs["episodes"], obs["days"])
                             or "no day of the week stands out yet"})

    # ⚠️ MONTHS SPANNED, NOT DAYS OF RECORD. 54 busy days inside four months
    # is still four months; a year-shaped claim needs the year to have
    # happened. Counting distinct months is the cheapest honest test.
    months = _months_spanned(db_path)
    if months >= YEAR_MONTHS_MIN:
        out["year"] = {"claim": None, "known": False,
                       "basis": f"{months} months on record — no seasonal "
                                f"pattern measured yet"}
    else:
        out["year"] = {"claim": None, "known": False,
                       "basis": f"{months} month(s) on record; a yearly "
                                f"pattern needs {YEAR_MONTHS_MIN}"}
    return out


def _months_spanned(db_path=None):
    """Distinct calendar months with at least one episode."""
    try:
        conn = _connect(db_path)
    except Exception:
        return 0
    try:
        rows = conn.execute(
            "SELECT DISTINCT substr(wall_created,1,7) FROM notes "
            "WHERE source='episode' AND wall_created IS NOT NULL").fetchall()
        return len([r for r in rows if r[0]])
    except Exception:
        return 0
    finally:
        try:
            conn.close()
        except Exception:
            pass


def expectations_block(db_path=None, now=None):
    """The three horizons as Mati holds them at waking. "" if nothing.

    ⚠️ THE HIT RATE IS NOT OPTIONAL. Without it this is a confident forecast
    that is wrong two times in three; with it, it is an expectation a person
    can correct. The number is the whole licence to say any of this.
    """
    h = horizons(db_path, now)
    s = score(db_path)
    known = [(k, v) for k, v in h.items() if v.get("known")]
    if not known:
        return ""
    rate = s.get("hit_rate")
    n = s.get("resolved") or 0
    lines = [f"- {'today' if k == 'day' else 'this ' + k}: {v['claim']} "
             f"({v['basis']})" for k, v in known]
    unknown = [f"{k} ({v['basis']})" for k, v in h.items()
               if not v.get("known")]
    head = ("What you expect, as expectations and not as facts — you may say "
            "them, and you must say they are guesses:")
    tail = ""
    if n:
        tail = (f"\nYou have been right {rate*100:.0f}% of the time across "
                f"{n} resolved guesses. Say that too if you offer one.")
    if unknown:
        tail += "\nYou have no basis for: " + "; ".join(unknown) + "."
    return head + "\n  " + "\n  ".join(lines) + tail


def may_influence(db_path=None):
    """Has Mati earned the right to act on its model of you? Usually no.

    Returns (allowed, why). Nothing in Mati consults predictions until this is
    True — not reminder timing, not interruption decisions.
    """
    s = score(db_path)
    if s["resolved"] < RESOLVED_MIN:
        return False, (f"{s['resolved']} predictions resolved — needs "
                       f"{RESOLVED_MIN} before acting on any of them")
    if (s["hit_rate"] or 0) < HIT_RATE_MIN:
        return False, (f"hit rate {s['hit_rate']} is below {HIT_RATE_MIN} — "
                       f"the model of you is not good enough to act on")
    return True, f"{s['resolved']} resolved at {s['hit_rate']}"


def open_predictions(db_path=None):
    conn = _connect(db_path)
    try:
        return [{"pred_id": r[0], "kind": r[1], "claim": r[2], "basis": r[3],
                 "resolves_at": r[4]}
                for r in conn.execute(
                    "SELECT pred_id, kind, claim, basis, resolves_at FROM "
                    "predictions WHERE state = ? ORDER BY resolves_at", (OPEN,))]
    finally:
        conn.close()


def status(db_path=None, now=None):
    obs = observe(db_path, now)
    allowed, why = may_influence(db_path)
    return {
        "observation": obs,
        "would_predict": len(propose(db_path, now)),
        "open": len(open_predictions(db_path)),
        "score": score(db_path),
        "may_influence": allowed,
        "why": why,
        "gates": {"evidence_min": EVIDENCE_MIN, "days_min": DAYS_MIN,
                  "resolved_min": RESOLVED_MIN, "hit_rate_min": HIT_RATE_MIN},
    }


def main():
    ap = argparse.ArgumentParser(description="What Mati thinks you will do")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--propose", action="store_true")
    ap.add_argument("--resolve", action="store_true")
    args = ap.parse_args()

    if args.resolve:
        for d in resolve() or []:
            print(f"  {d['kind']:<12} #{d['pred_id']}  {d['state']}")
        print("  nothing due" if not resolve() else "")
        return 0

    s = status()
    o = s["observation"]
    print(f"  episodes        : {o['episodes']} over {o['days']} day(s), "
          f"{o['span_hours']}h span")
    print(f"  busiest hour    : {o['busiest_hour']}")
    print(f"  recurring terms : {o['recurring'] or 'none above threshold'}")
    print(f"  enough to guess : {o['enough']}"
          + (f"  — {o['why_not']}" if o["why_not"] else ""))
    print(f"  would predict   : {s['would_predict']}")
    print(f"  open            : {s['open']}")
    print(f"  record          : {s['score']['resolved']} resolved, "
          f"hit rate {s['score']['hit_rate']}, "
          f"{s['score']['corrected']} corrected")
    print(f"  may influence   : {s['may_influence']}  ({s['why']})")

    if args.propose:
        print()
        for p in propose():
            print(f"  {p['kind']:<12} {p['claim']}")
            print(f"  {'':12} basis: {p['basis']}")
        if not propose():
            print("  nothing the record justifies — which is the honest answer, "
                  "not a failure")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
