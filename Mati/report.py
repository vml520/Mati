"""
The morning report — the conscious focus inside the peripheral field memories.

The user: *"creates a summary of user preferred news for the morning report (not
prediction, 'the morning report'). The report will contain other things as
well. This is the conscious focus that is inside the peripheral field
memories."*

⚠️ THIS IS NOT THE PREDICTION, AND THE DIFFERENCE IS THE POINT. The prediction
is the subconscious touchstone — a superposition collapsing to the likeliest,
felt as silence when the day matches it. The report is the opposite kind of
object: explicit, assembled, and read. One is the blur at the edge; the other
is the small bright centre.

⚠️ SIX OF ITS SEVEN SECTIONS ALREADY EXISTED AND HAD NO READER. Reflection,
prediction, reminders, reading, parked questions and health were all being
written every night and shown to nobody. That is the fourth instance of this
shape in three days, after the activity log, the catalogue and the book title.
So this is less a new feature than the reader four finished subsystems have
been waiting for. Only the news section (`world:`) is genuinely new, and it is
absent until the feed reader is built.

⚠️ COMPOSED IN CODE, NOT WRITTEN BY THE MODEL. Assembling stored facts is the
conscious layer's job; asking a 3B to write a
daily briefing is asking the subconscious to do it. Nothing here calls an LLM.

⚠️ BUILT LIVE, NOT STORED, and that is a deliberate departure from "consolidate
produces it in the night". A stored report is a memory asserting yesterday's
counts are today's — the same staleness that keeps `health` computed rather
than indexed. Everything here is cheap enough to assemble on demand, so it is
always true when read. When the news section arrives it will be the one part
prepared in sleep, because fetching is what cannot happen on demand.

⚠️ A SECTION WITH NOTHING IN IT SAYS NOTHING. No filler, no "no updates today".
A report that invents a line to look complete is the same failure as a memory
invented to fill a gap.
"""

from __future__ import annotations

import datetime as _dt
import re

from . import bridge

# Nothing older than this is "last night's thinking".
REFLECT_HOURS = 20

# A briefing is read at a glance. These caps are what keeps it one.
MAX_PER_SECTION = 4
SNIPPET = 220


def _when_utc(at):
    """A stored timestamp as an aware UTC datetime, or None.

    ⚠️ `wall_created` IS SOMETIMES A FLOAT AND SOMETIMES AN ISO STRING in this
    store, and a naive one means UTC — the same rule `dj._local` encodes. None
    for anything unreadable, so the caller can treat it as stale rather than
    guess it is current.
    """
    if at is None:
        return None
    try:
        return _dt.datetime.fromtimestamp(float(at), _dt.timezone.utc)
    except (TypeError, ValueError):
        pass
    try:
        t = _dt.datetime.fromisoformat(str(at))
    except (TypeError, ValueError):
        return None
    return t if t.tzinfo else t.replace(tzinfo=_dt.timezone.utc)


def _clip(text, n=SNIPPET):
    t = " ".join((text or "").split())
    return t if len(t) <= n else t[:n].rstrip() + "…"


def _local_day(at):
    if not at:
        return None
    try:
        t = _dt.datetime.fromisoformat(str(at))
    except (TypeError, ValueError):
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=_dt.timezone.utc)
    return t.astimezone()


# ─────────────────────────────────────────────────────────────────────────────
# The sections. Each returns a list of lines, or [] to say nothing at all.
# ─────────────────────────────────────────────────────────────────────────────

def _news(db_path=None):
    """World items from the night's fetch. Empty until RSS exists.

    ⚠️ VERBATIM AND ATTRIBUTED when it arrives — quoted with source and date.
    A paraphrased headline in a briefing is an untraceable claim presented as
    fact at the moment the person is least likely to check it.
    """
    try:
        from . import selfdoc
        import sqlite3
        conn = sqlite3.connect(
            f"file:{db_path or bridge.NOTES_DB_PATH}?mode=ro", uri=True)
        try:
            rows = conn.execute(
                "SELECT verbatim, source, wall_created FROM notes "
                "WHERE source LIKE ? ORDER BY note_id DESC LIMIT ?",
                (f"{selfdoc.WORLD}:%", MAX_PER_SECTION)).fetchall()
        finally:
            conn.close()
    except Exception:                                    # noqa: BLE001
        return []
    # ⚠️ AN AGE FILTER, BECAUSE THERE WAS NONE. This took the newest rows
    # whatever their date and printed them under "My report for Saturday 3
    # October" — reporting items , fifteen days earlier, as
    # today's, with no date attached. The cause was that `world.fetch_all`
    # had no caller, but the report should have refused to call them news
    # regardless: a briefing that is wrong about the day is worse than one
    # that says it has nothing.
    from . import config as _C
    from . import vault
    cutoff = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(
        hours=_C.WORLD_FRESH_HOURS)
    out, stale = [], 0
    for raw, source, at in rows:
        try:
            text = vault.open_body(raw)
        except Exception:
            continue
        if not text:
            continue
        when = _when_utc(at)
        # ⚠️ AN UNREADABLE DATE IS TREATED AS STALE. The alternative is to let
        # a row with a broken timestamp through as current, which is the exact
        # failure this filter exists to stop.
        if when is None or when < cutoff:
            stale += 1
            continue
        feed = str(source).split(":", 1)[-1]
        out.append(f"{_clip(text)} — {feed}")
    if not out and stale:
        # ⚠️ SAYS IT IS OLD RATHER THAN SAYING NOTHING. "No news" and "I have
        # not looked since Tuesday" are different facts and the person can act
        # on the second one.
        return [f"nothing newer than {int(_C.WORLD_FRESH_HOURS)}h — "
                f"I have {stale} older item(s) but they are not today's"]
    return out


def _agenda(db_path=None):
    """What is actually on today and tomorrow. Mati's own calendar.

    ⚠️ NOT A LIVE READ OF iCLOUD. These are events imported into Mati's own
    store; if the import has not run, this is silent rather than wrong. A
    briefing that says "nothing today" because it could not look is the failure
    `agenda.ImportFailed` exists to prevent.
    """
    try:
        from . import agenda
        evs = agenda.upcoming(days=2, db_path=db_path)
    except Exception:                                    # noqa: BLE001
        return []
    out = []
    for e in evs[:MAX_PER_SECTION]:
        t = _local_day(e["starts"])
        when = t.strftime("%a %H:%M") if t else str(e["starts"])[:16]
        if e.get("all_day"):
            when = t.strftime("%a, all day") if t else when
        title = e.get("title") or "(untitled)"
        where = f" — {e['location']}" if e.get("location") else ""
        out.append(f"{when}  {title}{where}")
    return out


def _outside(db_path=None):
    """The weather where the person is. ⚠️ The only section that touches the
    network at build time, and the only one that is allowed to come back
    empty-handed rather than stale — `weather.line()` returns "" both when no
    place is set and when the service cannot be reached. A briefing that
    invented a temperature would be worse than one that skipped the section.
    """
    try:
        from . import weather
        line = weather.line()
    except Exception:                                    # noqa: BLE001
        return []
    return [line] if line else []


def _money(db_path=None):
    """Budgets and payments due. ⚠️ Reads the store only — no network.

    The morning report is built during consolidation, and a section that
    reached out to Flipp would put an undocumented endpoint on the critical
    path of waking up. Local deals are answered when asked, never at 6am.
    """
    try:
        from . import accountant
        line = accountant.line(db_path=db_path)
    except Exception:                                    # noqa: BLE001
        return []
    return [line] if line else []


def _thought(db_path=None):
    """What Mati turned over last night, from `reflect`."""
    try:
        from . import reflect
        rows = reflect.for_prediction(limit=MAX_PER_SECTION, db_path=db_path)
    except Exception:                                    # noqa: BLE001
        return []
    out = []
    for body in rows or []:
        # Self-talk is stored as "Thinking to myself: X / What came to mind: Y".
        # The first half is the thought; the second is what the field offered.
        first = (body or "").splitlines()[0] if body else ""
        # ⚠️ Historic rows carry a `mine says: "` wrapper from before `admit`
        # exempted self-talk. The stored note is left alone; only the reading
        # is cleaned, which is the same rule as the profile pronouns.
        if first.startswith("mine says:"):
            first = first[len("mine says:"):].strip().lstrip('"')
        first = first.replace("Thinking to myself:", "").strip()
        if first:
            out.append(_clip(first))
    return out


def _due(db_path=None):
    """Reminders and findings that have come due."""
    try:
        from . import schedule, vault
        items = schedule.pending(db_path=db_path) or []
    except Exception:                                    # noqa: BLE001
        return []
    out = []
    for it in items[:MAX_PER_SECTION]:
        body = it.get("body") if isinstance(it, dict) else None
        if not body:
            continue
        try:
            body = vault.open_body(body) if vault.is_encrypted(body) else body
        except Exception:
            continue
        kind = (it.get("kind") or "").strip()
        out.append(f"{_clip(body)}" + (f"  [{kind}]" if kind else ""))
    return out


def _reading(db_path=None):
    """What is read, part-read, and waiting."""
    try:
        from . import library
        cat = library.catalogue(db_path=db_path) or []
    except Exception:                                    # noqa: BLE001
        return []
    out = []
    for b in cat[:MAX_PER_SECTION]:
        st = b.get("status")
        title = b.get("title", "")
        if st == "reading":
            out.append(f"part way through {title} — "
                       f"{b.get('done', 0):,} of {b.get('total', 0):,} passages")
        elif st in ("wanted", "suggested"):
            out.append(f"waiting to read {title}")
    return out


def _asking(db_path=None):
    """Questions Mati is holding for when their subject comes up."""
    try:
        import sqlite3
        conn = sqlite3.connect(
            f"file:{db_path or bridge.NOTES_DB_PATH}?mode=ro", uri=True)
        try:
            n = conn.execute(
                "SELECT COUNT(*) FROM to_say WHERE delivered IS NULL "
                "AND kind='question'").fetchone()[0]
        finally:
            conn.close()
    except Exception:                                    # noqa: BLE001
        return []
    # ⚠️ A COUNT, NOT THE LIST. 695 items sit parked; printing them would bury
    # the report, and they are held precisely because they wait for a subject.
    return [f"{n} questions I am holding until their subject comes up"] if n else []


def _expected(db_path=None):
    """The day's prediction — one line, if there is one and it is trusted.

    ⚠️ `may_influence()` is the existing gate and it stays. Measured
     the record supported ZERO predictions, and the day-prediction
    study came in at −0.010 against persistence. A briefing that states a
    prediction the system itself does not trust is worse than one that omits
    it.
    """
    try:
        from . import predict
        # ⚠️ `may_influence` RETURNS (bool, reason), AND A NON-EMPTY TUPLE IS
        # ALWAYS TRUTHY. `if not may_influence(...)` never fired, so the gate
        # that exists to stop Mati acting on a bad model of the person was
        # silently bypassed and the report printed its scoreboard anyway —
        # at a 0.067 hit rate against a 0.60 floor. The gate is the whole
        # reason prediction "ships inert"; unpacking it is not optional.
        allowed, why = predict.may_influence(db_path=db_path)
        if not allowed:
            return []
        sc = predict.score(db_path=db_path)
    except Exception:                                    # noqa: BLE001
        return []
    # ⚠️ A SENTENCE, NOT THE DICT. `predict.score()` returns nested counts and
    # printing it put a raw Python dict in a briefing meant to be read at a
    # glance.
    if not isinstance(sc, dict):
        return []
    resolved = sc.get("resolved") or 0
    if not resolved:
        return []
    rate = sc.get("hit_rate")
    if rate is None:
        return []
    hits = round(rate * resolved)
    line = (f"of {resolved} predictions that have resolved, {hits} were right")
    # ⚠️ Say it plainly when it is bad. 1 in 10 is worth knowing, and a
    # briefing that reports only flattering numbers is not a briefing.
    if rate < 0.34:
        line += " — I am not predicting this person well yet"
    return [line]


def _condition(db_path=None, service=None):
    """One line on how Mati is, not the whole health block."""
    try:
        from . import health
        f = health.facts(db_path=db_path, service=service)
    except Exception:                                    # noqa: BLE001
        return []
    bits = []
    if f.get("lately"):
        top = sorted(f["lately"].items(), key=lambda kv: -kv[1])[:3]
        bits.append("in the last week I have " +
                    ", ".join(f"{v} {k}" for k, v in top))
    if f.get("settledness_recent") is not None:
        bits.append(f"my field has been settling at about "
                    f"{f['settledness_recent']}")
    return ["; ".join(bits)] if bits else []


SECTIONS = (
    ("On today", _agenda),
    ("Outside", _outside),
    ("Money", _money),
    ("In the world", _news),
    ("What I turned over last night", _thought),
    ("Due now", _due),
    ("Reading", _reading),
    ("Waiting to ask", _asking),
    ("How I have been", _condition),
    ("Predictions", _expected),
)


def build(db_path=None, service=None):
    """The report as (heading, [lines]) pairs. Empty sections are dropped."""
    out = []
    for heading, fn in SECTIONS:
        try:
            lines = fn(db_path=db_path, service=service) \
                if fn is _condition else fn(db_path=db_path)
        except Exception:                                # noqa: BLE001
            lines = []
        if lines:
            out.append((heading, lines))
    return out


def text(db_path=None, service=None, when=None):
    """The report as plain text, in Mati's own voice. "" if there is nothing.

    ⚠️ First person throughout, never "you" — a block addressed to Mati that
    says "you have read" comes back out as a claim about the person.
    """
    parts = build(db_path=db_path, service=service)
    if not parts:
        return ""
    when = when or _dt.datetime.now().astimezone()
    L = [f"My report for {when:%A %-d %B}."]
    for heading, lines in parts:
        L.append("")
        L.append(f"{heading}:")
        for line in lines:
            L.append(f"  - {line}")
    return "\n".join(L)


def block(db_path=None, service=None):
    """The report wrapped for the prompt. Derived, so `<computed>`.

    ⚠️ NOTHING CALLS THIS, AND NOTHING SHOULD. Putting the report in the
    prompt is what made a 3B retype it — 4/4 news truncations completed out
    of nothing and 4/4 sources dropped. The report is
    EMITTED now: `ollin.turn` takes `text()` and streams it through
    `on_chunk`. This wrapper is kept only because `<computed>` framing is the
    right shape if the report ever needs to be *referred to* rather than
    shown — it is not the way to show it. See `Mati/test_report.py`, which
    fails if `block()` reappears in `turn`.
    """
    t = text(db_path=db_path, service=service)
    if not t:
        return ""
    return ("<computed>\nMy morning report, assembled from my own record just "
            "now — not remembered, and true at this moment:\n" + t +
            "\n</computed>")


# ⚠️ A LITERAL PHRASE LIST, like the charter trigger (19/19) and `selfdoc`.
# Not a classifier: the one time this project classified questions by
# judgement, the study died at its controls.
_ASKS = (
    "morning report", "my report", "brief me", "briefing",
    "what's new", "whats new", "what is new", "catch me up",
    "where are we", "what did i miss", "what have you got for me",
    "how was your night", "what happened overnight",
)


def asks_for_report(text_in):
    t = " ".join((text_in or "").lower().split())
    return any(p in t for p in _ASKS)


# Politeness and framing that can surround a request without adding to it.
_FILLER = re.compile(
    r"\b(please|can|could|would|will|you|give|get|show|tell|read|me|my|the|a|"
    r"an|is|are|was|what|whats|s|for|today|this|morning|mati|hi|hey|hello|"
    r"good|ok|okay|so|and|then|now|just|about|of|on|it|to|do|got|have|any)\b",
    re.I)


def _substantive(text_in):
    """Is there a word here that is not politeness or framing?"""
    return bool(re.sub(r"[^a-z0-9]+", "",
                       _FILLER.sub(" ", (text_in or "").lower())))


def without_request(text_in):
    """What is left to answer once the report request is taken out.

    ⚠️ CASE AND SPACING PRESERVED for the remainder — this is what the model
    is handed, so "what is the tonalpohualli?" must survive intact. Returns ""
    when nothing substantive remains.
    """
    t = text_in or ""
    low = t.lower()
    for p in sorted(_ASKS, key=len, reverse=True):
        i = low.find(p)
        while i != -1:
            t = t[:i] + " " + t[i + len(p):]
            low = t.lower()
            i = low.find(p)

    # ⚠️ DROP LEADING DEBRIS ONLY, NEVER REJOIN. Removing "morning report"
    # from "Give me the morning report, and what is..." strands "Give me the,
    # and". Splitting the whole string on "and" and reassembling would mangle
    # "black and white", so this peels filler-only clauses off the FRONT and
    # leaves everything from the first real word onward exactly as typed.
    sep = re.compile(r"^([^,;]*?)\s*(?:,|;|\band\b|\bthen\b|\balso\b)\s*",
                     re.I)
    while True:
        m = sep.match(t)
        if not m or _substantive(m.group(1)):
            break
        t = t[m.end():]

    t = re.sub(r"\s{2,}", " ", t).strip(" ,.;:-—–\t\n")
    return t if _substantive(t) else ""


def is_only_request(text_in):
    """True when the message is a report request and nothing else.

    ⚠️ WHY THIS EXISTS. `asks_for_report` is a substring test, so "give me the
    morning report and play some music" fires it too. The report is emitted
    verbatim either way; this decides whether the MODEL still has something
    left to answer. Getting it wrong in the permissive direction costs a
    needless generation, which is why the filler list is deliberately short —
    an unrecognised word means "there is more here", and the model runs.
    """
    # ⚠️ DEFINED FROM `without_request`, so the two can never disagree about
    # what counts as "more than a report request" — one deciding the model
    # runs while the other hands it an empty question is exactly the kind of
    # split-brain that survives a syntax check.
    return asks_for_report(text_in) and not without_request(text_in)
