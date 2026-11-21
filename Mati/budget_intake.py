"""
Collecting a budget that arrives in pieces, across turns, in ordinary English.

    python3 -m Mati.budget_intake "dinner at $40 per person per day for two, 3 months"

⚠️ WHY THIS EXISTS. People describe a budget the way they think about it — all
at once, in a paragraph:

    "Let's breakdown meal costs. I would like to spend enough to buy a coffee
     and a baked item for breakfast everyday per person. I would like to have
     a rolling budget for lunch that can cover from 0 to $50 group spending a
     day. Dinner should be the big one at $40 per person per day. The budget
     is for two people. It should span three months."

`accountant.hears_money` is built for *"set a grocery budget of 400 a month"* —
one name, one amount, one period. On the paragraph above it would return a
budget named **"lunch that can cover from"** for **$0.00 a month**: the `0`
taken out of "from 0 to $50", a sentence fragment used as the name, and the
$40/person/day dinner, the two people and the three months dropped entirely.

⚠️ AND THE POINT IS LARGER THAN BUDGETS. Nobody should have to learn patterned
speech as commands. Whether a service or conversation is wanted should be
understood from the language itself.

So the trigger here is NOT a command phrase. It is the SHAPE of the language:
several amounts, with units and periods, and a count of people. That shape is
a budget whether the sentence begins "set a budget", "let's break down", or
nothing at all. And once a budget is open, the next turns continue it without
needing any keyword — because a person mid-conversation does not re-announce
what they are doing.

⚠️ WHAT IT DOES NOT DO: guess an amount. "a coffee and a baked item for
breakfast everyday per person" names an item with NO figure, and inventing one
would be confabulation. It is collected as a known-unknown and asked about —
the setup principle: work with partial data, ask for the one missing thing
when it blocks something.
"""

from __future__ import annotations

import re
import sys

# ── money ────────────────────────────────────────────────────────────────────
# ⚠️ INTEGER CENTS THROUGHOUT, and never float dollars. The accountant's rule,
# and the reason is that 0.1 + 0.2 is not 0.3 and a budget that is wrong by a
# cent per item is wrong by real money over three months.
_MONEY = re.compile(r"\$\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)"
                    r"|(\d+(?:,\d{3})*(?:\.\d{1,2})?)\s*(?:dollars?|usd|bucks)",
                    re.I)

# "from 0 to $50" — a range. ⚠️ THE TOP OF A RANGE IS THE BUDGET. A rolling
# allowance of "0 to $50 a day" is a FIFTY-dollar budget that may go unspent;
# reading the bottom gives zero, which is what the accountant did.
_RANGE = re.compile(
    r"(?:from\s*)?\$?\s*(\d+(?:\.\d{1,2})?)\s*(?:to|-|–|up to)\s*"
    r"\$?\s*(\d+(?:\.\d{1,2})?)", re.I)

_PER_PERSON = re.compile(r"\b(?:per|each|a)\s+person\b|\bper\s+head\b"
                         r"|\beach\b(?=[^.]*\b(?:day|week|month)\b)", re.I)
_PER_DAY = re.compile(r"\b(?:per|a|each|every)\s*day\b|\bdaily\b|\beveryday\b"
                      r"|\bevery day\b", re.I)
_PER_WEEK = re.compile(r"\b(?:per|a|each|every)\s*week\b|\bweekly\b", re.I)
_PER_MONTH = re.compile(r"\b(?:per|a|each|every)\s*month\b|\bmonthly\b", re.I)

_WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
            "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
            "twelve": 12, "a": 1, "an": 1}

_PEOPLE = re.compile(
    r"\bfor\s+(\w+)\s+(?:people|persons?|of us|adults?)\b"
    r"|\b(\w+)\s+of\s+us\b"
    r"|\b(myself|me)\s+and\s+my\s+(?:wife|husband|partner|spouse)\b", re.I)

_SPAN = re.compile(
    r"\b(?:span|spans|over|for|across|lasting)\s+(\w+)\s+"
    r"(day|days|week|weeks|month|months|year|years)\b", re.I)

# The meals and categories a line item can be about. ⚠️ A NAME, NOT A
# FRAGMENT. The accountant's failure was using whatever text sat beside the
# number; these are words that actually name a thing being budgeted.
_ITEM = re.compile(
    r"\b(breakfast|lunch|dinner|supper|brunch|coffee|groceries|grocery|"
    r"meals?|food|eating out|snacks?|drinks?|transport|transportation|"
    r"rent|hotel|lodging|flights?|airfare|airline tickets?|tickets?|"
    r"utilities|internet|phone|insurance|medical|health|fun|entertainment|"
    r"clothing|household|pets?|car|fuel|gas|petrol)\b", re.I)

# ⚠️ A CLAUSE CAN NAME SEVERAL OF THESE, AND THE FIRST ONE IS NOT THE SUBJECT.
# In "a coffee and a baked item for BREAKFAST everyday per person", taking the
# first match gives `coffee` — so the budget would grow a "coffee" line and no
# "breakfast" line, for the same money. A meal is what is being budgeted; a
# coffee is what it is spent on.
_MEALS = {"breakfast", "lunch", "dinner", "supper", "brunch"}

# Headings, not lines. "Let's break down MEAL costs" opens the subject; it is
# not a line item called "meal" — and it comes with no figure, so as a line it
# would sit in the budget as a permanent unanswerable gap ("I still need what
# you want to allow for meal") while the real lines are already complete.
_GENERIC = {"meal", "meals", "food", "costs", "cost"}


def to_cents(text):
    """Dollars as written -> integer cents. ⚠️ Rounds, never truncates."""
    return int(round(float(str(text).replace(",", "")) * 100))


def _words_to_n(word):
    if word is None:
        return None
    w = word.strip().lower()
    if w.isdigit():
        return int(w)
    return _WORDNUM.get(w)


def people_in(text):
    """How many people the budget is for, or None if not said."""
    m = _PEOPLE.search(text or "")
    if not m:
        return None
    # "myself and my wife" is two, and is how people actually say it.
    if m.group(3):
        return 2
    return _words_to_n(m.group(1) or m.group(2))


def span_days(text):
    """How many days the budget covers, or None."""
    m = _SPAN.search(text or "")
    if not m:
        return None
    n = _words_to_n(m.group(1))
    if n is None:
        return None
    unit = m.group(2).lower().rstrip("s")
    return n * {"day": 1, "week": 7, "month": 30, "year": 365}[unit]


def _unit_of(clause):
    """(per_person, period) for one clause."""
    per_person = bool(_PER_PERSON.search(clause))
    if _PER_DAY.search(clause):
        return per_person, "day"
    if _PER_WEEK.search(clause):
        return per_person, "week"
    if _PER_MONTH.search(clause):
        return per_person, "month"
    return per_person, None


def items_in(text):
    """Every budget line the text actually states.

    Each is {name, cents, per_person, period, ranged, said}. `cents` is None
    when the person named an item without a figure — ⚠️ KEPT AS A
    KNOWN-UNKNOWN RATHER THAN DROPPED OR GUESSED, because "a coffee and a
    baked item for breakfast" is a real line with a real cost that only they
    can supply.
    """
    out = []
    # ⚠️ SPLIT INTO CLAUSES FIRST. A paragraph holds several lines, and the
    # unit words ("per person", "a day") attach to the nearest amount — not to
    # the whole message. Reading the paragraph as one thing is how "per person
    # per day" ended up labelled "month".
    for clause in re.split(r"(?<=[.;])\s+|\s+and\s+(?=\w+\s+(?:should|at|is))",
                           text or ""):
        if not clause.strip():
            continue
        found = [m.group(1).lower() for m in _ITEM.finditer(clause)]
        if not found:
            continue
        # A meal word wins over an ingredient in the same clause; a specific
        # name wins over a heading word.
        meals = [f for f in found if f.rstrip("s") in
                 {m.rstrip("s") for m in _MEALS}]
        specific = [f for f in found if f not in _GENERIC]
        name = (meals or specific or found)[0]
        per_person, period = _unit_of(clause)
        cents, ranged = None, False
        rng = _RANGE.search(clause)
        if rng:
            # The TOP of the range is the budget; see `_RANGE`.
            cents, ranged = to_cents(rng.group(2)), True
        else:
            mm = _MONEY.search(clause)
            if mm:
                cents = to_cents(mm.group(1) or mm.group(2))
        out.append({"name": name, "cents": cents, "per_person": per_person,
                    "period": period, "ranged": ranged,
                    "said": " ".join(clause.split())[:120]})
    # ⚠️ DROP THE HEADING ONCE A REAL LINE EXISTS. "meal costs" is the subject
    # of the paragraph; breakfast, lunch and dinner are the budget.
    if any(i["name"] not in _GENERIC for i in out):
        out = [i for i in out if i["name"] not in _GENERIC]
    # De-duplicate by name, preferring the entry that carries a figure.
    best = {}
    for it in out:
        prev = best.get(it["name"])
        if prev is None or (prev["cents"] is None and it["cents"] is not None):
            best[it["name"]] = it
    return list(best.values())


# Asking for the service in so many words. ⚠️ NOT the only way in — it is one
# signal among several, and `looks_like_budget` works without it.
_ASKS_BUDGET = re.compile(
    r"\b(?:help me|let'?s|can you|i want to|i'?d like to|we should)\b"
    r"[^.?!]{0,40}?\bbudget\b"
    r"|\b(?:create|make|set up|build|work out|plan)\b[^.?!]{0,20}?\bbudget\b",
    re.I)


def looks_like_budget(text):
    """Is this message a budget being described? From its SHAPE, not a phrase.

    ⚠️ NO COMMAND WORD IS REQUIRED, which is the whole point: nobody should
    have to learn patterned speech as commands. Several named items with
    amounts, periods, a count of people or a span — that shape is a budget
    whether the sentence opens "set a budget", "let's break down", or neither.

    ⚠️ AND IT IS DELIBERATELY NOT A SINGLE KEYWORD. "the budget is tight this
    month" names no items and no figures and is conversation.
    """
    t = text or ""
    items = items_in(t)
    figures = len(_MONEY.findall(t)) + (1 if _RANGE.search(t) else 0)
    signals = sum((
        len(items) >= 2,
        figures >= 2,
        bool(people_in(t)),
        bool(span_days(t)),
        bool(re.search(r"\bbudget(?:s|ing)?\b", t, re.I)),
    ))
    # ⚠️ AN EXPLICIT REQUEST OPENS ONE WITH NOTHING IN IT. "help me create a
    # budget for my move" has no figures and no items, and it is unmistakably a
    # request for this service. Requiring items would leave the opening turn
    # unrecognised and make the detail that follows carry every signal by
    # itself.
    if _ASKS_BUDGET.search(t):
        return True
    # Otherwise two independent signals, so a single stray number or the bare
    # word "budget" is not enough on its own.
    return signals >= 2 and bool(items)


# ── held across turns ────────────────────────────────────────────────────────
# ⚠️ A BUDGET ARRIVES IN PIECES AND THE PIECES DO NOT REANNOUNCE THEMSELVES.
# "Help me create a budget for my move" comes first, and the meal detail
# follows as a separate message. Treated independently, the second would have
# to carry every signal by itself. Once a budget is open, later turns extend it
# without needing any keyword.
_OPEN = {}


def open_budget(session="default"):
    return _OPEN.get(session or "default")


def clear(session=None):
    if session is None:
        _OPEN.clear()
    else:
        _OPEN.pop(session, None)


def absorb(text, session="default"):
    """Fold this message into the open budget, opening one if the shape fits.

    Returns the collected budget, or None if this message is not part of one.
    """
    key = session or "default"
    held = _OPEN.get(key)
    if held is None and not looks_like_budget(text):
        return None
    if held is None:
        held = {"items": {}, "people": None, "days": None, "label": None}
        _OPEN[key] = held
    for it in items_in(text):
        prev = held["items"].get(it["name"])
        # ⚠️ A LATER FIGURE REPLACES A MISSING ONE, and a later statement
        # replaces an earlier figure: people correct themselves mid-budget and
        # the most recent number is the one they mean.
        if prev is None or it["cents"] is not None:
            held["items"][it["name"]] = it
    for field, fn in (("people", people_in), ("days", span_days)):
        got = fn(text)
        if got:
            held[field] = got
    where = re.search(r"\b(?:to|in|for)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", text or "")
    if where and not held["label"]:
        held["label"] = where.group(1)
    return held


def total_cents(held):
    """What the stated lines come to over the whole span, or None.

    None when anything needed is missing — ⚠️ A PARTIAL TOTAL IS A WRONG
    TOTAL, and presenting one as the answer is how a $0 budget gets filed.
    """
    if not held or not held.get("days"):
        return None
    people = held.get("people") or 1
    days = held["days"]
    total = 0
    for it in held["items"].values():
        if it["cents"] is None or it["period"] is None:
            return None
        n = it["cents"] * (people if it["per_person"] else 1)
        per_day = {"day": 1.0, "week": 1 / 7.0, "month": 1 / 30.0}[it["period"]]
        total += int(round(n * per_day * days))
    return total


def _money(cents):
    return f"${cents // 100:,}.{cents % 100:02d}"


def missing(held):
    """What still has to be asked for, most blocking first."""
    gaps = []
    for it in held.get("items", {}).values():
        if it["cents"] is None:
            gaps.append(f"what you want to allow for {it['name']}")
        elif it["period"] is None:
            gaps.append(f"whether {it['name']} is per day, week or month")
    if not held.get("people"):
        gaps.append("how many people it covers")
    if not held.get("days"):
        gaps.append("how long it should span")
    return gaps


def say(held):
    """What Mati says back: the lines read correctly, then ONE question.

    ⚠️ COMPOSED IN CODE AND EMITTED, never handed to the model. Every figure
    here is arithmetic over what the person said, and a 3B asked to relay a
    budget rewrites the numbers — the morning-report lesson.

    ⚠️ ONE QUESTION, NOT A FORM. The setup principle: work with partial data
    and ask for the one missing thing that blocks something. A list of five
    questions is the wizard this project keeps refusing to build.
    """
    if not held or not held.get("items"):
        return None
    lines = []
    for it in sorted(held["items"].values(), key=lambda i: i["name"]):
        if it["cents"] is None:
            lines.append(f"  {it['name']} — no figure yet")
            continue
        unit = []
        if it["per_person"]:
            unit.append("per person")
        if it["period"]:
            unit.append(f"per {it['period']}")
        upto = "up to " if it["ranged"] else ""
        lines.append(f"  {it['name']} — {upto}{_money(it['cents'])}"
                     + (" " + " ".join(unit) if unit else ""))
    head = "Budget so far"
    if held.get("label"):
        head += f" for {held['label']}"
    if held.get("people"):
        head += f", {held['people']} people"
    if held.get("days"):
        d = held["days"]
        head += f", {d // 30} months" if d >= 60 else f", {d} days"
    out = [head + ":"] + lines
    total = total_cents(held)
    if total is not None:
        out.append(f"  total over the span — {_money(total)}")
    gaps = missing(held)
    if gaps:
        out.append("")
        out.append(f"I still need {gaps[0]}.")
    return "\n".join(out)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    text = " ".join(argv)
    if not text:
        print("  give me a budget sentence to parse")
        return 1
    print(f"  looks_like_budget: {looks_like_budget(text)}")
    held = absorb(text, session="cli")
    print(say(held) or "  nothing collected")
    clear("cli")
    return 0


if __name__ == "__main__":
    sys.exit(main())
