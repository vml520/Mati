"""
The accountant — a shopping assistant, not a bookkeeper.

Mati keeps its personal-assistant, confidant role here, and this is the
shopping assistant inside it — shopping budgets, payment schedules, finding
local deals, help searching for items.

⚠️ MONEY IS INTEGER CENTS, EVERYWHERE, ALWAYS. `0.1 + 0.2 != 0.3` and a budget
is a comparison against a limit. Formatting to "$12.34" happens at the edge,
in `money()`, and nowhere else. (Shopify's own agent surface uses the same
rule: "integers in the currency's ISO 4217 minor units".)

⚠️ "NO DATA" AND "SPENT NOTHING" ARE DIFFERENT ANSWERS. `spent()` returns None
when nothing has been recorded, and 0 only when something was. "$0 of $200
used" said about an envelope nobody has touched is a licence to overspend, and
it is the first thing the tests check.

⚠️ THE ANNEALER IS NOT USED HERE. Knapsack at 5–50 items is exact by dynamic
programming in milliseconds, and annealing returns no optimality certificate.
For money an approximate answer is a wrong answer.

⚠️ WHAT THIS MODULE DOES NOT OWN. Payment schedules are `agenda` events — a
bill is a recurring dated fact and §Bills here is a thin adapter, not storage.
Deal feeds are `world`. Nudges and quiet hours are `schedule` + `notify`.
Local circulars are `flipp`, behind its own firewall. If any of those grow a
table in this file, something has gone wrong.

⚠️ `forget()` IS THE ONE REAL DELETE IN MATI, BY DECISION. Everywhere else
corrections annotate and withdrawals hide. Spending is the exception because a
recorded purchase can be someone else's business, and a gift is the obvious
case. It is named plainly, it is the only one, and `correct()` beside it still
annotates rather than overwrites.
"""

from __future__ import annotations

import re
import sqlite3
import time

from . import bridge
from . import config as C

B_TABLE = "budgets"
S_TABLE = "spending"
W_TABLE = "watchlist"
P_TABLE = "price_seen"

PERIODS = ("week", "month", "once")

# What the watchlist calls "a deal worth mentioning" when no target is set:
# a listed original price at least this much above the current one.
DEFAULT_DROP = 0.20


# ── money at the edge ────────────────────────────────────────────────────────

def money(cents):
    """Integer cents to "$12.34". The ONLY place cents become a string.

    ⚠️ Negative amounts read as "-$3.00", not "$-3.00".
    """
    if cents is None:
        return "—"
    sign = "-" if cents < 0 else ""
    c = abs(int(cents))
    return f"{sign}${c // 100:,}.{c % 100:02d}"


def to_cents(text):
    """"$12.34", "12.34", "12" -> 1234, 1234, 1200. None if unreadable.

    ⚠️ ROUNDED, NEVER TRUNCATED. `int(8.70 * 100)` is **869** and
    `int(0.29 * 100)` is **28**, because neither has an exact binary form. A
    budget tool that loses a cent per entry is not a budget tool. (4.99 happens
    to be safe, which is exactly why picking one example and trusting it is not
    a check.)

    ⚠️ An unreadable amount is None, never zero — recording a purchase as
    free is worse than refusing it.
    """
    if text is None:
        return None
    if isinstance(text, int) and not isinstance(text, bool):
        return text
    s = str(text).strip().replace("$", "").replace(",", "").replace(" ", "")
    if not re.match(r"^-?\d+(\.\d{1,2})?$", s):
        return None
    try:
        return int(round(float(s) * 100))
    except ValueError:
        return None


# ── storage ──────────────────────────────────────────────────────────────────

def _conn(db_path=None):
    return sqlite3.connect(db_path or bridge.NOTES_DB_PATH, timeout=30)


def ensure(db_path=None):
    """Create the four tables. Idempotent; safe to call on every entry point."""
    conn = _conn(db_path)
    try:
        conn.execute(f"""CREATE TABLE IF NOT EXISTS {B_TABLE} (
                budget_id  INTEGER PRIMARY KEY,
                name       TEXT NOT NULL,
                limit_c    INTEGER NOT NULL,
                period     TEXT NOT NULL,
                created    TEXT,
                withdrawn  TEXT
            )""")
        conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS {B_TABLE}_name "
                     f"ON {B_TABLE}(name)")
        # ⚠️ `amount_c` and `at` stay CLEAR so totals work without the key —
        # the same split `agenda` uses for times, and `recall` for addresses.
        # `merchant` and `item` are SEALED: what you bought is confidant
        # material in a way that "$14.90 on the 3rd" is not.
        conn.execute(f"""CREATE TABLE IF NOT EXISTS {S_TABLE} (
                entry_id   INTEGER PRIMARY KEY,
                amount_c   INTEGER NOT NULL,
                merchant   TEXT,
                item       TEXT,
                budget     TEXT,
                at         TEXT NOT NULL,
                note       TEXT,
                corrected  TEXT,
                recorded   TEXT
            )""")
        conn.execute(f"CREATE INDEX IF NOT EXISTS {S_TABLE}_when "
                     f"ON {S_TABLE}(at)")
        conn.execute(f"CREATE INDEX IF NOT EXISTS {S_TABLE}_budget "
                     f"ON {S_TABLE}(budget)")
        conn.execute(f"""CREATE TABLE IF NOT EXISTS {W_TABLE} (
                watch_id   INTEGER PRIMARY KEY,
                item       TEXT NOT NULL,
                target_c   INTEGER,
                note       TEXT,
                added      TEXT,
                dropped    TEXT
            )""")
        conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS {W_TABLE}_item "
                     f"ON {W_TABLE}(item)")
        conn.execute(f"""CREATE TABLE IF NOT EXISTS {P_TABLE} (
                seen_id    INTEGER PRIMARY KEY,
                item       TEXT NOT NULL,
                cents      INTEGER NOT NULL,
                source     TEXT,
                merchant   TEXT,
                at         TEXT NOT NULL,
                mentioned  TEXT
            )""")
        conn.execute(f"CREATE INDEX IF NOT EXISTS {P_TABLE}_item "
                     f"ON {P_TABLE}(item, at)")
        conn.commit()
        return True
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def _key(item):
    """The watchlist's matching key. Case and spacing must not create a second
    entry for the same thing."""
    return " ".join(str(item or "").lower().split())


def _since(days):
    if days is None:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%S",
                         time.localtime(time.time() - days * 86400))


def _period_start(period):
    """Start of the current window, as a local ISO stamp. None for 'once'."""
    now = time.localtime()
    if period == "week":
        return _since(now.tm_wday)      # back to Monday
    if period == "month":
        return time.strftime(f"%Y-%m-01T00:00:00", now)
    return None


# ── budgets ──────────────────────────────────────────────────────────────────

def set_budget(name, cents, period="month", db_path=None):
    """Create or change an envelope. Returns its name, or None if refused."""
    name = " ".join(str(name or "").split())
    cents = to_cents(cents)
    if not name or cents is None or cents < 0 or period not in PERIODS:
        return None
    ensure(db_path)
    conn = _conn(db_path)
    try:
        conn.execute(
            f"INSERT INTO {B_TABLE} (name, limit_c, period, created) "
            f"VALUES (?,?,?,?) ON CONFLICT(name) DO UPDATE SET "
            f"limit_c=excluded.limit_c, period=excluded.period, withdrawn=NULL",
            (name, cents, period, bridge._now()))
        conn.commit()
        return name
    except sqlite3.Error:
        return None
    finally:
        conn.close()


def budgets(db_path=None, include_withdrawn=False):
    """Every envelope, with what it holds and what is left."""
    ensure(db_path)
    conn = _conn(db_path)
    try:
        where = "" if include_withdrawn else "WHERE withdrawn IS NULL"
        rows = conn.execute(
            f"SELECT name, limit_c, period, withdrawn FROM {B_TABLE} "
            f"{where} ORDER BY name").fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    out = []
    for name, limit_c, period, withdrawn in rows:
        used = spent(name, db_path=db_path)
        out.append({"name": name, "limit_c": limit_c, "period": period,
                    "spent_c": used,
                    "left_c": None if used is None else limit_c - used,
                    "withdrawn": withdrawn})
    return out


def remove_budget(name, db_path=None):
    """Withdraw an envelope. ⚠️ Never deletes — the spending history stays."""
    ensure(db_path)
    conn = _conn(db_path)
    try:
        cur = conn.execute(
            f"UPDATE {B_TABLE} SET withdrawn=? WHERE name=? AND withdrawn IS NULL",
            (bridge._now(), " ".join(str(name or "").split())))
        conn.commit()
        return cur.rowcount > 0
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def spent(name, since=None, db_path=None):
    """Cents spent against an envelope in its current window.

    ⚠️ RETURNS None WHEN NOTHING HAS BEEN RECORDED, AND 0 ONLY WHEN SOMETHING
    HAS. This is the single most important line in the module. "$0 of $200
    used" said about an envelope nobody has touched reads as permission.
    """
    ensure(db_path)
    name = " ".join(str(name or "").split())
    conn = _conn(db_path)
    try:
        row = conn.execute(
            f"SELECT period FROM {B_TABLE} WHERE name=?", (name,)).fetchone()
        start = since or (_period_start(row[0]) if row else None)
        sql = (f"SELECT COUNT(*), COALESCE(SUM(amount_c),0) FROM {S_TABLE} "
               f"WHERE budget=?")
        args = [name]
        if start:
            sql += " AND at >= ?"
            args.append(start)
        n, total = conn.execute(sql, args).fetchone()
        return None if not n else int(total)
    except sqlite3.Error:
        return None
    finally:
        conn.close()


def remaining(name, db_path=None):
    """(cents_left, limit_cents), or None when the envelope or data is absent."""
    ensure(db_path)
    conn = _conn(db_path)
    try:
        row = conn.execute(
            f"SELECT limit_c FROM {B_TABLE} WHERE name=? AND withdrawn IS NULL",
            (" ".join(str(name or "").split()),)).fetchone()
    except sqlite3.Error:
        return None
    finally:
        conn.close()
    if not row:
        return None
    used = spent(name, db_path=db_path)
    if used is None:
        return None                     # ⚠️ not (limit, limit). No data.
    return (row[0] - used, row[0])


def over_budget(db_path=None):
    """Envelopes past their limit, worst first. [] when none are, or none known."""
    out = [b for b in budgets(db_path=db_path)
           if b["left_c"] is not None and b["left_c"] < 0]
    out.sort(key=lambda b: b["left_c"])
    return out


# ── spending ─────────────────────────────────────────────────────────────────

def record(cents, merchant=None, item=None, budget=None, at=None, note=None,
           db_path=None):
    """Record one purchase. Returns the entry id, or None if refused.

    ⚠️ An unreadable amount is REFUSED, not stored as zero.
    """
    from . import vault
    cents = to_cents(cents)
    if cents is None:
        return None
    ensure(db_path)
    conn = _conn(db_path)
    try:
        cur = conn.execute(
            f"INSERT INTO {S_TABLE} (amount_c, merchant, item, budget, at, "
            f"note, recorded) VALUES (?,?,?,?,?,?,?)",
            (cents,
             vault.seal(merchant) if merchant else None,
             vault.seal(item) if item else None,
             " ".join(str(budget).split()) if budget else None,
             at or bridge._now(),
             vault.seal(note) if note else None,
             bridge._now()))
        conn.commit()
        return cur.lastrowid
    except sqlite3.Error:
        return None
    finally:
        conn.close()


def spending(since_days=30, budget=None, limit=50, db_path=None):
    """Recent purchases, newest first, unsealed."""
    from . import vault
    ensure(db_path)
    conn = _conn(db_path)
    try:
        sql = (f"SELECT entry_id, amount_c, merchant, item, budget, at, note, "
               f"corrected FROM {S_TABLE} WHERE 1=1")
        args = []
        start = _since(since_days)
        if start:
            sql += " AND at >= ?"; args.append(start)
        if budget:
            sql += " AND budget = ?"; args.append(" ".join(str(budget).split()))
        sql += " ORDER BY at DESC LIMIT ?"; args.append(int(limit))
        rows = conn.execute(sql, args).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    out = []
    for eid, amt, merch, item, bud, at, note, corrected in rows:
        out.append({
            "entry_id": eid, "cents": amt,
            "merchant": vault.open_body(merch) if merch else None,
            "item": vault.open_body(item) if item else None,
            "budget": bud, "at": at,
            "note": vault.open_body(note) if note else None,
            "corrected": corrected,
        })
    return out


def correct(entry_id, cents=None, merchant=None, item=None, budget=None,
            db_path=None):
    """Fix an entry. ⚠️ ANNOTATES — the original is kept in `corrected`.

    This is the `correction.py` rule applied to money: what was believed, and
    what replaced it, both stay visible. `forget()` is the only erasure.
    """
    from . import vault
    ensure(db_path)
    conn = _conn(db_path)
    try:
        row = conn.execute(
            f"SELECT amount_c, merchant, item, budget, corrected FROM {S_TABLE} "
            f"WHERE entry_id=?", (entry_id,)).fetchone()
        if not row:
            return False
        was = (f"was {money(row[0])}"
               + (f" at {vault.open_body(row[1])}" if row[1] else "")
               + (f" for {vault.open_body(row[2])}" if row[2] else "")
               + (f" under {row[3]}" if row[3] else "")
               + f" until {bridge._now()}")
        trail = (row[4] + " | " + was) if row[4] else was
        sets, args = ["corrected=?"], [trail]
        c = to_cents(cents)
        if c is not None:
            sets.append("amount_c=?"); args.append(c)
        if merchant is not None:
            sets.append("merchant=?"); args.append(vault.seal(merchant))
        if item is not None:
            sets.append("item=?"); args.append(vault.seal(item))
        if budget is not None:
            sets.append("budget=?"); args.append(" ".join(str(budget).split()))
        args.append(entry_id)
        conn.execute(f"UPDATE {S_TABLE} SET {', '.join(sets)} WHERE entry_id=?",
                     args)
        conn.commit()
        return True
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def forget(entry_id, db_path=None):
    """⚠️ THE ONE REAL DELETE IN MATI. Removes a purchase entirely.

    Everywhere else corrections annotate and withdrawals hide. Spending is the
    exception by decision: a recorded purchase can be someone else's business —
    a gift is the obvious case — and hiding a row that still sums into a total
    is not forgetting it.

    ⚠️ So it must really go. No tombstone, no `forgotten` column, nothing left
    that a later feature could resurrect and recite.
    """
    ensure(db_path)
    conn = _conn(db_path)
    try:
        cur = conn.execute(f"DELETE FROM {S_TABLE} WHERE entry_id=?",
                           (entry_id,))
        conn.commit()
        return cur.rowcount > 0
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def summary(days=30, db_path=None):
    """Where the money went: by envelope and by merchant. {} when nothing known."""
    rows = spending(since_days=days, limit=10000, db_path=db_path)
    if not rows:
        return {}
    by_budget, by_merchant = {}, {}
    for r in rows:
        by_budget[r["budget"] or "unassigned"] = (
            by_budget.get(r["budget"] or "unassigned", 0) + r["cents"])
        if r["merchant"]:
            by_merchant[r["merchant"]] = by_merchant.get(r["merchant"], 0) + r["cents"]
    return {
        "days": days,
        "entries": len(rows),
        "total_c": sum(r["cents"] for r in rows),
        "by_budget": dict(sorted(by_budget.items(), key=lambda kv: -kv[1])),
        "by_merchant": dict(sorted(by_merchant.items(),
                                   key=lambda kv: -kv[1])[:10]),
    }


# ── the watchlist — search, inverted ─────────────────────────────────────────

def watch(item, target=None, note=None, db_path=None):
    """Watch for an item, optionally under a target price. Returns its key."""
    from . import vault
    key = _key(item)
    if not key:
        return None
    ensure(db_path)
    conn = _conn(db_path)
    try:
        conn.execute(
            f"INSERT INTO {W_TABLE} (item, target_c, note, added) "
            f"VALUES (?,?,?,?) ON CONFLICT(item) DO UPDATE SET "
            f"target_c=excluded.target_c, note=excluded.note, dropped=NULL",
            # ⚠️ THE NOTE IS SEALED, THE ITEM IS NOT, AND THAT ASYMMETRY IS
            # DELIBERATE. `item` is the upsert key — `ON CONFLICT(item)` above
            # and `WHERE item=?` in `unwatch`/`history` — and `vault.seal` is
            # nondeterministic, so a sealed key would make every write a new
            # row and every lookup a miss. The note is never compared.
            (key, to_cents(target), vault.seal(note) if note else None,
             bridge._now()))
        conn.commit()
        return key
    except sqlite3.Error:
        return None
    finally:
        conn.close()


def watching(db_path=None):
    from . import vault
    """What is being watched, with the cheapest ever seen beside each."""
    ensure(db_path)
    conn = _conn(db_path)
    try:
        rows = conn.execute(
            f"SELECT item, target_c, note, added FROM {W_TABLE} "
            f"WHERE dropped IS NULL ORDER BY added DESC").fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    out = []
    for item, target, note, added in rows:
        best = cheapest(item, db_path=db_path)
        out.append({"item": item, "target_c": target,
                    "note": vault.open_body(note) if note else None,
                    "added": added, "cheapest": best})
    return out


def unwatch(item, db_path=None):
    """Stop watching. ⚠️ Withdraws — the price history it built survives."""
    ensure(db_path)
    conn = _conn(db_path)
    try:
        cur = conn.execute(
            f"UPDATE {W_TABLE} SET dropped=? WHERE item=? AND dropped IS NULL",
            (bridge._now(), _key(item)))
        conn.commit()
        return cur.rowcount > 0
    except sqlite3.Error:
        return False
    finally:
        conn.close()


# ── price history — what makes this more than a list ─────────────────────────

def price_seen(item, cents, source=None, merchant=None, at=None, db_path=None):
    """Record one price observation. Returns the id, or None if unreadable."""
    from . import vault
    cents = to_cents(cents)
    key = _key(item)
    if cents is None or not key:
        return None
    ensure(db_path)
    conn = _conn(db_path)
    try:
        cur = conn.execute(
            f"INSERT INTO {P_TABLE} (item, cents, source, merchant, at) "
            f"VALUES (?,?,?,?,?)",
            # ⚠️ `merchant` SEALED, `source` CLEAR. Where a person shops is
            # theirs; "flipp" is the name of a feed and says nothing about
            # them. `item` is the lookup key, as in `watchlist`.
            (key, cents, source, vault.seal(merchant) if merchant else None,
             at or bridge._now()))
        conn.commit()
        return cur.lastrowid
    except sqlite3.Error:
        return None
    finally:
        conn.close()


def history(item, db_path=None):
    """Every price seen for an item, oldest first."""
    from . import vault
    ensure(db_path)
    conn = _conn(db_path)
    try:
        rows = conn.execute(
            f"SELECT at, cents, source, merchant FROM {P_TABLE} "
            f"WHERE item=? ORDER BY at", (_key(item),)).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    return [{"at": a, "cents": c, "source": s,
             "merchant": vault.open_body(m) if m else None}
            for a, c, s, m in rows]


def cheapest(item, db_path=None):
    """The lowest price ever recorded for an item, or None if never seen.

    ⚠️ "That is the lowest I have seen it" is a claim only a record can
    support, and this is the record. ⚠️ It says nothing about prices Mati
    never observed, so the phrasing must stay "the lowest I have seen",
    never "the lowest there is".
    """
    h = history(item, db_path=db_path)
    if not h:
        return None
    return min(h, key=lambda r: r["cents"])


def is_a_good_price(item, cents, db_path=None):
    """(verdict, why) for a price, from the record alone. Never a guess.

    Returns one of "lowest", "good", "typical", "high", or "unknown" — and
    "unknown" is a real answer whenever there is not enough history.
    """
    cents = to_cents(cents)
    if cents is None:
        return ("unknown", "I could not read that price.")
    h = [r["cents"] for r in history(item, db_path=db_path)]
    if len(h) < 3:
        # ⚠️ Two observations cannot establish "typical". Refusing here is the
        # difference between a record and an opinion.
        return ("unknown", f"I have only seen {len(h)} price(s) for that.")
    lo, hi = min(h), max(h)
    mid = sorted(h)[len(h) // 2]
    if cents <= lo:
        return ("lowest", f"that is the lowest I have seen, under {money(lo)}.")
    if cents < mid:
        return ("good", f"below the {money(mid)} I usually see.")
    if cents <= hi:
        return ("typical", f"about the {money(mid)} I usually see.")
    return ("high", f"above the {money(hi)} I have seen it at.")


# ── deals: local circulars, then the feeds ───────────────────────────────────

def local_deals(query=None, limit=10, db_path=None):
    """Circular items near the person, matched to the watchlist if no query.

    ⚠️ RAISES `flipp.Unavailable` RATHER THAN RETURNING []. "No deals near
    you" and "I could not look" are different sentences and the caller must
    be able to tell them apart. The whole reason `flipp.py` is a separate
    file is so this is the only thing that can go wrong.
    """
    from . import flipp
    terms = [query] if query else [w["item"] for w in watching(db_path=db_path)]
    if not terms:
        return []
    found = []
    for t in terms[:5]:                  # a turn is not a crawl
        for d in flipp.search(t, limit=limit):
            d["watching"] = None if query else t
            found.append(d)
    # Remember what things cost, so `cheapest` has something to say later.
    for d in found:
        price_seen(d.get("watching") or d["name"], d["cents"],
                   source="flipp", merchant=d.get("merchant"), db_path=db_path)
    found.sort(key=lambda d: d["cents"])
    return found[:limit]


def matches(items=None, db_path=None):
    """Deal-feed items that hit the watchlist. Reads `world`, stores nothing new.

    ⚠️ A target price is checked against the deal's OWN price only. Comparing
    a $6 four-pack with a $3 single is the `item_weight` trap from `flipp.py`
    in another costume, and without a unit the comparison is refused.
    """
    from . import world
    watched = watching(db_path=db_path)
    if not watched:
        return []
    if items is None:
        try:
            items = world.playable("", k=200, db_path=db_path) or []
        except Exception:                                    # noqa: BLE001
            items = []
    out = []
    for w in watched:
        for it in items:
            text = f"{it.get('title','')} {it.get('summary','')}".lower()
            if w["item"] not in text:
                continue
            c = _price_in(text)
            if c is None:
                continue
            if w["target_c"] is not None and c > w["target_c"]:
                continue
            out.append({"item": w["item"], "cents": c,
                        "title": it.get("title"), "link": it.get("link"),
                        "target_c": w["target_c"]})
    return out


_PRICE = re.compile(r"\$\s?(\d{1,6}(?:\.\d{2})?)")


def _price_in(text):
    """The first price in a string, in cents, or None. ⚠️ Never a guess."""
    m = _PRICE.search(text or "")
    return to_cents(m.group(1)) if m else None


def track_search(page_url, item=None, target=None, db_path=None):
    """The person pastes a shopping-search URL; Mati tracks it. (ok, message).

    ⚠️ THE WATCHLIST IS USER-EXTENSIBLE HERE, and that is the point. A search
    someone ran themselves covers what they actually want, and a search URL is
    the most natural thing to hand over — it is already in the address bar.
    Both of these work:

        slickdeals.net/newsearch.php?q=espresso+machine&rss=1
            -> items titled "'espresso machine' search", prices in the titles
        allbirds.com/collections/mens-shoes.atom
            -> entries with a structured <price>: 130.00, 140.00

    ⚠️ NOTHING HERE WORKS AROUND A BLOCK. `world.discover_feed` reads only
    what a site advertises about itself. eBay and Craigslist return their
    block pages to this exactly as they do to everything else, and the honest
    answer for them is that they publish no feed Mati may read.
    """
    from . import world
    feed, how = world.discover_feed(page_url)
    if not feed:
        return False, how
    ok, msg = world.subscribe(feed, name=item or None)
    if not ok:
        return False, msg
    # The search is only half of it — what is being looked FOR is what makes a
    # crossing recognisable later.
    if item:
        watch(item, target=target, note=f"from {page_url}", db_path=db_path)
    return True, f"{msg} — {how}"


# ── bills: a thin adapter over `agenda`, NOT storage ─────────────────────────

def bill(title, cents, starts, rrule="FREQ=MONTHLY", db_path=None):
    """Put a recurring payment on the calendar. Returns the agenda event id.

    ⚠️ THREE FUNCTIONS AND NO TABLE. A bill is a recurring dated fact, which
    is exactly what `agenda` already stores, seals and recurs. If this group
    grows storage of its own, the design went wrong.
    """
    from . import agenda
    c = to_cents(cents)
    if c is None:
        return None
    return agenda.put(starts, f"{title} — {money(c)}", rrule=rrule,
                      ext_id=f"bill:{_key(title)}", source="accountant",
                      db_path=db_path)


def bills(days=30, db_path=None):
    from . import agenda
    return [e for e in agenda.upcoming(days=days, db_path=db_path)
            if (e.get("source") == "accountant")]


def due_soon(days=7, db_path=None):
    return bills(days=days, db_path=db_path)


# ── what Mati says ──────────────────────────────────────────────────────────

def line(db_path=None):
    """One sentence for the morning report, or ""."""
    bits = []
    over = over_budget(db_path=db_path)
    if over:
        b = over[0]
        bits.append(f"{b['name']} is {money(-b['left_c'])} over")
    else:
        live = [b for b in budgets(db_path=db_path) if b["left_c"] is not None]
        if live:
            b = min(live, key=lambda x: x["left_c"])
            bits.append(f"{money(b['left_c'])} left in {b['name']}")
    due = due_soon(days=7, db_path=db_path)
    if due:
        bits.append(f"{len(due)} payment(s) due this week")
    return "; ".join(bits)


def block(db_path=None):
    """The `<computed>` money block. ⚠️ Never stored.

    A balance written into the store becomes "I have $340 left" recited
    forever — stale the moment it is written. Same rule as `weather` and
    `health`, and the same provenance kind.

    ⚠️ First person, and never "you" — a block addressed to Mati comes back
    out as a claim about the person the moment Mati speaks.
    """
    bs = budgets(db_path=db_path)
    due = due_soon(days=14, db_path=db_path)
    watch_rows = watching(db_path=db_path)
    if not (bs or due or watch_rows):
        return ""
    L = ["<computed>",
         "Worked out from the record just now — not remembered, and true only "
         "at this moment:"]
    for b in bs:
        if b["spent_c"] is None:
            # ⚠️ THE RULE, SAID OUT LOUD. Not "$0 of $200 used".
            L.append(f"  - {b['name']}: {money(b['limit_c'])} per "
                     f"{b['period']}, with nothing recorded against it yet.")
        else:
            state = "over by " + money(-b["left_c"]) if b["left_c"] < 0 \
                else money(b["left_c"]) + " left"
            L.append(f"  - {b['name']}: {money(b['spent_c'])} of "
                     f"{money(b['limit_c'])} this {b['period']}, {state}.")
    for e in due[:5]:
        L.append(f"  - Due {str(e.get('starts'))[:10]}: {e.get('title')}")
    for w in watch_rows[:5]:
        s = f"  - Watching {w['item']}"
        if w["target_c"] is not None:
            s += f" under {money(w['target_c'])}"
        if w["cheapest"]:
            s += f"; cheapest I have seen is {money(w['cheapest']['cents'])}"
        L.append(s + ".")
    L.append("</computed>")
    return "\n".join(L)


# ⚠️ A LITERAL LIST, like `weather` and the charter injection. Money questions
# must not be guessed at by a classifier — a false positive puts budget figures
# into a turn that was never about money.
_ASKS = (
    "budget", "spent", "spend", "spending", "afford", "how much money",
    "left this month", "over budget", "my money", "bill", "bills", "due",
    "payment", "payments", "deal", "deals", "on sale", "sale price",
    "watchlist", "watching for", "price of", "cheaper", "cheapest",
    "shopping list", "groceries", "coupon",
    "presupuesto", "cuánto gasté", "cuanto gaste", "oferta", "ofertas",
)


def asks_about_money(text_in):
    t = (text_in or "").lower()
    return any(p in t for p in _ASKS)


# ── Hearing money in a sentence ──────────────────────────────────────────────
#
# ⚠️ A TURN HAS TO BE ABLE TO REACH THE WRITERS. A read block alone would
# recognise "set a grocery budget of 400", answer it, and store nothing:
# `set_budget`, `record`, `watch`, `correct` and `forget` are reached from a
# turn through here.
#
# ⚠️ A FALSE POSITIVE HERE RECORDS A WRONG NUMBER, which is worse than a miss
# in every other recogniser in this project. Money gets the literal treatment:
# an explicit verb AND an explicit amount, or nothing happens.

_AMOUNT = re.compile(
    r"(?:\$\s*)?(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)"
    r"\s*(?:dollars?|bucks?|usd)?", re.I)

_SET_BUDGET = re.compile(
    r"\b(?:set|make|create|start|give me|put)\b[^.?!]{0,40}?\bbudget\b"
    r"|\bbudget\b[^.?!]{0,24}?\b(?:of|at|to|is)\b", re.I)

_SPEND = re.compile(
    r"\b(?:i\s+)?(?:spent|paid|bought|purchased|picked up)\b"
    r"|\bthat (?:cost|was)\b|\bit cost\b", re.I)

_PERIOD = re.compile(r"\b(?:a|per|each|every)\s*(week|month)\b"
                     r"|\b(weekly|monthly)\b", re.I)

# "on groceries", "for groceries", "at Safeway"
_ON = re.compile(r"\b(?:on|for)\s+(?P<what>[a-z][\w' -]{1,28}?)"
                 r"(?=\s*(?:$|[.,;]|\bbudget\b|\bper\b|\ba\s+(?:week|month)))",
                 re.I)
_AT = re.compile(r"\bat\s+(?P<where>[A-Z][\w'&. -]{1,28}?)"
                 r"(?=\s*(?:$|[.,;]|\bon\b|\bfor\b))")


# A sentence ASKING about money rather than instructing. ⚠️ Checked before any
# write pattern, and a question is never a write.
_ASKING_MONEY = re.compile(
    r"^\s*(?:how much|how many|what|which|where|when|did|do|does|have|has|am|"
    r"is|are|can|could)\b"
    r"|\bhow much (?:have|did|do|has)\b"
    r"|\?\s*$", re.I)


def _amount_cents(text):
    """The first money amount in a sentence, in cents. None if there is none.

    ⚠️ `to_cents` ROUNDS and never truncates — `int(8.70 * 100)` is 869. That
    is already settled in this module and this just routes through it.
    """
    m = _AMOUNT.search(text or "")
    return to_cents(m.group(1).replace(",", "")) if m else None


# ⚠️ ENGLISH PUTS THE ENVELOPE'S NAME ON EITHER SIDE OF THE WORD. "a GROCERY
# budget" and "a budget FOR GROCERIES" are the same request, and the first
# form was unreachable — it has no "on"/"for" to anchor on, so the whole
# instruction came back as "I cannot tell what it is for".
_BUDGET_ADJ = re.compile(r"\b(?:a|an|the|my)?\s*([a-z][\w'-]{2,20})\s+budget\b",
                         re.I)
_BUDGET_FOR = re.compile(r"\bbudget\b\s*(?:of\s*\$?[\d.,]+\s*)?"
                         r"(?:for|on)\s+(?:a|an|the|my)?\s*"
                         r"([a-z][\w' -]{2,24}?)"
                         r"(?=\s*(?:$|[.,;]|\bof\b|\bat\b|\bper\b|\ba\s+(?:week|month)|"
                         r"\b(?:weekly|monthly)\b|\$|\d))", re.I)

# Words that are never an envelope name, however the sentence is shaped.
_NOT_A_NAME = {"a", "an", "the", "my", "new", "monthly", "weekly", "me",
               "that", "this", "some", "another", "total", "overall"}


def _budget_name(t):
    """The envelope a budget instruction names, or None."""
    for pat in (_BUDGET_FOR, _BUDGET_ADJ):
        m = pat.search(t)
        if not m:
            continue
        name = " ".join(m.group(1).split()).strip(" .,'").lower()
        name = re.sub(r"\b(?:a|an|the|my)\b", " ", name).strip()
        if name and name not in _NOT_A_NAME:
            return _singular(name) or None
    return None


def _singular(name):
    """Fold an obvious plural so one envelope does not become two.

    ⚠️ CONSERVATIVE ON PURPOSE. Naive rules — `s$ -> ''` then `ies$ -> y` —
    turn "gas" into "ga" and "groceries" into "grocerie". A wrong envelope name
    is worse than an unmerged one: the person sets a budget and is then told
    there is no data for it.

    So: only the last word, only when it is long enough to be a real plural,
    and never after s/ss/us/is — which is what makes "gas" safe.
    """
    parts = name.split()
    if not parts:
        return name
    w = parts[-1]
    if len(w) >= 5 and w.endswith("ies"):
        w = w[:-3] + "y"
    elif len(w) >= 5 and w.endswith("s") and not w.endswith(
            ("ss", "us", "is", "os")):
        w = w[:-1]
    return " ".join(parts[:-1] + [w])


def hears_money(text_in):
    """A money INSTRUCTION, or None. ⚠️ Never a guess — see the note above.

    Returns one of:
        {"verb": "budget",  "name": …, "cents": …, "period": …}
        {"verb": "spend",   "cents": …, "merchant": …, "item": …}
        {"verb": "unclear", "why": …}   — heard, but something is missing
    """
    t = " ".join((text_in or "").split())
    if not t:
        return None

    # ⚠️ "how much have I spent" IS A QUESTION, NOT A FAILED RECORDING. It
    # contains "spent", so `_SPEND` matches it and finds no amount — and
    # without this Mati would offer to record a purchase in answer to a request
    # for a total. Same shape as `play` matching "What role did X play".
    if _ASKING_MONEY.search(t):
        return None

    if _SET_BUDGET.search(t):
        cents = _amount_cents(t)
        name = _budget_name(t)
        p = _PERIOD.search(t)
        period = ((p.group(1) or p.group(2) or "month").replace("ly", "")
                  if p else "month")
        period = {"week": "week", "month": "month",
                  "weekly": "week", "monthly": "month"}.get(period, "month")
        # ⚠️ ASKS RATHER THAN ASSUMES. A budget with no amount, or no name, is
        # a request Mati cannot honour — and inventing either is how a number
        # nobody chose ends up governing their spending.
        if cents is None or not name:
            missing = "an amount" if cents is None else "what it is for"
            return {"verb": "unclear", "why": missing, "kind": "budget"}
        return {"verb": "budget", "name": name, "cents": cents,
                "period": period}

    if _SPEND.search(t):
        cents = _amount_cents(t)
        if cents is None:
            return {"verb": "unclear", "why": "how much", "kind": "spend"}
        at = _AT.search(t)
        on = _ON.search(t)
        return {"verb": "spend", "cents": cents,
                "merchant": at.group("where").strip() if at else None,
                "item": on.group("what").strip() if on else None}
    return None


def act(req, db_path=None):
    """Carry out a money instruction from `hears_money`. Returns a sentence.

    ⚠️ THIS IS THE WRITER A TURN REACHES. Everything below it is otherwise
    reachable only from the terminal.
    """
    if not req:
        return None
    verb = req.get("verb")

    if verb == "unclear":
        # ⚠️ ASKS FOR THE ONE MISSING THING. Mati works with limited data and
        # asks for the missing piece when it is needed — and the same sentence
        # can arrive half-formed any day. Guessing an amount is how a number
        # nobody chose starts governing their spending.
        if req.get("kind") == "budget":
            return (f"I can set that up — I just need {req.get('why')}. "
                    f"Say it like “a grocery budget of 400 a month”.")
        return (f"I can record that — I just need {req.get('why')}. "
                f"Say it like “I spent 24.50 at Safeway on groceries”.")

    if verb == "budget":
        name = set_budget(req["name"], req["cents"], period=req["period"],
                          db_path=db_path)
        if not name:
            # ⚠️ ASK, DO NOT REPORT. Clarify before an error reaches an error
            # handler. A budget that would not store is almost always a NAME
            # Mati read badly out of a longer sentence — a budget called "lunch
            # that can cover from", say — so the useful answer asks what to
            # call it, not that something failed.
            return (f"I have {money(req['cents'])} a {req['period']}, but I "
                    f"am not sure what to call it. What should this budget "
                    f"be named?")
        # ⚠️ READS BACK THE NUMBER IT STORED, not the one it was given. An
        # amount that failed `to_cents` and an amount that rounded are both
        # visible this way, which is the point of integer cents.
        return (f"{money(req['cents'])} a {req['period']} for {name}. "
                f"Tell me what you spend and I will keep the running total.")

    if verb == "spend":
        # ⚠️ ATTACH TO AN EXISTING ENVELOPE RATHER THAN INVENT ONE. A purchase
        # naming a budget nobody set would silently create it, and a budget
        # with a limit nobody chose is worse than no budget.
        budget = None
        item = (req.get("item") or "").strip().lower()
        if item:
            want = _singular(item)
            for b in budgets(db_path=db_path):
                if _singular(b["name"].lower()) == want:
                    budget = b["name"]
                    break
        eid = record(req["cents"], merchant=req.get("merchant"),
                     item=req.get("item"), budget=budget, db_path=db_path)
        if eid is None:
            # ⚠️ THE AMOUNT IS THE ONE THING ONLY THEY KNOW, so ask for it
            # rather than announcing a failure and dropping the spend. Silence
            # about money is how "no data" becomes "spent nothing".
            where = (f" at {req['merchant']}" if req.get("merchant")
                     else (f" on {req['item']}" if req.get("item") else ""))
            return f"How much was that{where}? I did not catch the amount."
        said = f"Recorded {money(req['cents'])}"
        if req.get("merchant"):
            said += f" at {req['merchant']}"
        if budget:
            left = remaining(budget, db_path=db_path)
            if left:
                cents_left, limit_c = left
                said += (f", against {budget} — {money(cents_left)} left of "
                         f"{money(limit_c)}")
        elif item:
            # ⚠️ SAYS IT IS UNFILED rather than silently losing the category.
            said += f" for {req['item']}, with no budget by that name"
        return said + "."
    return None


def status(db_path=None):
    from . import flipp
    bs = budgets(db_path=db_path)
    return {
        "budgets": len(bs),
        "budgets_with_data": sum(1 for b in bs if b["spent_c"] is not None),
        "over": len(over_budget(db_path=db_path)),
        "watching": len(watching(db_path=db_path)),
        "local_source": flipp.status(),
    }
