"""
Local store circulars, by postal code. THE ONLY LOCAL DEAL SOURCE THAT WORKS.

Craigslist, eBay, Facebook Marketplace and the classifieds sites are blocked
or dead to an automated reader. Flipp answers.

⚠️ THIS WHOLE FILE IS A FIREWALL, AND THAT IS ITS REASON FOR EXISTING.
`backflipp.wishabi.com` is the backend Flipp's own web app talks to. It is
undocumented: no key, no block, and no published terms either. That is a third
category, distinct from both the RSS feeds Mati subscribes to (offered) and
from Craigslist (which returns `<title>blocked</title>`, and working around
that would be the violation). Nothing here is being bypassed — but nothing
here is promised to keep working, either.

So the contract with the rest of Mati is narrow on purpose:

    every function returns a value or raises `Unavailable`.
    NOTHING ELSE IMPORTS urllib FOR THIS. When Flipp changes shape, ONE file
    changes, and the accountant says "I can't see local flyers right now"
    while budgets, the watchlist and the bills carry on untouched.

⚠️ DO NOT PUT THIS ON THE CRITICAL PATH OF ANYTHING.

⚠️ NEVER COMPARE PRICES ACROSS MERCHANTS WITHOUT THE UNIT. A real row: "Milk,
Restaurant Depot, $88.99, item_weight 3.0" — a wholesale case, not a carton.
A watchlist that reports "milk is $88.99 near you" is worse than silence.
`unit_price` returns None when the unit cannot be established, and None must
be shown as "not comparable", never as zero.

⚠️ THE POSTAL CODE IS A PERSONAL FACT AND FOLLOWS `weather.py`'S PRECEDENT: it
lives in settings, never in the field, and `forget_postal()` genuinely removes
it. A field cannot forget; settings can.

The postal code really does change the merchants, which is what makes this
local rather than national:

    78701 Austin   ALDI, CVS, Dollar General, La Michoacana Meat Market
    10001 NYC      Acme, BJ's Wholesale, Bravo Supermarket, C Town
    94102 SF       Andronico's, Foods Co, Grocery Outlet
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request

from . import config as C

API = "https://backflipp.wishabi.com/flipp/items/search"
UA = C.USER_AGENT
NET_TIMEOUT = 20

# Circulars change weekly; an hour is well inside that and keeps Mati off the
# network on repeated questions about the same item.
FRESH_SECONDS = 3600

_cache = {}          # (postal, query) -> (fetched_at, [deals])


class Unavailable(RuntimeError):
    """Flipp could not be read. ⚠️ NOT the same as "no deals near you"."""


# ── where to look ────────────────────────────────────────────────────────────

# Deliberately permissive: US ZIP, ZIP+4, and Canadian FSA/postal codes, since
# Flipp serves both. Rejecting a valid code is worse than passing a bad one to
# a service that will simply return nothing.
_POSTAL = re.compile(r"^(?:\d{5}(?:-\d{4})?|[A-Za-z]\d[A-Za-z] ?\d[A-Za-z]\d)$")


def set_postal(code):
    """Remember the postal code to search. Returns it, or None if malformed."""
    code = " ".join(str(code or "").split()).upper()
    if not _POSTAL.match(code):
        return None
    # Only the first five digits are ever sent: ZIP+4 identifies a building,
    # and the extra four buy nothing here — circulars are regional.
    code = code.split("-")[0]
    C.save_settings(flipp_postal=code)
    _cache.clear()
    return code


def postal():
    """The saved postal code, or None. None means do not look, and do not guess."""
    return (C._settings().get("flipp_postal") or None)


def forget_postal():
    """Delete it. ⚠️ `save_settings` merges and cannot remove, so this rewrites."""
    import os
    cur = C._settings()
    had = "flipp_postal" in cur
    cur.pop("flipp_postal", None)
    os.makedirs(os.path.dirname(C.SETTINGS_PATH), exist_ok=True)
    tmp = C.SETTINGS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cur, f, indent=2)
    os.replace(tmp, C.SETTINGS_PATH)
    _cache.clear()
    return had


# ── reading a circular ───────────────────────────────────────────────────────

def _cents(value):
    """Dollars (however Flipp expressed them) to integer cents, or None.

    ⚠️ ROUNDED, NEVER TRUNCATED, AND NEVER LEFT AS A FLOAT. Measured:
    `int(8.70 * 100)` is **869** and `int(0.29 * 100)` is **28**. That lost
    cent is the whole reason the first money rule exists.
    """
    if value is None:
        return None
    try:
        return int(round(float(value) * 100))
    except (TypeError, ValueError):
        return None


def _deal(raw):
    price = _cents(raw.get("current_price"))
    if price is None:
        # ⚠️ A row with no price is not a free item. Dropped.
        return None
    was = _cents(raw.get("original_price"))
    from . import untrusted
    return {
        # ⚠️ Product and merchant names come from a retailer's own feed.
        "name": untrusted.clean(raw.get("name"), max_len=140),
        "merchant": untrusted.clean(raw.get("merchant_name"), max_len=80) or None,
        "cents": price,
        "was_cents": was if (was and was > price) else None,
        "story": untrusted.clean(raw.get("sale_story"), max_len=140) or None,
        "weight": raw.get("item_weight"),
        "valid_to": (raw.get("valid_to") or "")[:10] or None,
        "image": raw.get("clean_image_url") or None,
        "source": "flipp",
    }


def search(query, code=None, limit=20):
    """Local circular items matching a query, cheapest first.

    Raises `Unavailable` when Flipp cannot be read — which is NOT the same as
    an empty list, and the caller must say different things about the two.
    """
    code = code or postal()
    if not code:
        return []
    key = (code, (query or "").strip().lower())
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < FRESH_SECONDS:
        return hit[1][:limit]

    q = urllib.parse.urlencode({"locale": "en-us", "postal_code": code,
                                "q": query or ""})
    req = urllib.request.Request(f"{API}?{q}", headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=NET_TIMEOUT) as r:
            data = json.loads(r.read().decode("utf-8", "replace"))
    except Exception as e:                                   # noqa: BLE001
        raise Unavailable(f"could not read the local flyers "
                          f"({type(e).__name__})") from e
    if not isinstance(data, dict):
        # ⚠️ An undocumented endpoint may change shape without warning. That
        # is a failure to report, not a reason to crash the turn.
        raise Unavailable("the local flyers came back in a shape I don't know")

    out = []
    for raw in (data.get("items") or []):
        d = _deal(raw) if isinstance(raw, dict) else None
        if d and d["name"]:
            out.append(d)
    out.sort(key=lambda d: d["cents"])
    _cache[key] = (time.time(), out)
    return out[:limit]


# ── saying it safely ─────────────────────────────────────────────────────────

def unit_price(deal):
    """Cents per unit of weight, or None when it cannot be established.

    ⚠️ None IS THE IMPORTANT RETURN VALUE. Flipp's `item_weight` is
    unit-less and frequently absent, and the case that made this function
    necessary was real: "Milk, Restaurant Depot, $88.99, weight 3.0". Without
    a unit, two prices are not comparable, and saying so is the correct
    answer rather than a shortcoming.
    """
    w = deal.get("weight")
    try:
        w = float(w)
    except (TypeError, ValueError):
        return None
    if w <= 0:
        return None
    return deal["cents"] / w


def comparable(deals):
    """Split into (comparable, not_comparable). Never silently drops either."""
    yes = [d for d in deals if unit_price(d) is not None]
    no = [d for d in deals if unit_price(d) is None]
    return yes, no


def money(cents):
    """Integer cents to "$4.99". The ONLY place cents become a string."""
    if cents is None:
        return "—"
    return f"${cents // 100}.{cents % 100:02d}"


def describe(deal):
    """One line about one deal, with the merchant always attached.

    ⚠️ THE MERCHANT IS NEVER OPTIONAL IN THE SENTENCE. "Milk, $88.99" is
    alarming; "Milk, $88.99 at Restaurant Depot" is obviously a wholesale
    case. The context is what makes the number readable.
    """
    s = f"{deal['name']} {money(deal['cents'])}"
    if deal.get("merchant"):
        s += f" at {deal['merchant']}"
    if deal.get("was_cents"):
        s += f" (was {money(deal['was_cents'])})"
    if deal.get("valid_to"):
        s += f", through {deal['valid_to']}"
    return s


def status():
    code = postal()
    return {"postal": code, "set": bool(code), "cached_searches": len(_cache),
            "source": "flipp (undocumented; failures are reported, not fatal)"}
