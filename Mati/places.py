"""
Places near a point, from OpenStreetMap via Overpass. Open data, no key.

⚠️ THIS FILE IS A FIREWALL, on the `flipp.py` pattern. Overpass is a SHARED
PUBLIC endpoint run on donated infrastructure under a fair-use policy. It is
genuinely open — no key, no account, no terms to breach by querying it — and
that is exactly why an app shipped to many people must not hammer it. So:

    every function returns a value or raises `Unavailable`.
    one query at a time, rate-limited, cached, and a small `[out:json]` body.
    when it fails, Mati says so and everything else carries on.

⚠️ WHY NOT MKLocalSearch, WHICH IS RIGHT THERE AND NEEDS NO NETWORK LIBRARY.
Because it is BROKEN for this purpose and it fails SILENTLY. Measured
: `setRegion_` is accepted, reads back exactly as set, and is then
ignored — searching Austin (30.2672,-97.7431) and New York (40.7128,-74.0060)
returned byte-identical results, both in the device's OWN metro area. 25
plausible museums came
back and everything looked like success. Only running two distant regions and
comparing revealed it. MapKit is still used for ROUTING, where coordinates are
honoured (see `directions.py`); it is not used for search.

⚠️ OSM HAS NO RATINGS, AND THAT IS NOT A GAP TO FILL WITH A GUESS. There is no
"top-rated" here because the data holds no reviews. Yelp Fusion would supply
them and now costs $299/month minimum, which is disqualified. So Mati lists
what is NEAR, never what is BEST, and the wording has to keep that honest.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request

from . import config as C

ENDPOINT = "https://overpass-api.de/api/interpreter"
UA = C.USER_AGENT
NET_TIMEOUT = 35

# ⚠️ FAIR USE, ENFORCED IN CODE. Overpass asks for moderate load and nothing
# here needs more: a person asks where the nearest pharmacy is, not for a
# continent. One query every few seconds, and answers kept for an hour.
MIN_GAP_SECONDS = 3.0
FRESH_SECONDS = 3600
MAX_RADIUS_M = 8000

_last_call = [0.0]
_cache = {}


class Unavailable(RuntimeError):
    """Overpass could not be read. ⚠️ NOT the same as "nothing near you"."""


# What a person asks for, and the OSM tags that mean it. ⚠️ A plain list, not
# a classifier — an unrecognised word is answered with "I don't know that
# kind of place", which is better than searching for the wrong tag.
KINDS = {
    "restaurant":  [("amenity", "restaurant")],
    "food":        [("amenity", "restaurant"), ("amenity", "fast_food")],
    "cafe":        [("amenity", "cafe")],
    "coffee":      [("amenity", "cafe")],
    "bar":         [("amenity", "bar"), ("amenity", "pub")],
    "pharmacy":    [("amenity", "pharmacy")],
    "hospital":    [("amenity", "hospital")],
    "doctor":      [("amenity", "doctors")],
    "bank":        [("amenity", "bank")],
    "atm":         [("amenity", "atm")],
    "fuel":        [("amenity", "fuel")],
    "gas":         [("amenity", "fuel")],
    "parking":     [("amenity", "parking")],
    "supermarket": [("shop", "supermarket")],
    "grocery":     [("shop", "supermarket"), ("shop", "convenience")],
    "museum":      [("tourism", "museum")],
    "park":        [("leisure", "park")],
    "library":     [("amenity", "library")],
    "hotel":       [("tourism", "hotel")],
    "cinema":      [("amenity", "cinema")],
    "theatre":     [("amenity", "theatre")],
    "post":        [("amenity", "post_office")],
    "toilets":     [("amenity", "toilets")],
    "playground":  [("leisure", "playground")],
    "gym":         [("leisure", "fitness_centre")],
    "bookshop":    [("shop", "books")],
    "hardware":    [("shop", "hardware"), ("shop", "doityourself")],
}


def kinds():
    """Every kind of place Mati can look for, alphabetically."""
    return sorted(KINDS)


def _resolve(query):
    """A typed word -> OSM tag pairs, or None. ⚠️ Never a guess."""
    q = " ".join(str(query or "").lower().split())
    if q in KINDS:
        return KINDS[q]
    # Singularise the obvious plural rather than demanding exact words.
    if q.endswith("s") and q[:-1] in KINDS:
        return KINDS[q[:-1]]
    for k in KINDS:
        if k in q.split():
            return KINDS[k]
    return None


def _ql(tags, lat, lon, radius):
    parts = "".join(
        f'node(around:{radius},{lat},{lon})["{k}"="{v}"];'
        f'way(around:{radius},{lat},{lon})["{k}"="{v}"];'
        for k, v in tags)
    return f"[out:json][timeout:25];({parts});out center tags 40;"


def _haversine_m(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, asin, sqrt
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = (sin(dlat / 2) ** 2
         + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2)
    return 2 * 6371000 * asin(sqrt(a))


def near(query, lat, lon, radius=2000, limit=8):
    """Places of a kind near a point, nearest first.

    Raises `Unavailable` when Overpass cannot be read. Returns [] when the
    query is a kind Mati knows and there is genuinely nothing nearby — the
    caller must say different things about those two.
    """
    tags = _resolve(query)
    if not tags:
        return None                      # ⚠️ None = "I don't know that kind"
    radius = max(100, min(int(radius), MAX_RADIUS_M))
    key = (tuple(tags), round(lat, 3), round(lon, 3), radius)
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < FRESH_SECONDS:
        return hit[1][:limit]

    gap = time.time() - _last_call[0]
    if gap < MIN_GAP_SECONDS:
        time.sleep(MIN_GAP_SECONDS - gap)
    _last_call[0] = time.time()

    body = urllib.parse.urlencode({"data": _ql(tags, lat, lon, radius)})
    req = urllib.request.Request(ENDPOINT, data=body.encode(),
                                 headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=NET_TIMEOUT) as r:
            data = json.loads(r.read().decode("utf-8", "replace"))
    except Exception as e:                                   # noqa: BLE001
        raise Unavailable(f"could not reach the map data "
                          f"({type(e).__name__})") from e
    if not isinstance(data, dict):
        raise Unavailable("the map data came back in a shape I don't know")

    out = []
    for el in (data.get("elements") or []):
        t = el.get("tags") or {}
        name = (t.get("name") or "").strip()
        if not name:
            # ⚠️ An unnamed node is a real map feature and a useless answer.
            continue
        c = el.get("center") or el
        plat, plon = c.get("lat"), c.get("lon")
        if plat is None or plon is None:
            continue
        # ⚠️ ANYONE CAN EDIT OPENSTREETMAP. A café renamed to an
        # instruction is a trivial edit, and it would arrive here as a name.
        from . import untrusted
        name = untrusted.clean(name, max_len=120)
        if not name:
            continue
        out.append({
            "name": name,
            "lat": plat, "lon": plon,
            "metres": int(_haversine_m(lat, lon, plat, plon)),
            "kind": t.get("amenity") or t.get("shop") or t.get("tourism")
                    or t.get("leisure"),
            "address": " ".join(x for x in (t.get("addr:housenumber"),
                                            t.get("addr:street")) if x) or None,
            "hours": t.get("opening_hours"),
            "phone": t.get("phone") or t.get("contact:phone"),
            "source": "openstreetmap",
        })
    out.sort(key=lambda p: p["metres"])
    _cache[key] = (time.time(), out)
    return out[:limit]


def describe(place, imperial=False):
    """One line about one place. ⚠️ NEAREST, never 'best' — see the header."""
    # ⚠️ FEET UNDER A FIFTH OF A MILE. Rounding to one decimal turned a
    # pharmacy 40 m away into "0.0 mi", which reads as a formatting bug and
    # tells the person nothing. Same rule as `directions.distance`.
    m = place["metres"]
    if imperial:
        d = f"{m / 1609.34:.1f} mi" if m >= 322 else f"{int(m * 3.28)} feet"
    else:
        d = f"{m / 1000:.1f} km" if m >= 300 else f"{m} m"
    s = f"{place['name']} — {d}"
    if place.get("address"):
        s += f", {place['address']}"
    if place.get("hours"):
        s += f" ({place['hours']})"
    return s


def status():
    return {"endpoint": ENDPOINT, "kinds": len(KINDS),
            "cached": len(_cache),
            "note": "OpenStreetMap has no ratings; Mati lists NEAREST, "
                    "never 'best'"}
