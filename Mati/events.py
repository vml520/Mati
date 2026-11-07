"""
Concerts and live events near a place. OFF until the person brings a key.

⚠️ THIS MODULE IS INERT BY DEFAULT AND THAT IS THE DESIGN, not a limitation.
Ticketmaster's Discovery API is free, and it still needs a Consumer Key. A key
shipped inside a public Apache-2.0 build is readable by anyone, gets revoked
the first time somebody abuses it, and takes every install down together. So
`credentials.py` holds the one rule — Mati ships no keys — and this module is
one of the three things that rule unblocks.

With no key, every function here returns cleanly and `why_off()` says exactly
where to get one and what it costs. Nothing raises, nothing pretends, and
nothing is silently empty.

⚠️ A FIREWALL, like `flipp.py` and `places.py`. Everything raises
`Unavailable`; nothing else in Mati imports urllib for events. When
Ticketmaster changes or a key is revoked, one file changes and the sentence
Mati says is "I can't see events right now".

⚠️ SENDING A LOCATION PLUS "WHAT IS ON TONIGHT" IS A REAL DISCLOSURE, larger
than a gridded weather lookup — it says where someone is and that they intend
to go out. So this is never on an idle loop and never part of the morning
report; it runs when asked, with the coarse saved postcode or place, never a
device GPS reading. `weather.py` set that precedent and the accountant reused
it.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request

from . import config as C

API = "https://app.ticketmaster.com/discovery/v2/events.json"
UA = C.USER_AGENT
NET_TIMEOUT = 20
SERVICE = "ticketmaster"

# A radius someone would actually travel for an evening.
DEFAULT_RADIUS_MILES = 25
MAX_RESULTS = 20


class Unavailable(RuntimeError):
    """Events could not be read. ⚠️ NOT the same as "nothing is on"."""


def enabled():
    """Is there a key? ⚠️ Answerable with the vault locked — see credentials."""
    from . import credentials
    return credentials.have(SERVICE)


def why_off():
    """The sentence to say when this is off. Empty string when it is on.

    ⚠️ IT NAMES THE KEYLESS PATH TOO, because for a lot of people that is the
    right answer and the key is not. The user: "I'd rather not use
    anything that requires keys or passwords." A message that only says "go
    make an account" would push someone toward the thing they declined.
    """
    from . import credentials
    if enabled():
        return ""
    return (credentials.needed(SERVICE) + "  Or skip the key entirely and "
            "follow a venue, promoter or listings site directly with "
            "`/events follow <url>` — that needs no account, though it only "
            "covers places you choose rather than everything nearby.")


def follow(url, name=None, db_path=None):
    """Follow a venue's own listings. (ok, message). ⚠️ No key, no account.

    Measured  — this path is REAL BUT PATCHY, and saying so is part
    of the feature. Songkick's artist RSS returns 406, Bandsintown's is dead
    and the Kennedy Center returns 403; SF Station's feed answers with ten
    items and Pitchfork publishes one. So it works where a site chooses to
    publish, which is a smaller promise than "concerts near you" and an
    honest one.

    ⚠️ ALL EXISTING MACHINERY. An iCal feed goes to `agenda.import_ics`, so
    gigs land beside meetings and bills and get the same reminders; anything
    else goes through `world.discover_feed` + `subscribe`. Nothing new is
    stored and no third table appears.
    """
    from . import agenda, world
    import urllib.request

    ok, why = world.is_public_url(url)
    if not ok:
        return False, why

    # An.ics is a calendar, not a feed — and Mati already imports those.
    looks_ical = url.lower().split("?")[0].endswith((".ics", ".ical"))
    if looks_ical:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=NET_TIMEOUT) as r:
                text = r.read(4_000_000).decode("utf-8", "replace")
        except Exception as e:                               # noqa: BLE001
            return False, f"that calendar could not be read ({type(e).__name__})"
        try:
            n = agenda.import_ics(text, db_path=db_path)
        except agenda.ImportFailed as e:
            return False, str(e)
        return True, (f"imported {n} event(s) into the calendar — they will "
                      f"show up beside everything else on the day")

    feed, how = world.discover_feed(url)
    if not feed:
        return False, (f"{how}. If they publish a calendar, an .ics address "
                       f"works here too.")
    ok, msg = world.subscribe(feed, name=name)
    return (ok, f"{msg} — {how}") if ok else (False, msg)


def _get(params):
    from . import credentials
    from . import vault
    try:
        key = credentials.get(SERVICE)
    except vault.Locked as e:
        # ⚠️ "Locked" is not "not set up". Passing this through unchanged is
        # the whole reason credentials.get distinguishes them.
        raise Unavailable(str(e)) from e
    if not key:
        return None
    q = dict(params, apikey=key, size=str(MAX_RESULTS))
    req = urllib.request.Request(f"{API}?{urllib.parse.urlencode(q)}",
                                 headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=NET_TIMEOUT) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            # ⚠️ A rejected key is a DIFFERENT problem from no key, and the
            # person needs to hear which one.
            raise Unavailable("Ticketmaster rejected that key — it may be "
                              "wrong or revoked; `/key set ticketmaster …`") from e
        if e.code == 429:
            raise Unavailable("Ticketmaster is rate-limiting; try later") from e
        raise Unavailable(f"Ticketmaster returned {e.code}") from e
    except Exception as e:                                   # noqa: BLE001
        raise Unavailable(f"could not reach Ticketmaster "
                          f"({type(e).__name__})") from e


def _event(raw):
    dates = (raw.get("dates") or {}).get("start") or {}
    venues = ((raw.get("_embedded") or {}).get("venues") or [{}])
    v = venues[0] if venues else {}
    price = None
    for p in (raw.get("priceRanges") or []):
        lo, hi = p.get("min"), p.get("max")
        if lo is not None:
            # ⚠️ Integer cents, the convention `accountant.py` establishes.
            price = {"min_c": int(round(float(lo) * 100)),
                     "max_c": int(round(float(hi))) * 100 if hi else None,
                     "currency": p.get("currency")}
            break
    from . import untrusted
    return {
        # ⚠️ Event and venue names are submitted by promoters.
        "name": untrusted.clean(raw.get("name"), max_len=140),
        # ⚠️ Taken as the API gives it — local to the VENUE, not converted.
        # Re-deriving a local day from UTC is the bug this project shipped
        # twice, and a gig's date belongs to the city it is in.
        "date": dates.get("localDate"),
        "time": dates.get("localTime"),
        "venue": untrusted.clean(v.get("name"), max_len=80) or None,
        "city": untrusted.clean((v.get("city") or {}).get("name"), max_len=60) or None,
        "genre": next((((c.get("genre") or {}).get("name"))
                       for c in (raw.get("classifications") or [])), None),
        "price": price,
        "link": raw.get("url"),
    }


def near(lat, lon, radius_miles=DEFAULT_RADIUS_MILES, keyword=None,
         classification=None, days=14):
    """Events near a point. [] when nothing is on; None when there is no key.

    ⚠️ THREE DIFFERENT ANSWERS, kept apart: None ("not set up"), [] ("nothing
    on"), and `Unavailable` ("I could not look"). Collapsing any two of them
    produces a confident sentence that is false.
    """
    import datetime
    start = datetime.datetime.now(datetime.timezone.utc)
    end = start + datetime.timedelta(days=int(days))
    p = {
        "latlong": f"{lat},{lon}",
        "radius": str(int(radius_miles)),
        "unit": "miles",
        "startDateTime": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "endDateTime": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sort": "date,asc",
    }
    if keyword:
        p["keyword"] = keyword
    if classification:
        p["classificationName"] = classification
    data = _get(p)
    if data is None:
        return None
    out = [_event(e) for e in
           ((data.get("_embedded") or {}).get("events") or [])]
    return [e for e in out if e["name"]]


def tonight(lat, lon, radius_miles=DEFAULT_RADIUS_MILES):
    return near(lat, lon, radius_miles, days=1)


def music(lat, lon, radius_miles=DEFAULT_RADIUS_MILES, days=14):
    return near(lat, lon, radius_miles, classification="music", days=days)


def describe(e):
    """One line. ⚠️ Price stated only when the API gave one — never estimated."""
    from . import accountant
    s = e["name"]
    if e.get("date"):
        s += f" — {e['date']}"
        if e.get("time"):
            s += f" {e['time'][:5]}"
    if e.get("venue"):
        s += f", {e['venue']}"
    if e.get("city"):
        s += f" ({e['city']})"
    if e.get("price") and e["price"].get("min_c"):
        s += f", from {accountant.money(e['price']['min_c'])}"
    return s


_ASKS = (
    "concert", "concerts", "gig", "live music", "show tonight",
    "what's on tonight", "whats on tonight", "events near", "any events",
    "tickets for", "playing tonight", "tour dates",
    "concierto", "conciertos", "qué hay esta noche", "que hay esta noche",
)


def asks_about_events(text_in):
    t = (text_in or "").lower()
    return any(p in t for p in _ASKS)


def status():
    return {"enabled": enabled(), "service": SERVICE,
            "why_off": why_off() or None,
            "note": "never on an idle loop; a location plus 'what is on "
                    "tonight' is a real disclosure"}
