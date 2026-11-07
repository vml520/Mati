"""
The weather where the person is, worked out at the moment it is asked.

The user asked whether local weather was in the world module. It was not, and it
does not belong there — `world.py` INGESTS, and everything it takes in becomes
a stored note with a 30-day life. A temperature is true for about an hour. A
weather reading written into the store would be recited back days later as a
remembered fact, which is the exact failure this project keeps guarding
against: a stored fact outliving its truth.

So this follows `health.py` instead — COMPUTED, NEVER STORED — and arrives
under `<computed>`, the provenance kind that already exists for things worked
out rather than remembered.

Source: Open-Meteo. Free, keyless, no account, global. :

    api.weather.gov                       404 for 19.43,-99.13 — US only
    forecast.weather.gov/xml/current_obs  empty response
    api.open-meteo.com                    answered, global          ✓

⚠️ THE LOCATION LIVES IN SETTINGS, NOT IN THE PROFILE. Every other personal
fact Mati holds is imprinted into the field and injected each turn. A field
cannot forget — `forget_place()` would be a promise nothing could keep. In
`mati_settings.json` a location is one key that can actually be deleted, it
never enters θ₀, and it costs no tokens on turns nobody asked about the sky.

⚠️ THE COORDINATES ARE COARSENED, AND IT COSTS NOTHING. Measured: sending
19.43,-99.13 returns 19.437609,-99.10715 — the service snaps to its own grid
about 2.4 km away regardless. Precision past two decimals buys accuracy of
exactly zero and gives away where someone sleeps, so it is rounded before it is
ever saved, not merely before it is sent.

⚠️ A FAILED FETCH SAYS NOTHING. No cached value is passed off as current, no
number is estimated. Offline, Mati has nothing to say about the weather, and
that is the honest answer.

⚠️ LOCAL TIMES COME FROM THE SERVICE, ALREADY LOCAL. `timezone=auto` returns
wall-clock strings for the place being asked about. They are printed as given
and never re-converted — re-deriving a local day from UTC is the bug this
project has now shipped twice.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request

from . import config as C

FORECAST_API = "https://api.open-meteo.com/v1/forecast"
GEOCODE_API = "https://geocoding-api.open-meteo.com/v1/search"
UA = C.USER_AGENT
NET_TIMEOUT = 15

# How coarse the saved coordinates are. 2 decimals is ~1.1 km, which is finer
# than the forecast grid and so loses nothing; see the header.
PLACES = 2

# A reading is good for this long. Open-Meteo updates every 15 minutes and
# `current.interval` says so; half an hour keeps Mati off the network on most
# turns without ever serving something stale enough to be wrong.
FRESH_SECONDS = 1800

_cache = {}          # (lat, lon, units) -> (fetched_at, facts)


class Unavailable(RuntimeError):
    """No reading could be taken. ⚠️ NOT the same as fair weather."""


# ⚠️ WMO codes, not prose from the service. The API returns an integer; the
# words are chosen here so they stay in Mati's register and stay stable.
CODES = {
    0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "freezing fog",
    51: "drizzling lightly", 53: "drizzling", 55: "drizzling heavily",
    56: "freezing drizzle", 57: "heavy freezing drizzle",
    61: "raining lightly", 63: "raining", 65: "raining heavily",
    66: "freezing rain", 67: "heavy freezing rain",
    71: "snowing lightly", 73: "snowing", 75: "snowing heavily",
    77: "snow grains",
    80: "passing showers", 81: "showers", 82: "violent showers",
    85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorms", 96: "thunderstorms with hail",
    99: "thunderstorms with heavy hail",
}


def _describe(code):
    return CODES.get(code, "hard to characterise")


# ── where the person is ──────────────────────────────────────────────────────

def find_place(name, count=5):
    """Candidate places for a typed name. ⚠️ The request carries the name only.

    Returns [] when nothing matches, which is different from a network failure;
    a failure raises, so the caller never says "no such place" about an outage.
    """
    q = urllib.parse.urlencode({"name": name, "count": str(count),
                                "language": "en", "format": "json"})
    req = urllib.request.Request(f"{GEOCODE_API}?{q}", headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=NET_TIMEOUT) as r:
            data = json.loads(r.read().decode())
    except Exception as e:                                   # noqa: BLE001
        raise Unavailable(f"could not look that place up ({type(e).__name__})") from e
    out = []
    for r in data.get("results", []):
        if r.get("latitude") is None or r.get("longitude") is None:
            continue
        # ⚠️ Deduped. Open-Meteo returns name/admin1/country separately and
        # they often repeat — "Mexico City, Mexico City, Mexico" reads like a
        # bug even though every field is correct.
        parts, label = [], ""
        for x in (r.get("name"), r.get("admin1"), r.get("country")):
            if x and x not in parts:
                parts.append(x)
        label = ", ".join(parts)
        out.append({
            "label": label,
            "country_code": r.get("country_code"),
            # Coarsened HERE, at the moment of discovery, so a precise
            # coordinate is never written to disk in the first place.
            "latitude": round(float(r["latitude"]), PLACES),
            "longitude": round(float(r["longitude"]), PLACES),
        })
    return out


def set_place(place):
    """Remember where to ask about. Takes a `find_place` result, or a name.

    ⚠️ Saved to settings, never imprinted. See the header.
    """
    if isinstance(place, str):
        hits = find_place(place, count=1)
        if not hits:
            return None
        place = hits[0]
    units = place.get("units") or (
        # The only countries that would find Celsius surprising. Everywhere
        # else gets metric, and either can be changed afterwards.
        "imperial" if (place.get("country_code") or "").upper()
        in ("US", "BS", "BZ", "KY", "LR", "PW", "FM", "MH") else "metric")
    # ⚠️ THE COUNTRY CODE IS SAVED TOO, and it is not for weather. `radio.near`
    # needs it: without a country filter the nearest station to the test
    # location came back from 574km away, because the rows fetched are the
    # world's most-clicked and hold almost none from the user's own region.
    # The user: *"as long as we get nearby stations first before stations from
    # outside of the the US, or whatever country the user happens to be in."*
    # Two characters answers that, and asking a second time for a fact already
    # on screen would not.
    C.save_settings(weather_place=place["label"],
                    weather_lat=round(float(place["latitude"]), PLACES),
                    weather_lon=round(float(place["longitude"]), PLACES),
                    weather_country=(place.get("country_code") or "").upper()[:2]
                    or None,
                    weather_units=units)
    _cache.clear()
    return place["label"]


def forget_place():
    """Delete the location. ⚠️ This is why it was never put in the field."""
    cur = C._settings()
    # ⚠️ `weather_country` BELONGS IN THIS LIST AND IS EASY TO MISS.
    # `set_place` writes five keys and this once removed only four, so
    # "forget where I am" left the country code behind — and the whole reason
    # the location lives in settings rather than the field is that settings
    # CAN forget. A delete that leaves part of the thing is the failure this
    # design exists to avoid. `test_weather` had been reporting it.
    removed = [k for k in ("weather_place", "weather_lat", "weather_lon",
                           "weather_units", "weather_country") if k in cur]
    for k in removed:
        cur.pop(k)
    # ⚠️ `save_settings` MERGES, so it cannot remove a key. Written whole.
    import os
    os.makedirs(os.path.dirname(C.SETTINGS_PATH), exist_ok=True)
    tmp = C.SETTINGS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cur, f, indent=2)
    os.replace(tmp, C.SETTINGS_PATH)
    _cache.clear()
    return bool(removed)


def where():
    """The saved place, or None. None means: do not ask, and do not guess."""
    s = C._settings()
    lat, lon = s.get("weather_lat"), s.get("weather_lon")
    if lat is None or lon is None:
        return None
    return {"label": s.get("weather_place") or "there",
            "latitude": round(float(lat), PLACES),
            "longitude": round(float(lon), PLACES),
            "units": s.get("weather_units") or "metric"}


# ── the reading ──────────────────────────────────────────────────────────────

def _fetch(lat, lon, units):
    q = {
        "latitude": f"{lat:.{PLACES}f}",
        "longitude": f"{lon:.{PLACES}f}",
        "current": "temperature_2m,apparent_temperature,weather_code,is_day",
        "daily": ("temperature_2m_max,temperature_2m_min,"
                  "precipitation_probability_max,weather_code,sunrise,sunset"),
        "timezone": "auto",
        "forecast_days": "1",
    }
    if units == "imperial":
        q["temperature_unit"] = "fahrenheit"
        q["wind_speed_unit"] = "mph"
    req = urllib.request.Request(f"{FORECAST_API}?{urllib.parse.urlencode(q)}",
                                 headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=NET_TIMEOUT) as r:
            return json.loads(r.read().decode())
    except Exception as e:                                   # noqa: BLE001
        raise Unavailable(f"could not reach the weather service "
                          f"({type(e).__name__})") from e


def _clock(iso):
    """"06:58" from "2026-09-10T06:58". ⚠️ Already local; not re-converted.

    Zero-padded 24-hour, matching `activity.local_parts`, so every time Mati
    shows reads the same way.
    """
    m = re.search(r"T(\d{2}:\d{2})", iso or "")
    return m.group(1) if m else None


def facts(force=False):
    """What is true outside right now. Raises `Unavailable` rather than guess.

    Returns {} — not an exception — when no place is set, because that is a
    settled state and not a failure.
    """
    place = where()
    if not place:
        return {}
    key = (place["latitude"], place["longitude"], place["units"])
    hit = _cache.get(key)
    if hit and not force and time.time() - hit[0] < FRESH_SECONDS:
        return dict(hit[1], age_minutes=int((time.time() - hit[0]) / 60))

    data = _fetch(place["latitude"], place["longitude"], place["units"])
    cur = data.get("current") or {}
    day = data.get("daily") or {}

    def first(k):
        v = day.get(k)
        return v[0] if isinstance(v, list) and v else None

    deg = "°F" if place["units"] == "imperial" else "°C"
    out = {
        "place": place["label"],
        "units": deg,
        "now": cur.get("temperature_2m"),
        "feels_like": cur.get("apparent_temperature"),
        "sky": _describe(cur.get("weather_code")),
        "is_day": bool(cur.get("is_day")),
        "high": first("temperature_2m_max"),
        "low": first("temperature_2m_min"),
        "rain_chance": first("precipitation_probability_max"),
        "day_sky": _describe(first("weather_code")),
        "sunrise": _clock(first("sunrise")),
        "sunset": _clock(first("sunset")),
        "observed_at": _clock(cur.get("time")),
        "age_minutes": 0,
    }
    _cache[key] = (time.time(), out)
    return out


def line(force=False):
    """One sentence, or "" — for the morning report and anywhere else brief."""
    try:
        f = facts(force=force)
    except Unavailable:
        return ""
    if not f or f.get("now") is None:
        return ""
    s = f"{f['sky']} and {f['now']:.0f}{f['units']} in {f['place']}"
    if f.get("high") is not None and f.get("low") is not None:
        s += f", between {f['low']:.0f} and {f['high']:.0f} today"
    if f.get("rain_chance"):
        s += f", {f['rain_chance']}% chance of rain"
    return s


def block():
    """The `<computed>` weather lines, in Mati's own voice.

    ⚠️ First person, and the place named rather than addressed. A block written
    with "you" in it comes back out as a claim about the person the moment
    Mati speaks — that is how 291 self-asked questions became "291 questions
    you asked".

    ⚠️ Facts only. This says what is outside; it does not say to take a coat.
    """
    try:
        f = facts()
    except Unavailable:
        return ""
    if not f or f.get("now") is None:
        return ""
    L = ["<computed>",
         f"The weather in {f['place']}, taken just now — not remembered, and "
         f"true only for the next hour or so:"]
    L.append(f"  - It is {f['sky']}, {f['now']:.0f}{f['units']}"
             + (f", feeling like {f['feels_like']:.0f}{f['units']}"
                if f.get("feels_like") is not None
                and abs(f["feels_like"] - f["now"]) >= 2 else "") + ".")
    if f.get("high") is not None and f.get("low") is not None:
        L.append(f"  - Today runs between {f['low']:.0f} and "
                 f"{f['high']:.0f}{f['units']}, {f['day_sky']}.")
    if f.get("rain_chance") is not None:
        L.append(f"  - Chance of rain today: {f['rain_chance']}%.")
    if f.get("sunrise") and f.get("sunset"):
        # ⚠️ Tenseless on purpose. "The sun rose at 06:23" is wrong before
        # dawn and "rises" is wrong after it, and the block is read at any
        # hour — so it states the times and commits to no tense at all.
        L.append(f"  - Sunrise {f['sunrise']}, sunset {f['sunset']}, "
                 f"local time there.")
    L.append("</computed>")
    return "\n".join(L)


# ── when to bother ───────────────────────────────────────────────────────────

# ⚠️ A LITERAL LIST, on the pattern the charter injection already uses. A
# learned or fuzzy trigger would put a network call and 5 lines of prompt on
# turns that were never about the sky.
_ASKS = (
    "weather", "forecast", "temperature", "how hot", "how cold", "how warm",
    "raining", "going to rain", "rain today", "snowing", "going to snow",
    "sunny", "cloudy", "outside today", "take a coat", "take an umbrella",
    "need a jacket", "need an umbrella", "sunrise", "sunset", "humid",
    "el clima", "el tiempo", "va a llover", "hace frío", "hace calor",
)


def asks_about_weather(text_in):
    t = (text_in or "").lower()
    return any(p in t for p in _ASKS)


def status():
    p = where()
    if not p:
        return {"place": None, "set": False}
    return {"place": p["label"], "units": p["units"], "set": True,
            "coordinates": f"{p['latitude']}, {p['longitude']} "
                           f"({PLACES} decimals, ~1 km)",
            "cached_readings": len(_cache)}
