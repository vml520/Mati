"""
Getting there. Routes, spoken if asked, handed off for the actual driving.

⚠️ THE HONEST ANSWER TO "DOES MAPKIT DO TURN-BY-TURN": NO, AND NOTHING DOES.
`MKNavigation`, `MKNavigationSession`, `MKGuidance` and `MKTurnByTurn` are ALL
ABSENT from MapKit. Apple ships live, position-triggered guidance in Maps.app
and exposes no API for it. What `MKDirections` gives is the whole route AT
ONCE:

    MKRoute       distance, expectedTravelTime, name, polyline, advisoryNotices
    MKRouteStep   instructions ("Turn right onto W Cesar Chavez St"), distance

A MANIFEST IS THE WHOLE SCOPE HERE, and that is a decision rather than a
shortfall. Live guidance belongs to a future phone app; this module answers
questions and lays out the turns, MapQuest-style.

⚠️ SO THERE IS NO LOCATION TRACKING IN MATI AT ALL, and that is worth
stating because it is load-bearing for the privacy claim rather than
incidental. Live guidance would mean taking CoreLocation updates and
advancing the step list as the person moves — a continuous stream of exactly
where someone is. **Mati reads no device location anywhere.** A route is
computed from a start and an end it was told about, once, and then it is a
list of steps like any other text.

A laptop is also the wrong object for it: something that expects to be
watched while driving is the only feature here whose failure mode is a
collision. On a phone, with a phone's assumptions, that calculus changes —
which is why it is deferred rather than refused.

So the division, for this release:

    Mati        says how far, how long, and what the turns are — and will
                 SPEAK them, because it does have a voice (see below)
    Maps.app     does the actual navigation, if the person wants to drive it

⚠️ THE VOICE IS ALREADY ON THE MACHINE. `/usr/bin/say` ships with macOS, comes
with many voices, is fully offline, needs no account and costs nothing. It is
OFF by default here anyway — software that starts talking unprompted is worse
than software that does not talk at all.

⚠️ MKLocalSearch IS NOT USED, and `places.py` explains why: its region
argument is silently ignored. Routing honours coordinates; search does not.
Geocoding a typed place name reuses `weather.find_place` — already measured,
already keyless, no second geocoder to keep working.
"""

from __future__ import annotations

import json
import subprocess
import urllib.parse
import urllib.request

from . import config as C

# Transport, as MapKit numbers them.
MODES = {"driving": 1, "walking": 2, "transit": 4}
DEFAULT_MODE = "driving"

OFFLINE_TIMEOUT = 20
# ⚠️ localhost ONLY, and this is the exact INVERSE of `world.is_public_url`.
# A routing engine is something the person runs on their own machine; a
# "local engine" on someone else's host is a stranger reading every journey.
OFFLINE_DEFAULT = "http://127.0.0.1:8002"


class Unavailable(RuntimeError):
    """No route could be worked out. ⚠️ NOT the same as 'you cannot get there'."""


# ── the offline switch ───────────────────────────────────────────────────────

def offline():
    """(enabled, engine_url). Off unless the person turned it on."""
    s = C._settings()
    return (bool(s.get("directions_offline")),
            s.get("directions_engine") or OFFLINE_DEFAULT)


def set_offline(on, engine_url=None):
    """Turn the local routing engine on or off. Returns (enabled, url).

    ⚠️ REFUSES A NON-LOCAL ENGINE. The whole point of offline routing is that
    no journey leaves the machine; pointing it at a remote host would quietly
    invert that while still being called "offline".
    """
    url = engine_url or offline()[1]
    u = urllib.parse.urlparse(url)
    host = (u.hostname or "").lower()
    if on and host not in ("localhost", "127.0.0.1", "::1"):
        return None
    C.save_settings(directions_offline=bool(on), directions_engine=url)
    return (bool(on), url)


def engine_ready():
    """(ok, detail) — is a local routing engine actually answering?

    ⚠️ ASKED BEFORE IT IS TRUSTED. Nothing is installed on this machine, so
    the ordinary state of the offline path is "enabled and nothing there",
    and that must read as a plain sentence rather than a stack trace.
    """
    on, url = offline()
    if not on:
        return False, "offline routing is off"
    try:
        req = urllib.request.Request(f"{url}/status",
                                     headers={"User-Agent": C.USER_AGENT})
        with urllib.request.urlopen(req, timeout=4) as r:
            r.read(2048)
        return True, f"a routing engine is answering at {url}"
    except Exception as e:                                   # noqa: BLE001
        return False, (f"offline routing is on, but nothing is answering at "
                       f"{url} ({type(e).__name__})")


# ── finding a place ──────────────────────────────────────────────────────────

def locate(text):
    """A typed place name -> (lat, lon, label), or None.

    ⚠️ Reuses `weather.find_place` — one geocoder, already measured, already
    keyless. A second one would be a second thing to keep working.
    """
    from . import weather
    t = " ".join(str(text or "").split())
    if not t:
        return None
    # "30.2672,-97.7431" is a place too.
    if "," in t:
        a, _, b = t.partition(",")
        try:
            return (float(a), float(b), t)
        except ValueError:
            pass
    try:
        hits = weather.find_place(t, count=1)
    except Exception as e:                                   # noqa: BLE001
        raise Unavailable(
            f"I could not look that place up ({type(e).__name__}). Which "
            f"town or country is it in?") from e
    if not hits:
        return None
    h = hits[0]
    return (h["latitude"], h["longitude"], h["label"])


def here():
    """Where the person is, from settings. ⚠️ Never taken from the device.

    `weather.py` settled this and the accountant reused it: a location is
    asked for, coarsened, kept out of the field, and genuinely forgettable.
    An assistant that silently reads GPS to save a question is the pattern
    this project has refused twice already.
    """
    from . import weather
    p = weather.where()
    return (p["latitude"], p["longitude"], p["label"]) if p else None


# ── routes ───────────────────────────────────────────────────────────────────

def _route_mapkit(o_lat, o_lon, d_lat, d_lon, mode):
    """MKDirections, via the objc bridge. No key, no account, no install."""
    try:
        import objc
        from Foundation import NSBundle, NSRunLoop, NSDate
    except ImportError as e:
        raise Unavailable("the map framework is not available here") from e

    if not NSBundle.bundleWithPath_(
            "/System/Library/Frameworks/MapKit.framework").load():
        raise Unavailable("MapKit would not load")

    # ⚠️ HAND-REGISTERED BLOCK SIGNATURE. PyObjC has no metadata for this
    # selector and fails with "Argument 2 is a block, but no signature
    # available" — the same trap EventKit's notification handler hit.
    try:
        objc.registerMetaDataForSelector(
            b"MKDirections", b"calculateDirectionsWithCompletionHandler:",
            {"arguments": {2: {"callable": {
                "retval": {"type": b"v"},
                "arguments": {0: {"type": b"^v"}, 1: {"type": b"@"},
                              2: {"type": b"@"}}}}}})
    except Exception:                                        # noqa: BLE001
        pass                      # already registered in this process

    MKPlacemark = objc.lookUpClass("MKPlacemark")
    MKMapItem = objc.lookUpClass("MKMapItem")
    MKDirectionsRequest = objc.lookUpClass("MKDirectionsRequest")
    MKDirections = objc.lookUpClass("MKDirections")

    def item(lat, lon):
        return MKMapItem.alloc().initWithPlacemark_(
            MKPlacemark.alloc().initWithCoordinate_((lat, lon)))

    req = MKDirectionsRequest.alloc().init()
    req.setSource_(item(o_lat, o_lon))
    req.setDestination_(item(d_lat, d_lon))
    req.setTransportType_(MODES.get(mode, 1))

    import threading
    done, box = threading.Event(), {}

    def handler(resp, err):
        box["err"] = err
        box["routes"] = list(resp.routes()) if resp else []
        done.set()

    MKDirections.alloc().initWithRequest_(req) \
        .calculateDirectionsWithCompletionHandler_(handler)

    loop = NSRunLoop.currentRunLoop()
    for _ in range(300):                                     # ~30s ceiling
        if done.is_set():
            break
        loop.runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.1))
    if not done.is_set():
        raise Unavailable("the route took too long to work out")
    if box.get("err") is not None and not box.get("routes"):
        raise Unavailable(
            "no route came back for that. Are both places right, and should "
            "I be routing by car?")
    routes = box.get("routes") or []
    if not routes:
        # ⚠️ ASKS WHICH PLACE, because "no route between those" is far more
        # often two places Mati resolved wrongly than two places with no
        # road between them. Naming them back is what lets the person see it.
        raise Unavailable(
            "I could not get a route for those two. Can you give me the "
            "place names more exactly — a city or a street address?")

    r = routes[0]
    steps = []
    for s in list(r.steps()):
        text = s.instructions()
        if text:
            steps.append({"say": str(text), "metres": int(s.distance())})
    return {
        "metres": int(r.distance()),
        "seconds": int(r.expectedTravelTime()),
        "via": str(r.name()) if r.name() else None,
        "steps": steps,
        "engine": "mapkit",
    }


def _route_offline(o_lat, o_lon, d_lat, d_lon, mode):
    """A local Valhalla-style engine. ⚠️ Only ever 127.0.0.1 — see set_offline."""
    _, url = offline()
    costing = {"driving": "auto", "walking": "pedestrian",
               "transit": "multimodal"}.get(mode, "auto")
    body = json.dumps({
        "locations": [{"lat": o_lat, "lon": o_lon},
                      {"lat": d_lat, "lon": d_lon}],
        "costing": costing,
        "directions_options": {"units": "kilometers"},
    }).encode()
    req = urllib.request.Request(
        f"{url}/route", data=body,
        headers={"Content-Type": "application/json", "User-Agent": C.USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=OFFLINE_TIMEOUT) as r:
            data = json.loads(r.read().decode("utf-8", "replace"))
    except Exception as e:                                   # noqa: BLE001
        raise Unavailable(f"the local routing engine did not answer "
                          f"({type(e).__name__})") from e
    trip = (data or {}).get("trip") or {}
    legs = trip.get("legs") or []
    if not legs:
        raise Unavailable("the local engine returned no route")
    steps = []
    for leg in legs:
        for m in (leg.get("maneuvers") or []):
            if m.get("instruction"):
                steps.append({"say": m["instruction"],
                              "metres": int((m.get("length") or 0) * 1000)})
    summ = trip.get("summary") or {}
    return {
        "metres": int((summ.get("length") or 0) * 1000),
        "seconds": int(summ.get("time") or 0),
        "via": None,
        "steps": steps,
        "engine": "local",
    }


def route(destination, origin=None, mode=DEFAULT_MODE):
    """How to get somewhere. Raises `Unavailable` rather than inventing a route.

    Offline first when it is turned on, and ⚠️ IT DOES NOT SILENTLY FALL BACK
    — someone who switched routing to a local engine did so to keep their
    journeys off the network, and quietly asking Apple instead would defeat
    exactly the thing they asked for. It fails, and says which engine failed.
    """
    d = locate(destination)
    if not d:
        return None
    o = locate(origin) if origin else here()
    if not o:
        raise Unavailable("I don't know where to start from — set a place "
                          "with /weather set, or name one")

    on, _ = offline()
    if on:
        ok, detail = engine_ready()
        if not ok:
            raise Unavailable(detail)
        out = _route_offline(o[0], o[1], d[0], d[1], mode)
    else:
        out = _route_mapkit(o[0], o[1], d[0], d[1], mode)

    out["from"], out["to"], out["mode"] = o[2], d[2], mode
    # Kept so `hand_off` can build a Maps URL without geocoding a second time.
    out["_origin"], out["_dest"] = (o[0], o[1]), (d[0], d[1])
    return out


# ── saying it ────────────────────────────────────────────────────────────────

def _imperial():
    from . import weather
    p = weather.where()
    return bool(p and p.get("units") == "imperial")


def distance(metres):
    if _imperial():
        mi = metres / 1609.34
        return f"{mi:.1f} miles" if mi >= 0.2 else f"{int(metres * 3.28)} feet"
    return (f"{metres / 1000:.1f} km" if metres >= 300 else f"{metres} m")


def duration(seconds):
    m = int(round(seconds / 60))
    if m < 60:
        return f"{m} min"
    return f"{m // 60}h {m % 60:02d}m"


def summary(r):
    """One sentence. ⚠️ Distance and time, never an arrival time — that would
    be a promise about traffic Mati has no basis for."""
    s = f"{distance(r['metres'])}, about {duration(r['seconds'])}"
    if r.get("via"):
        s += f" via {r['via']}"
    return s


def speak(text, voice=None):
    """Say something aloud. ⚠️ OFF UNLESS ASKED, and never on its own.

    `/usr/bin/say` is offline, free, and needs no account. It is also the one
    thing in Mati that makes noise in a room, so nothing calls this except a
    person explicitly asking to be read to.
    """
    if not text:
        return False
    cmd = ["/usr/bin/say"]
    if voice:
        cmd += ["-v", str(voice)]
    try:
        # Not awaited — a long route would otherwise block the turn.
        subprocess.Popen(cmd + [str(text)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:                                        # noqa: BLE001
        return False


def hand_off(r):
    """Open the route in Maps.app, where Apple's real turn-by-turn lives.

    ⚠️ THIS IS THE ANSWER TO "HOW DOES MATI NAVIGATE", not a consolation.
    Live guidance has no public API, and a laptop is the wrong thing to be
    watching at a junction. Mati plans; Maps drives.
    """
    o, d = r.get("_origin"), r.get("_dest")
    if not (o and d):
        return False
    flag = {"driving": "d", "walking": "w", "transit": "r"}.get(r.get("mode"), "d")
    url = (f"maps://?saddr={o[0]},{o[1]}&daddr={d[0]},{d[1]}&dirflg={flag}")
    try:
        subprocess.Popen(["/usr/bin/open", "-g", url],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:                                        # noqa: BLE001
        return False


def block(user_text=None):
    """The `<computed>` block. ⚠️ Never stored — a route is true for an hour.

    ⚠️ First person, and no "you": a block addressed to Mati becomes a claim
    about the person the moment Mati speaks it.
    """
    on, url = offline()
    L = ["<computed>",
         "Worked out just now, not remembered:"]
    if on:
        ok, detail = engine_ready()
        L.append(f"  - {detail}.")
    L.append("  - I can work out distance, time and the list of turns. I "
             "cannot give live guidance while someone is moving — no such "
             "interface exists, and a laptop is the wrong thing to watch at "
             "a junction. I hand the route to Maps for that.")
    L.append("  - I can read the turns aloud if asked.")
    L.append("</computed>")
    return "\n".join(L)


_ASKS = (
    "directions", "how do i get to", "how far is", "how long to get",
    "route to", "drive to", "walk to", "navigate", "take me to",
    "nearest", "near me", "closest", "around here", "nearby",
    "cómo llego", "como llego", "qué tan lejos", "que tan lejos", "cerca de mí",
)


def asks_about_directions(text_in):
    t = (text_in or "").lower()
    return any(p in t for p in _ASKS)


def status():
    on, url = offline()
    ok, detail = engine_ready()
    return {"offline_enabled": on, "engine": url, "engine_ready": ok,
            "detail": detail, "voice": "/usr/bin/say (offline, off by default)",
            "live_guidance": "not available from any API; handed to Maps.app"}
