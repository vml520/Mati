"""
Radio, played inside Mati. ⚠️ NO APPLICATION OPENS.

The user: *"I'd rather not open Apple Music. Is there a different way to play
music… I would rather keep this as simple as possible."*

There is, and it is simpler than any of the alternatives. :

    afplay <stream>     AudioFileOpen failed ('wht?')  — it plays FILES
    mpv / ffplay / vlc  absent, and installing one would be this project's
                        first runtime dependency
    AVFoundation        ⭐ playing in 0.5s, pauses cleanly, already on the
                        machine, and runs INSIDE this process

So an `AVPlayer` holds the stream in the server itself. No window, no Dock
icon, nothing to activate and nothing to quit — the whole "if Mati opens an
app it must be able to close it" problem simply does not arise, because no app
opens.

⚠️ CORRECTION : AN EARLIER NOTE HERE SAID MUSIC.APP HAS NO
SCRIPTABLE MINI PLAYER. IT DOES — `miniplayer` is in its dictionary. That
claim came from `sdef`, which on this machine needs Xcode and silently
returned ZERO BYTES, so a grep over nothing looked like a definitive absence.
The dictionary is readable directly at
`Music.app/Contents/Resources/com.apple.Music.sdef` (45,766 bytes).

⚠️ THE DECISION STANDS ANYWAY, for a better reason than the wrong one. A mini
player is still Music.app: a process to launch, a Dock icon, and something to
quit. An in-process AVPlayer opens nothing at all, which is what the user asked
for. The reasoning was right and one of its supports was false.

⚠️ APPLE MUSIC IS STILL USED, JUST NOT OPENED. `dj.py` reads the library to
learn what the person likes; that is a query, not a takeover. Playing is what
moved here.

Stations come from the **Radio Browser** API — community-maintained, public
domain, no key, no account. The user found it. Measured: keyless, and it returns
`url_resolved` with bitrate and codec, so a dead or terrible stream can be
filtered out before anything is played rather than after.

⚠️ THE FILTERS ARE NOT OPTIONAL. The database is open, so it is also messy:
`hidebroken` drops stations that failed their last check, `order=clickcount`
prefers ones people actually listen to, and a bitrate floor keeps a 32kbps
stream from being offered as music. Without these the first result is often
silence.
"""

from __future__ import annotations

import json
import random
import re
import threading
import time
import urllib.parse
import urllib.request

from . import config as C

# ⚠️ MIRRORS, NOT ONE HOST. Radio Browser is volunteer-run and individual
# servers go down; the project's own advice is to pick among them. Tried in
# order, and a failure moves on rather than becoming "no stations exist".
MIRRORS = ("https://de1.api.radio-browser.info",
           "https://nl1.api.radio-browser.info",
           "https://at1.api.radio-browser.info")

# ⚠️ They ask for a descriptive User-Agent so they can see who is calling.
# Being identifiable is the rent for a free community service.
UA = C.USER_AGENT
NET_TIMEOUT = 20

MIN_BITRATE = 96          # below this, music sounds like a phone call
LIMIT = 30

_player = None            # the AVPlayer, held for the life of the process
_current = None           # what it is playing

# ⚠️ AVPlayer NEEDS A RUN LOOP, AND THAT IS WHY THERE WAS NO SOUND.
#
# The user: *"the last time I asked for music there was no sound."*
# Measured the same day, playing a 320kbps stream from a plain script:
#
#     without a run loop   rate=1.00  player.status=0  item.status=0   (6s, no error)
#     pumping the loop     rate=1.00  player.status=1  item.status=1   (ready at 5.5s)
#
# Status 0 is `Unknown`. The asset never loads, no error is ever set, and the
# player sits there silently forever. The server is a threaded HTTP server with
# no CFRunLoop running anywhere, so this is what it did every time.
#
# ⚠️ AND NOTHING NOTICED, because `playing()` tested `rate() > 0`. `rate` is the
# REQUESTED rate — it is 1.0 the instant `play()` is called, whether or not a
# single byte has arrived. So Mati reported the station as playing while the
# room was silent. The bug had no reader.
#
# ⚠️ AND IT IS THE **MAIN** THREAD'S LOOP, which cost a wrong fix first.
# My first attempt marshalled every AVFoundation call onto a dedicated thread
# that ran its own loop — reasoning that a run loop is per-thread, which is
# true and not the point. Measured: with the player built on that thread and
# its loop pumped there, every station still timed out at 12s. With the same
# player built on that thread and the MAIN loop pumped instead, ready at 5.5s.
# AVFoundation does its asset work against the main run loop wherever the
# player was created.
#
# So the loop is pumped by the SERVER's main thread, in `service_actions()` —
# the hook `serve_forever` already calls on every pass. See `server.py`.
# Nothing here needs its own thread; a lock is enough, because the original
# code called AVPlayer from request threads and that part was never the bug.
_PLAYER_LOCK = threading.RLock()

# How long to wait for a stream to become ready before admitting it has not.
# ⚠️ 5.5s was the MEASURED time to ready on a good stream, so this is not
# generous — it is about twice the observed case.
READY_TIMEOUT = 12.0
POLL = 0.1


def pump(seconds=0.0):
    """Service the main run loop once. ⚠️ CALL ONLY FROM THE MAIN THREAD.

    This is what makes a stream load. `seconds=0.0` drains whatever is pending
    and returns immediately, so the caller's own loop keeps its cadence —
    `server.service_actions()` runs every `poll_interval`, which is the same
    0.5s cadence the measurement above used.

    ⚠️ Returns False when AVFoundation is not present, so a caller can stop
    asking rather than raising on every pass.
    """
    try:
        from CoreFoundation import CFRunLoopRunInMode, kCFRunLoopDefaultMode
    except Exception:                                        # noqa: BLE001
        return False
    CFRunLoopRunInMode(kCFRunLoopDefaultMode, seconds, False)
    return True


def _with_player(fn, timeout=None):
    """Touch the player under the lock. ⚠️ Any thread; see the note above."""
    with _PLAYER_LOCK:
        return fn()


# Kept so the call sites read the same whichever way this is wired.
_on_audio = _with_player


class Unavailable(RuntimeError):
    """Radio Browser could not be reached. ⚠️ NOT "no such station"."""


# ── finding something to play ────────────────────────────────────────────────

# How many rows to pull when ordering by distance. ⚠️ LARGER THAN `LIMIT` ON
# PURPOSE: sorting 30 rows by distance gives "the nearest of the 30 most
# clicked", which is not what was asked for and reads identically in the
# output. 400 was measured to work in one request.
GEO_LIMIT = 400


def _get(path, params, limit=None, order=None):
    q = dict(params, hidebroken="true",
             order=order or "clickcount",
             reverse="true", limit=str(limit or LIMIT))
    err = None
    for base in MIRRORS:
        url = f"{base}{path}?{urllib.parse.urlencode(q)}"
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=NET_TIMEOUT) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except Exception as e:                               # noqa: BLE001
            err = e
            continue
    raise Unavailable(f"could not reach any radio directory "
                      f"({type(err).__name__})")


def _clean(rows):
    """Usable stations only, best first. ⚠️ Names come from strangers."""
    from . import untrusted
    out = []
    for s in rows or ():
        url = s.get("url_resolved") or s.get("url")
        if not url or not url.startswith(("http://", "https://")):
            continue
        try:
            bitrate = int(s.get("bitrate") or 0)
        except (TypeError, ValueError):
            bitrate = 0
        if bitrate and bitrate < MIN_BITRATE:
            continue
        name = untrusted.clean(s.get("name"), max_len=80)
        if not name:
            continue
        # ⚠️ `geo_distance` MUST SURVIVE THIS, and it did not. `near()` sorted
        # on a key this function had already dropped, so every row scored
        # infinity, the sort was a no-op and every `km` came out None — a
        # distance feature that did nothing. Caught only because a `%6.1f`
        # crashed on None; `%s` would have printed "None" and shipped.
        #
        # ⚠️ NUMBERS, SO NOT PUT THROUGH `untrusted.clean`. That is for text a
        # stranger typed. These are coerced to float instead, which is the
        # equivalent guard for a number: a hostile value becomes None rather
        # than reaching a sort comparator.
        def _num(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        out.append({
            "name": name,
            "url": url,
            "bitrate": bitrate or None,
            "codec": (s.get("codec") or "").upper() or None,
            "country": untrusted.clean(s.get("country"), max_len=40) or None,
            "state": untrusted.clean(s.get("state"), max_len=40) or None,
            "tags": untrusted.clean(s.get("tags"), max_len=80) or None,
            "votes": s.get("votes"),
            "geo_distance": _num(s.get("geo_distance")),
            "geo_lat": _num(s.get("geo_lat")),
            "geo_long": _num(s.get("geo_long")),
        })
    return out


def by_genre(genre, limit=8):
    """Stations for a tag. [] when the tag is real and has nothing good."""
    g = " ".join(str(genre or "").split()).lower()
    if not g:
        return []
    return _clean(_get(f"/json/stations/bytag/{urllib.parse.quote(g)}", {}))[:limit]


def by_country(country, limit=8):
    c = " ".join(str(country or "").split())
    if not c:
        return []
    return _clean(_get(
        f"/json/stations/bycountryexact/{urllib.parse.quote(c)}", {}))[:limit]


def by_language(language, limit=8):
    l = " ".join(str(language or "").split()).lower()
    if not l:
        return []
    return _clean(_get(
        f"/json/stations/bylanguage/{urllib.parse.quote(l)}", {}))[:limit]


def search(name, limit=8):
    n = " ".join(str(name or "").split())
    if not n:
        return []
    return _clean(_get("/json/stations/search", {"name": n}))[:limit]


# ⚠️ WHAT THE LAST SEARCH FOUND, so "the next station" has something to mean.
#
# The user: *"there needs to be verbal navigation for the stations found in a
# search; such as 'play another station'… 'play the next station.'"* A pattern
# can recognise "next"; only state can answer it. This is the state.
#
# ⚠️ IN MEMORY, AND THAT IS HONEST. A restart loses the cursor, and the right
# behaviour then is to say the list is gone rather than to resurrect a
# yesterday's search as though it were this conversation.
_last_found = []
_last_at = -1
_last_wanted = None


def remember_found(rows, wanted=None):
    """Hold a search result set for `next_found` to walk."""
    global _last_found, _last_at, _last_wanted
    _last_found = list(rows or ())
    _last_at = -1
    _last_wanted = wanted
    return len(_last_found)


def mark_played(station):
    """Note which of the held results is on, so "next" means next from here."""
    global _last_at
    for i, r in enumerate(_last_found):
        if r.get("url") == (station or {}).get("url"):
            _last_at = i
            return i
    return _last_at


def next_found():
    """The next station in the last search. (station, remaining) or (None, why).

    ⚠️ DOES NOT WRAP. Silently returning to the top makes a list of four feel
    infinite and plays the same station again without saying so. Running out
    is an answer.
    """
    if not _last_found:
        return None, "I do not have a list of stations to move through."
    if _last_at + 1 >= len(_last_found):
        return None, (f"That was the last of {len(_last_found)}"
                      + (f" for “{_last_wanted}”" if _last_wanted else "") + ".")
    return _last_found[_last_at + 1], len(_last_found) - _last_at - 2


def nth_found(n):
    """The n-th station of the last search, 1-based. None if out of range.

    ⚠️ THE USER ASKED FOR THIS IN HIS OWN WORDS and there was nothing to answer
    with: *"I mean the first of the hip hop stations that you presented"* and
    *"Let me hear the first station again"*. The list was held and only ever
    walked FORWARD, so a station already named could not be returned to.
    """
    if not _last_found or n is None:
        return None
    i = int(n) - 1
    if 0 <= i < len(_last_found):
        return _last_found[i]
    return None


def back_found():
    """The station before the current one. (station, None) or (None, why)."""
    if not _last_found:
        return None, "I do not have a list of stations to move through."
    if _last_at <= 0:
        return None, "That was the first one I found."
    return _last_found[_last_at - 1], None


def listing(limit=8):
    """The last search, as named options. [] when there is no list."""
    return [{"n": i + 1, "name": r.get("name"), "km": r.get("km")}
            for i, r in enumerate(_last_found[:limit])]


def found_count():
    return len(_last_found)


def remaining():
    """How many are left AFTER the one playing.

    ⚠️ `found_count() - 1` is not this. It reported "5 more" on the first
    station and still "5 more" on the third, because it never moved.
    """
    if not _last_found:
        return 0
    return max(0, len(_last_found) - _last_at - 1)


# How wide a distance band counts as "the same distance" when ranking. Inside
# a band, the better match wins; across bands, nearer wins on a tie.
BAND_KM = 500


def _norm(s):
    """Letters and digits only, lowercased. ⚠️ `hip hop`, `hip-hop` and
    `HipHop` are the same genre and the directory spells it all three ways."""
    return re.sub(r"[^a-z0-9]+", "", str(s or "").lower())


def _match(row, want):
    """How well a station matches a genre, 0..2. Arithmetic, no judgement.

    ⚠️ IT IS THE *SHARE* OF TAGS, NOT THE COUNT. A station carrying ten
    genres matches every one of them weakly, and that is exactly the case
    that beat a real hip hop station on distance alone. One tag that matches
    out of one beats one out of ten.
    """
    w = _norm(want)
    if not w:
        return 0.0
    tags = [t for t in (row.get("tags") or "").split(",") if t.strip()]
    hits = sum(1 for t in tags if w in _norm(t))
    share = hits / len(tags) if tags else 0.0
    # ⚠️ THE NAME IS WORTH A WHOLE POINT. "West Coast – G-Funk & Hip-Hop" and
    # "181.FM - Old School HipHop" say what they are; a tag list is what
    # somebody typed into a community database.
    in_name = 1.0 if w in _norm(row.get("name")) else 0.0
    return round(share + in_name, 3)


def near(limit=8, tag=None, name=None, lat=None, lon=None, countrycode=None):
    """Stations nearest the person, closest first. [] when location is unknown.

    the user: *"have the search list stations by distance from user."*

    ⚠️ THE LOCATION COMES FROM SETTINGS, NEVER THE DEVICE. `weather.where()`
    settled this and `directions.here()` reused it: asked for, coarsened,
    forgettable, out of the field. Reading GPS to save a question is the
    pattern this project has refused three times now.

    ⚠️ SORTED HERE, NOT BY THE API. `order=geodist` is accepted and silently
    ignored — measured  from the development machine, the eight
    rows came back in
    NAME order with distances of 8.6M, 3.3M, 944k, 2.2M… So the sort is local,
    over `GEO_LIMIT` rows, with `has_geo_info` guaranteeing every row has
    coordinates.

    ⚠️ `geo_distance` IS IN METRES. 8,679,204 for a Netherlands station is not
    kilometres — Earth is 40,000km around. Verified against my own haversine
    on six stations: identical to one decimal place.

    ⚠️ HALF THE DIRECTORY HAS NO COORDINATES — measured 15 of 30 US stations.
    `has_geo_info=true` excludes them rather than sorting them as if they were
    at (0, 0), which would place every unknown station in the Gulf of Guinea
    and rank it above anything real.
    """
    if lat is None or lon is None:
        try:
            from . import directions
            where = directions.here()
        except Exception:                                    # noqa: BLE001
            where = None
        if not where:
            return []
        lat, lon = where[0], where[1]
        # ⚠️ COUNTRY FIRST, AND THE SAVED LOCATION ALREADY KNOWS IT. The user:
        # *"it doesn't matter how far away the closest popular station is, as
        # long as we get nearby stations first before stations from outside of
        # the the US."* So the country narrows the pull before distance orders
        # it. Measured from the development machine: unfiltered the nearest
        # is 574km; filtered to US it is 37km. Same query, same sort — a
        # different sample.
        if countrycode is None:
            try:
                from . import config as _C
                countrycode = (_C._settings().get("weather_country") or None)
            except Exception:                                # noqa: BLE001
                pass

    params = {"has_geo_info": "true",
              "geo_lat": f"{float(lat):.4f}",
              "geo_long": f"{float(lon):.4f}"}
    # ⚠️ "NEAREST OF WHAT WAS FETCHED", AND THE SAMPLE IS VISIBLE IN THE
    # ANSWER. Measured from the development machine: with no country filter
    # the closest result is 574km, because 400 rows of GLOBAL most-clicked
    # contain almost no stations from its own country. With `countrycode=US`
    # the same query returns one at 37km. So this is
    # nearest-of-the-most-listened unless a caller narrows it,
    # and `countrycode` is how they narrow it.
    if countrycode:
        params["countrycode"] = str(countrycode).upper()[:2]
    if tag:
        params["tag"] = " ".join(str(tag).split()).lower()
    if name:
        params["name"] = " ".join(str(name).split())

    rows = _clean(_get("/json/stations/search", params, limit=GEO_LIMIT))
    for r in rows:
        d = r.get("geo_distance")
        r["km"] = round(d / 1000.0, 1) if d is not None else None

    # ⚠️ DISTANCE WAS THE ONLY KEY, AND IT PICKED A CHILL STATION FOR HIP HOP.
    # The user: *"That's not hip hop."* They were right. Measured for `hip hop`:
    #
    #   Radio ACID - Everything Chill  583km   14 votes  hip hop 1 of ~10 tags
    #   West Coast – G-Funk & Hip-Hop  711km   15        3 of 5, and in the name
    #   senshiphop                     717km  140        2 of 4, and in the name
    #   181.FM - Old School HipHop     1008km 1441       its ONLY tag
    #
    # The tag filter is binary — every one of these IS tagged hip hop — so
    # sorting the survivors by distance threw away the only signal that says
    # which is actually a hip hop station. A station carrying ten genres
    # matches all of them weakly.
    #
    # ⚠️ DISTANCE STILL MATTERS, IN BUCKETS. The user also asked for nearby
    # first, and both are honoured: match decides, then 500km bands, then the
    # directory's own vote count. A 583km station does not beat a better match
    # 128km further out, but a tie does go to the nearer one.
    want = tag or name
    if want:
        for r in rows:
            r["match"] = _match(r, want)
        rows.sort(key=lambda r: (
            -r["match"],
            int((r["km"] or 1e6) // BAND_KM),
            -(r.get("votes") or 0)))
    else:
        # ⚠️ A row with no distance sorts LAST, not first. `or 0` would make
        # an unknown distance the nearest thing in the list.
        rows.sort(key=lambda r: r["km"] if r["km"] is not None else float("inf"))
    return rows[:limit]


# ── playing it, in this process ──────────────────────────────────────────────

def _make_player(url):
    import objc
    from Foundation import NSBundle, NSURL
    if not NSBundle.bundleWithPath_(
            "/System/Library/Frameworks/AVFoundation.framework").load():
        raise Unavailable("the audio framework would not load")
    AVPlayer = objc.lookUpClass("AVPlayer")
    u = NSURL.URLWithString_(url)
    if u is None:
        raise Unavailable("that stream address is not usable")
    return AVPlayer.playerWithURL_(u)


def play(station):
    """Start a stream. (ok, message). ⚠️ Replaces whatever was playing.

    ⚠️ NO APPLICATION IS LAUNCHED. The player lives here, so stopping is
    immediate and there is nothing left on screen afterwards.
    """
    global _player, _current
    if isinstance(station, str):
        found = search(station, limit=1) or by_genre(station, limit=1)
        if not found:
            # ⚠️ ASKS, EVEN HERE. This was left as a bare statement on the
            # reasoning that the low-level player has no conversation to hold
            # — and the user overruled it: *"Have radio and dj ask too."* They are
            # right that the caller does not always rephrase it, so a plain
            # report reaches the person unchanged and reads as "your request
            # does not exist" when it means "I did not match your words".
            return False, (f"nothing here matched {station!r} — is that a "
                           f"genre, a place, or a station's name?")
        station = found[0]
    stop()

    def _start():
        global _player
        _player = _make_player(station["url"])
        _player.play()
        return True

    try:
        _on_audio(_start)
    except Unavailable as e:
        return False, str(e)
    except Exception as e:                                   # noqa: BLE001
        return False, f"the player would not start: {e}"

    # ⚠️ WAIT FOR SOUND, DO NOT ASSUME IT. This used to return the moment
    # `play()` was called — which is why "no sound" reported as success. A
    # stream measured 5.5s to ReadyToPlay, so returning early is returning
    # before there is anything to hear.
    deadline = time.time() + READY_TIMEOUT
    while time.time() < deadline:
        state = _on_audio(_item_state)
        if state["item"] == 1:                       # ReadyToPlay
            _current = station
            bits = station.get("bitrate")
            return True, (f"{station['name']}"
                          + (f" — {bits}kbps" if bits else "")
                          + (f", {station['country']}"
                             if station.get("country") else ""))
        if state["item"] == 2 or state["error"]:     # Failed
            _on_audio(_tear_down)
            return False, (f"{station['name']} would not play"
                           + (f" — {state['error']}" if state["error"] else ""))
        time.sleep(POLL)

    _on_audio(_tear_down)
    # ⚠️ NAMED AS A TIMEOUT, NOT AS A REFUSAL. The station may be fine and the
    # network slow, and saying "could not find it" would be false.
    return False, (f"{station['name']} did not start within "
                   f"{READY_TIMEOUT:.0f}s — the stream may be down")


def _item_state():
    """(player status, item status, error) — read ON the audio thread."""
    if _player is None:
        return {"player": None, "item": None, "error": None}
    item = _player.currentItem()
    err = None
    try:
        e = item.error() if item else None
        if e is not None:
            err = str(e.localizedDescription())
    except Exception:                                        # noqa: BLE001
        pass
    return {"player": int(_player.status()),
            "item": int(item.status()) if item else None,
            "error": err}


def _tear_down():
    global _player
    if _player is not None:
        try:
            _player.pause()
        except Exception:                                    # noqa: BLE001
            pass
    _player = None
    return True


def stop():
    """Silence, immediately. ⚠️ Nothing to close — see the module docstring."""
    global _current
    was = _current
    if _player is not None:
        try:
            _on_audio(_tear_down, timeout=5.0)
        except Exception:                                    # noqa: BLE001
            pass
    _current = None
    return was


def playing():
    """What is on, or None. ⚠️ Asks the PLAYER, never a remembered flag.

    ⚠️ AND NOT `rate()` EITHER, WHICH IS THE BUG THIS DOCSTRING MISSED. `rate`
    is the REQUESTED rate: it reads 1.0 the instant `play()` is called, before
    a single byte has arrived and whether or not any ever does. Asking the
    player was right; asking it the wrong question reported silence as music
    for as long as radio has existed here. Readiness is `item.status == 1`.
    """
    if _player is None or _current is None:
        return None
    try:
        state = _on_audio(_item_state, timeout=5.0)
    except Exception:                                        # noqa: BLE001
        return None
    if state["item"] != 1:
        # Not ready, or failed, or the stream dropped.
        return None
    try:
        if float(_on_audio(lambda: _player.rate(), timeout=5.0)) <= 0.0:
            return None
    except Exception:                                        # noqa: BLE001
        return None
    return dict(_current)


def volume(level=None):
    """Read or set volume, 0.0–1.0. None reads it.

    ⚠️ ON THE AUDIO THREAD like everything else that touches the player —
    see the run-loop note at the top. A setter that lands on the wrong thread
    is the quiet version of the no-sound bug.
    """
    if _player is None:
        return None
    try:
        if level is None:
            return round(float(_on_audio(lambda: _player.volume(),
                                         timeout=5.0)), 2)
        lvl = max(0.0, min(1.0, float(level)))
        _on_audio(lambda: _player.setVolume_(lvl), timeout=5.0)
        return lvl
    except Exception:                                        # noqa: BLE001
        return None


def describe(station):
    s = station["name"]
    if station.get("bitrate"):
        s += f" — {station['bitrate']}kbps"
    if station.get("codec"):
        s += f" {station['codec']}"
    if station.get("country"):
        s += f", {station['country']}"
    return s


def status():
    now = playing()
    return {"playing": now["name"] if now else None,
            "source": "Radio Browser (community, no key)",
            "player": "AVFoundation, in-process — no application opens",
            "mirrors": len(MIRRORS),
            "min_bitrate": MIN_BITRATE}
