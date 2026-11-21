"""
Radio, played inside Mati. No application opens.

Streams play through an `AVPlayer` held in the server process itself. There is
no window, no Dock icon, nothing to activate and nothing to quit, so stopping
is immediate and leaves nothing behind. AVFoundation is already on every Mac;
nothing is installed. (`afplay` plays files, not streams, and mpv, ffplay or
VLC would each be a runtime dependency.)

Apple Music's library may still be read (`dj.py`) to learn what the person
likes. That is a query; Music.app is never opened or driven.

Stations come from the **Radio Browser** API: community-maintained, public
domain, no key, no account. It returns `url_resolved` with bitrate and codec,
so dead or poor streams are filtered out before anything plays.

The filters are required, because the directory is open and therefore messy:
`hidebroken` drops stations that failed their last check, `order=clickcount`
prefers stations people actually listen to, and a bitrate floor keeps a
low-quality stream from being offered as music.
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

# Several mirrors, tried in order. Radio Browser is volunteer-run and single
# servers go down; a failure moves on to the next rather than reporting that
# no stations exist.
MIRRORS = ("https://de1.api.radio-browser.info",
           "https://nl1.api.radio-browser.info",
           "https://at1.api.radio-browser.info")

# Radio Browser asks callers for a descriptive User-Agent.
UA = C.USER_AGENT
NET_TIMEOUT = 20

MIN_BITRATE = 96          # below this, music sounds like a phone call
LIMIT = 30

_player = None            # the AVPlayer, held for the life of the process
_current = None           # what it is playing

# ⚠️ AVPlayer NEEDS THE MAIN RUN LOOP. Without it the item stays at status 0
# (`Unknown`): the asset never loads, no error is set, and the player is
# silent indefinitely. AVFoundation does its asset work against the MAIN run
# loop wherever the player was created, so a loop pumped on another thread
# does not help.
#
# The server's main thread pumps it in `service_actions()`, the hook
# `serve_forever` calls on every pass (see `server.py`). Player calls may come
# from any request thread; a lock serialises them.
#
# `rate()` is the REQUESTED rate and reads 1.0 as soon as `play()` is called,
# so it cannot tell whether sound is coming out. Readiness is
# `item.status == 1` (ReadyToPlay); see `playing()`.
_PLAYER_LOCK = threading.RLock()

# How long to wait for a stream to become ready before reporting that it has
# not. A healthy stream is typically ready in about 5 seconds.
READY_TIMEOUT = 12.0
POLL = 0.1


def pump(seconds=0.0):
    """Service the main run loop once. ⚠️ CALL ONLY FROM THE MAIN THREAD.

    This is what makes a stream load. `seconds=0.0` drains whatever is pending
    and returns immediately, so the caller's own loop keeps its cadence
    (`server.service_actions()` runs every `poll_interval`).

    Returns False when AVFoundation is not present, so a caller can stop
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

# How many rows to pull when ordering by distance. Larger than `LIMIT` on
# purpose: sorting 30 rows by distance would give "the nearest of the 30 most
# clicked", not the nearest stations. 400 fits in one request.
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
        # The geo fields are kept: `near()` sorts on `geo_distance`.
        #
        # They are numbers, so they are coerced to float rather than put
        # through `untrusted.clean` (which is for text). A hostile value
        # becomes None rather than reaching a sort comparator.
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


# What the last search found, so "play the next station", "the third one" and
# "go back" have something to refer to.
#
# Held in memory only. After a restart the list is gone, and Mati says so
# rather than reviving an old search as though it were this conversation.
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

    Does not wrap: returning silently to the top would replay a station
    without saying so. Running out is an answer.
    """
    if not _last_found:
        return None, "I do not have a list of stations to move through."
    if _last_at + 1 >= len(_last_found):
        return None, (f"That was the last of {len(_last_found)}"
                      + (f" for “{_last_wanted}”" if _last_wanted else "") + ".")
    return _last_found[_last_at + 1], len(_last_found) - _last_at - 2


def nth_found(n):
    """The n-th station of the last search, 1-based. None if out of range.

    Lets a person return to a station already offered: "the first of the hip
    hop stations", "let me hear the first station again".
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

    Counts from the current position, not from the start of the list.
    """
    if not _last_found:
        return 0
    return max(0, len(_last_found) - _last_at - 1)


# How wide a distance band counts as "the same distance" when ranking. Inside
# a band, the better match wins; across bands, nearer wins on a tie.
BAND_KM = 500


def _norm(s):
    """Letters and digits only, lowercased. `hip hop`, `hip-hop` and `HipHop`
    are the same genre, and the directory spells it all three ways."""
    return re.sub(r"[^a-z0-9]+", "", str(s or "").lower())


def _match(row, want):
    """How well a station matches a genre, 0..2. Arithmetic, no judgement.

    It is the SHARE of tags, not the count. A station carrying ten genres
    matches each of them only weakly; one matching tag out of one beats one
    out of ten.
    """
    w = _norm(want)
    if not w:
        return 0.0
    tags = [t for t in (row.get("tags") or "").split(",") if t.strip()]
    hits = sum(1 for t in tags if w in _norm(t))
    share = hits / len(tags) if tags else 0.0
    # The genre in the station's NAME is worth a whole point: a name like
    # "Old School HipHop" says what the station is, while tags are whatever
    # someone typed into a community database.
    in_name = 1.0 if w in _norm(row.get("name")) else 0.0
    return round(share + in_name, 3)


def near(limit=8, tag=None, name=None, lat=None, lon=None, countrycode=None):
    """Stations nearest the person, closest first. [] when location is unknown.

    The location comes from settings, never the device: asked for, coarsened,
    forgettable, and kept out of the field (`weather.where()`,
    `directions.here()`).

    Sorted here, not by the API: `order=geodist` is accepted and ignored. The
    sort runs over `GEO_LIMIT` rows, and `has_geo_info=true` guarantees each
    has coordinates. Stations without coordinates (about half the directory)
    are excluded rather than sorted as if they sat at (0, 0).

    `geo_distance` is in metres.
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
        # Country first: stations in the person's own country come before
        # stations abroad. The saved location already knows the country, and
        # it narrows the pull before distance orders it.
        if countrycode is None:
            try:
                from . import config as _C
                countrycode = (_C._settings().get("weather_country") or None)
            except Exception:                                # noqa: BLE001
                pass

    params = {"has_geo_info": "true",
              "geo_lat": f"{float(lat):.4f}",
              "geo_long": f"{float(lon):.4f}"}
    # The result is the nearest of what was fetched. Without a country filter
    # the pull is the globally most-listened stations, which contain few
    # local ones; `countrycode` narrows it.
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

    # Ranking: how well the station matches the genre decides first, then
    # 500 km distance bands, then the directory's vote count. The tag filter
    # is binary (every row returned carries the tag), so distance alone would
    # prefer a nearby station that lists ten genres over a farther one that
    # is plainly the genre asked for. Within a band, a tie goes to the nearer.
    want = tag or name
    if want:
        for r in rows:
            r["match"] = _match(r, want)
        rows.sort(key=lambda r: (
            -r["match"],
            int((r["km"] or 1e6) // BAND_KM),
            -(r.get("votes") or 0)))
    else:
        # A row with no distance sorts last, not first.
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

    No application is launched. The player lives here, so stopping is
    immediate and nothing is left on screen afterwards.
    """
    global _player, _current, _paused
    if isinstance(station, str):
        found = search(station, limit=1) or by_genre(station, limit=1)
        if not found:
            # Phrased as a question, because callers often pass it to the
            # person unchanged: "nothing matched your words" must not read as
            # "what you asked for does not exist".
            return False, (f"nothing here matched {station!r} — is that a "
                           f"genre, a place, or a station's name?")
        station = found[0]
    stop(keep_paused=True)

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

    # Wait for sound rather than assume it: success is reported only once the
    # stream reaches ReadyToPlay.
    deadline = time.time() + READY_TIMEOUT
    while time.time() < deadline:
        state = _on_audio(_item_state)
        if state["item"] == 1:                       # ReadyToPlay
            _current = station
            _paused = None                           # sound again: no pause
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
    # Reported as a timeout, not a refusal: the station may be fine and the
    # network slow.
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


def stop(keep_paused=False):
    """Silence, immediately. Nothing to close; see the module docstring.

    Stop also ends a pause: stop means off. `keep_paused` is for `play()`
    alone, which clears the player before tuning and must keep the pause
    while it does (see `resume`).
    """
    global _current, _paused, _tuning
    was = _current or _paused
    if _player is not None:
        try:
            _on_audio(_tear_down, timeout=5.0)
        except Exception:                                    # noqa: BLE001
            pass
    _current = None
    if not keep_paused:
        _paused = None
        _tuning = None
    return was


# Pause is a stop that remembers the station. A live stream has nothing to
# resume: a paused AVPlayer would come back minutes behind the broadcast, or
# fail once its buffer expired. So the player is torn down as `stop` does,
# the station is kept, and Play tunes it fresh, which takes a few seconds.
_paused = None
_tuning = None          # the paused station, while Play is tuning it


def pause():
    """Silence, keeping the station for `resume`. The station, or None."""
    global _paused
    was = _current
    if was is None:
        return None
    stop()
    _paused = was
    return was


def resume():
    """Play the paused station again, live. (ok, message).

    The pause stays in place while the stream tunes: `play` clears it only
    at ReadyToPlay, and `paused()` reports `tuning` meanwhile. Otherwise the
    page would see neither playing nor paused for those seconds and take down
    the label, the controls and the presets.

    A Stop during tuning wins: it clears the pause, and a tune that then
    fails does not bring it back.
    """
    global _tuning
    st = _paused
    if st is None:
        return False, "nothing is paused"
    _tuning = st
    try:
        return play(st)
    finally:
        _tuning = None


def paused():
    if not _paused:
        return None
    d = dict(_paused)
    d["tuning"] = _tuning is _paused
    return d


def playing():
    """What is on, or None. Asks the player, never a remembered flag.

    Readiness is `item.status == 1`, not `rate()`: `rate` is the REQUESTED
    rate and reads 1.0 as soon as `play()` is called, whether or not any sound
    ever arrives.
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

    Goes through the player lock like everything else that touches the
    player; see the run-loop note at the top.
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
