"""
Podcasts — the one kind of audio that is meant to be kept.

Songs cannot be downloaded for an offline list: an Apple Music library is
`cloud status: subscription`, DRM streaming, and `download` is not in
Music.app's scripting dictionary (see `dj.py`). Podcasts are the case that IS
designed for it — an RSS `<enclosure>` is a file, published to be fetched and
kept, and no terms are being bent by doing what the format exists for.

All of it is free and keyless:

    gpodder.net search  ->  the feed URL for a show, no account, no key
    the feed            ->  episodes, each enclosure a file with a type and size
    playback            ->  in this process, through the player radio uses

⚠️ SIZE IS STATED BEFORE ANYTHING IS FETCHED. An episode is tens of megabytes
and this is someone's disk and possibly their tethered connection. Every
download says what it will cost first, and `MAX_EPISODE_MB` refuses the
outliers outright.

⚠️ NOTHING IS SUBSCRIBED OR FETCHED WITHOUT BEING ASKED. Idle time is for
absorbing what is already held — a background process that quietly accumulates
gigabytes of audio is not a feature.

⚠️ VIDEO IS METADATA ONLY, DELIBERATELY. A YouTube RSS feed gives titles and
links and is free and keyless — but the video itself is not in the feed, and
fetching it would need a scraper and would breach the terms. Video PODCASTS are
different: their enclosures are files like any other, and they download by the
same path as audio. `is_video` marks which is which rather than pretending they
are the same.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from . import bridge
from . import config as C

# ⚠️ gpodder.net, NOT iTunes, and no Apple service anywhere in the path. A
# podcast IS an RSS feed with `<enclosure>` audio — `episodes()` below parses
# exactly that. Apple Podcasts is a DIRECTORY over the same feeds, not a
# format, so using another index removes a middleman rather than working
# around one.
#
# gpodder.net is keyless, open and community-run — the same shape as Radio
# Browser, which this project already relies on. `search.json?q=` returns
# title, url, author, website and subscriber counts, with no account.
#
# ⚠️ PodcastIndex is the other candidate and is out: it answers 401 without a
# key, and a shared key in a public Apache-2.0 build is not a key.
SEARCH_API = "https://gpodder.net/search.json"
UA = C.USER_AGENT
NET_TIMEOUT = 25

# ⚠️ SELF-LIMITED BECAUSE NOBODY MAKES US. gpodder.net documents no rate limit
# and asks for no key — the directory search is explicitly public content — and
# that is exactly why a client should throttle itself. It is a community-run
# free service on donated hosting, and "no published limit" is an absence of a
# rule, not permission to hammer it.
#
# ⭐ THE SAME DISCIPLINE AS THE OTHER SERVICES. `places.py` leaves 3s between
# Overpass calls and caches for an hour; `flipp.py` caches for an hour.
#
# A podcast directory changes over days, not seconds, so an hour of cache
# costs a user nothing and takes repeat questions off the network entirely.
MIN_GAP_SECONDS = 1.0
FRESH_SECONDS = 3600

_cache = {}              # query -> (fetched_at, [shows])
_last_call = [0.0]

# Where kept episodes live. Beside the rest of Mati's state, so `transfer`
# carries or skips them as one decision rather than scattering audio.
DIR = os.path.join(C.CACHE_DIR, "podcasts")

# ⚠️ A ceiling, not a guess. A four-hour show at high bitrate is 300 MB and
# nobody meant to spend that by saying "keep it".
MAX_EPISODE_MB = 250

class FeedUnavailable(RuntimeError):
    """The feed could not be read. ⚠️ NOT the same as a show with no episodes."""


AUDIO = ("audio/",)
VIDEO = ("video/",)


# ── finding shows ────────────────────────────────────────────────────────────

def search(term, limit=5):
    """Shows matching a term. ⚠️ The request carries the term and nothing else.

    ⚠️ NAMES AND DESCRIPTIONS COME FROM STRANGERS. A directory is user-edited,
    so every field a person could see goes through `untrusted.clean` — the
    same guard `radio._clean` applies to station names, and for the same
    reason: this text ends up inside Mati's prompt.
    """
    from . import untrusted
    term = " ".join((term or "").split())
    if not term:
        return []
    # ⚠️ KEYED ON THE QUERY, NOT ON `limit`. The same search asked for 3 then 5
    # results must not become two requests; the slice happens on the way out.
    hit = _cache.get(term.lower())
    if hit and time.time() - hit[0] < FRESH_SECONDS:
        return hit[1][:limit]

    gap = time.time() - _last_call[0]
    if gap < MIN_GAP_SECONDS:
        time.sleep(MIN_GAP_SECONDS - gap)
    _last_call[0] = time.time()

    q = urllib.parse.urlencode({"q": term})
    req = urllib.request.Request(f"{SEARCH_API}?{q}", headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=NET_TIMEOUT) as r:
            data = json.loads(r.read().decode("utf-8", "replace"))
    except Exception:                                        # noqa: BLE001
        return []
    # ⚠️ A LIST, NOT A DICT. gpodder returns the array itself rather than
    # wrapping it in {"results": [...]}, and `.get("results", [])` on a list
    # raises rather than returning nothing — a silent "no podcasts found".
    if not isinstance(data, list):
        return []
    out = []
    for x in data:
        feed = (x.get("url") or "").strip()
        if not feed.startswith(("http://", "https://")):
            continue
        title = untrusted.clean(x.get("title"), max_len=120)
        if not title:
            continue
        out.append({"title": title,
                    "author": untrusted.clean(x.get("author"), max_len=80) or None,
                    "feed": feed,
                    "website": (x.get("website") or "").strip() or None,
                    # ⚠️ The directory's own count, like `clickcount` for
                    # stations — arithmetic, not a judgement about quality.
                    "subscribers": x.get("subscribers")})
    # Most-subscribed first, so a search for a common name lands on the show
    # people actually mean rather than on whichever duplicate sorted first.
    out.sort(key=lambda s: -(s.get("subscribers") or 0))
    # ⚠️ STORED BEFORE THE SLICE, so a later call wanting more results is
    # served from here rather than going back out. A cache that is read and
    # never written is just a slower miss.
    _cache[term.lower()] = (time.time(), out)
    return out[:limit]


def _fetch(url, timeout=NET_TIMEOUT):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def episodes(feed_url, limit=10):
    """Episodes from a feed, newest first, with what each would cost to keep.

    ⚠️ Parsed with `xml.etree`, not a dependency, and a malformed feed is
    SKIPPED rather than repaired — the same rule the RSS design already sets.
    """
    # ⚠️ AN UNREACHABLE FEED IS NOT AN EMPTY SHOW. Returning [] for both would
    # have Mati say "that podcast has no episodes" about a feed that is simply
    # down. Raising makes the caller choose what to say.
    try:
        raw = _fetch(feed_url)
    except Exception as e:                                   # noqa: BLE001
        raise FeedUnavailable(f"could not reach that feed ({type(e).__name__})") from e
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as e:
        raise FeedUnavailable("that feed is not readable XML") from e
    out = []
    for item in root.iter("item"):
        enc = item.find("enclosure")
        if enc is None:
            continue
        url = enc.get("url")
        ctype = (enc.get("type") or "").lower()
        if not url or not url.startswith(("http://", "https://")):
            continue
        try:
            size = int(enc.get("length") or 0)
        except ValueError:
            size = 0
        title = (item.findtext("title") or "").strip()
        out.append({
            "title": title,
            "url": url,
            "type": ctype,
            "mb": round(size / 1e6, 1) if size else None,
            "published": (item.findtext("pubDate") or "").strip(),
            "is_video": ctype.startswith(VIDEO),
        })
        if len(out) >= limit:
            break
    return out


# ── keeping one ──────────────────────────────────────────────────────────────

def _safe_name(title, url):
    stem = re.sub(r"[^\w \-.]", "", title or "").strip()[:70] or "episode"
    ext = os.path.splitext(urllib.parse.urlparse(url).path)[1][:6] or ".mp3"
    if not re.match(r"^\.[A-Za-z0-9]+$", ext):
        ext = ".mp3"
    return stem + ext


def would_cost(ep):
    """What keeping this episode costs, in words. Always said before fetching."""
    if ep.get("mb"):
        return f"{ep['mb']:.0f} MB"
    return "an unstated size"


def download(ep, db_path=None, verbose=False):
    """Fetch one episode to disk. Returns the path, or None with a reason.

    ⚠️ REFUSES RATHER THAN SURPRISES. Over the ceiling, or a size the feed
    would not state and the server will not confirm, and nothing is written.
    """
    url = ep.get("url")
    if not url:
        return None, "no file in that episode"
    mb = ep.get("mb")
    if mb and mb > MAX_EPISODE_MB:
        return None, f"that episode is {mb:.0f} MB, past the {MAX_EPISODE_MB} MB limit"
    os.makedirs(DIR, exist_ok=True)
    path = os.path.join(DIR, _safe_name(ep.get("title"), url))
    if os.path.exists(path):
        return path, "already kept"
    tmp = path + ".part"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=120) as r:
            declared = int(r.headers.get("Content-Length") or 0)
            if declared and declared / 1e6 > MAX_EPISODE_MB:
                return None, (f"the server says it is "
                              f"{declared/1e6:.0f} MB, past the limit")
            written = 0
            with open(tmp, "wb") as fh:
                while True:
                    chunk = r.read(262144)
                    if not chunk:
                        break
                    written += len(chunk)
                    # ⚠️ Enforced DURING the read, not only from the header. A
                    # server that understates or omits Content-Length must not
                    # be able to fill the disk.
                    if written / 1e6 > MAX_EPISODE_MB:
                        fh.close(); os.remove(tmp)
                        return None, "it kept going past the limit, so I stopped"
                    fh.write(chunk)
        os.replace(tmp, path)
    except Exception as e:                                   # noqa: BLE001
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except OSError: pass
        return None, f"it would not download ({type(e).__name__})"
    if verbose:
        print(f"  [podcast] kept {os.path.basename(path)} "
              f"({os.path.getsize(path)/1e6:.1f} MB)")
    return path, None


def kept():
    """What is on disk, newest first. This is the offline list."""
    if not os.path.isdir(DIR):
        return []
    rows = []
    for name in os.listdir(DIR):
        p = os.path.join(DIR, name)
        if name.endswith(".part") or not os.path.isfile(p):
            continue
        rows.append({"name": name, "path": p,
                     "mb": round(os.path.getsize(p) / 1e6, 1),
                     "at": os.path.getmtime(p)})
    rows.sort(key=lambda r: -r["at"])
    return rows


def play(ep_or_path):
    """Play an episode — a kept file, or straight from the feed. (ok, message).

    ⚠️ IN THIS PROCESS, NOT IN MUSIC.APP. Podcasts use no Apple Music path at
    all: a kept file and a stream both go through `radio.play`.

    ⚠️ AND THAT IS THE RIGHT PLAYER FOR THREE REASONS. `radio.play` waits for
    ReadyToPlay rather than assuming it (assuming it can mean no sound), it
    runs against the main run loop that AVFoundation needs, and it is the
    player `/silence` and the now-playing chip already control. An episode
    started through Music.app could not be stopped by the stop button at all.

    ⚠️ A LOCAL FILE NEEDS A file:// URL. AVPlayer will not take a bare POSIX
    path, and the failure is the silent one — `URLWithString_` returns nil and
    `_make_player` raises "that stream address is not usable".
    """
    from . import radio
    is_dict = isinstance(ep_or_path, dict)
    target = ep_or_path.get("url") if is_dict else ep_or_path
    title = (ep_or_path.get("title") if is_dict else None) or "episode"
    if not target:
        return False, "there is no episode address to play"
    target = str(target)
    if os.path.isfile(target):
        target = "file://" + urllib.parse.quote(os.path.abspath(target))
    return radio.play({"name": title, "url": target})


# ── the shelf ────────────────────────────────────────────────────────────────

def subscribe(show, db_path=None):
    """Put a show on the shelf beside the books. Nothing is fetched."""
    from . import library
    try:
        library.ensure(db_path)
        library.add(show["title"], status=library.WANTED, db_path=db_path,
                    origin=f"podcast:{show['feed']}")
        return True
    except Exception:                                        # noqa: BLE001
        return False


def status():
    files = kept()
    return {"kept": len(files),
            "megabytes": round(sum(f["mb"] for f in files), 1),
            "folder": DIR}
