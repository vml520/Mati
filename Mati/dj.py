"""
Music: requests, a record of what played, and ranking by count.

The design: take requests and rank by how often something was asked for;
otherwise play radio by genre. And the shape a request should have:

    "Mati play that music station from Saturday night"
    "Sure thing. They played one of your favourite songs, 'Not Like Us'.
     I love that song. Here you go."   <plays station>

⚠️ RANKING IS ARITHMETIC, NOT JUDGEMENT. `COUNT(*)` over what was asked for.
Nothing here asks a model what the person would like; that is the one thing a
3B must not be doing between a person and their music.

⚠️ GENRE RADIO PLAYS IN-PROCESS, through `radio.py`, and opens no
application. The remaining Music.app scripting in this module (replaying a
recorded stream, the library readers below) launches Music only when asked,
and closes what it opened (see "opening an app means being able to close it").

⚠️ PLAYBACK IS AUDIBLE AND CAN HAPPEN WHILE SOMEONE IS ON A CALL. Unasked
playback goes through the same per-kind consent gate as an interruption.
Asked-for playback does not: being told to play something IS the permission.

⚠️ LIBRARY FACTS. Music.app may answer 0 tracks while it is still loading
its library, so a count is only trusted once `_library_ready()` says so.
Subscription tracks are DRM streams, not files, and Music.app's scripting
dictionary has no `download` command, so offline download is not offered.
"""

from __future__ import annotations

import json
import re
import sqlite3
import subprocess
import time
import urllib.parse
import urllib.request

from . import config as C
from .field import notes as N

from . import bridge

class Locked(RuntimeError):
    """The record exists but cannot be read. NOT an empty history."""


TABLE = "played"

# radio-browser: free, keyless, no personal data in the request. Verified
# returning real streams with codec and bitrate.
RADIO_API = "https://de1.api.radio-browser.info/json/stations/search"
UA = C.USER_AGENT
NET_TIMEOUT = 12

# A station under this bitrate is usually not worth hearing.
MIN_BITRATE = 48

# How long a stream gets to actually start before it is called a failure.
START_TIMEOUT = 8.0


# ── talking to Music.app ─────────────────────────────────────────────────────

def _osa(script_lines, *args, timeout=20):
    """Run AppleScript with arguments passed as ARGV, never interpolated.

    ⚠️ Same rule as `notify.py`: a track name or a station name is text from a
    person or an API, and building a script string around it is an injection
    waiting to happen. `on run argv` keeps it data.
    """
    cmd = ["/usr/bin/osascript"]
    for line in script_lines:
        cmd += ["-e", line]
    cmd += ["--", *[str(a) for a in args]]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception:                                        # noqa: BLE001
        return None
    if r.returncode != 0:
        return None
    return (r.stdout or "").strip()


def available():
    """Is there a Music.app to talk to?"""
    import os
    return os.path.isdir("/System/Applications/Music.app") or \
        os.path.isdir("/Applications/Music.app")


def now_playing():
    """(state, track, artist) — or None if Music is not running."""
    out = _osa(["on run argv",
                'tell application "Music"',
                "if it is running then",
                'set s to (player state as string)',
                'set n to ""',
                'set a to ""',
                "try",
                "set n to name of current track",
                "set a to artist of current track",
                "end try",
                'return s & "\\t" & n & "\\t" & a',
                "else",
                'return "stopped\\t\\t"',
                "end if",
                "end tell",
                "end run"])
    if out is None:
        return None
    parts = (out.split("\t") + ["", "", ""])[:3]
    return {"state": parts[0], "track": parts[1], "artist": parts[2]}


def stop(close=True):
    """Stop playback, and by default CLOSE whatever Mati opened to do it.

    ⚠️ `close=True` is the default: opening an application on someone's
    behalf is only acceptable if it can be handed back. Stopping the audio but
    leaving Music sitting in the Dock is taking the screen and not returning
    it.

    ⚠️ QuickTime is stopped too. "Stop the movie" reaches here, and a video is
    the case where a whole window is covering what the person was doing.
    """
    ok = _osa(["on run argv", 'tell application "Music" to stop',
               "end run"]) is not None
    if _running("QuickTime Player"):
        _osa(["on run argv",
              'tell application "QuickTime Player" to '
              'if (count documents) > 0 then stop document 1',
              "end run"], timeout=15)
    if close:
        closed = close_player()
        return {"stopped": ok, "closed": closed}
    return {"stopped": ok, "closed": []}


# ── the library ──────────────────────────────────────────────────────────────

def library_size():
    """How many tracks Music.app holds, or None if it could not be asked.

    ⚠️ None, NOT ZERO. 0 is a real answer, and a Music.app that has not
    finished loading would otherwise be indistinguishable from an empty
    library. A caller must be able to tell "you have no music" from "I could
    not look".
    """
    out = _osa(["on run argv",
                'tell application "Music" to return (count of tracks of library playlist 1) as string',
                "end run"], timeout=40)
    try:
        return int(out)
    except (TypeError, ValueError):
        return None


def find_track(query, limit=5):
    """Tracks whose name or artist contains the words. Deterministic."""
    q = " ".join((query or "").split())
    if not q:
        return []
    out = _osa(["on run argv",
                "set q to item 1 of argv",
                'tell application "Music"',
                "set out to \"\"",
                "set ts to (every track of library playlist 1 whose name contains q or artist contains q)",
                "repeat with i from 1 to (count of ts)",
                "if i > 12 then exit repeat",
                "set t to item i of ts",
                'set out to out & (name of t) & "\\t" & (artist of t) & "\\t" & '
                '(played count of t) & "\\t" & (genre of t) & linefeed',
                "end repeat",
                "return out",
                "end tell",
                "end run"], q, timeout=60)
    rows = []
    for line in (out or "").splitlines():
        # ⚠️ PAD, DO NOT REQUIRE. `_osa` ends with `.strip()`, which eats the
        # TRAILING TAB when the last field is empty — so a track with no genre
        # arrived as 3 fields, failed a `>= 4` check, and became invisible.
        # A downloaded podcast has no genre, so the one thing this was just
        # taught to find was the one thing it could not see.
        bits = (line.split("\t") + ["", "", "", ""])[:4]
        if bits[0].strip():
            try:
                played = int(bits[2])
            except ValueError:
                played = 0
            rows.append({"name": bits[0], "artist": bits[1],
                         "played": played, "genre": bits[3]})
    # ⚠️ Ranked by PLAY COUNT, which is the person's own history — not by
    # a model's opinion of what they would like.
    rows.sort(key=lambda r: -r["played"])
    return rows[:limit]


def favourites(limit=5, at_least=4):
    """The most-played tracks. The library's own count, nothing inferred."""
    out = _osa(["on run argv",
                "set n to (item 1 of argv) as integer",
                'tell application "Music"',
                "set out to \"\"",
                "set ts to (every track of library playlist 1 whose played count > n)",
                "repeat with i from 1 to (count of ts)",
                "if i > 40 then exit repeat",
                "set t to item i of ts",
                'set out to out & (played count of t) & "\\t" & (name of t) & '
                '"\\t" & (artist of t) & linefeed',
                "end repeat",
                "return out",
                "end tell",
                "end run"], at_least, timeout=90)
    rows = []
    for line in (out or "").splitlines():
        bits = (line.split("\t") + ["", "", ""])[:3]
        if bits[1].strip():
            try:
                rows.append({"played": int(bits[0]), "name": bits[1],
                             "artist": bits[2]})
            except ValueError:
                continue
    rows.sort(key=lambda r: -r["played"])
    return rows[:limit]


# ── the one permitted library read ───────────────────────────────────────────

TASTE_KEY = "music_genres"
TASTE_AT = "music_genres_at"


def taste():
    """The genres imported at setup, or None if it was never run.

    ⚠️ None AND [] ARE DIFFERENT. None means the import never ran; [] means it
    ran and the library had nothing to say. The same distinction
    `accountant.spent` exists to protect.
    """
    s = C._settings()
    if s.get(TASTE_AT) is None:
        return None
    return list(s.get(TASTE_KEY) or ())


def import_taste(limit=12, force=False):
    """Read genres from the music library ONCE, at setup. (ok, message).

    The single permitted read of the person's music data, for setup only. It
    is not reachable from `media`, and it refuses to run twice.

    ⚠️ GENRES ONLY. Not titles, not artists, not play counts, not ratings.
    A genre list is enough to suggest a first set of presets, and it is the
    least that will do that: "what you listen to" is a profile, "you have
    jazz in your library" is a dial position.

    ⚠️ IT GOES TO SETTINGS, NOT THE FIELD. A personal fact lives in settings
    because a field cannot forget (the same rule as the weather place).
    `forget_taste()` is why that matters.

    ⚠️ REFUSES TO RUN TWICE unless explicitly forced, and records WHEN rather
    than just a flag: a date is auditable and a boolean is not.
    """
    s = C._settings()
    if s.get(TASTE_AT) and not force:
        return False, (f"I already read your library once, on "
                       f"{str(s[TASTE_AT])[:10]}. I do not do it again.")
    if not available():
        return False, "the music library is not available on this machine"

    # ⚠️ A COLD MUSIC.APP ANSWERS 0, NOT AN ERROR, and this is a ONE-SHOT. The
    # first call may be the one that LAUNCHES Music and read 0 against a real
    # library. Marking the import done on that read would record the genres
    # as "none" permanently and never ask again.
    if not _library_ready():
        return False, ("your music library is not answering yet — Music needs "
                       "to be open. Try again in a moment.")

    out = _osa(["on run argv",
                'tell application "Music"',
                "set out to \"\"",
                "set gs to (genre of every track of library playlist 1)",
                "repeat with g in gs",
                'set out to out & (g as text) & linefeed',
                "end repeat",
                "return out",
                "end tell",
                "end run"], timeout=120)
    counts = {}
    for line in (out or "").splitlines():
        g = " ".join(line.split()).lower()
        if g:
            counts[g] = counts.get(g, 0) + 1
    ranked = [g for g, _n in sorted(counts.items(), key=lambda kv: -kv[1])]
    genres = ranked[:limit]

    # ⚠️ AN EMPTY READ IS NOT MARKED DONE. `_library_ready` above says the
    # library HAS tracks, so no genres means they are untagged — but this is a
    # one-shot, and recording "none" forever on a read that may simply have
    # come too early is the expensive direction to be wrong in. Nothing is
    # saved, so asking again costs a sentence.
    if not genres:
        return False, ("I reached your library but it gave me no genres — the "
                       "tracks may be untagged. Nothing recorded, so you can "
                       "ask me again.")

    C.save_settings(**{TASTE_KEY: genres, TASTE_AT: N._now()})
    return True, (f"Noted {len(genres)} genres from your library, most common "
                  f"first: {', '.join(genres[:5])}. I will not read it again.")


def forget_taste():
    """Delete the imported genres. ⚠️ This is why they are in settings."""
    cur = C._settings()
    removed = [k for k in (TASTE_KEY, TASTE_AT) if k in cur]
    for k in removed:
        cur.pop(k)
    if not removed:
        return []
    import json as _json
    import os as _os
    _os.makedirs(_os.path.dirname(C.SETTINGS_PATH), exist_ok=True)
    tmp = C.SETTINGS_PATH + ".tmp"
    with open(tmp, "w") as f:
        _json.dump(cur, f, indent=2)
    _os.replace(tmp, C.SETTINGS_PATH)
    return removed


# ── internet radio ───────────────────────────────────────────────────────────

def stations(genre, limit=5):
    """Stations for a genre, from radio-browser. Nothing personal is sent."""
    params = urllib.parse.urlencode({"tagList": genre, "limit": str(limit * 3),
                                     "hidebroken": "true", "order": "clickcount",
                                     "reverse": "true"})
    req = urllib.request.Request(f"{RADIO_API}?{params}",
                                 headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=NET_TIMEOUT) as r:
            data = json.loads(r.read().decode())
    except Exception:                                        # noqa: BLE001
        return []
    out = []
    for s in data:
        url = s.get("url_resolved") or s.get("url")
        if not url or not url.startswith(("http://", "https://")):
            continue
        if int(s.get("bitrate") or 0) < MIN_BITRATE:
            continue
        out.append({"name": s.get("name", "").strip(), "url": url,
                    "codec": s.get("codec"), "bitrate": s.get("bitrate"),
                    "country": s.get("country")})
        if len(out) >= limit:
            break
    return out


# ── playlists and reviews ────────────────────────────────────────────────────
#
# Playlists and ratings live in the media host, so they are read here, in the
# module that already reads the library, rather than in a table of their own.
# Music.app's scripting dictionary covers:
#
#     user playlist / library playlist / folder playlist   all scriptable
#     make, duplicate, add                                 all present
#     rating, rating kind, played count, disliked          all readable
#
# ⚠️ `rating kind` IS THE WHOLE DESIGN. Its enumeration is exactly two values:
# `user` and `computed`. A COMPUTED rating is Apple's inference from play
# counts. It is not an opinion, and reporting it as one would put words in the
# person's mouth about their own taste. `reviews()` returns ONLY `user`
# ratings, and `rating_of` always says which kind it found.
#
# ⚠️ NO FRIENDS API EXISTS: the dictionary has nothing for friends, followers
# or social. A SHARED playlist is still a playlist, so that is the reachable
# version.
#
# ⚠️ READING REQUIRES MUSIC.APP TO BE RUNNING, which means these calls can
# launch it. `_note_launch` records that, so `close_player` can put it back:
# the rule about opening applications applies to reads too.

def _library_ready():
    """Is Music.app actually able to answer about the library yet?

    ⚠️ A COLD MUSIC.APP ANSWERS 0, NOT AN ERROR. The call that launches Music
    can read 0 against a real library; asking too early is indistinguishable
    from an empty library. `library_size()` is the readiness signal, and a
    caller that cannot get a real count must say so rather than say "none".
    """
    n = library_size()
    return bool(n)


STARS = 20            # Music stores 0-100; 20 per star


def _stars(rating):
    """0-100 -> 0-5 stars. None when unrated, which is NOT zero stars."""
    try:
        r = int(rating)
    except (TypeError, ValueError):
        return None
    return None if r <= 0 else round(r / STARS, 1)


def playlists(limit=30):
    """The person's own playlists, largest first.

    ⚠️ Returns None when Music could not be asked, [] when there are genuinely
    none. See `_library_ready` — a cold Music.app answers 0.

    ⚠️ `user playlist` only — library, folder and subscription playlists are
    Apple's containers, not lists the person made.
    """
    _note_launch("Music")
    # ⚠️ None means "could not ask", [] means "none exist".
    if not _library_ready():
        return None
    out = _osa(["on run argv",
                'tell application "Music"',
                'set out to ""',
                "set ps to every user playlist",
                "repeat with i from 1 to (count of ps)",
                f"if i > {int(limit)} then exit repeat",
                "set pl to item i of ps",
                'set out to out & (name of pl) & "\t" & '
                '(count of tracks of pl) & "\t" & (time of pl) & linefeed',
                "end repeat",
                "return out",
                "end tell",
                "end run"], timeout=60)
    if out is None:
        return []
    from . import untrusted
    rows = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) < 2 or not parts[0].strip():
            continue
        try:
            n = int(parts[1])
        except ValueError:
            continue
        rows.append({"name": untrusted.clean(parts[0], max_len=90),
                     "tracks": n,
                     "length": parts[2].strip() if len(parts) > 2 else None})
    rows.sort(key=lambda r: -r["tracks"])
    return [r for r in rows if r["name"]]


def playlist(name, limit=40):
    """What is in one playlist. ⚠️ None if there is no playlist by that name —
    which is different from a playlist that is empty."""
    n = " ".join(str(name or "").split())
    if not n:
        return None
    _note_launch("Music")
    out = _osa(["on run argv",
                "set nm to item 1 of argv",
                'tell application "Music"',
                'if not (exists user playlist nm) then return "NOPLAYLIST"',
                'set out to ""',
                "set ts to every track of user playlist nm",
                "repeat with i from 1 to (count of ts)",
                f"if i > {int(limit)} then exit repeat",
                "set t to item i of ts",
                'set out to out & (name of t) & "\t" & (artist of t) & "\t" & '
                '(rating of t) & "\t" & (rating kind of t) & linefeed',
                "end repeat",
                "return out",
                "end tell",
                "end run"], n, timeout=90)
    if out is None or out.strip() == "NOPLAYLIST":
        return None
    from . import untrusted
    rows = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) < 2 or not parts[0].strip():
            continue
        kind = (parts[3].strip().lower() if len(parts) > 3 else "")
        rows.append({
            "title": untrusted.clean(parts[0], max_len=120),
            "artist": untrusted.clean(parts[1], max_len=90) or None,
            "stars": _stars(parts[2] if len(parts) > 2 else None),
            # ⚠️ Carried, never flattened — see the header.
            "rating_kind": "user" if "user" in kind else
                           ("computed" if "comput" in kind else None),
        })
    return [r for r in rows if r["title"]]


def make_playlist(name, titles):
    """Create a playlist from track titles already in the library.

    (created, added, message). ⚠️ Refuses to overwrite an existing playlist —
    a playlist is something a person built, and silently replacing one is
    destroying work.
    """
    n = " ".join(str(name or "").split())
    if not n:
        return False, 0, "a playlist needs a name"
    wanted = [" ".join(str(x).split()) for x in (titles or []) if str(x).strip()]
    if not wanted:
        return False, 0, "no tracks named"
    _note_launch("Music")
    exists = _osa(["on run argv",
                   'tell application "Music" to return (exists user playlist '
                   "(item 1 of argv)) as text",
                   "end run"], n, timeout=30)
    if (exists or "").strip().lower() == "true":
        return False, 0, f"“{n}” already exists — I will not overwrite it"

    lines = ["on run argv",
             "set nm to item 1 of argv",
             'tell application "Music"',
             "set pl to make new user playlist with properties {name:nm}",
             "set added to 0"]
    for i, w in enumerate(wanted, start=2):
        lines += [
            f"set q{i} to item {i} of argv",
            f"set ts{i} to (every track of library playlist 1 whose name is q{i})",
            f"if (count of ts{i}) > 0 then",
            f"duplicate (item 1 of ts{i}) to pl",
            "set added to added + 1",
            "end if"]
    lines += ["return added as text", "end tell", "end run"]
    got = _osa(lines, n, *wanted, timeout=120)
    if got is None:
        return False, 0, "Music would not make the playlist"
    try:
        added = int((got or "0").strip())
    except ValueError:
        added = 0
    # ⚠️ Says how many of the asked-for tracks were actually found. A playlist
    # reported as made when it is half empty is a quiet failure.
    return True, added, (f"made “{n}” with {added} of {len(wanted)} track(s)"
                         if added else
                         f"made “{n}”, but none of those tracks are in the library")


def rating_of(title, artist=None):
    """What the person thinks of a track, if anything. None if not found.

    ⚠️ ALWAYS REPORTS `kind`. A `computed` rating is Apple's inference from
    play counts, not an opinion — see the header.
    """
    q = " ".join(str(title or "").split())
    if not q:
        return None
    _note_launch("Music")
    out = _osa(["on run argv",
                "set q to item 1 of argv",
                'tell application "Music"',
                "set ts to (every track of library playlist 1 whose name contains q)",
                'if (count of ts) is 0 then return "NOTRACK"',
                "set t to item 1 of ts",
                'return (name of t) & "\t" & (artist of t) & "\t" & '
                '(rating of t) & "\t" & (rating kind of t) & "\t" & '
                '(played count of t) & "\t" & (disliked of t)',
                "end tell",
                "end run"], q, timeout=60)
    if out is None or out.strip() == "NOTRACK":
        return None
    parts = out.split("\t")
    if len(parts) < 4:
        return None
    from . import untrusted
    kind = parts[3].strip().lower()
    try:
        plays = int(parts[4]) if len(parts) > 4 else None
    except ValueError:
        plays = None
    return {"title": untrusted.clean(parts[0], max_len=120),
            "artist": untrusted.clean(parts[1], max_len=90) or None,
            "stars": _stars(parts[2]),
            "rating_kind": "user" if "user" in kind else
                           ("computed" if "comput" in kind else None),
            "played_count": plays,
            "disliked": (len(parts) > 5 and "true" in parts[5].strip().lower())}


def rate(title, stars):
    """Record what the person thinks. (ok, message).

    ⚠️ Writes a USER rating, which is the only kind that is an opinion. 0
    clears it rather than meaning "nought stars" — unrated and hated are
    different, and `rating_of` reports them differently.
    """
    q = " ".join(str(title or "").split())
    try:
        s = float(stars)
    except (TypeError, ValueError):
        return False, "a rating is 0 to 5 stars"
    if not q or not (0 <= s <= 5):
        return False, "a rating is 0 to 5 stars"
    _note_launch("Music")
    out = _osa(["on run argv",
                "set q to item 1 of argv",
                "set r to (item 2 of argv) as integer",
                'tell application "Music"',
                "set ts to (every track of library playlist 1 whose name contains q)",
                'if (count of ts) is 0 then return "NOTRACK"',
                "set t to item 1 of ts",
                "set rating of t to r",
                'return (name of t)',
                "end tell",
                "end run"], q, str(int(round(s * STARS))), timeout=60)
    if out is None or out.strip() == "NOTRACK":
        return False, f"“{q}” is not in the library"
    return True, (f"{out.strip()} — {s:g} star(s)" if s else
                  f"{out.strip()} — rating cleared")


def reviews(limit=20):
    """Only what the person actually rated. ⚠️ `user` ratings, never computed.

    ⚠️ None when Music could not be asked; [] when nothing is rated.

    This is the honest answer to "what do I like" — Apple's computed ratings
    would fill this list with inferences and present them as opinions.
    """
    _note_launch("Music")
    if not _library_ready():
        return None                       # ⚠️ see _library_ready
    out = _osa(["on run argv",
                'tell application "Music"',
                'set out to ""',
                "set ts to (every track of library playlist 1 whose rating > 0 "
                "and rating kind is user)",
                "repeat with i from 1 to (count of ts)",
                f"if i > {int(limit)} then exit repeat",
                "set t to item i of ts",
                'set out to out & (name of t) & "\t" & (artist of t) & "\t" & '
                '(rating of t) & "\t" & (played count of t) & linefeed',
                "end repeat",
                "return out",
                "end tell",
                "end run"], timeout=120)
    if out is None:
        return None
    from . import untrusted
    rows = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) < 3 or not parts[0].strip():
            continue
        rows.append({"title": untrusted.clean(parts[0], max_len=120),
                     "artist": untrusted.clean(parts[1], max_len=90) or None,
                     "stars": _stars(parts[2]),
                     "played_count": (int(parts[3]) if len(parts) > 3
                                      and parts[3].strip().isdigit() else None),
                     "rating_kind": "user"})
    rows.sort(key=lambda r: -(r["stars"] or 0))
    return rows


# ── opening an app means being able to close it ──────────────────────────────
#
# A media request may open Music only if Mati can also quit it when it is
# done.
#
# ⚠️ ONLY WHAT MATI OPENED. If Music was already running when Mati arrived,
# the person opened it and it is not Mati's to close: they may have a queue,
# a playlist, a window arrangement. So the state is recorded BEFORE launching
# and only a launch Mati caused is ever undone.
#
# Music.app's MiniPlayer is a UI window, not a scriptable property, so "use
# the small one instead" is not available. Closing afterwards is the honest
# alternative.
_OURS = set()


def _running(app):
    out = _osa(["on run argv",
                'tell application "System Events" to return '
                '(exists application process (item 1 of argv)) as text',
                "end run"], app, timeout=10)
    return (out or "").strip().lower() == "true"


def _note_launch(app):
    """Call BEFORE opening. Records whether this launch is Mati's doing."""
    if not _running(app):
        _OURS.add(app)


def close_player(app=None):
    """Quit what Mati opened. Returns the apps actually closed.

    ⚠️ Never closes an app the person had open already — see `_OURS`.
    """
    closed = []
    for name in ([app] if app else list(_OURS)):
        if name not in _OURS:
            continue
        if _running(name):
            _osa(["on run argv",
                  'tell application (item 1 of argv) to quit',
                  "end run"], name, timeout=20)
            closed.append(name)
        _OURS.discard(name)
    return closed


def opened_by_mati():
    """What Mati currently has open. For `/health` and for saying so."""
    return sorted(_OURS)


def play_video(name, url, db_path=None):
    """Video goes to QuickTime, not Music.

    ⚠️ Music.app WILL play an .mp4 — verified, it reports `playing` — but it
    plays it as though it were a song, in a music window. QuickTime is where a
    person expects a video, and routing by KIND rather than by whichever app
    happens to accept the file is the difference between working and seeming
    to work.
    """
    if not url or not url.startswith(("http://", "https://")):
        return False
    _note_launch("QuickTime Player")
    out = _osa(["on run argv",
                'tell application "QuickTime Player"',
                "set d to open location (item 1 of argv)",
                "activate",
                'return "ok"',
                "end tell",
                "end run"], url, timeout=40)
    if out == "ok":
        remember_play(kind="video", name=name, detail=url, db_path=db_path)
        return True
    return False


def play_station(name, url, db_path=None):
    """Play an internet radio stream through Music.app, and record it."""
    if not url or not url.startswith(("http://", "https://")):
        return False
    _note_launch("Music")
    if _osa(["on run argv",
             'tell application "Music" to open location (item 1 of argv)',
             "end run"], url, timeout=25) is None:
        return False
    # ⚠️ VERIFY THAT IT IS ACTUALLY PLAYING. The command not erroring is not
    # the same as sound coming out: a stream can accept `open location`
    # without complaint and sit at `stopped`. Reporting it as playing would
    # also write it into the history as though it had played, poisoning "what
    # played on Saturday night" with something nobody heard.
    for _ in range(int(START_TIMEOUT / 0.5)):
        time.sleep(0.5)
        np = now_playing()
        if np and np.get("state") == "playing":
            remember_play(kind="station", name=name, detail=url,
                          db_path=db_path)
            return True
    stop()
    return False


def play_track(name, artist=None, db_path=None):
    """Play a track from the library by name. Records what was played."""
    ok = _osa(["on run argv",
               "set n to item 1 of argv",
               'tell application "Music"',
               "set ts to (every track of library playlist 1 whose name is n)",
               "if (count of ts) is 0 then return \"none\"",
               "play (item 1 of ts)",
               "return \"ok\"",
               "end tell",
               "end run"], name, timeout=40)
    if ok == "ok":
        remember_play(kind="track", name=name, detail=artist, db_path=db_path)
        return True
    return False


# ── the record: what played, and when ────────────────────────────────────────

def ensure(db_path=None):
    conn = sqlite3.connect(db_path or bridge.NOTES_DB_PATH, timeout=30)
    try:
        conn.execute(f"""CREATE TABLE IF NOT EXISTS {TABLE} (
                play_id INTEGER PRIMARY KEY,
                at      TEXT NOT NULL,
                kind    TEXT,
                name    TEXT,
                detail  TEXT)""")
        conn.execute(f"CREATE INDEX IF NOT EXISTS {TABLE}_at ON {TABLE}(at)")
        conn.commit()
        return True
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def remember_play(kind, name, detail=None, db_path=None):
    """⚠️ THIS IS WHAT MAKES "that station from Saturday night" ANSWERABLE.

    Without a record of what played and when, that request cannot be resolved
    at all, and neither can ranking by count.

    Name and detail are SEALED: what someone listens to is as personal as a
    note body. The time stays clear so the history can be searched by when.

    ⚠️ RECORDING MUST NEVER BREAK PLAYING. `vault.seal` raises when the vault
    is locked, and this runs AFTER playback has already started, so a failure
    here would turn music that is playing into a turn that reports an error.
    The write is best-effort: losing one line of listening history is a small
    loss, and it is not worth the turn.
    """
    try:
        from . import vault
        ensure(db_path)
        conn = sqlite3.connect(db_path or bridge.NOTES_DB_PATH, timeout=30)
        try:
            conn.execute(f"INSERT INTO {TABLE} (at, kind, name, detail) "
                         f"VALUES (?,?,?,?)",
                         (bridge._now(), kind,
                          vault.seal(name) if name else None,
                          vault.seal(detail) if detail else None))
            conn.commit()
            return True
        except sqlite3.Error:
            return False
        finally:
            conn.close()
    except Exception as e:                              # noqa: BLE001
        # ⚠️ Silent to the caller, visible to a developer.
        import sys as _s
        print(f"  [dj] could not record the play: {e!r}", file=_s.stderr)
        return None


def _local(at):
    import datetime as _dt
    try:
        t = _dt.datetime.fromisoformat(str(at))
    except (TypeError, ValueError):
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=_dt.timezone.utc)
    return t.astimezone()


def history(limit=20, db_path=None):
    from . import vault
    ensure(db_path)
    conn = sqlite3.connect(
        f"file:{db_path or bridge.NOTES_DB_PATH}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            f"SELECT at, kind, name, detail FROM {TABLE} "
            f"ORDER BY play_id DESC LIMIT ?", (int(limit),)).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    out = []
    unreadable = 0
    for at, kind, name, detail in rows:
        try:
            name = vault.open_body(name) if name else None
            detail = vault.open_body(detail) if detail else None
        except Exception:
            # ⚠️ A LOCKED VAULT IS NOT AN EMPTY HISTORY. Skipping silently
            # made "I have no record of what played" the answer when the
            # record was there and simply sealed.
            unreadable += 1
            continue
        t = _local(at)
        out.append({"at": at, "when": t, "kind": kind,
                    "name": name, "detail": detail})
    if unreadable and not out:
        raise Locked(f"{unreadable} plays are sealed and this process holds "
                     f"no key — that is why nothing came back, not because "
                     f"nothing played")
    return out


# ⚠️ LOCAL DAYS, not UTC. "Saturday night" is a local idea, and matching on the
# UTC date would put an evening play on the following day — an easy fault
# anywhere a day is derived from a timestamp.
_WEEKDAYS = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
             "friday": 4, "saturday": 5, "sunday": 6}


def what_played(when_words, db_path=None):
    """Resolve "that station from Saturday night" against the record.

    Returns the most recent play matching the day named, or None. ⚠️ None
    rather than a guess: playing the wrong thing because a phrase was
    ambiguous is worse than asking which one was meant.
    """
    words = (when_words or "").lower()
    day = next((d for d in _WEEKDAYS if d in words), None)
    evening = any(w in words for w in ("night", "evening", "tonight"))

    # ⚠️ RELATIVE DAYS ARE NOT WEEKDAYS, and "yesterday" is the one people
    # actually say. Resolved against the clock to a DATE rather than a weekday
    # number, because "yesterday" on a Monday is Sunday and matching
    # `weekday() == 6` would also match last Sunday.
    # ⚠️ LOCAL AND TIMEZONE-AWARE, because `history` returns aware local times
    # via `_local` — it stores UTC and converts on the way out. A naive
    # `datetime.now()` here raised "can't compare offset-naive and
    # offset-aware" on the first comparison, and a naive `.date()` would have
    # silently answered yesterday's question around midnight.
    import datetime as _dt
    now = _dt.datetime.now().astimezone()
    today = now.date()
    on_date = None
    hours = None            # (from, to) on a 24h clock, when one is named

    if "yesterday" in words or "last night" in words or "last evening" in words:
        on_date = today - _dt.timedelta(days=1)
        if "night" in words or "evening" in words:
            hours = (17, 24)
    elif ("today" in words or "this morning" in words
            or "this afternoon" in words or "this evening" in words
            or "tonight" in words or "earlier" in words):
        on_date = today

    # ⚠️ PART OF DAY, AND IT MUST OUTRANK THE `evening` FLAG. "this morning"
    # contains neither "night" nor "evening", but "this evening" does, and
    # without explicit hours an evening flag would accept any play after 17:00
    # on ANY day for a question that named today.
    if "morning" in words:
        hours = (0, 12)
    elif "afternoon" in words:
        hours = (12, 17)
    elif "evening" in words or "tonight" in words or "night" in words:
        hours = (17, 24)
    # ⚠️ "earlier" IS NOT A PART OF DAY — it means before now, today. Giving it
    # a window would make "play what I had on earlier" miss anything from the
    # last hour, which is the likeliest thing meant.
    if "earlier" in words and "morning" not in words:
        hours = None

    for row in history(limit=200, db_path=db_path):
        t = row.get("when")
        if not t:
            continue
        if on_date is not None and t.date() != on_date:
            continue
        if day is not None and t.weekday() != _WEEKDAYS[day]:
            continue
        if hours is not None and not (hours[0] <= t.hour < hours[1]):
            continue
        if hours is None and evening and t.hour < 17:
            continue
        if "earlier" in words and t > now:
            continue
        return row
    return None


def most_asked(limit=5, db_path=None):
    """⚠️ RANKING BY COUNT, exactly as specified. Arithmetic, no judgement."""
    from . import vault
    ensure(db_path)
    conn = sqlite3.connect(
        f"file:{db_path or bridge.NOTES_DB_PATH}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            f"SELECT name, COUNT(*) n FROM {TABLE} WHERE name IS NOT NULL "
            f"GROUP BY name ORDER BY n DESC LIMIT ?", (int(limit) * 4,)
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    # ⚠️ Sealing is SALTED, so identical names do not group in SQL. They are
    # opened and counted here instead — the same reason `rapport` needed a
    # hash column for its lookup.
    tally = {}
    for name, _n in rows:
        try:
            plain = vault.open_body(name)
        except Exception:
            continue
        if plain:
            tally[plain] = tally.get(plain, 0) + 1
    return sorted(({"name": k, "times": v} for k, v in tally.items()),
                  key=lambda r: -r["times"])[:limit]


def status(db_path=None):
    ensure(db_path)
    conn = sqlite3.connect(
        f"file:{db_path or bridge.NOTES_DB_PATH}?mode=ro", uri=True)
    try:
        n = conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
    except sqlite3.Error:
        n = 0
    finally:
        conn.close()
    size = library_size() if available() else None
    out = {"plays_recorded": n, "music_app": available(),
           "library_tracks": size}
    if size is None:
        out["library_unreadable"] = True   # ⚠️ said, not implied by a 0
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Hearing a request
# ─────────────────────────────────────────────────────────────────────────────

# ⚠️ A LITERAL PATTERN, NOT A CLASSIFIER. This is the shape that scored 19/19
# for the charter trigger and 100% for `selfdoc`, and the one time this project
# asked a model to classify intent the study died at its controls. Playing
# music is also the first IRREVERSIBLE thing Mati does, so the bar for
# deciding "they asked me to" is higher than for deciding what to say.
_ASK = re.compile(
    r"\b(?:please\s+)?(play|put on|start playing|stop|pause)\b", re.I)
_STOP = re.compile(r"\b(stop|pause)\b", re.I)
# "that station from Saturday night", "the one from yesterday"
_RECALL = re.compile(
    r"\b(?:that|the)\s+(?:\w+\s+){0,3}?(?:station|one|thing)\b[^.?!]*", re.I)
_GENRE = re.compile(
    r"\b(jazz|blues|rock|classical|soul|funk|hip[- ]?hop|rap|reggae|"
    r"electronic|ambient|country|folk|metal|pop|latin|salsa|bossa|"
    r"lo[- ]?fi|chill|news|talk)\b", re.I)
_BY = re.compile(r"\bby\s+([A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*){0,3})")


# ⚠️ WORDS THE MUSIC PATH MUST NOT CLAIM. "play me a movie" is not a request
# for a SONG called "a movie". Video belongs to `archive_org`, and
# `hears_request` declines rather than guessing.
_VIDEO = re.compile(r"\b(movie|movies|film|films|video|videos|documentary|"
                    r"documentaries|episode of|show|series|pel[ií]cula|"
                    r"pel[ií]culas|cine)\b", re.I)


def wants_video(text):
    """Is this about moving pictures rather than music? If so, DECLINE."""
    return bool(_VIDEO.search(text or ""))


def hears_request(text):
    """What the person asked for, or None. Never a guess.

    ⚠️ None is the safe answer and the common one. A false positive starts
    audio nobody asked for; a false negative costs one clarifying sentence.
    """
    t = " ".join((text or "").split())
    if not t or not _ASK.search(t):
        return None
    # ⚠️ DECLINE VIDEO OUTRIGHT. A stop is still a stop — "stop the movie"
    # should halt whatever is playing — but nothing else here may answer a
    # request for moving pictures.
    if _VIDEO.search(t) and not _STOP.search(t):
        return None
    if _STOP.search(t) and not re.search(r"\bplay\b", t, re.I):
        return {"kind": "stop"}
    m = _RECALL.search(t)
    if m and re.search(r"\b(from|on|last)\b", t, re.I):
        return {"kind": "recall", "when": t}
    g = _GENRE.search(t)
    if g:
        return {"kind": "station", "genre": g.group(1).lower()}
    b = _BY.search(t)
    if b:
        return {"kind": "track", "what": b.group(1)}
    # "play Not Like Us" — whatever follows the verb, if anything substantial.
    after = re.split(r"\b(?:play|put on|start playing)\b", t, 1, re.I)
    if len(after) > 1:
        what = after[1].strip(" .,!?\"'")
        what = re.sub(r"^(?:me|us|some|the|a|an)\s+", "", what, flags=re.I)
        if len(what.split()) >= 1 and len(what) > 2:
            return {"kind": "track", "what": what}
    return None


def answer_request(req, db_path=None):
    """Do it, and say what was done. Returns a sentence, or None.

    ⚠️ EVERY BRANCH REPORTS WHAT ACTUALLY HAPPENED. A station that would not
    start says so; a track that is not in the library says so. The one thing
    this must never do is claim to be playing something that is not.
    """
    if not req or not available():
        return None
    kind = req.get("kind")

    if kind == "stop":
        # ⚠️ Say that the app was closed, not just that audio stopped. If
        # Mati took over the screen, handing it back is the part the person
        # actually notices.
        from . import radio
        was = radio.stop()                 # the in-process stream, if any
        out = stop()                       # Music/QuickTime, if Mati opened them
        if out.get("closed"):
            return f"Stopped, and closed {', '.join(out['closed'])}."
        return f"Stopped {was['name']}." if was else "Stopped."

    if kind == "recall":
        try:
            row = what_played(req.get("when", ""), db_path=db_path)
        except Locked:
            return ("I have played things, but the record is sealed just now "
                    "and I cannot read it back.")
        if not row:
            return ("I do not have a record of what was playing then — "
                    "tell me the station and I will remember it next time.")
        if row.get("kind") == "station" and row.get("detail"):
            if play_station(row["name"], row["detail"], db_path=db_path):
                return f"{row['name']} again. Here you go."
            return f"That was {row['name']}, but it will not start just now."
        return f"That was {row['name']}."

    if kind == "world":
        # ⚠️ PLAYS THE FILE THE FEED CARRIED, not a search result. The
        # reference in the reply and the thing that plays are the same object.
        from . import world as _world
        found = _world.playable(req.get("what", ""), k=1, db_path=db_path)
        if not found:
            return None
        it = found[0]
        played = (play_video if it.get("kind") == "video" else play_station)(
            it["title"], it["url"], db_path=db_path)
        if played:
            what = "video" if it.get("kind") == "video" else "episode"
            return f"That {what} — {it['title']}, from {it['feed']}. Here you go."
        return f"I have {it['title']} from {it['feed']}, but it will not play."

    if kind == "station":
        # ⚠️ RADIO PLAYS IN-PROCESS, AND NO APPLICATION OPENS. `radio.py`
        # holds an AVPlayer inside the server, so there is no window to take
        # over the screen and nothing to quit afterwards.
        from . import radio
        try:
            found = radio.by_genre(req["genre"], limit=4)
        except radio.Unavailable as e:
            return f"I could not reach the radio directory ({e})."
        for st in found:
            ok, msg = radio.play(st)
            if ok:
                remember_play(kind="station", name=st["name"],
                              detail=st["url"], db_path=db_path)
                return f"{msg}."
        # ⚠️ TWO DIFFERENT FAILURES, AND BOTH OFFER THE NEXT MOVE. Stations
        # that exist and will not start is a true network fact and useless on
        # its own: the person is holding a dead end they cannot act on.
        # Nothing matching the words is Mati having misread them, which
        # `media._ask_instead` asks about.
        if found:
            return (f"I found {len(found)} {req['genre']} stations and none "
                    f"of them would start — shall I look for something near "
                    f"you instead, or try another genre?")
        from . import media
        return media._ask_instead(req["genre"], "station")

    if kind == "track":
        hits = find_track(req["what"], limit=3)
        if not hits:
            return (f"“{req['what']}” is not in the library. I can find a "
                    f"station instead if you name a kind of music.")
        top = hits[0]
        if play_track(top["name"], top.get("artist"), db_path=db_path):
            extra = ""
            if top.get("played", 0) >= 4:
                extra = f" You have played it {top['played']} times."
            return f"{top['name']} — {top['artist']}.{extra} Here you go."
        return f"I found {top['name']} but it would not play."
    return None
