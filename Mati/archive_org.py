"""
Free films that are free to take — the Internet Archive, keyless.

The user asked about yt-dlp, ProxiTok and free movie sites like Tubi. Measuring
those turned up one fact that reframes the whole question:

    ⚠️ FOR OPENLY-LICENSED VIDEO YOU DO NOT NEED A RIPPER AT ALL.
    THE FILE IS ALREADY PUBLISHED.

    archive.org/metadata/night_alarm
      licence  http://creativecommons.org/licenses/publicdomain/
      files    night_alarm.mp4  382.0 MB   direct download
               night_alarm_512kb.mp4  266.5 MB   direct download

An extractor exists precisely to pull video out of sites that CHOSE NOT to
publish a file. Where the file is published, plain HTTP is the whole story —
which is also why `podcast.py` already downloads video podcast enclosures with
no extractor anywhere in this codebase.

:

    archive.org feature_films      28,437 items, keyless, direct files
    PeerTube (framatube.org)       23,831 videos, open API AND an RSS feed
    ProxiTok (3 known instances)   000, 000, 000 — nothing is running
    yt-dlp / ffmpeg                not installed

    YouTube channel RSS            404 on all three channel IDs tried —
                                   and ⚠️ the channel page STILL ADVERTISES
                                   that exact URL in <link rel="alternate">.
                                   Even when it worked it carried titles and
                                   links only, never a file. Shorts have no
                                   feed of their own at all.
    Twitch official API            401 Unauthorized — needs a Client-ID
    twitch.tv/feeds/<user>         200, but it is the app HTML, 0 items
    twitchrss.appspot.com          200, 1 item — ⚠️ a THIRD-PARTY bridge on
                                   someone's App Engine instance, not Twitch.
                                   Worse than Flipp: Flipp is at least
                                   first-party.

⚠️ "ADVERTISED" IS NOT "AVAILABLE", and YouTube is the clean example: its own
page publishes a feed URL that 404s. `world.discover_feed` will happily hand
back an address that does not answer, so a caller must treat a discovered feed
as a claim to check rather than a fact.

⚠️ PEERTUBE NEEDS NO CODE AT ALL. `/feeds/videos.xml` is ordinary RSS and
`world.subscribe` already handles it — 20 items on the first try. Anything
added here for PeerTube would be a second copy of `world.py`.

⚠️ WHAT THIS MODULE WILL NOT DO, and the reason is one line: Tubi is a
legitimate free service whose catalogue is licensed from studios and delivered
under DRM. Taking video out of it means circumventing that protection, which
is a different category from every other source in this project — not a terms
question like Craigslist, but an anti-circumvention one. A public Apache-2.0
release cannot carry that. Nothing here touches a protected stream, and the
three sources above need no circumvention to begin with.
"""

from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request

from . import config as C

SEARCH = "https://archive.org/advancedsearch.php"
META = "https://archive.org/metadata"
DOWNLOAD = "https://archive.org/download"
UA = C.USER_AGENT
NET_TIMEOUT = 30

DIR = os.path.join(C.CACHE_DIR, "films")

# ⚠️ A ceiling, like `podcast.MAX_EPISODE_MB`. A feature film at archive
# quality is hundreds of megabytes and nobody meant to spend that by saying
# "play it". The 512kb derivatives are preferred for exactly this reason.
MAX_FILM_MB = 900

# Collections that are openly licensed by definition. ⚠️ A LITERAL LIST —
# archive.org also hosts material under other terms, and searching everything
# would return items this module has no business downloading.
COLLECTIONS = {
    "films":      "feature_films",
    "movies":     "feature_films",
    "shorts":     "short_films",
    "cartoons":   "animationandcartoons",
    "noir":       "film_noir",
    "serials":    "serials",
    "newsreels":  "newsandpublicaffairs",
    "silent":     "silent_films",
    "scifi":      "SciFi_Horror",
    "westerns":   "westerns_movies",
    # ⚠️ EVERY SHELF HERE WAS FICTION, which is why "play nasa videos" came
    # back with sci-fi B-movies — searching `feature_films` for "nasa" finds
    # *Voyage to the Planet of Prehistoric Women*, not a launch. Measured
    # : `collection:nasa AND mediatype:movies` holds 13,771 items and
    # nothing in this project had ever pointed at it.
    #
    # ⚠️ ALL PUBLIC DOMAIN, which is the standing rule for this module. NASA
    # material is US government work; Prelinger is explicitly released.
    "nasa":       "nasa",
    "space":      "nasa",
    "prelinger":  "prelinger",
    "archival":   "prelinger",
    "educational": "prelinger",
    "computers":  "computerchronicles",
}

# Words that NAME a shelf when they appear in a request. ⚠️ Longest first, so
# "educational" is not matched as "ed" by a shorter key later.
_SHELF_WORDS = tuple(sorted(COLLECTIONS, key=len, reverse=True))

PLAYABLE = (".mp4", ".m4v", ".ogv", ".webm", ".mkv")


class Unavailable(RuntimeError):
    """The archive could not be read. ⚠️ NOT the same as "nothing matched"."""


def shelf_for(text):
    """(shelf KEY, what is left) — which shelf a request names, if any.

    ⚠️ RETURNS THE KEY, NOT THE COLLECTION ID. `search()` resolves keys
    through `COLLECTIONS`, so handing it the id would work for "nasa" (key and
    id are the same word) and silently fail for "films" — `COLLECTIONS["films"]`
    is "feature_films", and `COLLECTIONS.get("feature_films")` is None, which
    `search` reports as "I don't know that shelf".

    ⚠️ THIS IS WHAT "play nasa videos" NEEDED. The request names a SHELF, not
    a title, and searching the default shelf for the shelf's own name is how
    it came back with fiction. "nasa videos" -> ("nasa", ""), which browses
    the collection instead of searching it for the word "nasa".
    """
    import re as _re
    low = " ".join((text or "").lower().split())
    if not low:
        return None, ""
    for word in _SHELF_WORDS:
        if _re.search(rf"\b{_re.escape(word)}\b", low):
            rest = _re.sub(rf"\b{_re.escape(word)}\b", " ", low)
            # ⚠️ The kind words go too. "nasa videos" leaves "videos", and
            # searching the NASA shelf for "videos" is the same mistake one
            # level down.
            rest = _re.sub(r"\b(video|videos|footage|clip|clips|film|films|"
                           r"movie|movies|documentary|documentaries)\b",
                           " ", rest)
            return word, " ".join(rest.split())
    return None, low


def collections():
    return sorted(COLLECTIONS)


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=NET_TIMEOUT) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except Exception as e:                                   # noqa: BLE001
        raise Unavailable(f"could not reach the archive "
                          f"({type(e).__name__})") from e


def search(query=None, collection="films", limit=10):
    """Public-domain films matching a query. [] when nothing matched."""
    coll = COLLECTIONS.get(str(collection or "films").lower())
    if not coll:
        return None                      # ⚠️ None = "I don't know that shelf"
    q = f'collection:{coll} AND mediatype:movies'
    if query:
        # Quoted so a stray colon cannot rewrite the query.
        safe = re.sub(r'["\\]', "", str(query))
        q += f' AND ({safe})'
    # ⚠️ ASK FOR MORE THAN WILL BE RETURNED when there is a query, because the
    # ranking below has to have something to re-order. The archive sorts by
    # DOWNLOADS, which is popularity and not relevance.
    rows = int(limit) * (6 if query else 1)
    url = (f"{SEARCH}?" + urllib.parse.urlencode({
        "q": q, "rows": str(rows), "output": "json",
        "sort[]": "downloads desc"}) +
        "&fl[]=identifier&fl[]=title&fl[]=year&fl[]=downloads&fl[]=licenseurl"
        "&fl[]=subject&fl[]=description")
    data = _get(url)
    docs = ((data or {}).get("response") or {}).get("docs") or []
    out = []
    for d in docs:
        if not d.get("identifier"):
            continue
        from . import untrusted
        out.append({
            "id": d["identifier"],
            # ⚠️ archive.org titles are supplied by uploaders.
            "title": untrusted.clean(d.get("title"), max_len=140) or d["identifier"],
            "year": d.get("year"),
            "downloads": d.get("downloads"),
            "subject": d.get("subject"),
            "licence": d.get("licenseurl") or "public domain collection",
            "page": f"https://archive.org/details/{d['identifier']}",
        })

    # ⚠️ THE ARCHIVE MATCHES LOOSELY AND RANKS BY DOWNLOADS, so the thing
    # actually about the subject lands anywhere. Measured for `physics` on the
    # Prelinger shelf:
    #
    #   1. Automotive Service (1940)   subjects: Occupations, Car culture   ← won
    #   2. Case of Spring Fever        subjects: PHYSICS: Springs
    #   3. Make Mine Freedom           subjects: Cold War, Capitalism
    #   4. A is for Atom (1953)        subjects: Atomic-nuclear, PHYSICS    ← right
    #
    # The user: *"I asked for a video about physics; it played a short fiction
    # film."* Same disease as the radio tag filter — a binary match, then a
    # popularity sort, with relevance thrown away in between.
    if query:
        for r in out:
            r["match"] = _topic_match(r, query)
        # ⚠️ NOTHING THAT MATCHES NOTHING. The archive's loose match returns
        # rows for any term — "cooking" on the Prelinger shelf came back as
        # *Time Savers for House Makers*, *Design for Dreaming* and *Classic
        # Television Commercials*, all scoring 0.00. Handing those over is how
        # a request for physics played a sci-fi B-movie. An empty answer lets
        # the caller say "nothing on the shelves for that", which is true.
        #
        # ⚠️ THE SCORE ONLY READS title AND subject, so an item that matches
        # in its description alone scores 0 and is dropped with the rest. That
        # is a real false negative and the acceptable direction to err: the
        # alternative is playing something unrelated and calling it the answer.
        hits = [r for r in out if r["match"] > 0]
        out = hits
        out.sort(key=lambda r: (-r["match"], -(r.get("downloads") or 0)))
    return out[:int(limit)]


# What an uploader's free-text `rights` must look like to be repeated as a
# licence. ⚠️ Anything else is their prose, not a licence this module relies on.
_LOOKS_LIKE_LICENCE = re.compile(
    r"\b(public domain|creative commons|\bcc[\s-]?(?:0|by)|copyright[- ]free|"
    r"no known copyright|licen[cs]e)\b", re.I)


def _rights(raw):
    """A `rights` string fit to say aloud, or None.

    ⚠️ Tags are stripped rather than escaped — this is going into speech, not
    into HTML, and an escaped anchor read aloud is no better than a raw one.

    ⚠️ AND MOST `rights` TEXT IS NOT A LICENCE. One item came back as
    "**FAIR USE NOTICE**…", which Mati read out mid-sentence as the licence —
    mangled by the clipping AND wrong, because an uploader asserting fair use
    is not a public-domain grant. This module's basis is the COLLECTION, which
    is openly licensed by definition; uploader prose is only repeated when it
    actually reads as a licence, and otherwise the collection basis is stated
    instead. Being vaguer is better than quoting half a notice as a grant.
    """
    if not raw:
        return None
    from . import untrusted
    text = re.sub(r"<[^>]+>", " ", str(raw))
    text = re.sub(r"[*_`#]+", " ", text)          # markdown emphasis
    text = untrusted.clean(text, max_len=120) or ""
    if not text or not _LOOKS_LIKE_LICENCE.search(text):
        return None
    return text


def _topic_match(row, query):
    """How well an item is ABOUT the query, 0..3. Arithmetic, no judgement.

    ⚠️ `subject` IS THE STRONG FIELD and it is the one the archive does not
    sort on. "A is for Atom" carries the subject `Physics` outright; the
    automotive film that outranked it carries none.
    """
    terms = [w for w in re.split(r"\W+", str(query or "").lower()) if len(w) > 2]
    if not terms:
        return 0.0
    title = str(row.get("title") or "").lower()
    subj = row.get("subject")
    subj = " ".join(subj) if isinstance(subj, (list, tuple)) else str(subj or "")
    subj = subj.lower()
    score = 0.0
    for w in terms:
        if w in subj:
            score += 2.0        # the archive's own topical label
        if w in title:
            score += 1.0
    return round(score / len(terms), 3)


def files(identifier):
    """Playable files for one item, smallest first. ⚠️ Size always stated."""
    data = _get(f"{META}/{urllib.parse.quote(str(identifier))}")
    meta = (data or {}).get("metadata") or {}
    out = []
    for f in (data or {}).get("files") or []:
        name = f.get("name") or ""
        if not name.lower().endswith(PLAYABLE):
            continue
        try:
            mb = int(f.get("size") or 0) / 1e6
        except (TypeError, ValueError):
            mb = 0
        out.append({
            "name": name,
            "mb": round(mb, 1) if mb else None,
            "url": f"{DOWNLOAD}/{identifier}/{urllib.parse.quote(name)}",
        })
    # ⚠️ PLAYABLE FIRST, THEN SMALL — not small first. Sorting on size alone
    # picked `dressed_to_kill.ogv` (306 MB) over `dressed_to_kill_512kb.mp4`
    # (311.8 MB) to save five megabytes, and QuickTime cannot open.ogv at
    # all. A file that saves 2% and then will not play is not the smaller
    # option, it is the broken one.
    NATIVE = (".mp4", ".m4v")
    out.sort(key=lambda f: (not f["name"].lower().endswith(NATIVE),
                            f["mb"] is None, f["mb"] or 0))
    return {
        "id": identifier,
        "title": meta.get("title") or identifier,
        "year": meta.get("year"),
        # ⚠️ STATED, NOT ASSUMED. If an item's licence is not what this module
        # expects, the caller can see that rather than being told it is fine.
        # ⚠️ `rights` IS FREE TEXT AN UPLOADER TYPED, and it reaches a spoken
        # reply. Measured: one item's rights field is an `<a href="https://…">`
        # anchor, which Mati read out as raw HTML mid-sentence. `licenseurl`
        # is a URL and safe; `rights` has to be cleaned like any other string
        # from a stranger — the same rule `_clean` applies to station names.
        "licence": meta.get("licenseurl") or _rights(meta.get("rights"))
                   or "public domain collection",
        "files": out,
    }


def best_file(identifier, max_mb=MAX_FILM_MB):
    """The smallest playable file under the ceiling, or (None, why).

    ⚠️ SMALLEST, NOT BEST QUALITY. A 382 MB master and a 266 MB derivative are
    the same film; on a machine with 9 GB of RAM and a metered connection the
    derivative is the right default, and the full file is a deliberate choice.
    """
    info = files(identifier)
    if not info["files"]:
        return None, "that item has no playable video file"
    for f in info["files"]:
        if f["mb"] is None or f["mb"] <= max_mb:
            return dict(f, title=info["title"], licence=info["licence"]), None
    smallest = info["files"][0]
    return None, (f"the smallest copy is {smallest['mb']:.0f} MB, past the "
                  f"{max_mb} MB limit")


def would_cost(f):
    return f"{f['mb']:.0f} MB" if f and f.get("mb") else "an unstated size"


def playable(identifier, max_mb=MAX_FILM_MB):
    """What the APP needs to play a film itself. (dict, None) or (None, why).

    ⚠️ NO APPLICATION IS INVOLVED. The user asked whether video could go through
    AVFoundation like the radio does. It can — AVKit and AVPlayerView are both
    present — but putting a window in the server would turn a headless daemon
    into a GUI app with a Dock icon and an NSApplication run loop competing
    with `serve_forever`. That is the opposite of "keep this as simple as
    possible".

    The app already has a `<video>` element, and archive.org serves these
    directly: measured **HTTP 206 with `content-range`**, so seeking works,
    and a plain `<video>` needs no CORS for playback. So Mati hands over a
    URL and the page plays it — no new framework, no window, nothing to quit.
    """
    f, why = best_file(identifier, max_mb=max_mb)
    if not f:
        return None, why
    return {"url": f["url"], "title": f["title"], "licence": f["licence"],
            "megabytes": f.get("mb"), "kind": "video"}, None


# Shelves a silent-era reel is drawn from. ⚠️ ALL SILENT BY ERA, which is the
# whole point: the user wants this running WHILE the radio does, and a talkie
# would fight it. Muting is still enforced in the app, because archive.org
# prints of silent films often carry an added music track.
#
# ⚠️ `shorts` IS DELIBERATELY NOT HERE. `short_films` sorted by downloads
# returns, in its first four: "These Girls Are Fools (1950), exploitation short
# with nude scenes" and "1940s French stag film: whipping, spanking fetish".
# Measured  while building this. The Chaplin shorts people actually
# want are in `silent` anyway.
REEL_SHELVES = ("cartoons", "silent")

# ⚠️ A SCREEN FOR UNATTENDED CHOICES ONLY, AND THE LINE IS WHO CHOSE.
#
# The user: *"I just don't want to accidentally play something
# inappropriate. The user should be able to watch any available film by name,
# or subject matter, or genre etc — the same for videos and cartoons."*
#
# So the test is not WHAT the material is, it is WHO picked it:
#
#   Mati picked it   a bare "play a movie", "play some shorts", the reel,
#                     a shelf browsed with nothing named -> SCREENED
#   you picked it     a title, a subject, a genre, a shelf plus a search
#                     term -> NOT screened, ever
#
# An assistant that refuses a named request is deciding what an adult may
# watch. One that puts a stag film on the wall because it sorted highest is
# not making a decision at all — it is failing to.
#
# ⚠️ THE WORD LIST IS NOT ENOUGH, which is why there is a second list.
# `silent_films` holds "The Birth of a Nation" (1915) high in its downloads —
# a KKK-glorifying film with a completely innocuous title. No keyword screen
# catches that, so a few things are named. The list is short and specific by
# design: it is not a ratings board, it is a few known cases that would
# otherwise end up on a wall unprompted.
_REEL_REJECT = re.compile(
    r"\b(nude|nudity|naked|stag|fetish|erotic|exploitation|porn|xxx|"
    r"burlesque|striptease|sexploitation|adults? only)\b", re.I)

_REEL_BY_NAME = ("birth of a nation",)

# The late end of "from the era". Generous on purpose — the Fleischer and
# Famous Studios Popeyes run into the late fifties and are exactly the thing.
REEL_MAX_YEAR = 1965

# ⚠️ A SIZE CAP IS HOW "SHORT" GETS ENFORCED, because archive.org does not
# publish runtime in the search row. With only the year filter the reel came
# back as *Phantom of the Opera* (463 MB), *20,000 Leagues* (429 MB) and
# *Caligari* (222 MB) — right era, and features rather than shorts, and a lot
# of bandwidth to leave looping. Measured: Popeyes land at 26–29 MB and the
# Chaplin two-reelers at 88 MB, so this keeps both and drops the features.
REEL_MAX_MB = 120


def _year_of(f):
    """A four-digit year from a search row, or None. ⚠️ archive.org `year` is
    sometimes a full date, sometimes a string, sometimes absent."""
    raw = str(f.get("year") or "")
    m = re.search(r"\b(1[89]\d{2}|20\d{2})\b", raw)
    return int(m.group(1)) if m else None


def ok_unattended(title):
    """May MATI choose this on its own? ⚠️ Not a judgement on the film.

    Named requests never come through here — see the note above.
    """
    low = " ".join((title or "").lower().split())
    if not low:
        return False
    if _REEL_REJECT.search(low):
        return False
    return not any(n in low for n in _REEL_BY_NAME)


# ── Studio pictures ──────────────────────────────────────────────────────────
#
# The user: *"Unless someone is looking for something different,
# 'film' and 'movie' must be something from major film studios."*
#
# ⚠️ THE METADATA CANNOT DO THIS, which is why the list is written out.
# Measured: `publisher` and `creator` are ABSENT on most rows — of the six
# most-downloaded feature films, four have neither. There is nothing to filter
# a studio on.
#
# ⚠️ AND THE DEFAULT SHELF IS WORSE THAN UNHELPFUL. `feature_films` sorted by
# downloads opens with "Sex Madness" (1938) and "Dr. Sex." — the same trouble
# the reel had, reaching the named path too.
#
# ⚠️ WHAT THIS CAN AND CANNOT BE. A current studio film is under copyright and
# is not legitimately available; this module refuses Tubi on exactly that
# ground. What IS available is the set of major-studio pictures whose copyright
# lapsed — a real and well-known group, mostly through renewal failures. So
# "from major film studios" is honoured as far as the law allows, and the
# docstring says so rather than quietly delivering something narrower.
#
# Every identifier here was verified present on archive.org  with its
# download count; none is a guess.
STUDIO_PICKS = (
    ("his_girl_friday", "His Girl Friday", 1940, "Columbia"),
    ("meet_john_doe", "Meet John Doe", 1941, "Warner Bros."),
    ("TheStranger_0", "The Stranger", 1946, "RKO"),
    ("MyManGodfrey1936", "My Man Godfrey", 1936, "Universal"),
    ("royal_wedding", "Royal Wedding", 1951, "MGM"),
    ("till_the_clouds_roll_by", "Till the Clouds Roll By", 1946, "MGM"),
    ("beneath_the_12-mile_reef", "Beneath the 12-Mile Reef", 1953,
     "20th Century Fox"),
    ("NothingSacred", "Nothing Sacred", 1937, "Selznick / United Artists"),
    ("Charade_1953", "Charade", 1963, "Universal"),
    ("The_General_Buster_Keaton", "The General", 1926, "United Artists"),
    ("doa_ipod", "D.O.A.", 1950, "United Artists"),
    ("Night.Of.The.Living.Dead_1080p", "Night of the Living Dead", 1968,
     "Continental"),
)


def studio_pick(limit=1, max_mb=MAX_FILM_MB, rng=None):
    """Playable major-studio pictures in the public domain. [] if none load.

    ⚠️ SHUFFLED, so "play a movie" twice is not the same film twice. Seeded
    from `random` by default; a caller passing `rng` gets a reproducible order,
    which is what the tests use.
    """
    import random as _random
    picks = list(STUDIO_PICKS)
    (rng or _random).shuffle(picks)
    out = []
    for ident, title, year, studio in picks:
        if len(out) >= limit:
            break
        try:
            got, _why = playable(ident, max_mb=max_mb)
        except Exception:                                    # noqa: BLE001
            continue
        if not got:
            continue
        out.append({"url": got["url"],
                    "title": f"{title} ({year})",
                    "studio": studio,
                    "licence": got["licence"],
                    "megabytes": got.get("megabytes"),
                    "kind": "video"})
    return out


def reel(limit=5, max_mb=REEL_MAX_MB):
    """A short loop of era shorts and cartoons. [] if nothing is playable.

    the user: *"there should be a short playlist of shorts and
    cartoons from the era that play in a loop. This way it can play at the
    same time as listening to music."*

    ⚠️ EVERY ITEM IS CHECKED FOR A PLAYABLE FILE BEFORE IT IS OFFERED. This is
    not caution for its own sake — the bare "play some videos" request
    returned a Chaplin short whose only formats were unsupported, and said so
    after the fact. A reel that stops on its third item is worse than a
    shorter reel.

    ⚠️ COSTS ONE REQUEST PER CANDIDATE, so it is built once and handed over
    whole rather than resolved as it plays.
    """
    # ⚠️ ROUND-ROBIN, NOT SHELF BY SHELF. Draining `cartoons` first filled
    # every slot before `silent` was looked at, so a reel asked for as "shorts
    # and cartoons" came back all cartoons.
    per_shelf, seen = {}, set()
    for shelf in REEL_SHELVES:
        try:
            per_shelf[shelf] = list(search(None, collection=shelf,
                                           limit=limit * 4) or [])
        except Exception:                                    # noqa: BLE001
            per_shelf[shelf] = []

    candidates = []
    for i in range(max((len(v) for v in per_shelf.values()), default=0)):
        for shelf in REEL_SHELVES:
            rows = per_shelf.get(shelf) or []
            if i < len(rows):
                candidates.append(rows[i])

    out = []
    for f in candidates:
        if len(out) >= limit:
            break
        if f["id"] in seen:
            continue
        seen.add(f["id"])
        # ⚠️ SCREENED BEFORE THE NETWORK CALL, so a rejected item costs
        # nothing and cannot reach the reel by timing.
        if not ok_unattended(describe(f)):
            continue
        # ⚠️ "FROM THE ERA", WHICH IS WHAT WAS ASKED FOR. The cartoons shelf
        # is topped by `Elephants Dream` (2006) and `Sintel` (2010) — Blender
        # demos, openly licensed and three generations wrong. A missing year
        # is allowed through: plenty of genuine prints have none, and
        # rejecting on absence would thin the reel to nothing.
        year = _year_of(f)
        if year is not None and year > REEL_MAX_YEAR:
            continue
        try:
            got, _why = playable(f["id"], max_mb=max_mb)
        except Exception:                                    # noqa: BLE001
            continue
        if not got:
            continue
        out.append({"url": got["url"],
                    "title": describe(f),
                    "licence": got["licence"],
                    "megabytes": got.get("megabytes"),
                    "kind": "video"})
    return out


def play(identifier, max_mb=MAX_FILM_MB):
    """Play via QuickTime. ⚠️ FOR THE TERMINAL ONLY — it has no browser.

    The app uses `playable()` instead and plays it in the page. This path
    opens an application, so `dj` records the launch and can close it again.
    """
    from . import dj
    f, why = best_file(identifier, max_mb=max_mb)
    if not f:
        return False, why
    ok = dj.play_video(f["title"], f["url"])
    return (bool(ok),
            f"playing {f['title']} ({would_cost(f)}, {f['licence']})"
            if ok else "that would not open")


def keep(identifier, max_mb=MAX_FILM_MB, verbose=False):
    """Download one film. (path, None) or (None, why).

    ⚠️ SIZE IS STATED AND ENFORCED DURING THE READ, not only from the header —
    the same rule `podcast.download` already follows, for the same reason: a
    server that understates Content-Length must not be able to fill the disk.
    """
    f, why = best_file(identifier, max_mb=max_mb)
    if not f:
        return None, why
    os.makedirs(DIR, exist_ok=True)
    safe = re.sub(r"[^\w \-.]", "", f["name"])[:80] or "film.mp4"
    path = os.path.join(DIR, safe)
    if os.path.exists(path):
        return path, "already kept"
    tmp = path + ".part"
    try:
        req = urllib.request.Request(f["url"], headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=120) as r:
            written = 0
            with open(tmp, "wb") as fh:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written / 1e6 > max_mb:
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
        print(f"  [film] kept {safe} ({os.path.getsize(path)/1e6:.1f} MB)")
    return path, None


def kept():
    if not os.path.isdir(DIR):
        return []
    rows = []
    for n in sorted(os.listdir(DIR)):
        p = os.path.join(DIR, n)
        if n.endswith(".part") or not os.path.isfile(p):
            continue
        rows.append({"name": n, "path": p,
                     "mb": round(os.path.getsize(p) / 1e6, 1)})
    return rows


def describe(item):
    s = str(item.get("title"))[:60]
    if item.get("year"):
        s += f" ({item['year']})"
    return s


# ⚠️ THE NATURAL PHRASINGS WERE MISSING, and that is how a film request ended
# up in Apple Music: "play me a movie" matched none of these, so `dj` — which
# matches on "play" — took it and searched for a song by that name.
_ASKS = ("free movie", "free film", "public domain film", "old movie",
         "watch a film", "watch a movie", "classic film", "silent film",
         "play a movie", "play me a movie", "play a film", "play me a film",
         "put on a movie", "put on a film", "show me a movie", "a movie",
         "a film", "some movie", "movie night", "watch something",
         "película gratis", "pelicula gratis", "una película", "una pelicula",
         "ver una película", "ver una pelicula")


def asks_about_films(text_in):
    t = (text_in or "").lower()
    return any(p in t for p in _ASKS)


def status():
    f = kept()
    return {"kept": len(f), "megabytes": round(sum(x["mb"] for x in f), 1),
            "folder": DIR, "collections": collections(),
            "note": "openly-licensed only; the files are published, so no "
                    "extractor is involved"}
