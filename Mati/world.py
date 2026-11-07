"""
Reading the world — RSS, including the audio and video in it.

The user: *"I want to make sure that Mati's responses include audio and video
references from its RSS feeds, in addition to servicing."*

So a world item is not only text. When a feed carries an `<enclosure>` or a
`<media:content>`, that file travels WITH the note — and a reply that draws on
the item can say "there is an episode about that" and then play it, rather than
paraphrasing something the person cannot reach.

⚠️ NOTHING IS CONFIGURED OUT OF THE BOX, and that is what keeps the README
claim true. `Mati/README.md` says Mati makes no outbound network connections;
with no feed configured it still makes none. Adding one is the person's act,
and the user's wording for what it becomes is: *"minimal RSS connections with no
personal information is ever used."* Both halves are enforced here — a bounded
number of fetches, and a request that carries a feed URL and nothing else.

⚠️ CONSCIOUS-LAYER MATERIAL, NOT THE FIELD'S. The user: *"The RSS feed is not the
same as the field prediction. This one is verbatim and a lookup."* So world
items are archived from birth, lexical-only, never imprinted. θ₀ is not
deformed by the news, and the volume bound is affordable
precisely because the field is not involved.

⚠️ AND IT IS THE WEAKEST THING MATI HOLDS. Weaker than a book the person
chose, because nobody vetted it. The provenance label says where it came from
and when, every time.
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

SOURCE = "world"
UA = C.USER_AGENT
NET_TIMEOUT = 20

# ⚠️ THE BOUND IS THE DESIGN, not a tuning knob. Measured  the store
# grows ~104 notes/day, mostly one book read once. Two active news feeds
# unbounded is ~25,000 notes a year and the record of a person becomes a stale
# news archive with a diary attached.
MAX_PER_DAY = 40
EXPIRE_DAYS = 30

# Where subscriptions live. A file the person owns, not a hardcoded list — a
# suggestion list is still an editorial position, so there is no default.
FEEDS_PATH = os.path.join(C.CACHE_DIR, "feeds.json")

_MEDIA_NS = "{http://search.yahoo.com/mrss/}"


class FeedUnavailable(RuntimeError):
    """The feed could not be read. ⚠️ NOT the same as a feed with no items."""


# ── subscriptions ────────────────────────────────────────────────────────────

def feeds():
    """The person's feeds. Empty unless they added one."""
    try:
        with open(FEEDS_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:                                        # noqa: BLE001
        return []
    return [f for f in data if isinstance(f, dict) and f.get("url")]


def is_public_url(url):
    """(ok, why) — is this an address out in the world? ⚠️ THE SSRF GUARD.

    ⚠️ THIS LIVES ON ITS OWN BECAUSE `subscribe` IS NO LONGER THE ONLY DOOR.
    It used to be, and the check was inlined there. `discover_feed` then
    started FETCHING a pasted URL before `subscribe` ever saw it, so
    `http://127.0.0.1:8756/` was retrieved by Mati from itself — the guard
    was still in the codebase and no longer on the path. Any new function
    that fetches a URL from outside calls this FIRST.
    """
    u = urllib.parse.urlparse(url or "")
    if u.scheme not in ("http", "https") or not u.hostname:
        return False, "that has to be an http or https address"
    host = u.hostname.lower().strip("[]")
    if (host in ("localhost", "127.0.0.1", "0.0.0.0", "::1", "::")
            or host.endswith(".local") or host.endswith(".internal")
            or host.startswith(("127.", "10.", "192.168.", "169.254.",
                                "fc", "fd", "fe80:"))
            or re.match(r"^172\.(1[6-9]|2\d|3[01])\.", host)):
        return False, "that address is on this machine, not the world"
    return True, "ok"


def subscribe(url, name=None):
    """Add a feed. ⚠️ Refuses anything that is not a public http(s) URL.

    A feed URL pointing at `127.0.0.1:8756` would make Mati fetch itself, and
    a `file:` URL would make it read the disk. This is the one place in Mati
    where an arbitrary URL arrives from outside, so the allow-list is here —
    and it is the exact INVERSE of the local-services rule, where only
    localhost is permitted.
    """
    ok, why = is_public_url(url)
    if not ok:
        return False, why
    u = urllib.parse.urlparse(url)
    host = u.hostname.lower()
    have = feeds()
    if any(f["url"] == url for f in have):
        return True, "already subscribed"
    have.append({"url": url, "name": name or host, "added": bridge._now()})
    os.makedirs(os.path.dirname(FEEDS_PATH), exist_ok=True)
    with open(FEEDS_PATH, "w", encoding="utf-8") as fh:
        json.dump(have, fh, indent=1)
    return True, f"subscribed to {name or host}"


# ── turning a page someone pasted into a feed ────────────────────────────────

# ⚠️ AUTODISCOVERY IS THE STANDARD, AND IT IS WHAT MAKES THIS GENERAL. A page
# that publishes a feed says so in its own <head>:
#     <link rel="alternate" type="application/rss+xml" href="...">
# Measured  — every shopping page tried carried one:
#     allbirds.com/collections/mens-shoes  -> /collections/mens.atom
#     slickdeals.net/deals/                -> /newsearch.php?mode=frontpage…
#     dealnews.com/                        -> /?rss=1
_FEED_LINK = re.compile(
    r"""<link\b[^>]*?(?=[^>]*\brel\s*=\s*["']?alternate)"""
    r"""[^>]*?type\s*=\s*["']application/(?:rss|atom)\+xml["'][^>]*>""",
    re.I | re.S)
_HREF = re.compile(r"""href\s*=\s*["']([^"']+)["']""", re.I)

# ⚠️ Measured: allbirds' and slickdeals' <head> both exceed 200 KB. This is the
# hard stop for a page that never closes its head at all.
HEAD_CEILING = 2_000_000

# Platform patterns, tried only when a page advertises nothing. These are
# published conventions, not guesses at private endpoints.
_PATTERNS = (
    # Shopify: every collection is an Atom feed with a structured <price>.
    (re.compile(r"^(https?://[^/]+/collections/[^/?#]+)", re.I), "{0}.atom"),
    # WordPress / WooCommerce search.
    (re.compile(r"^(https?://[^/]+/)\?s=", re.I), None),
)


def discover_feed(page_url, timeout=NET_TIMEOUT):
    """Find the feed a page publishes. Returns (feed_url, how) or (None, why).

    ⚠️ THIS ONLY EVER READS WHAT A SITE ADVERTISES ABOUT ITSELF. It does not
    parse listings, and it does not invent endpoints — if a page publishes no
    feed, the answer is that it publishes no feed. Sites that block automated
    access stay blocked; nothing here works around anything.

    ⚠️ `<head>` ONLY, and capped. The tag is always in the head, and reading a
    whole shopping page to find it would pull megabytes.
    """
    ok, why = is_public_url(page_url)
    if not ok:
        return None, why

    req = urllib.request.Request(page_url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            # ⚠️ STREAMED UNTIL `</head>`, NOT A FLAT SLICE. A 200 KB read
            # missed the feed link on BOTH pages first tried: modern shopping
            # pages inline so much JSON and CSS that the <head> alone runs
            # past a quarter-megabyte, and the tag sat beyond the cut. Reading
            # to the real end of the head fixes it; the ceiling keeps a page
            # with no </head> at all from being read forever.
            chunks, got = [], 0
            while got < HEAD_CEILING:
                block = r.read(65536)
                if not block:
                    break
                chunks.append(block)
                got += len(block)
                if b"</head>" in block or b"</HEAD>" in block:
                    break
            head = b"".join(chunks).decode("utf-8", "replace")
            final = r.geturl()
    except Exception as e:                                   # noqa: BLE001
        return None, f"that page could not be read ({type(e).__name__})"

    # ⚠️ A redirect can leave the allow-list — check where we ACTUALLY landed.
    ok, why = is_public_url(final)
    if not ok:
        return None, why

    # Some sites ARE the feed already.
    if head.lstrip()[:400].lower().startswith(("<?xml", "<rss", "<feed")):
        return page_url, "that page is already a feed"

    cut = head.split("</head>", 1)[0]
    for tag in _FEED_LINK.findall(cut):
        m = _HREF.search(tag)
        if not m:
            continue
        href = m.group(1).replace("&amp;", "&").strip()
        if href:
            return urllib.parse.urljoin(final, href), "the page publishes one"

    for pat, tmpl in _PATTERNS:
        m = pat.match(final)
        if not m:
            continue
        if tmpl:
            return tmpl.format(m.group(1)), "a known feed address for that site"
        joined = final + ("&" if "?" in final else "?") + "feed=rss2"
        return joined, "a known feed address for that site"

    return None, "that page does not publish a feed"


_OG = re.compile(
    r"""<meta\b[^>]*?(?:property|name)\s*=\s*["']og:(\w+)["'][^>]*?"""
    r"""content\s*=\s*["']([^"']{0,400})["']""", re.I | re.S)
_OG_REV = re.compile(
    r"""<meta\b[^>]*?content\s*=\s*["']([^"']{0,400})["'][^>]*?"""
    r"""(?:property|name)\s*=\s*["']og:(\w+)["']""", re.I | re.S)


def preview(page_url, timeout=NET_TIMEOUT):
    """What a link is, from its own OpenGraph tags. (dict, None) or (None, why).

    The ordinary link-preview mechanism — one fetch of a URL the person was
    already handed, exactly what Messages and Slack do. It is not scraping a
    search page and it touches nothing that blocks.

    Measured  against a real listing:
        og:title        Motley Crue: From the Beginning Vinyl: $18
        og:description  It's $17 under list price.
        og:image        https://d.dlnws.com/…product-trimmed.png
        og:site_name    dealnews

    ⚠️ ON DEMAND, NEVER AUTOMATICALLY, and this is a rule about the caller
    rather than about this function. Fetching a listing tells that retailer
    somebody is interested in that item; doing it for every item in every
    feed cycle would hand a shopping profile to every retailer in the list,
    for things nobody even looked at. Preview the one link that was asked
    about.

    ⚠️ THE SAME SSRF GUARD as `discover_feed`, before the fetch and again
    after redirects. A pasted URL is outside input.
    """
    ok, why = is_public_url(page_url)
    if not ok:
        return None, why
    req = urllib.request.Request(page_url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            # ⚠️ Capped and stopped at </head>. The measurement that produced
            # this function read 200 KB to find four tags; `discover_feed`
            # already learned that a flat slice is the wrong shape.
            chunks, got = [], 0
            while got < HEAD_CEILING:
                block = r.read(65536)
                if not block:
                    break
                chunks.append(block)
                got += len(block)
                if b"</head>" in block or b"</HEAD>" in block:
                    break
            head = b"".join(chunks).decode("utf-8", "replace")
            final = r.geturl()
    except Exception as e:                                   # noqa: BLE001
        return None, f"that page could not be read ({type(e).__name__})"

    ok, why = is_public_url(final)
    if not ok:
        return None, why

    cut = head.split("</head>", 1)[0]
    tags = {}
    for k, v in _OG.findall(cut):
        tags.setdefault(k.lower(), v.strip())
    for v, k in _OG_REV.findall(cut):
        tags.setdefault(k.lower(), v.strip())
    if not tags:
        # ⚠️ A page with no OpenGraph is not a broken page.
        return None, "that page says nothing about itself"
    import html as _html
    from . import untrusted
    # ⚠️ OpenGraph tags are written by whoever owns the page.
    out = {k: untrusted.clean(_html.unescape(v), max_len=400)
           for k, v in tags.items()
           if k in ("title", "description", "site_name", "type")}
    if "image" in tags:
        out["image"] = _html.unescape(tags["image"])[:500]
    out["url"] = final
    return out, None


# ── reading a feed ───────────────────────────────────────────────────────────

def _media_of(item):
    """The playable file in this item, if there is one.

    ⚠️ TWO PLACES TO LOOK. Podcasts use `<enclosure>`; YouTube and many video
    feeds use `<media:content>`. Checking only the first would silently make
    every video feed text-only, which is exactly what the user asked to avoid.
    """
    enc = item.find("enclosure")
    if enc is not None and enc.get("url"):
        return enc.get("url"), (enc.get("type") or "").lower()
    mc = item.find(_MEDIA_NS + "content")
    if mc is not None and mc.get("url"):
        return mc.get("url"), (mc.get("type") or mc.get("medium") or "").lower()
    return None, ""


def _kind(ctype, url=""):
    t = (ctype or "").lower()
    if t.startswith("audio") or t == "audio":
        return "audio"
    if t.startswith("video") or t == "video":
        return "video"
    if url.lower().endswith((".mp3", ".m4a", ".aac", ".ogg")):
        return "audio"
    if url.lower().endswith((".mp4", ".m4v", ".webm")):
        return "video"
    return ""


def _text(item, *tags):
    for t in tags:
        v = item.findtext(t)
        if v and v.strip():
            return " ".join(re.sub(r"<[^>]+>", " ", v).split())
    return ""


def read(feed_url, limit=20):
    """Items from one feed. Raises rather than returning [] on failure."""
    req = urllib.request.Request(feed_url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=NET_TIMEOUT) as r:
            raw = r.read()
    except Exception as e:                                   # noqa: BLE001
        raise FeedUnavailable(f"could not reach {feed_url} "
                              f"({type(e).__name__})") from e
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as e:
        raise FeedUnavailable(f"{feed_url} is not readable XML") from e

    feed_name = _text(root.find("channel") or root, "title") or feed_url
    out = []
    # RSS <item> and Atom <entry>, both.
    nodes = list(root.iter("item")) or list(
        root.iter("{http://www.w3.org/2005/Atom}entry"))
    for item in nodes:
        title = _text(item, "title", "{http://www.w3.org/2005/Atom}title")
        if not title:
            continue
        summary = _text(item, "description", "summary",
                        "{http://www.w3.org/2005/Atom}summary",
                        _MEDIA_NS + "description")
        link = _text(item, "link", "guid") or ""
        murl, mtype = _media_of(item)
        out.append({
            "title": title,
            "summary": summary[:600],
            "link": link,
            "feed": feed_name,
            "media": murl,
            "media_kind": _kind(mtype, murl or ""),
            "published": _text(item, "pubDate", "published", "updated"),
        })
        if len(out) >= limit:
            break
    return out


# ── keeping what matters to this person ──────────────────────────────────────

# Abbreviations a person writes in a profile, and the words a headline uses
# instead. ⚠️ Both directions are needed: the short form for "AI model" and
# the long form for "artificial intelligence".
_EXPANSIONS = {
    "ai": ("artificial intelligence", "machine learning", "neural"),
    "ml": ("machine learning",),
    "cs": ("computer science",),
    "vr": ("virtual reality",),
}


def _interests(db_path=None):
    """The profile's own words. ⚠️ Never a hardcoded topic list."""
    try:
        from . import vault
        rows = bridge.profile_notes(db_path=db_path) or []
    except Exception:                                        # noqa: BLE001
        return set()
    # ⚠️ FROM THE INTERESTS SENTENCE ONLY, NOT THE WHOLE PROFILE. Scraping
    # every profile fact produced `asked`, `called`, `interested` — function
    # words from "They asked me to talk to them like this" — and a filter
    # containing `asked` matches most of the news. The profile states the
    # interests in one sentence; that sentence is the filter.
    words = set()
    for fact in rows:
        m = re.search(r"interested in\s+(.+)", fact or "", re.I)
        if not m:
            continue
        for part in re.split(r"[,;]| and ", m.group(1)):
            part = part.strip(" .").lower()
            # ⚠️ TWO CHARACTERS IS NOW SAFE because `_matches` uses word
            # boundaries. The old floor of three silently discarded "AI".
            if len(part) >= 2:
                # An abbreviation cannot match the words it stands for, so the
                # expansion goes in beside it — "AI" alone would never have
                # matched "artificial intelligence" either.
                for extra in _EXPANSIONS.get(part, ()):
                    words.add(extra)
                words.add(part)
                # multi-word topics also match on their rarest word
                for w in re.findall(r"[a-z]{5,}", part):
                    words.add(w)
    return words


# ⚠️ WORD BOUNDARIES, NOT SUBSTRINGS, AND THAT IS WHAT MAKES SHORT INTERESTS
# SAFE. `any(w in hay...)` meant "ai" would have matched said, rain, chair,
# available and again — so `_interests` dropped anything under three
# characters to protect against it. The guard was on the wrong side: it made
# **AI**, a stated interest in the user's own profile, permanently unmatchable.
# : "New AI model beats humans at protein folding" did not
# match, and neither did "Artificial intelligence advances".
#
# A trailing `\w*` keeps the stem behaviour the substring test gave for free —
# "genetic" still finds "genetics" — while "ai" now only matches the word.
def _boundary(word):
    return re.compile(r"\b" + re.escape(word) + r"\w*", re.I)


_MATCH_CACHE = {}


def _matches(item, interests):
    if not interests:
        return False
    hay = f"{item['title']} {item['summary']}"
    for w in interests:
        pat = _MATCH_CACHE.get(w)
        if pat is None:
            pat = _MATCH_CACHE[w] = _boundary(w)
        if pat.search(hay):
            return True
    return False


def ensure_columns(db_path=None):
    conn = sqlite3.connect(db_path or bridge.NOTES_DB_PATH, timeout=30)
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(notes)")}
        if "media" not in cols:
            # The playable file. SEALED: it is part of what the person holds.
            conn.execute("ALTER TABLE notes ADD COLUMN media TEXT")
        if "media_kind" not in cols:
            # ⚠️ CLEAR, deliberately. "Do you have any video about X" has to be
            # answerable without the key, the same rule as `notes.source`.
            conn.execute("ALTER TABLE notes ADD COLUMN media_kind TEXT")
        conn.commit()
        return True
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def ingest(items, db_path=None, verbose=False):
    """Keep the items that touch this person's interests. Returns how many.

    ⚠️ ARCHIVED FROM BIRTH AND UNIMPRINTED. World material must not deform θ₀,
    take a working-set slot, or compete with things that happened with the
    person. Lexical search reads it regardless.
    """
    from . import vault
    ensure_columns(db_path)
    interests = _interests(db_path)
    now = bridge._now()
    stamp = time.time()
    kept = 0
    conn = sqlite3.connect(db_path or bridge.NOTES_DB_PATH, timeout=30)
    try:
        bridge._ensure_schema(conn)
        for it in items:
            if kept >= MAX_PER_DAY:
                break
            if not _matches(it, interests):
                continue
            src = f"{SOURCE}:{it['feed']}"
            # ⚠️ CLEANED BEFORE IT BECOMES A MEMORY. A feed title used to go
            # in verbatim, so a hostile item was not a one-turn attack — it
            # was stored and could resurface unbidden for weeks. See
            # `untrusted.py` for the three breaks that were demonstrated.
            from . import untrusted
            body = untrusted.clean(it["title"])
            if it["summary"]:
                body += " — " + untrusted.clean(it["summary"], max_len=400)
            if not body.strip():
                continue          # nothing but structure; drop the item
            # ⚠️ Keyed on the link so a re-read does not duplicate. Feeds
            # repeat the same item for days.
            dup = conn.execute(
                "SELECT 1 FROM notes WHERE source=? AND verbatim IS NOT NULL "
                "AND media IS ? LIMIT 1", (src, vault.seal(it["media"])
                                           if it["media"] else None)).fetchone()
            if dup and it["media"]:
                continue
            conn.execute(
                "INSERT INTO notes (body, source, wall_created, verbatim, "
                "archived, triplet_w0, triplet_w1, triplet_w2, media, "
                "media_kind) VALUES (?,?,?,?,?,0.0,0.0,0.0,?,?)",
                (vault.seal(body), src, now, vault.seal(body), stamp,
                 vault.seal(it["media"]) if it["media"] else None,
                 it["media_kind"] or None))
            kept += 1
        conn.commit()
    except sqlite3.Error as e:
        if verbose:
            print(f"  [world] {e}")
        return 0
    finally:
        conn.close()
    if verbose:
        print(f"  [world] kept {kept} of {len(items)}")
    return kept


def fetch_all(db_path=None, verbose=False):
    """Read every subscribed feed. For sleep and idle, never a turn."""
    total, failed = 0, []
    for f in feeds():
        try:
            total += ingest(read(f["url"]), db_path=db_path, verbose=verbose)
        except FeedUnavailable as e:
            failed.append(str(e))
    expire(db_path=db_path)
    return {"kept": total, "failed": failed, "feeds": len(feeds())}


def expire(db_path=None):
    """⚠️ THE ONLY MEMORY KIND MATI DELETES, and it needs the exception.

    Corrections annotate and never erase, so that an error cannot resurface
    unmarked. A news item is different: it is true AS OF a date, it is not the
    person's and not Mati's, and "today X happened" is simply wrong six months
    later. A conversation ABOUT an item is a lived episode and is untouched.
    """
    cutoff = time.time() - EXPIRE_DAYS * 86400
    conn = sqlite3.connect(db_path or bridge.NOTES_DB_PATH, timeout=30)
    try:
        n = conn.execute(
            "DELETE FROM notes WHERE source LIKE ? AND archived < ?",
            (f"{SOURCE}:%", cutoff)).rowcount
        conn.commit()
        return n
    except sqlite3.Error:
        return 0
    finally:
        conn.close()


# ── what a reply can say about it ────────────────────────────────────────────

def playable(query, k=3, db_path=None):
    """World items matching the query that Mati can actually PLAY.

    ⚠️ This is what makes a reference real rather than decorative. A reply
    saying "there is an episode about that" is only worth saying if the
    episode can then be played, and that needs the file, not a summary of it.
    """
    from . import notes as _notes, vault
    try:
        hits = _notes.search(query, k=k * 4, db_path=db_path)
    except Exception:                                        # noqa: BLE001
        return []
    out = []
    conn = sqlite3.connect(
        f"file:{db_path or bridge.NOTES_DB_PATH}?mode=ro", uri=True)
    try:
        for h in hits:
            if not str(h.get("source", "")).startswith(SOURCE + ":"):
                continue
            row = conn.execute(
                "SELECT media, media_kind FROM notes WHERE note_id=?",
                (h["note_id"],)).fetchone()
            if not row or not row[0]:
                continue
            try:
                url = vault.open_body(row[0])
            except Exception:
                continue
            out.append({"title": h.get("body", "")[:90], "url": url,
                        "kind": row[1] or "media",
                        "feed": h["source"].split(":", 1)[-1]})
            if len(out) >= k:
                break
    except sqlite3.Error:
        return out
    finally:
        conn.close()
    return out


# ── being asked for the news ─────────────────────────────────────────────────

# ⚠️ THIS MODULE HAD NO RECOGNISER, and that is the whole bug the user hit.
# They asked "Is there any news for me?" and Mati said "Of course. Let me put
# that together." — then nothing happened, because nothing routed the
# question. `report.asks_for_report` returns False for "news", `world.block`
# returns "" when no stored item matches, and no code ran at all. The model
# filled the silence with a promise because nothing contradicted it.
#
# ⚠️ THE SAME SHAPE AS radio AND podcast: a capability with no front door can
# only be reached by a terminal command, never by asking.
_ASKS_NEWS = (
    "any news", "the news", "what's new", "whats new", "news for me",
    "anything new", "what happened today", "anything happening",
    "headlines", "current events", "what's going on in the world",
    "qué hay de nuevo", "que hay de nuevo", "las noticias", "alguna noticia",
)


# ⚠️ A SUBSTRING LIST CANNOT SEE AN ADJECTIVE, and that is how the exact
# question the user asked was missed. "any news" is in the list above; they said
# *"Have you heard any interesting news?"* — so the literal "any news" is not
# present and the whole news path stayed shut. Mati then answered from
# ordinary notes with "I haven't come across anything particularly
# interesting", which is a worse failure than silence: it sounded like an
# answer about the news.
#
# ⚠️ STILL NOT A CLASSIFIER. The project's rule stands — the one time
# questions were classified by judgement the study died at its controls. This
# is the same literal vocabulary with up to two words allowed between the
# parts, which is what "interesting" needed.
_NEWS_RE = re.compile(
    r"\bany(?:thing)?\s+(?:\w+\s+){0,2}news\b"
    r"|\bheard\s+(?:any(?:thing)?\s+)?(?:\w+\s+){0,2}(?:news|interesting)\b"
    r"|\bany(?:thing)\s+(?:\w+\s+){0,2}(?:happening|going on)\b"
    r"|\bwhat(?:'s| is)\s+(?:\w+\s+){0,2}(?:happening|going on)\s+in the world\b",
    re.I)


def asks_about_news(text_in):
    t = (text_in or "").lower()
    return any(p in t for p in _ASKS_NEWS) or bool(_NEWS_RE.search(t))


def recent(k=8, topic=None, db_path=None):
    """The newest feed items. ⚠️ NOT `playable` — that is a different question.

    `playable` answers "which stored item can Mati actually PLAY", so it
    filters to items carrying an audio or video file. A news article has no
    enclosure, so it can never pass that filter — and `news_block` was calling
    it, which is why 19 stored items reported as "nothing stored that
    matches".

    ⚠️ It was also passing the QUESTION as a search query, so "any news" was
    being looked up as a phrase. A general request has no topic; it wants the
    latest. A topic is optional and narrows it.
    """
    from . import vault
    conn = sqlite3.connect(
        f"file:{db_path or bridge.NOTES_DB_PATH}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT note_id, source, body, wall_created FROM notes "
            "WHERE source LIKE ? ORDER BY note_id DESC LIMIT ?",
            (f"{SOURCE}:%", int(k) * 6)).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()

    words = [w for w in re.findall(r"[a-z]{4,}", (topic or "").lower())]
    out = []
    for nid, src, body, when in rows:
        try:
            text = vault.open_body(body) if body else ""
        except Exception:                                    # noqa: BLE001
            continue                                         # locked; skip it
        if not text:
            continue
        if words and not any(w in text.lower() for w in words):
            continue
        out.append({"title": text, "feed": str(src).split(":", 1)[-1],
                    "when": when, "note_id": nid})
        if len(out) >= k:
            break
    return out


def news_block(text_in=None, db_path=None):
    """What Mati can truthfully say when asked for the news.

    ⚠️ THE EMPTY CASE IS THE IMPORTANT ONE. With no feed subscribed there is
    no news, and the honest answer says so and says what would change it — a
    promise to "put it together" is exactly what happened before and it was
    false. This block is `<computed>` because it is worked out from the
    subscription list and the store, not remembered.
    """
    have = feeds()
    # ⚠️ `recent`, not `playable` — see that function. A general news request
    # wants the latest, not a phrase search for the words of the question.
    topic = text_in if (text_in and not asks_about_news(text_in)) else None
    items = recent(k=8, topic=topic, db_path=db_path) or []

    L = ["<computed>"]
    if not have:
        # ⚠️ Stated as a fact about the setup, not as an apology or a plan.
        L.append("I follow no news feeds yet, so I have no news — not none "
                 "today, none at all. Nothing is configured out of the box "
                 "on purpose. I can follow one if the person adds it with "
                 "`/world add <url>`; until then I should say I have none "
                 "rather than offer to look.")
        L.append("</computed>")
        return "\n".join(L)

    if not items:
        L.append(f"I follow {len(have)} feed(s) and have nothing stored that "
                 f"matches. That means nothing has come in on this topic, not "
                 f"that I have not looked.")
        L.append("</computed>")
        return "\n".join(L)

    from . import untrusted
    L.append("Things I read in a feed. This is the weakest thing I hold — "
             "nobody vetted it and I did not live it:")
    for it in items:
        title = untrusted.clean(it.get("title"))
        if title:
            L.append(f"  - [from {untrusted.clean(str(it.get('feed')), 60)}] "
                     f"{title}")
    L.append("</computed>")
    return "\n".join(L)


def block(query, db_path=None):
    """The `<world>` block for a turn — what is held, and that it can be played.

    ⚠️ Labelled as the weakest thing Mati holds, and attributed to the feed
    and the day. A headline spoken with the standing of a book is the failure
    that would make everything else untrustworthy.
    """
    items = playable(query, db_path=db_path)
    if not items:
        return ""
    lines = ["<world>",
             "Things I read in a feed. This is the weakest thing I hold — "
             "nobody vetted it and I did not live it. I may say what it is, "
             "say where it came from, and offer to play it:"]
    # ⚠️ CLEANED AGAIN HERE, on purpose. Notes ingested before the guard
    # existed are still in the store, and a second pass costs nothing.
    from . import untrusted
    for it in items:
        title = untrusted.clean(it["title"])
        if not title:
            continue
        lines.append(f"  - [{untrusted.clean(str(it['kind']), 20)}, "
                     f"from {untrusted.clean(str(it['feed']), 60)}] {title}")
    lines.append("If one of these answers what was asked, I say so and offer "
                 "to play it. I do not describe what is in it — I have only "
                 "its title and summary, not the recording.")
    lines.append("</world>")
    return "\n".join(lines)


def status(db_path=None):
    conn = sqlite3.connect(
        f"file:{db_path or bridge.NOTES_DB_PATH}?mode=ro", uri=True)
    try:
        n = conn.execute("SELECT COUNT(*) FROM notes WHERE source LIKE ?",
                         (f"{SOURCE}:%",)).fetchone()[0]
        media = conn.execute(
            "SELECT media_kind, COUNT(*) FROM notes WHERE source LIKE ? "
            "AND media_kind IS NOT NULL GROUP BY media_kind",
            (f"{SOURCE}:%",)).fetchall()
    except sqlite3.Error:
        return {}
    finally:
        conn.close()
    return {"items": n, "feeds": len(feeds()),
            "playable": {k: v for k, v in media}}
