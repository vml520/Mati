"""
An encyclopedia on this machine, if the person wants one.

    python3 -m Mati.encyclopedia --status
    python3 -m Mati.encyclopedia --look "propositional logic"

⚠️ OFF UNTIL CONNECTED, and there is nothing to turn on by default. This needs
a binary Mati does not ship and a file Mati does not download. Until both
exist `status()` says so and every other call returns nothing.

⭐ THE UNIT IS A SUBJECT, NOT WIKIPEDIA. Kiwix publishes per-subject builds —
mathematics at 56 MB, physics 51, chemistry 22, astronomy 339, history 701 —
next to the 119 GB complete dump. The subject builds are the ones that belong
here: the library exists so the person can manage what Mati specialises in,
and a 56 MB maths reference IS a specialisation. The complete dump is a
different product and it does not fit an 8 GB machine already running a 3B.

⚠️ NOTHING LEAVES THIS MACHINE, AND THAT IS ENFORCED, NOT INTENDED. The whole
reason this is a local server rather than Wikipedia's search API is that a
research query says what the person is writing about — the first outbound call
that would carry their own words. `_loopback_only` is what keeps the README's
host table true, so it is checked on every request and not merely at setup.

⚠️ GPL, AT ARM'S LENGTH. `libzim`/`python-libzim` are GPLv3 and importing one
would put this Apache-2.0 package in a contested position. `kiwix-serve` is a
separate process spoken to over a socket — the same arrangement Mati already
has with Ollama — so no GPL code is linked and none is redistributed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from . import config as C
from . import untrusted

NET_TIMEOUT = 8          # it is on localhost; slower than this means broken
START_TIMEOUT = 25       # a cold ZIM index can take a moment to mount
DEFAULT_PORT = 8099      # ⚠️ NOT 8756 — that is Mati's own server
SNIPPET_MAX = 300
ARTICLE_MAX = 3500       # about a page; the window is the real constraint

_proc = None
_lock = threading.Lock()

# ── the invariant ───────────────────────────────────────────────────────────

_LOOPBACK = ("127.0.0.1", "::1", "localhost", "0:0:0:0:0:0:0:1")


def _loopback_only(url):
    """The URL, if it goes nowhere. Raises otherwise.

    ⚠️ THIS IS THE FUNCTION THAT KEEPS THE PRIVACY CLAIM TRUE. The server
    address is a setting, and a setting is a thing a person can be talked into
    changing — point it at a host on the internet and every lookup silently
    becomes an outbound request carrying the topic they are writing about,
    with no other symptom. The README says nothing about you is ever sent; if
    that is to stay true it has to be checked here, on every call.

    ⚠️ AND IT CHECKS THE HOSTNAME LITERALLY, NOT BY RESOLVING IT. A name that
    resolves to 127.0.0.1 today can resolve elsewhere tomorrow, and a DNS
    lookup is itself a disclosure. Only these four spellings pass.
    """
    p = urllib.parse.urlsplit(url)
    if p.scheme != "http":
        raise ValueError(f"the encyclopedia must be plain http on this "
                         f"machine, not {p.scheme!r}")
    host = (p.hostname or "").lower()
    if host not in _LOOPBACK:
        raise ValueError(f"the encyclopedia must be on this machine; "
                         f"{host!r} is somewhere else")
    return url


# ── what is installed ───────────────────────────────────────────────────────

def zim_dir():
    """Where the person keeps their .zim files."""
    s = C._settings().get("encyclopedia_dir")
    return os.path.expanduser(s) if s else os.path.join(
        C.CACHE_DIR, "encyclopedias")


def zims():
    """The encyclopedias actually present, newest build first.

    ⚠️ A DIRECTORY, NOT A LIST OF FILES. These are downloaded by hand and
    replaced wholesale every few months — there is no delta update — so the
    person drops the new `..._2026-09.zim` in and deletes the old one. Reading
    the directory means that just works instead of needing a settings edit.
    """
    d = zim_dir()
    if not os.path.isdir(d):
        return []
    got = []
    for name in os.listdir(d):
        if name.endswith(".zim") and not name.startswith("."):
            full = os.path.join(d, name)
            try:
                got.append({"name": name[:-4], "path": full,
                            "bytes": os.path.getsize(full)})
            except OSError:
                pass
    return sorted(got, key=lambda z: z["name"], reverse=True)


_FLAVOURS = {"mini": "article introductions", "nopic": "full text",
             "maxi": "full text with pictures"}
_MONTHS = ("January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December")


def title_of(name):
    """A person's name for a ZIM file. `wikipedia_en_mathematics_mini_2026-06`
    -> "Encyclopedia of mathematics (Wikipedia, article introductions, June
    2026)". Unrecognised names come back as they are."""
    m = re.match(r"^([a-z]+)_([a-z]{2,3})_(.+?)_(mini|nopic|maxi)_(\d{4})-(\d{2})$",
                 name or "")
    if not m:
        return name
    source, _lang, subject, flavour, year, month = m.groups()
    subject = subject.replace("_", " ")
    when = f"{_MONTHS[int(month) - 1]} {year}" if 1 <= int(month) <= 12 else year
    return (f"Encyclopedia of {subject} ({source.capitalize()}, "
            f"{_FLAVOURS[flavour]}, {when})")


def binary():
    """Path to `kiwix-serve`, or None. Checked, not assumed."""
    s = C._settings().get("kiwix_serve")
    if s:
        s = os.path.expanduser(s)
        return s if os.access(s, os.X_OK) else None
    for d in (os.environ.get("PATH") or "").split(os.pathsep):
        if not d:
            continue
        p = os.path.join(d, "kiwix-serve")
        if os.access(p, os.X_OK):
            return p
    for p in ("/usr/local/bin/kiwix-serve", "/opt/homebrew/bin/kiwix-serve",
              os.path.expanduser("~/bin/kiwix-serve")):
        if os.access(p, os.X_OK):
            return p
    return None


def _port():
    return int(C._settings().get("encyclopedia_port") or DEFAULT_PORT)


def base_url():
    return _loopback_only(f"http://127.0.0.1:{_port()}")


# ── talking to it ───────────────────────────────────────────────────────────

def _get(path, timeout=NET_TIMEOUT):
    """GET from the local server. Bytes, or None. ⚠️ Never raises upward."""
    # ⚠️ NOTHING UNTIL CONNECTED, AND CHECKED HERE, ON THE REQUEST PATH. The
    # header promises every call returns nothing until the program and a file
    # both exist — but `search` only asked whether something answered on the
    # port, so with neither present it searched whatever was listening there:
    # another install's kiwix-serve, or one still serving files since deleted.
    if not connected():
        return None
    try:
        url = _loopback_only(base_url() + path)
    except ValueError:
        return None
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mati"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except Exception:
        return None


def _answering(timeout=1.5):
    """Is something listening on the port? A socket probe, not a request."""
    try:
        with socket.create_connection(("127.0.0.1", _port()), timeout):
            return True
    except OSError:
        return False


def serving():
    """True if a kiwix-serve is up and answering for us."""
    return _answering() and _get("/catalog/v2/root.xml", timeout=3) is not None


def start(wait=START_TIMEOUT):
    """Bring up kiwix-serve over whatever ZIMs are present. True if serving.

    ⚠️ SPAWNED ON DEMAND AND NOT SUPERVISED. A long-lived child that Mati is
    responsible for restarting is the worker-thread problem again; this one is
    allowed to die, and the next lookup starts it afresh. `serving()` is the
    only state that matters and it is measured, never remembered.

    ⚠️ IT DOES NOT ADOPT A SERVER IT DID NOT START, it just uses it. Somebody
    running their own kiwix-serve on this port is a feature, not a conflict —
    but then `stop()` must not kill it, which is why `_proc` is the only thing
    `stop()` will touch.
    """
    global _proc
    with _lock:
        if _answering():
            return serving()
        exe, files = binary(), zims()
        if not exe or not files:
            return False
        cmd = [exe, "--port", str(_port()), "--address", "127.0.0.1",
               "--nolibrarybutton"] + [z["path"] for z in files]
        try:
            _proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
        except Exception:
            _proc = None
            return False
    # ⚠️ POLLED, NOT SLEPT. Mounting a full-text index over a 700 MB subject
    # build is not instant, and a fixed sleep is either a stall or a flake.
    import time
    deadline = time.time() + wait
    while time.time() < deadline:
        if _proc is not None and _proc.poll() is not None:
            return False                      # it exited; no point waiting
        if serving():
            return True
        time.sleep(0.4)
    return False


def stop():
    """Stop the server Mati started. ⚠️ Never one it merely found."""
    global _proc
    with _lock:
        p, _proc = _proc, None
    if p is None or p.poll() is not None:
        return False
    try:
        p.terminate()
        try:
            p.wait(timeout=5)
        except Exception:
            p.kill()
        return True
    except Exception:
        return False


_warming = None


def warm():
    """Start the server in the background if it can be. True if it tried.

    ⚠️ BECAUSE `start()` ON THE TURN PATH IS A TWENTY-FIVE SECOND STALL.
    Mounting a full-text index is slow enough to feel like a hang, and latency
    is the thing people actually notice — a lookup that blocks the first reply
    of a session would read as Mati freezing, not as a cold cache. Called on
    waking, so by the time anyone asks for a reference it is already up.
    """
    global _warming
    if not connected() or _answering():
        return False
    if _warming is not None and _warming.is_alive():
        return True
    _warming = threading.Thread(target=start, daemon=True)
    _warming.start()
    return True


def _ensure(block_for=0.0):
    """Serving? ⚠️ Does NOT wait for a cold start unless asked to."""
    if _answering():
        return serving()
    if block_for <= 0:
        warm()
        return False
    return start(wait=block_for)


def books():
    """The encyclopedias the server has mounted: [{name, title, articles}]."""
    raw = _get("/catalog/v2/entries?count=100")
    if not raw:
        return []
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return []
    out = []
    for entry in root.iter():
        if not entry.tag.endswith("}entry") and entry.tag != "entry":
            continue
        got = {}
        for child in entry:
            tag = child.tag.rsplit("}", 1)[-1]
            if tag in ("title", "name", "articleCount", "mediaCount"):
                got[tag] = (child.text or "").strip()
        if got.get("name") or got.get("title"):
            out.append({"name": got.get("name") or got.get("title"),
                        "title": untrusted.clean(got.get("title") or "", 120),
                        "articles": got.get("articleCount") or ""})
    return out


# ── search ──────────────────────────────────────────────────────────────────

def _item_fields(node):
    """Pull title/link/snippet out of an RSS <item> or an Atom <entry>."""
    got = {"title": "", "link": "", "snippet": ""}
    for child in node:
        tag = child.tag.rsplit("}", 1)[-1].lower()
        text = (child.text or "").strip()
        if tag == "title" and not got["title"]:
            got["title"] = text
        elif tag == "link":
            got["link"] = got["link"] or text or (child.get("href") or "")
        elif tag in ("description", "summary", "content", "snippet"):
            got["snippet"] = got["snippet"] or text
    return got


_BOOK_PATH = re.compile(r"/(?:raw|content|viewer)?/?([^/]+)/(?:A/)?(.+)$")


def _split_link(link, fallback_book=None):
    """(book, path) out of whatever shape the link came in. ⚠️ Tolerant."""
    if not link:
        return fallback_book, None
    link = urllib.parse.urlsplit(link).path or link
    link = link.lstrip("/")
    parts = link.split("/")
    # /content/<book>/<path...>  or  /raw/<book>/content/<path...>
    if parts and parts[0] in ("content", "raw", "viewer"):
        parts = parts[1:]
    if parts and parts[0] == "content":
        parts = parts[1:]
    if len(parts) >= 2:
        book, rest = parts[0], "/".join(parts[1:])
        if rest.startswith("content/"):
            rest = rest[len("content/"):]
        return book, rest
    if len(parts) == 1 and fallback_book:
        return fallback_book, parts[0]
    return fallback_book, None


def search(query, k=5, book=None, wait=0.0):
    """Titles matching a query: [{title, book, path, snippet}]. [] if none.

    ⚠️ FULL TEXT, WHICH IS THE WHOLE REASON FOR THE SERVER. The Xapian index
    inside the ZIM is what makes this better than a filename match, and it is
    also why reading the file directly in Python was never the easy option.
    """
    q = " ".join((query or "").split())
    if len(q) < 2 or not _ensure(block_for=wait):
        return []
    args = {"pattern": q, "pageLength": max(1, min(int(k), 25)),
            "format": "xml"}
    if book:
        args["books.name"] = book
    else:
        # ⚠️ EVERY MOUNTED BOOK, NOT "the default one". With several subject
        # encyclopedias shelved, a question about entropy should reach physics
        # and chemistry both — omitting the selector searches the whole library.
        args["books.filter.lang"] = "eng"
    raw = _get("/search?" + urllib.parse.urlencode(args))
    if not raw:
        return []
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return []
    hits = []
    for node in root.iter():
        tag = node.tag.rsplit("}", 1)[-1].lower()
        if tag not in ("item", "entry"):
            continue
        f = _item_fields(node)
        if not f["title"]:
            continue
        bk, path = _split_link(f["link"], book)
        hits.append({
            "title": untrusted.clean(f["title"], 160),
            "book": bk, "path": path,
            "snippet": untrusted.clean(re.sub(r"<[^>]+>", " ", f["snippet"]),
                                       SNIPPET_MAX),
        })
        if len(hits) >= k:
            break
    return hits


_SCRIPT = re.compile(r"(?is)<(script|style|head)\b.*?</\1\s*>")
_BR = re.compile(r"(?i)<(br|/p|/div|/h[1-6]|/li|/tr)\s*/?>")
_TAGS = re.compile(r"(?s)<[^>]+>")


def _to_text(html):
    """Readable prose out of a ZIM article's HTML."""
    s = html.decode("utf-8", "replace") if isinstance(html, bytes) else html
    s = _SCRIPT.sub(" ", s)
    s = _BR.sub("\n", s)
    s = _TAGS.sub(" ", s)
    import html as _h
    s = _h.unescape(s)
    return s


def article(book, path, max_chars=ARTICLE_MAX):
    """The prose of one article, or None.

    ⚠️ THROUGH `clean_block`, BECAUSE THIS IS SOMEBODY ELSE'S TEXT. It is an
    encyclopedia and so about as benign as outside text gets — but the ZIM
    came off the internet, the person could have built it themselves, and the
    rule does not get to depend on the source seeming fine. Structure is
    stripped so nothing in it can close Mati's block and open another.
    """
    if not book or not path:
        return None
    safe = urllib.parse.quote(path, safe="/._-~")
    raw = (_get(f"/raw/{urllib.parse.quote(book)}/content/{safe}")
           or _get(f"/content/{urllib.parse.quote(book)}/{safe}"))
    if not raw:
        return None
    return untrusted.clean_block(_to_text(raw), max_len=max_chars) or None


def look_up(query, k=3, wait=0.0):
    """Search, then read the best hit. {title, book, text, others} or None.

    ⚠️ ONE ARTICLE, NOT THREE. The window is the constraint the prompt budget
    already loses to — the provenance legend alone is 38% of it. The other
    titles are named so Mati can offer them, not pasted in.
    """
    hits = search(query, k=k, wait=wait)
    if not hits:
        return None
    for h in hits:
        text = article(h["book"], h["path"])
        if text:
            return {"title": h["title"], "book": h["book"], "text": text,
                    "others": [x["title"] for x in hits
                               if x["title"] != h["title"]][:4]}
    # Found titles but could not read any body — say that rather than nothing.
    return {"title": hits[0]["title"], "book": hits[0]["book"],
            "text": None, "others": [h["title"] for h in hits[1:]][:4]}


# ── the conversational edge ─────────────────────────────────────────────────

# ⚠️ A LITERAL TRIGGER, like every other one in this codebase. "What is a
# truth table" is a question Llama can answer from its weights and does not
# need a reference for; this fires when the person asks for the ENCYCLOPEDIA,
# which is a different and much rarer sentence.
_LOOK_RE = re.compile(
    r"\b(?:look\s+up|look\s+that\s+up|check\s+the\s+encyclopedia|"
    r"what\s+does\s+(?:the\s+)?(?:encyclopedia|wikipedia)\s+say\s+about|"
    r"in\s+the\s+encyclopedia)\b[\s:,]*(?P<q>.*)$", re.I)


def asks_about_lookup(text_in):
    """Did they ask for a reference? The query, or None."""
    m = _LOOK_RE.search(" ".join((text_in or "").split()))
    if not m:
        return None
    q = (m.group("q") or "").strip(" .?!,:;\"'")
    q = re.sub(r"^(?:on|about|for|the)\s+", "", q, flags=re.I)
    return q or None


def connected():
    """Is there anything to look things up in? ⚠️ Measured, not remembered."""
    return bool(binary()) and bool(zims())


def block(text_in, db_path=None):
    """A reference, if one was asked for. ⚠️ Emitted as a quotation.

    ⚠️ LABELLED `reference`, NOT `studied`. Mati has not read this; it looked
    it up a second ago. The provenance legend names every label precisely so
    that distinction survives into the reply, and claiming to have studied
    something it fetched would be the exact overclaim the legend exists for.
    """
    q = asks_about_lookup(text_in)
    if not q or not connected():
        return None
    if not _answering():
        # ⚠️ NOT A FAILURE, AND IT MUST NOT READ AS ONE. The books are here;
        # the index is still mounting. Saying "nothing matches" would be a
        # false negative about the person's own library.
        warm()
        return ("<reference>\nMy encyclopedia is still opening — the index "
                "takes a moment on a cold start. Ask me again shortly and I "
                "will have it.\n</reference>")
    got = look_up(q)
    if not got:
        return (f"<reference>\nI have an encyclopedia here but nothing in it "
                f"matches “{q}”.\n</reference>")
    if not got["text"]:
        return (f"<reference>\nThe encyclopedia has an entry called "
                f"“{got['title']}” but I could not read it.\n</reference>")
    more = (f"\nOther entries I did not open: {', '.join(got['others'])}."
            if got["others"] else "")
    return (f"<reference>\nFrom my offline encyclopedia ({got['book']}), "
            f"the entry “{got['title']}”. I looked this up just now; I have "
            f"not studied it, and I am quoting it, not remembering it:\n\n"
            f"{got['text']}{more}\n</reference>")


def status():
    """What is and is not in place. For `/encyclopedia` and setup."""
    exe, files = binary(), zims()
    total = sum(z["bytes"] for z in files)
    return {
        "binary": exe,
        "zim_dir": zim_dir(),
        "zims": files,
        "bytes": total,
        "connected": bool(exe and files),
        "serving": serving() if (exe and files) else False,
        "port": _port(),
    }


def describe():
    """One short paragraph on where this stands, in Mati's voice."""
    s = status()
    if not s["binary"] and not s["zims"]:
        return ("I have no encyclopedia. It needs the kiwix-serve program and "
                "at least one .zim file — I ship neither, and nothing about "
                "this goes online once it is set up.")
    if not s["binary"]:
        n = len(s["zims"])
        return (f"I can see {n} encyclopedia file{'s' if n != 1 else ''} but "
                f"not the kiwix-serve program needed to read them.")
    if not s["zims"]:
        return (f"kiwix-serve is installed, but there are no .zim files in "
                f"{s['zim_dir']} yet.")
    mb = s["bytes"] / 1e6
    names = ", ".join(z["name"].replace("wikipedia_en_", "")
                      for z in s["zims"][:4])
    return (f"I have {len(s['zims'])} offline encyclopedia"
            f"{'s' if len(s['zims']) != 1 else ''} here — {names} — "
            f"{mb:.0f} MB in all, and I can search the full text without "
            f"going online.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--look", metavar="QUERY")
    ap.add_argument("--search", metavar="QUERY")
    ap.add_argument("--books", action="store_true")
    ap.add_argument("--stop", action="store_true")
    a = ap.parse_args(argv)

    if a.stop:
        print("  stopped" if stop() else "  nothing of mine was running")
        return 0
    if a.status or not (a.look or a.search or a.books):
        s = status()
        print(f"  binary   {s['binary'] or '— not installed'}")
        print(f"  zim dir  {s['zim_dir']}")
        for z in s["zims"]:
            print(f"    {z['bytes']/1e6:8.0f} MB  {z['name']}")
        if not s["zims"]:
            print("    (no .zim files)")
        print(f"  port     {s['port']}")
        print(f"  serving  {s['serving']}")
        print(f"\n  {describe()}")
        return 0
    if a.books:
        got = books()
        print(f"  {len(got)} mounted")
        for b in got:
            print(f"    {b['name']}  {b['articles']} articles")
        return 0
    if a.search:
        for h in search(a.search):
            print(f"  {h['title']}\n      {h['book']} :: {h['path']}")
            if h["snippet"]:
                print(f"      {h['snippet'][:140]}")
        return 0
    got = look_up(a.look)
    if not got:
        print("  nothing")
        return 1
    print(f"  {got['title']}  ({got['book']})")
    print("  " + "\n  ".join((got["text"] or "—").splitlines()[:12]))
    if got["others"]:
        print(f"\n  also: {', '.join(got['others'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
