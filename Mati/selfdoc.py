"""
What Mati is, made retrievable — and the way in for anything else.

The goal is for Mati to be aware of itself AND its environment. Everything is
in there, including how the machinery works, and the external world arrives
through RSS feeds by the same door.

⚠️ WITHOUT THIS, THE CORPUS HOLDS EVERYTHING EXCEPT MATI. Books studied,
questions it asked itself, episodes, the charter, notes about the person — and
nothing about Mati. An expert on whatever it has read that could not say what
it was, while README, PRINCIPLES, DESIGN and ARCHITECTURE sit on disk
describing it exactly. Retrieval is not the problem — lexical scores 0.838
against the field's 0.036. The subject has to be put in the corpus.

⚠️ NO IMPRINT, AND THAT IS WHAT MAKES IT AFFORDABLE. Studying imprints each
passage into θ₀ at ~48s a piece; the README alone would be hours. Lexical
retrieval reads `verbatim` and never touches the field, so this writes
archived-tier notes from birth: searchable immediately, no field cost, no
reindex, nothing competing for θ₀. Minutes rather than days.

⚠️ STATIC ONLY. What Mati IS goes here. How Mati is DOING does not — a stored
health snapshot is a memory asserting "queue_depth was 0" forever, stale the
moment it is written and recited as fact afterwards. That lives in `health.py`
and is computed fresh when asked, under `<computed>`.

⚠️ VOICE IS RE-AIMED ON THE WAY IN. These documents say "Mati does X" and
address a developer as "you". Both are wrong when Mati reads them back — "291
self-asked questions" comes back as "291 questions YOU asked". Third person
becomes first; second-person instructions to a developer are dropped rather
than rewritten, because a mangled imperative is worse than an absent one.

⚠️ BUILT FOR MORE THAN DOCUMENTS. `ingest()` takes a `kind` and a title, so an
RSS item lands through the same door with the same provenance discipline. What
changes for an external source is the LABEL — the world is not Mati, and
something read from a feed must never arrive wearing "this is my own".
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import time

from . import bridge

# Provenance prefixes. The label a memory carries is decided here and travels
# with it forever, so the kinds are explicit rather than inferred from a path.
SELF = "self"                 # what Mati is: its own documentation
WORLD = "world"               # RSS and anything else external

# The documents that describe what Mati IS. Development notes are excluded
# on purpose — anything written TO a developer ABOUT a project teaches Mati,
# once indexed, to describe itself as a work item with open tickets.
#
# ⚠️ THE PACKAGE FOLDER IS WHEREVER THIS FILE IS, NEVER A NAME WRITTEN IN. A
# path that names the folder resolves in one copy and silently misses in every
# other: `index_self` skips a missing document without complaint, so the
# assistant would know its architecture and nothing else about itself.
_PKG = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
SELF_DOCS = (
    (f"{_PKG}/README.md", "what I am and how I work"),
    (f"{_PKG}/PRINCIPLES.md", "what I am for"),
    (f"{_PKG}/DESIGN.md", "how I am built"),
    ("ARCHITECTURE.md", "how my memory is arranged"),
)

# Its own private memories, which belong in an answer about itself beside its
# documentation: what it reflected on (`mine`, `mine:charter`), its library
# (`mati:`), and what it measured about itself when it was built
# (`self-check`). Never episodes — what happened with the person is not
# self-knowledge.
_OWN_EXACT = ("mine", "self-check")
_OWN_PREFIXES = ("mine:", bridge.MINE_PREFIX)

# Paragraph size. Big enough to answer from, small enough that one retrieval
# does not eat the window.
CHUNK_CHARS = 700

# A paragraph telling a developer what to do is not a fact about Mati.
_IMPERATIVE = re.compile(
    r"\b(you should|you must|you can|you need|don'?t forget|make sure|"
    r"run this|see the|TODO|next step|do not ship)\b", re.I)

# Third person -> first. Order matters: possessive before bare.
_REAIM = (
    (re.compile(r"\bOlin'?s\b"), "my"),
    (re.compile(r"\bMati\b"), "I"),
)

# Verbs that follow "Mati" in this documentation, in the third person. Named
# rather than matched by a general rule so that "I pass", "I address" and
# "I access" survive intact.
_THIRD_PERSON_VERB = re.compile(
    r"\bI (run|hold|keep|store|remember|use|need|work|live|make|take|write|"
    r"read|know|learn|answer|speak|sit|come|go|stay|start|stop|ask|reply|"
    r"record|recall|imprint|reindex|carry|show|say|treat|refuse|accept|"
    r"depend|belong|arrive|change|grow|require|include|contain|report)s\b")


def _reaim(text):
    """Third person to first, and drop instructions aimed at a developer."""
    out = []
    for para in re.split(r"\n\s*\n", text):
        p = " ".join(para.split())
        if not p or len(p) < 60:
            continue
        if _IMPERATIVE.search(p):
            continue                       # an imperative, not a fact
        for pat, rep in _REAIM:
            p = pat.sub(rep, p)
        # ⚠️ VERB AGREEMENT, OR IT READS AS BROKEN ENGLISH. "Mati runs on
        # your own computer" became "I runs on your own computer" — visible in
        # the very first indexed passage. A blunt substitution needs a blunt
        # repair, and a NAMED verb list rather than a general `\w+s -> \w+`
        # rule, which would maul "I pass", "I address", "I access".
        p = re.sub(r"\bI is\b", "I am", p)
        p = re.sub(r"\bI has\b", "I have", p)
        p = re.sub(r"\bI does\b", "I do", p)
        p = re.sub(r"\bI isn't\b", "I am not", p)
        p = re.sub(r"\bI doesn't\b", "I do not", p)
        p = re.sub(r"\bI hasn't\b", "I have not", p)
        p = re.sub(_THIRD_PERSON_VERB, lambda m: "I " + m.group(1), p)
        out.append(p)
    return out


def _chunks(paras, size=CHUNK_CHARS):
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 1 > size and buf:
            yield buf.strip()
            buf = ""
        buf += p + " "
    if buf.strip():
        yield buf.strip()


def clear(kind=SELF, db_path=None):
    """Remove a kind's notes, so re-indexing replaces rather than duplicates."""
    conn = sqlite3.connect(db_path or bridge.NOTES_DB_PATH, timeout=30)
    try:
        n = conn.execute("DELETE FROM notes WHERE source LIKE ?",
                         (f"{kind}:%",)).rowcount
        conn.commit()
        return n
    except sqlite3.Error:
        return 0
    finally:
        conn.close()


def ingest(text, title, kind=SELF, db_path=None, reaim=True, verbose=False):
    """Write one source as retrievable, lexical-only notes. Returns the count.

    ⚠️ ARCHIVED FROM BIRTH AND UNIMPRINTED. These are reference, not
    experience: Mati did not live them and they must not deform θ₀ or take
    working-set slots from things that happened with the person. Lexical search
    reads them regardless — it never excluded archived notes.
    """
    from . import vault
    paras = _reaim(text) if reaim else [
        " ".join(p.split()) for p in re.split(r"\n\s*\n", text) if p.strip()]
    source = f"{kind}:{title}"
    now = bridge._now()
    stamp = time.time()
    conn = sqlite3.connect(db_path or bridge.NOTES_DB_PATH, timeout=30)
    n = 0
    try:
        bridge._ensure_schema(conn)
        for chunk in _chunks(paras):
            # ⚠️ The triplet columns are NOT NULL. They are a provenance
            # measurement of where θ₀ sat when a memory was MADE, and these
            # were not made by living anything — so they are zero, honestly,
            # rather than a fabricated reading. Nothing consults them for an
            # archived note: retrieval here is lexical.
            conn.execute(
                "INSERT INTO notes (body, source, wall_created, verbatim, "
                "archived, triplet_w0, triplet_w1, triplet_w2) "
                "VALUES (?,?,?,?,?,0.0,0.0,0.0)",
                (vault.seal(chunk), source, now, vault.seal(chunk), stamp))
            n += 1
        conn.commit()
    except sqlite3.Error as e:
        if verbose:
            print(f"  [selfdoc] {title}: {e}")
        return 0
    finally:
        conn.close()
    if verbose:
        print(f"  [selfdoc] {source}: {n} passages")
    return n


def _root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _digest(root):
    """What the documents say right now, as one hash."""
    h = hashlib.sha256()
    for rel, _title in SELF_DOCS:
        h.update(rel.encode("utf-8"))
        p = os.path.join(root, rel)
        if os.path.exists(p):
            with open(p, "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()


def _digest_path(db_path=None):
    return os.path.join(os.path.dirname(db_path or bridge.NOTES_DB_PATH),
                        "selfdoc.sha256")


def _indexed_count(db_path=None):
    try:
        conn = sqlite3.connect(
            f"file:{db_path or bridge.NOTES_DB_PATH}?mode=ro", uri=True)
        try:
            return conn.execute("SELECT count(*) FROM notes WHERE source LIKE ?",
                                (f"{SELF}:%",)).fetchone()[0]
        finally:
            conn.close()
    except sqlite3.Error:
        return 0


def refresh(root=None, db_path=None, verbose=False):
    """Re-index the documents if they changed since the last index. A count, or None.

    ⚠️ NOTHING ELSE KEEPS THIS CURRENT. Indexed once, the passages describe
    the documents as they were then, and the assistant goes on describing
    itself from an old README. So the server calls this when it has the key:
    at start, and again on a late unlock. Unchanged documents cost one hash.

    ⚠️ NEVER WHILE LOCKED. The passages are sealed, and clearing the old ones
    before the new ones can be written would leave it with no documentation at
    all.
    """
    from . import vault
    try:
        if vault.is_set_up() and not vault.unlocked():
            return None
    except Exception:                                    # noqa: BLE001
        return None
    root = root or _root()
    digest = _digest(root)
    marker = _digest_path(db_path)
    try:
        with open(marker, encoding="utf-8") as fh:
            done = fh.read().strip()
    except OSError:
        done = ""
    if done == digest and _indexed_count(db_path):
        return None
    n = index_self(root=root, db_path=db_path, verbose=verbose)
    if n:
        try:
            with open(marker, "w", encoding="utf-8") as fh:
                fh.write(digest)
        except OSError:
            pass
    return n


def index_self(root=None, db_path=None, verbose=False):
    """(Re)index everything that says what Mati is. Idempotent."""
    root = root or _root()
    clear(SELF, db_path=db_path)
    total = 0
    for rel, title in SELF_DOCS:
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            if verbose:
                print(f"  [selfdoc] missing: {rel}")
            continue
        with open(path, encoding="utf-8", errors="replace") as fh:
            total += ingest(fh.read(), title, kind=SELF,
                            db_path=db_path, verbose=verbose)
    return total


def status(db_path=None):
    conn = sqlite3.connect(
        f"file:{db_path or bridge.NOTES_DB_PATH}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT source, COUNT(*) FROM notes WHERE source LIKE ? "
            "OR source LIKE ? GROUP BY source",
            (f"{SELF}:%", f"{WORLD}:%")).fetchall()
    except sqlite3.Error:
        return {}
    finally:
        conn.close()
    return {s: n for s, n in rows}


# ─────────────────────────────────────────────────────────────────────────────
# Asking about itself
# ─────────────────────────────────────────────────────────────────────────────

# ⚠️ THE MOST NATURAL SELF-QUESTIONS RETRIEVE NOTHING. "who are you" and "what
# are you for" are entirely stopwords — there is no rare word to match on, so
# lexical search builds an empty query and returns zero hits over a corpus that
# answers both.
#
# A LITERAL PHRASE LIST, NOT A CLASSIFIER. The charter trigger is the same
# shape and scores 19/19 both ways, while classifying questions by judgement
# does not hold up. A phrase list is dull, checkable, and wrong in ways that
# are obvious rather than subtle.
_ASKS_ABOUT_SELF = (
    "who are you", "what are you",
    "about yourself", "about you", "yourself", "your own",
    "what do you do", "how do you work", "how were you made",
    "how are you built", "what are you for", "your purpose", "your principles",
    "your charter", "your memory", "your field", "your health",
    "how are you doing", "how do you feel", "know thyself",
    "your machinery", "your training", "your architecture",
)


def asks_about_self(text):
    """Is this a question about the assistant itself? Literal, checkable.

    ⚠️ "Who is <name>" uses the name this person gave it, read at call time.
    """
    from . import config as C
    t = " ".join((text or "").lower().split())
    name = C.assistant_name().lower()
    return (any(p in t for p in _ASKS_ABOUT_SELF)
            or f"who is {name}" in t or f"what is {name}" in t)


def _is_own(source):
    """Its own documentation, or one of its own private memories."""
    s = str(source or "")
    return (s.startswith(f"{SELF}:") or s in _OWN_EXACT
            or s.startswith(_OWN_PREFIXES))


def about_self(query, k=4, db_path=None):
    """What it knows about itself: its documentation and its own memories.

    Used when the query is about the assistant. Falls back to the opening
    passage of each document when the query has no searchable words of its
    own — the "who are you" case, and the reason this exists rather than
    leaving it to `search`.

    ⚠️ ITS OWN PRIVATE MEMORIES COUNT, not only what was written about it. A
    reflection it had last night is self-knowledge in a way a README is not,
    so the newest one always comes along, labelled as its own by `source`.
    """
    from . import notes, vault
    hits = [h for h in notes.search(query, k=k * 3, db_path=db_path)
            if _is_own(h.get("source"))][:k]
    if hits and any(not str(h.get("source", "")).startswith(f"{SELF}:")
                    for h in hits):
        return hits
    conn = sqlite3.connect(
        f"file:{db_path or bridge.NOTES_DB_PATH}?mode=ro", uri=True)
    try:
        rows = [] if hits else conn.execute(
            "SELECT note_id, body, source FROM notes WHERE source LIKE ? "
            "GROUP BY source ORDER BY note_id LIMIT ?",
            (f"{SELF}:%", int(k))).fetchall()
        rows += conn.execute(
            "SELECT note_id, body, source FROM notes WHERE source = 'mine' "
            "ORDER BY wall_created DESC LIMIT 1").fetchall()
    except sqlite3.Error:
        return hits
    finally:
        conn.close()
    out = list(hits)
    seen = {h.get("note_id") for h in hits}
    for note_id, body, source in rows:
        if note_id in seen:
            continue
        try:
            text = vault.open_body(body)
        except Exception:
            continue
        if text:
            out.append({"note_id": note_id, "body": text, "source": source,
                        "verbatim": text, "digest": None, "hits": [],
                        "score": 0.0, "when_said": None})
    return out
