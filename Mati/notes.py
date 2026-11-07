"""
The notes module: finding what is held. Lexical, exact, and the primary index.

⚠️ THIS IS WHERE LEXICAL RETRIEVAL ALWAYS BELONGED, AND IT SPENT FIVE DAYS
SOMEWHERE ELSE. The user: "the notes module was supposed to be the lexical
retrieval — I don't know how it got lost in the mix." It was not lost. The
notes module has been FIELD-ADDRESSED since the initial commit — `field/notes.py`
still says so: "a pure index of physical addresses, retrieval is
nearest-neighbor search in triplet space R³". Lexical `search()` arrived on
 and landed in `bridge.py`, which is the field↔LLM coupling and had no
business owning an index. The design was right; the code put the pieces in the
wrong boxes.

WHY LEXICAL IS PRIMARY, measured  on a harness validated at lexical
0.84 (two earlier probe sets were rejected at 0.15 and 0.29 before one passed):

    field    4/111 = 0.036      (chance 0.009)
    lexical 93/111 = 0.838

The field address is marginally above chance and 23x worse on identical cues
and pool. It also falls to 0.00 after a single turn of drift, because θ₀'s
motion IS the learning signal — an address anchored to a learning substrate is
stale by construction. See `ARCHITECTURE.md`.

⚠️ WHAT THIS MODULE CANNOT DO, and it is the field's remaining case: everything
here is CUED. A memory arriving UNASKED is untested and is the one thing a
lexical index cannot do at all.

⚠️ NOT FUZZY, DELIBERATELY. "On demand" means the asker already knows a word
that is in the thing, and a search that quietly returns near-misses is how you
end up confidently reading out the wrong memory.

Naming: this is `Mati/notes.py`, the RETRIEVAL module. `Mati/field/notes.py`
is the field-addressing primitive it was carved away from, and stays under
`field/` because that is what it is — substrate.
"""

from __future__ import annotations

import os
import re
import threading
import sqlite3
import time

from . import schedule, vault
from .field.notes import _now


# ⚠️ RESOLVED AT CALL TIME, NOT IMPORTED. Holding a module-level copy of
# _db() broke every caller that REDIRECTS the store by assigning
# `bridge.NOTES_DB_PATH` — five of them: test_tiers, test_server, test_coldstart,
# test_transfer, and two constitution diagnostics. They kept pointing bridge at a
# temp database while search quietly went on reading the real one. `test_tiers`
# caught it by raising `vault.Locked`; the others would have READ THE LIVE STORE
# and reported a pass. One source of truth, read when it is needed.
def _db(db_path=None):
    from . import bridge
    return bridge.NOTES_DB_PATH if db_path is None else db_path


# Module state for the lexical index: the document-frequency cache and the
# on-device POS tagger. They belong with the code that uses them — leaving
# them behind in `bridge` was the one real defect in the extraction, and it
# surfaced as NameError: _IDF rather than as a silent wrong answer.
_IDF = {"n_notes": 0, "df": {}}
_TAGGER = None
# ⚠️ Guards the setString_/enumerate pair above. See `nouns`.
_TAGGER_LOCK = threading.Lock()
_NOUNISH = {"Noun", "PlaceName", "PersonalName", "OrganizationName",
            "Number", "OtherWord"}


def _ensure(conn):
    """Schema, borrowed from the coupling layer at call time.

    Imported inside the function rather than at module scope: `bridge` imports
    THIS module, and the schema is shared by both retrieval paths, so it stays
    where the writer lives.
    """
    from .bridge import _ensure_schema
    return _ensure_schema(conn)


SEARCH_STOP = {
    "what", "when", "where", "which", "who", "how", "why", "did", "does",
    "do", "was", "were", "is", "are", "the", "a", "an", "of", "to", "in",
    "on", "it", "and", "or", "that", "this", "for", "with", "as", "at",
    "by", "be", "not", "about", "me", "my", "you", "your", "i", "we",
    "tell", "say", "said", "remember", "recall", "know",
}



# Decrypted bodies, keyed by their ciphertext.
#
# ⚠️ WHY THIS EXISTS. `vault.open_body` shells out to `openssl` — ONE SUBPROCESS
# PER BODY — and `search` scans every note. Measured  at 1,144 notes:
# 9.2s per search, 8ms/note, and no better on the second call. That alone is
# survivable; what was not is that `inquiry.resolve_from_later` runs one search
# PER PENDING QUESTION, and there were 129 of them. 129 x 9.2s is twenty
# minutes of an idle pass spent spawning processes, which is exactly how long
# the worker appeared to hang while `/health` reported it alive with no errors.
#
# Same defect class as the one already fixed in `recall` (101s -> 73ms at 8.9k
# notes) — that one could skip decryption entirely by scoring on addresses.
# Lexical search cannot: it needs the words. So the fix is to pay once.
#
# Keyed on the CIPHERTEXT, so a rewritten body misses naturally and no
# invalidation logic can go stale. Bounded, because an unbounded plaintext
# cache in a program whose whole point is encryption at rest is a bad trade.
_BODY_CACHE = {}


# ⚠️ A HARD ENTRY CEILING, KEPT ONLY AS A BACKSTOP. The real limit is
# `cache_budget_bytes()` — the user: *"the warming edge should be set by machine
# capacity, but a year is a perfect floor."* This stays so that a pathological
# store of millions of tiny notes cannot grow the dicts without bound even
# while the byte budget is satisfied.
_BODY_CACHE_MAX = 2_000_000

# Plaintext bytes held. ⚠️ TRACKED, NOT RECOMPUTED. Summing `len` over every
# cached value on each insert is O(n) per insert and O(n²) over a warm-up —
# at 6,600 entries that is 22 million string reads to warm the store, which
# would have made the segmented warming slower than the lump it replaced.
_CACHE_BYTES = 0


def _open_cached(stored):
    """`vault.open_body`, but not once per call per note."""
    global _CACHE_BYTES
    key = stored if isinstance(stored, (str, bytes)) else repr(stored)
    hit = _BODY_CACHE.get(key)
    if hit is not None:
        return hit
    body = vault.open_body(stored)
    _BODY_CACHE[key] = body
    _CACHE_BYTES += len(body) if isinstance(body, str) else 0
    if (_CACHE_BYTES > cache_budget_bytes()
            or len(_BODY_CACHE) > _BODY_CACHE_MAX):
        _evict()
    return body


def _drop(key):
    """Remove one cached entry, keeping the byte count honest."""
    global _CACHE_BYTES
    val = _BODY_CACHE.pop(key, None)
    if isinstance(val, str):
        _CACHE_BYTES -= len(val)
    _VERB_SEGMENT.pop(key, None)
    return val is not None


def cache_budget_bytes():
    """How much decrypted memory this machine may hold. The user's rule:
    *"The warming edge should be set by machine capacity, but a year is a
    perfect floor."*

    So the ceiling is the machine's, not a constant someone picked — and the
    year is guaranteed underneath it by `_evict`, which will not drop a
    memory from inside the year while any older one is still cached.

    ⚠️ A FRACTION OF TOTAL RAM, AND A SMALL ONE. The 8GB Mac is the stated
    floor for Mati and Ollama's 3B already takes most of it
    so this cannot budget against "free" memory
    it is about to lose to the model. 1% is 80 MB on that machine, which at
    the measured ~271 bytes an entry is ~300,000 memories — years of
    conversation — while being small enough that being wrong about it costs
    nothing anyone would notice.

    ⚠️ AND IT IS STILL BOUNDED ABOVE. Plaintext in RAM is the thing the vault
    exists to avoid; a bigger machine is a reason to hold more of the past,
    not a reason to hold all of it unconditionally.
    """
    try:
        total = (os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        total = 8 * 1024 ** 3          # assume the documented floor machine
    return int(min(max(total * CACHE_RAM_FRACTION, CACHE_MIN_BYTES),
                   CACHE_MAX_BYTES))


# 1% of RAM, between 16 MB and 512 MB. See `cache_budget_bytes`.
CACHE_RAM_FRACTION = 0.01
CACHE_MIN_BYTES = 16 * 1024 ** 2
CACHE_MAX_BYTES = 512 * 1024 ** 2

# Which segment a cached verbatim came from, by cache key. ⚠️ THE ONLY WAY
# EVICTION KNOWS WHAT IS OLD. The cache is keyed on ciphertext and carries no
# date, so without this `_evict` cannot tell a memory from this week from one
# from last year — see the measurement in `_evict`.
_VERB_SEGMENT = {}

# Set when the RAG-ready year does not fit in the budget at all, so the year
# floor could not be honoured. Surfaced by `/health`; see `_evict`.
year_overflowed = False


def _cache_bytes():
    """Plaintext currently held, from the running total kept by `_open_cached`."""
    return _CACHE_BYTES


def _evict():
    """Make room, oldest memories first, never breaking the year floor.

    ⚠️ THIS DROPPED THE NEWEST MEMORIES FIRST UNTIL IT WAS MEASURED. The
    previous version removed the oldest-INSERTED half, which sounds like the
    oldest memories and is the exact opposite: `warm_verbatim` runs
    newest-segment-first, so the first entries inserted are this month's.
    Measured  — warm segment 0 (newest 30 days) and segment 2 (days
    60-90), then evict:

        before   segment0 300/777   segment2 300/710
        after    segment0   0/777   segment2 300/710

    The most valuable half went and the older half stayed. Under sustained
    pressure that is a thrash, not a policy: warm this month, evict this
    month, re-warm it.

    Now eviction walks `_VERB_SEGMENT` from the highest segment down — real
    recency, from the only record of it — and stops as soon as it is under
    budget.

    ⚠️ BODIES ARE NEVER EVICTED HERE. Every search needs every body, so
    dropping one buys a few hundred bytes and costs a re-decryption on the
    next query. Verbatims are the elastic part.

    ⚠️ AND THE YEAR IS A FLOOR, NOT A PREFERENCE. Entries inside
    `VERBATIM_HOT_DAYS` are given up only when there is nothing older left to
    give, and that case sets `year_overflowed` rather than quietly shrinking
    what Mati can be held to — a machine too small for a year of verbatim is
    a fact the owner should be told, not a silent degradation.
    """
    global year_overflowed
    budget = cache_budget_bytes()
    if _cache_bytes() <= budget:
        return 0
    hot = hot_segments()
    # Oldest first, and within the year only after everything older is gone.
    order = sorted(_VERB_SEGMENT.items(), key=lambda kv: -kv[1])
    dropped = 0
    for key, seg in order:
        if _cache_bytes() <= budget:
            break
        if seg < hot:
            continue                    # the floor: not while older remains
        dropped += _drop(key)
    if _cache_bytes() > budget:
        # Nothing older left to give. The year itself does not fit, and that
        # is a fact about the machine rather than a thing to absorb quietly.
        year_overflowed = True
        for key, _seg in order:
            if _cache_bytes() <= budget:
                break
            dropped += _drop(key)
    return dropped


def _cached_only(stored):
    """The plaintext if it is ALREADY decrypted, else None. Never decrypts.

    ⚠️ THE POINT IS THE ABSENCE OF WORK. Cold verbatim is scored only when
    something else has already paid to open it, so an old archive can be
    searchable-if-warm without any search ever stalling on it.
    """
    key = stored if isinstance(stored, (str, bytes)) else repr(stored)
    return _BODY_CACHE.get(key)


def forget_bodies():
    """Drop the plaintext cache — call when the vault locks.

    ⚠️ CLEARS THE SIDE TABLES TOO. `_VERB_SEGMENT` left populated would make
    `_evict` believe entries are cached that are not, and `_CACHE_BYTES` left
    non-zero would make every later insert look over budget and evict
    immediately — a locked-then-unlocked vault would come back with a cache
    that could never fill.
    """
    global _CACHE_BYTES, year_overflowed
    _BODY_CACHE.clear()
    _VERB_SEGMENT.clear()
    _CACHE_BYTES = 0
    year_overflowed = False


# ── Verbatim is read in time segments ────────────────────────────────────────
#
# The user: *"We should split the verbatim into time segments, so we don't have
# to decrypt everything at once"* and *"the last year should be RAG ready"*.
#
# Decrypting every verbatim in the store is 23.9s cold at 2,361 of them —
# 10.1 ms each — and that number grows forever. Paying it in one lump was the
# only reason scoring the exact words looked expensive.
#
# So the exact words are read in two tiers:
#
#   HOT   — anything inside `VERBATIM_HOT_DAYS`. Always decrypted, always
#           scored, warmed at startup. Bounded by a YEAR OF CONVERSATION
#           rather than by all history, so the cost stops growing.
#   COLD  — older. Scored only if `_cached_only` already has it, and warmed a
#           segment at a time by the idle worker. A cold archive can never
#           stall a search.
#
# ⚠️ TODAY THE WHOLE STORE IS HOT, and that is worth saying plainly rather
# than implying a speed-up that has not happened: the store began ,
# so at ~10 weeks old every note is inside the year and startup still pays the
# full 23.9s. What this changes is that the figure is now bounded instead of
# unbounded — it is a cap for later, applied now so it is in place before it
# is needed.
VERBATIM_HOT_DAYS = 365.0
# Warming granularity for the cold tier, in days. One segment per idle pass,
# newest first — the oldest material is the least likely to be asked for, and
# the first to be archived out of the working set anyway.
VERBATIM_SEGMENT_DAYS = 30.0


def _iso_days_ago(days, now=None):
    import datetime as dt
    return dt.datetime.fromtimestamp(
        (time.time() if now is None else now) - days * 86400.0,
        dt.timezone.utc).isoformat(timespec="seconds")


def hot_segments():
    """How many segments make up the always-searchable year."""
    return int(round(VERBATIM_HOT_DAYS / VERBATIM_SEGMENT_DAYS))


def warm_verbatim(segment=0, db_path=None, now=None, budget=300):
    """Decrypt one time segment so its exact words become searchable.

    `segment` 0 is the most recent `VERBATIM_SEGMENT_DAYS`, 1 the segment
    before that, and so on — so segments 0..`hot_segments()-1` are the year
    the user asked to be RAG-ready, and anything beyond is bonus archive.

    Returns (decrypted, remaining_in_segment). `decrypted` 0 with `remaining`
    0 means this segment is finished or empty, which is how the caller knows
    to advance.

    ⚠️ BUDGETED, BECAUSE A SEGMENT IS NOT A KNOWN SIZE. Decryption is ~10.1 ms
    each; 300 is ~3s, which an idle pass can hold without making an arriving
    turn wait on it. A heavy month of conversation takes several passes
    instead of one long one — which is the entire point of segmenting.
    """
    db_path = _db(db_path)
    if not os.path.exists(db_path):
        return 0, 0
    newer = _iso_days_ago(segment * VERBATIM_SEGMENT_DAYS, now)
    older = _iso_days_ago((segment + 1) * VERBATIM_SEGMENT_DAYS, now)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT verbatim FROM notes WHERE verbatim IS NOT NULL "
            "AND wall_created <= ? AND wall_created > ? "
            "ORDER BY wall_created DESC", (newer, older)).fetchall()
    except sqlite3.Error:
        return 0, 0
    finally:
        conn.close()
    done = 0
    for i, (v,) in enumerate(rows):
        if _cached_only(v) is not None:
            continue
        if done >= budget:
            return done, len(rows) - i
        # ⚠️ STOP AT CAPACITY RATHER THAN EVICT TO MAKE ROOM FOR THIS. Past
        # the year, warming is opportunistic: an older segment must never push
        # a newer one out, or warming and eviction fight and the cache churns
        # with nothing gained. Inside the year it cannot happen — `_evict`
        # protects the floor — so this only ever bites on the archive.
        if segment >= hot_segments() and _cache_bytes() >= cache_budget_bytes():
            return done, len(rows) - i
        try:
            _open_cached(v)
            # ⚠️ RECORDED AFTER THE OPEN SUCCEEDS, and keyed the same way
            # `_open_cached` keys it, or `_evict` cannot find the entry it is
            # being asked to drop.
            key = v if isinstance(v, (str, bytes)) else repr(v)
            if key in _BODY_CACHE:
                _VERB_SEGMENT[key] = segment
            done += 1
        except Exception:
            continue          # a sealed row is not a reason to stop warming
    return done, 0


def oldest_segment(db_path=None, now=None):
    """The segment holding the oldest note with a verbatim, or 0 if none.

    ⚠️ THIS IS WHAT STOPS WARMING WALKING FOREVER now that the edge is set by
    capacity rather than by a fixed year. Without it, "keep going while there
    is room" on a small store means a SELECT per idle pass against segments
    that cannot contain anything, for the life of the process.
    """
    db_path = _db(db_path)
    if not os.path.exists(db_path):
        return 0
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        row = conn.execute("SELECT min(wall_created) FROM notes "
                           "WHERE verbatim IS NOT NULL").fetchone()
    except sqlite3.Error:
        return 0
    finally:
        conn.close()
    if not row or not row[0]:
        return 0
    try:
        import datetime as dt
        when = dt.datetime.fromisoformat(str(row[0]))
        if when.tzinfo is None:
            when = when.replace(tzinfo=dt.timezone.utc)
        age = ((time.time() if now is None else now) - when.timestamp()) / 86400.0
    except (TypeError, ValueError):
        return 0
    return int(age // VERBATIM_SEGMENT_DAYS)


def warming_edge(db_path=None, now=None):
    """How many segments warming should cover: the year, or more if it fits.

    the user: *"The warming edge should be set by machine capacity, but a year
    is a perfect floor."* So: never fewer than the year, never more than the
    store actually holds, and between those two the budget decides — checked
    as warming goes rather than predicted, since segment sizes vary.
    """
    return max(hot_segments(), oldest_segment(db_path, now) + 1)


def verbatim_warmth(db_path=None, now=None):
    """(warm, total) verbatims inside the RAG-ready year. For `/health`.

    ⚠️ THE YEAR, NOT THE STORE. Reporting against all history would make a
    fully-warmed Mati look permanently behind as the archive grows, which is
    the shape of the staleness metric that could never reach zero.
    """
    db_path = _db(db_path)
    if not os.path.exists(db_path):
        return 0, 0
    cutoff = _iso_days_ago(VERBATIM_HOT_DAYS, now)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT verbatim FROM notes WHERE verbatim IS NOT NULL "
            "AND wall_created > ?", (cutoff,)).fetchall()
    except sqlite3.Error:
        return 0, 0
    finally:
        conn.close()
    return sum(1 for (v,) in rows if _cached_only(v) is not None), len(rows)


_IDF_MAX = 8.0          # a stem never seen before is treated as very rare


def _idf(db_path=None):
    """How much each stem is worth. Common words are worth almost nothing."""
    import math
    conn = sqlite3.connect(_db(db_path))
    try:
        n = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
    except sqlite3.Error:
        return _IDF["df"]
    finally:
        conn.close()
    if _IDF["df"] and abs(n - _IDF["n_notes"]) < max(1, _IDF["n_notes"] // 10):
        return _IDF["df"]
    df = {}
    conn = sqlite3.connect(_db(db_path))
    try:
        for (stored,) in conn.execute("SELECT body FROM notes"):
            try:
                body = _open_cached(stored)
            except Exception:
                continue
            for w in {schedule._stem(x)
                      for x in re.findall(r"[a-z0-9'\-]+", (body or "").lower())}:
                df[w] = df.get(w, 0) + 1
    except sqlite3.Error:
        pass
    finally:
        conn.close()
    _IDF["n_notes"] = n
    _IDF["df"] = {w: min(math.log((n + 1) / (c + 1)) + 0.5, _IDF_MAX)
                  for w, c in df.items()}
    return _IDF["df"]



# How many words a question is allowed to be about. Three or four content
# words is what a person's question actually turns on; the rest is grammar.
FOCUS_TERMS = 4


def _nouns(text):
    """The nouns and names in a sentence, or None if the tagger is unavailable.

    ⚠️ RARITY ALONE CANNOT FIND THE SUBJECT, which is why this exists.
    Measured  on the user's own question, the rarest stems were:

        tweaked 7.84   let' 6.04   calmecac 5.23   training 4.75

    "tweaked" is rare because nobody says it, not because the question is about
    it — and "let'" is a tokenisation artefact of "Let's". Ranking by rarity
    picked both over the subject. Grammar knows the difference: what a question
    is ABOUT is a noun. It is also what separates a command from a query — a
    command is carried by its verb, a question by its noun.
    """
    global _TAGGER
    try:
        import NaturalLanguage as NL
        from Foundation import NSString, NSMakeRange
        # ⚠️ ONE THREAD AT A TIME, AND THIS SEGFAULTED THE SERVER ON
        # `_TAGGER` is a module global and the call is two steps:
        # `setString_` then `enumerateTagsInRange_`. The server is a
        # ThreadingHTTPServer PLUS a worker thread, so a chat turn and a
        # background pass could interleave — one swapped the string out from
        # under the other's enumeration, and CoreNLP dereferenced null:
        #
        #   EXC_BAD_ACCESS SIGSEGV at 0x0
        #   CoreNLP::CompositeTagger::getAttributedTokenAtIndex
        #   -[NLTagger enumerateTagsInRange:unit:scheme:options:usingBlock:]
        #
        # ⚠️ NO try/except COULD HAVE CAUGHT THAT. A segfault is not a Python
        # exception — it killed the process with no traceback, which is why
        # the log ended mid-turn and looked like a clean exit. Prevention is
        # the only option available, so the pair is made atomic.
        with _TAGGER_LOCK:
            if _TAGGER is None:
                _TAGGER = NL.NLTagger.alloc().initWithTagSchemes_(
                    [NL.NLTagSchemeLexicalClass])
        # ⚠️ ASK THE TAGGER WHERE THE WORDS ARE. Zipping its tags against a
        # a separate regex put them out of step: "Let's" is one token to the
        # regex and two to the tagger, so every tag after an apostrophe was
        # attached to the wrong word. Measured  — 9 tokens against 10
        # tags, and `calmecac` inherited its neighbour's part of speech, which
        # is exactly the word the question was about.
            _TAGGER.setString_(NSString.stringWithString_(text))
            found = set()

            def _keep(tag, rng, _stop):
                if str(tag) in _NOUNISH:
                    w = text[rng.location:rng.location + rng.length]
                    if len(w) > 1:
                        found.add(schedule._stem(w.lower()))

            _TAGGER.enumerateTagsInRange_unit_scheme_options_usingBlock_(
                NSMakeRange(0, len(text)), NL.NLTokenUnitWord,
                NL.NLTagSchemeLexicalClass,
                NL.NLTaggerOmitWhitespace | NL.NLTaggerOmitPunctuation, _keep)
            return found or None
    except Exception:
        return None                 # not macOS, or the framework is missing


def asking_part(text):
    """The part of a message that ASKS, when it also does something else.

    ⚠️ A TURN IS OFTEN TWO THINGS. The user's message was a statement about what
    they had changed, followed by a question:

        "I tweaked your training recall to make it more like memories. Let's
         see if this helps with your recall of the text. What does the book say
         about the calmecac?"

    Searching the whole thing retrieves for the statement as much as for the
    question, and the statement is about training and recall — so the notes
    that came back were about tweaking recall, not about the calmecac.

    A question mark is the reliable signal, and a leading interrogative is the
    fallback. Where neither is present the message is not asking, and the whole
    of it stands — a command is carried by its verb and has no subject to find.
    """
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", (text or "").strip())
             if p.strip()]
    if len(parts) < 2:
        return text
    asked = [p for p in parts if p.rstrip().endswith("?")]
    if not asked:
        asked = [p for p in parts if re.match(
            r"(?i)^(what|who|where|when|which|why|how|do|does|did|can|could|"
            r"is|are|was|were|tell me|explain)\b", p)]
    return " ".join(asked) if asked else text


def _focus(want, db_path=None, query=""):
    """What the question is about: its nouns, then the rarest of those.

    Falls back to rarity alone where the tagger is unavailable, which is worse
    but not broken — see `_nouns` for what that costs.
    """
    if len(want) <= FOCUS_TERMS:
        return want
    nouns = _nouns(query) if query else None
    if nouns:
        narrowed = want & nouns
        if narrowed:
            want = narrowed
    if len(want) <= FOCUS_TERMS:
        return want
    idf = _idf(db_path)
    return set(sorted(want, key=lambda w: -idf.get(w, _IDF_MAX))[:FOCUS_TERMS])



# ⚠️ AN EPISODE CONTAINS THE QUESTION, SO IT MATCHES THE QUESTION BY
# CONSTRUCTION. Every episode body opens `I was asked: <the person's words,
# verbatim>`. Ask the same thing twice and the record of the FIRST asking is a
# near-perfect lexical match for the second — not because it holds the answer,
# but because it holds the question. Matching it is matching yourself.
#
# , ten questions about the book, three recall slots each:
# 13 of 30 slots went to Mati's own past turns and 12 to the book, with 4 of
# 10 questions retrieving NO book content at all. The split was not random —
# bare topic questions ("Who were the tlamatinime?") reached the book, and
# questions about the reading itself ("do you remember reading…", "what does
# the book say…") were crowded out, because those phrasings are exactly the
# ones an episode echoes back.
#
# That is "it cannot remember reading the book", and it is self-reinforcing:
# each asking writes a new episode containing the question, which outranks the
# material on the next asking. One of the crowding episodes read "I do not
# actually remember reading the book" — a denial, recalled as evidence and
# repeated.
#
# DISCOUNTED, NOT DROPPED. "What did I ask you about last week?" is a real
# question and the asked half is its answer, so it still counts — at a
# fraction, so it can no longer outbid material that actually holds the
# subject. The reply and what happened count in full: those are Mati's, not
# an echo.
ASKED_ECHO = 0.25


# Swept  over ten book questions (episodes / book notes / questions
# returning no book at all, three slots each):
#
#     0.00   ep  5   book 14   no-book 4/10
#     0.25   ep  5   book 14   no-book 4/10
#     0.50   ep  4   book 16   no-book 4/10
#     1.00   ep  3   book 19   no-book 3/10
#
# ⚠️ CONTROLLED FOR FLOODING, because "more book notes" is trivially gamed
# by making every chunk match its own title. Eight unrelated questions (the
# person's name, the principles, arithmetic, the vault) took the SAME 5 of 24
# slots at 0.5 and at 1.0 — raising the weight does not spread the book into
# questions that are not about it.
SOURCE_WEIGHT = 1.0

# Weight on a word that appears ONLY in a memory's verbatim — the exact
# exchange — and not in the gist Mati kept. Fallback pass only; see
# `_search_pass`.
#
# ⚠️ WHY IT IS DISCOUNTED RATHER THAN 1.0. A verbatim-only word is one Mati
# HEARD but did not keep: it may be the person's passing phrasing, or the
# question rather than the answer. Matching it is a weaker claim than matching
# the gist, and this tier is only ever consulted when the gist pass returned
# nothing at all — so there is no stronger evidence alongside it to correct an
# over-confident hit.
#
# ⚠️ SWEPT, on 20 real episodes cued by the RAREST word that
# appears in their verbatim and not in their gist. The control is the same 20
# notes cued by their rarest GIST word, which must not get worse:
#
#     weight   verbatim-only cue   gist cue (control)
#     0.00           1/20                19/20          <- shipped until today
#     0.25          15/20                19/20
#     0.50          20/20                19/20          <- chosen
#     1.00          20/20                19/20
#     2.00          20/20                17/20          <- verbatim outbids gist
#
# 0.5 is the smallest weight that reaches the ceiling, and harm is measurable
# at 4x it. ⚠️ THE 2.00 ROW IS THE REASON THIS IS NOT 1.0 "for safety": over
# here, a word Mati merely HEARD starts beating a word it KEPT, and the
# control is the only thing that shows it.
#
# ⚠️ AND THE FIRST SWEEP OF THIS WAS A BAD INSTRUMENT. It picked cue words by
# LENGTH, so cues like "consciousness" and "conversation" were common words
# every note matched; it reported 1/20 at every weight and would have had me
# conclude the fix did not work. The scorer ranks by RARITY, so the probe has
# to as well.
VERBATIM_WEIGHT = 0.5

# How far a corrected memory falls behind an uncorrected one that matches
# equally well. Deliberately NOT zero: the user's rule is that the error does not
# disappear, it stops being what comes to mind. `/recall` and `/source` still
# reach it.
CORRECTED_WEIGHT = 0.35


def _episode_halves(body):
    """An episode split into (what it echoes back, what it is a record of)."""
    echo, rest = [], []
    for line in (body or "").split("\n"):
        t = line.strip()
        if t.startswith("I was asked:"):
            echo.append(t[len("I was asked:"):])
        else:
            rest.append(t)
    return " ".join(echo), " ".join(rest)


def _search_pass(query, k=5, db_path=None, sources=None, deep=False):
    """Find memories containing the words in `query`. Lexical, exact, instant.

    Scored by how many DISTINCT query words a memory contains, then by how
    recent it is. Deliberately not fuzzy: "on demand" means the asker already
    knows a word that is in the thing, and a search that quietly returns
    near-misses is how you end up confidently reading out the wrong memory.

    Returns a list of dicts: note_id, body, source, hits, score.
    """
    db_path = _db(db_path)
    if not os.path.exists(db_path):
        return []
    want = {schedule._stem(w)
            for w in re.findall(r"[a-z0-9'\-]+", (query or "").lower())
            if len(w) > 1 and w not in SEARCH_STOP}
    if not want:
        return []
    # ⚠️ THE QUESTION IS ITS RAREST WORDS. Weighting was not enough: a note
    # matching ten ordinary words still outscored one matching the single word
    # the question was ABOUT, because the numerator grows with every match.
    #
    # Measured  on the user's own message — "calmecac" found the right
    # notes 5 of 5, and the same question inside a 30-word conversational turn
    # found 1 of 5. That is "if I don't use the keyword search it can't find
    # anything": retrieval did not fail, TALKING NORMALLY WAS PENALISED.
    #
    # So filler is dropped rather than down-weighted. A long question is
    # searched on what it is about, and padding it costs nothing.
    # If the message asks something, the question is what to search on.
    want = _focus(want, db_path, asking_part(query))
    conn = sqlite3.connect(db_path)
    try:
        _ensure(conn)
        # `verbatim` travels with every hit so the FOCUSED path can show
        # the exact words. Gist and focus are different functions, not
        # two sizes of one: the gist is how a memory surfaces, the
        # verbatim is what focusing on it yields.
        sql = ("SELECT note_id, body, source, wall_created, verbatim, digest "
               "FROM notes")
        args = ()
        if sources:
            sql += " WHERE source IN (%s)" % ",".join("?" * len(sources))
            args = tuple(sources)
        sql += " ORDER BY note_id DESC"
        rows = conn.execute(sql, args).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()

    out = []
    unreadable = 0
    for note_id, stored, source, created, stored_verbatim, stored_digest in rows:
        try:
            body = _open_cached(stored)
        except Exception:
            # ⚠️ A LOCKED VAULT USED TO LOOK LIKE AN EMPTY STORE. Every body
            # failed to decrypt, every one was skipped, and search returned
            # zero hits with no error — which is exactly what "the terminal
            # cannot recall any memory". Not breaking is
            # right; being silent about it is not.
            unreadable += 1
            continue
        def _stems(t):
            return {schedule._stem(w)
                    for w in re.findall(r"[a-z0-9'\-]+", (t or "").lower())}

        # An episode is scored on what it RECORDS; the question it quotes back
        # counts at `ASKED_ECHO`. See the note above `_episode_halves`.
        # ⚠️ THE TITLE OF A DOCUMENT IS NOT IN THE DOCUMENT. Measured
        # : 1 of 400 book notes contains the words "Aztec Thought
        # and Culture" in its body. The title lives in `source`, search scored
        # `body` alone, and so NAMING THE BOOK COULD NOT MATCH THE BOOK — the
        # slots went to `studied:asked about religion` and the like, which at
        # least shared an ordinary word. "What does the book say about X" and
        # "what did you learn from <title>" are the two most natural ways to
        # ask, and both were unanswerable by construction.
        #
        # Discounted, because every chunk of a document matches its title
        # equally: it is a topic signal, not evidence that THIS passage is the
        # apposite one, and at full weight it would flatten 1,744 notes into a
        # tie broken by recency.
        src_words = re.sub(r"^studied:|\.txt$|[_/]", " ", source or "")
        if source == "episode":
            echo, rest = _episode_halves(body)
            have_full, echo_only = _stems(rest), _stems(echo) - _stems(rest)
        else:
            have_full, echo_only = _stems(body), set()
        title_only = _stems(src_words) - have_full - echo_only
        # ⚠️ THE EXACT WORDS, AND ONLY ON THE FALLBACK PASS. Measured
        #  on the real store: 34% of the words in a verbatim do not
        # appear in its gist, and a note was reachable by a verbatim-only word
        # 0 times out of 12 — against 8 of 12 by a gist word. The verbatim was
        # decrypted lazily AFTER scoring and attached to the hit, so it could
        # only ever be reached by a note you had already found. Write-only
        # memory.
        #
        # ⚠️ NOT ON THE NORMAL PASS, FOR TWO MEASURED REASONS. Decrypting
        # every verbatim costs 23.9s cold (0.002s warm) — the same shape as
        # the 101s regression that made `recall` decrypt every body. And the
        # verbatim holds the WHOLE exchange including the question, so folding
        # it into every score is the dilution this function has already been
        # fixed for twice. The fallback runs only when the gist pass found
        # nothing, so an ordinary turn pays neither cost.
        # ⚠️ A QUERY NEVER DECRYPTS A VERBATIM. `_cached_only` returns what is
        # already open and never does work, so the exact words cost a set
        # difference and nothing else. All decryption happens in
        # `warm_verbatim`, one time segment per idle pass.
        #
        # ⚠️ THE FIRST VERSION DECRYPTED HOT ROWS ON DEMAND AND THAT WAS THE
        # WHOLE PROBLEM RESTATED. Measured: a cold search took 73.0s against
        # 0.36s warm, because "hot" is every note in a ten-week-old store —
        # the year window bounds the archive, it does not make the current
        # year cheap. The user: *"split the verbatim into time segments, so we
        # don't have to decrypt everything at once."* A per-query decryption
        # is decrypting everything at once, wearing a bound.
        #
        # The cost of being wrong this way is bounded and self-correcting: a
        # verbatim not yet warm is not yet searchable by its exact words, and
        # becomes searchable a segment later. The cost of the other way was a
        # 73-second sentence.
        verb_only = set()
        if deep and stored_verbatim is not None:
            text = _cached_only(stored_verbatim)
            if text:
                verb_only = (_stems(text)
                             - have_full - echo_only - title_only)
        # ⚠️ THE READING SUMMARY WAS RETURNED AND NEVER SCORED. `digest` came
        # back on every hit but its own words made nothing findable — so a
        # condensation Mati wrote about a passage could not be reached by
        # anything it said. Asked for: the library, the study notes
        # AND the reading summaries should all be searchable by Mati.
        #
        # ⚠️ CACHED-ONLY, LIKE VERBATIM, FOR THE SAME MEASURED REASON. A
        # per-query decryption is decrypting everything at once wearing a
        # bound — it made a cold search 73 seconds. A digest not yet warm is
        # not yet searchable by its words and becomes so a segment later.
        dig_only = set()
        if deep and stored_digest is not None:
            dtext = _cached_only(stored_digest)
            if dtext:
                dig_only = (_stems(dtext)
                            - have_full - echo_only - title_only - verb_only)
        have = have_full | echo_only | title_only | verb_only | dig_only
        hits = want & have
        if not hits:
            continue
        # ⚠️ WEIGHTED BY RARITY. The old score was len(hits)/len(want) — every
        # matched word worth the same — so "what have you been reading" reduced
        # to {been, have, reading} and EVERY note holding those three scored a
        # perfect 1.00. Dozens tied, the tie broke on note_id DESC, and the
        # most recent chatter buried the memory of reading the book. Measured
        # : 0 of 4 natural phrasings could find it.
        #
        # `been` and `have` did no work and outvoted `reading` two to one.
        # Weighting by how rare a stem is in the store fixes exactly that:
        # `tlamatinime` counts, `have` barely does.
        # ⚠️ Weighting only separates PARTIAL matches. A note containing
        # every query stem still scores 1.00, so a query whose only content
        # word is common ("what have you been reading") cannot be ranked at
        # all. That is why the present tense moved out of retrieval and into
        # the waking block — see orient.
        idf = _idf(db_path)
        num = (sum(idf.get(w, _IDF_MAX) for w in (want & have_full))
               + ASKED_ECHO * sum(idf.get(w, _IDF_MAX)
                                  for w in (want & echo_only))
               + SOURCE_WEIGHT * sum(idf.get(w, _IDF_MAX)
                                     for w in (want & title_only))
               # ⚠️ DISCOUNTED, LIKE THE TITLE TIER AND FOR THE SAME REASON.
               # Matching a word Mati merely HEARD is weaker evidence than
               # matching the gist it kept — it may be a word the person used
               # in passing and Mati never took up. At full weight a verbatim
               # hit would outrank a real gist hit on the fallback pass, which
               # is the one case where the gist pass already came up empty and
               # cannot correct it.
               + VERBATIM_WEIGHT * sum(idf.get(w, _IDF_MAX)
                                       for w in (want & verb_only)))
        # ⚠️ NORMALISED BY THE BEST POSSIBLE MATCH, NOT BY THE WHOLE QUERY.
        # Dividing by every stem in the question meant each extra word you
        # typed made the word that mattered count for less. Measured
        #  on the user's own message:
        #
        #     "calmecac"                          5 of 5 hits were the right ones
        #     "What does the book say about..."   5 of 5
        #     their actual 30-word message          1 of 5
        #
        # That is "if I don't use the keyword search it can't find anything":
        # retrieval did not fail, TALKING NORMALLY WAS PENALISED. A
        # conversational question carries filler that a bare keyword does not,
        # and the filler was outvoting the subject.
        #
        # Dividing by the weight of the rarest few stems instead means a
        # question is scored on whether it found what it was ABOUT, and padding
        # it with ordinary words costs nothing.
        top = sorted((idf.get(w, _IDF_MAX) for w in want), reverse=True)[:3]
        den = sum(top) or 1.0
        # ⚠️ DECRYPTED LAZILY. `recall` once decrypted every body on every
        # call — 101s at 8.9k notes. Only the rows that actually MATCHED get
        # their verbatim opened, so this is a handful per turn, not the store.
        try:
            exact = _open_cached(stored_verbatim) if stored_verbatim else None
        except Exception:
            exact = None
        try:
            dig = _open_cached(stored_digest) if stored_digest else None
        except Exception:
            dig = None
        # ⚠️ DENSITY BREAKS THE TIE, AND WITHOUT IT A ONE-WORD QUERY IS A
        # COIN TOSS DECIDED BY RECENCY. `score` saturates at 1.0 when every
        # query word is present, so "tlamatinime" scored 1.000 for all 98 book
        # passages AND for every recent episode that mentioned it once — and
        # the sort fell through to `ORDER BY note_id DESC`, handing all four
        # slots to the newest chatter. Naming a specific thing is exactly when
        # retrieval matters most, and it was the case with no resolution at all.
        #
        # Density is occurrences of the matched stems per 100 words: a passage
        # ABOUT the tlamatinime uses the word repeatedly; a conversation that
        # mentions it uses it once. Tie-break only — it never reorders notes
        # whose scores actually differ.
        _wc = max(1, len((body or "").split()))
        _occ = sum(1 for w in re.findall(r"[a-z0-9'\-]+", (body or "").lower())
                   if schedule._stem(w) in hits)
        out.append({"note_id": note_id, "body": body, "source": source,
                    "verbatim": exact, "digest": dig,
                    "density": 100.0 * _occ / _wc,
                    "hits": sorted(hits), "score": num / den,
                    "when": created, "when_said": when_said(created)})
    # More query words matched wins; ties go to the more recent memory, which
    # `ORDER BY note_id DESC` above has already arranged.
    # ⚠️ A CORRECTION MUST TRAVEL WITH THE MEMORY IT CORRECTS, AND ON THIS
    # PATH IT DID NOT. `bridge.recall` has attached corrections since they were
    # built — but that was when the FIELD was the primary index. Lexical became
    # primary on  (0.838 against the field's 0.036), and this
    # function had never heard of `corrects`. So the channel that actually
    # answers questions was the one blind to every correction the user had made.
    #
    # The user's framing, and it is the right one: a verified correction should
    # outweigh the original. The error does not disappear — it stops being what
    # surfaces, unless something goes deliberately looking. "I said pass-a-port
    # when I was twelve, and what I remember is the learning."
    #
    # Attached to the BODY rather than returned beside it, so no consumer can
    # render the error without it — the same rule, and the same reason, as in
    # `recall`.
    if out:
        try:
            from . import correction
            ids = [r["note_id"] for r in out[:k * 3]]
            conn = sqlite3.connect(f"file:{_db(db_path)}?mode=ro", uri=True)
            try:
                marks = ",".join("?" * len(ids))
                fixes = {cid: vault.open_body(ctext) for cid, ctext in
                         conn.execute(
                             "SELECT corrects, body FROM notes WHERE corrects "
                             f"IN ({marks})", ids)}
            finally:
                conn.close()
            for r in out:
                fix = fixes.get(r["note_id"])
                if fix:
                    r["body"] = correction.annotate(r["body"], fix)
                    r["corrected"] = True
                    # ⚠️ OUTWEIGHED, NOT ERASED. The user: "any verified
                    # correction to memory should outweigh the original
                    # incorrect memory. The incorrect memory doesn't disappear,
                    # it just isn't recalled unless the LLM is doing active
                    # memory lookup."
                    #
                    # So a corrected memory is DEMOTED rather than dropped: it
                    # loses to anything uncorrected that also matches, and it
                    # still carries its correction if it does surface. `/recall`
                    # and `/source` reach it unchanged, which is the deliberate
                    # lookup they described — the twelve-year-old's "pass-a-port"
                    # is still there to be remembered, it is simply not what
                    # comes to mind now.
                    r["score"] *= CORRECTED_WEIGHT
        except (sqlite3.Error, vault.Locked):
            pass          # a sealed or busy store must not lose the search
        except Exception as exc:
            # ⚠️ A BLANKET `except: pass` HID THIS VERY BLOCK FAILING. The
            # demotion referenced a constant that did not exist, raised
            # NameError, and was swallowed — corrections looked applied and the
            # ranking silently did not change. Programming errors are surfaced;
            # only the two expected runtime conditions are absorbed.
            raise

    out.sort(key=lambda r: (-r["score"], -r.get("density", 0.0)))
    # ⚠️ WHAT MATI SAID ABOUT A BOOK IS NOT THE BOOK, AND OUTRANKS IT. With
    # the echo discounted, 4 of 10 book questions still returned no book at
    # all — this time on Mati's OWN past statements, which score legitimately
    # and, at three slots, can take every one. "Do you remember reading about
    # the tlamatinime?" filled all three with prior turns about tlamatinime.
    #
    # That is a memory loop with nothing outside it: the answers were built
    # from earlier answers, so a confabulation could be recalled as its own
    # corroboration, and a denial ("I do not actually remember reading the
    # book") could be recalled and repeated as fact.
    #
    # So episodes take at most half the slots WHEN THERE IS MATERIAL TO TAKE
    # THE REST. A purely conversational question, where nothing else matched,
    # is unaffected — the cap cannot invent material that does not exist.
    eps = [r for r in out if (r["source"] or "") == "episode"]
    oth = [r for r in out if (r["source"] or "") != "episode"]
    if eps and oth:
        # ⚠️ A CEILING, NOT A FLOOR. Written first as "take the best `cap`
        # episodes, then material", which FORCED an episode into every result
        # — book content fell 14 to 10 across the same ten questions, because
        # a slot three book notes had earned was handed to a past turn. Only
        # the EXCESS is trimmed: if episodes have not taken more than half,
        # nothing is touched.
        cap = max(1, k // 2)
        top = out[:k]
        if sum(1 for r in top if (r["source"] or "") == "episode") > cap:
            keep_ids = {id(r) for r in
                        [r for r in top if (r["source"] or "") == "episode"][:cap]}
            picked = [r for r in top
                      if (r["source"] or "") != "episode" or id(r) in keep_ids]
            spare = [r for r in oth if id(r) not in {id(x) for x in picked}]
            picked += spare[:k - len(picked)]
            rest = [r for r in out if id(r) not in {id(x) for x in picked}]
            out = sorted(picked, key=lambda r: -r["score"]) + rest
    if unreadable and not out:
        raise vault.Locked(
            f"{unreadable} memories are sealed and this process holds no key — "
            f"that is why nothing was found, not because nothing matched")
    return out[:k]


def search(query, k=5, db_path=None, sources=None):
    """Find memories containing the words in `query`. Lexical, exact, instant.

    Scored by how many DISTINCT query words a memory contains, weighted by
    rarity, then by how recent it is. Deliberately not fuzzy: "on demand"
    means the asker already knows a word that is in the thing, and a search
    that quietly returns near-misses is how you end up confidently reading out
    the wrong memory.

    Returns a list of dicts: note_id, body, source, hits, score.

    ⚠️ THE EXACT WORDS ARE SEARCHABLE, AND THAT TAKES DELIBERATE WORK.
    the user raised it. Measured on the real store:

        words in a verbatim but not in its gist      34%
        notes reachable by a rare verbatim-only word  1/20
        the same notes reachable by a gist word      19/20   (control)

    `verbatim` was selected, decrypted lazily AFTER scoring, and attached to
    whatever had already matched — so the exact words could only be read off a
    memory found some other way. Asking for what was actually SAID, rather
    than how Mati summarised it, found nothing. Write-only memory.

    Now `_search_pass(deep=True)` scores verbatim-only words at
    `VERBATIM_WEIGHT`, which took that 1/20 to 20/20 with the control
    unchanged. See the sweep above the constant.

    ⚠️ ALWAYS ON, NOT A FALLBACK, AND THE FALLBACK WAS TRIED FIRST. Running it
    only when the gist pass found NOTHING left the measurement at 0/12,
    because a rare word still matches *something* — just not the right thing.
    A wrong memory confidently returned is worse than an empty result, so
    "found something" is not evidence the deep pass was unnecessary. Any
    score threshold instead of "empty" would fire on ordinary queries anyway.

    ⚠️ THE COST IS DECRYPTION, AND IT IS PAID AT STARTUP. Reading every
    verbatim is 23.9s cold and 0.002s warm; measured per query, deep is
    0.53s against plain's 0.49s. `OlinService` warms it with a deep search
    on boot, for exactly the reason it already warmed the body cache: this is
    a cost Mati should pay, never the person in their first sentence.
    """
    return _search_pass(query, k=k, db_path=db_path, sources=sources,
                        deep=True)


def when_said(stored, now=None):
    """When a memory was made, in words a person would use.

    ⚠️ A RECALLED MEMORY MUST CARRY ITS TIME. Without it `search`
    returned body, source, hits and score, so Mati was handed the content of a
    memory with nothing about when it happened — and then asked what day it
    was, or what came first. It was not failing to work that out; it had never
    been told. This is the missing half.

    Local, because the stored column is UTC and nobody thinks in UTC. Relative
    while relative is meaningful, absolute once it is not — "yesterday" is
    useful, "412 days ago" is not.
    """
    from datetime import datetime, timedelta
    if not stored:
        return None
    try:
        at = datetime.fromisoformat(stored).astimezone()
    except (TypeError, ValueError):
        return None                 # unreadable is unreadable — do not guess
    ref = datetime.now().astimezone() if now is None else now
    delta = ref - at
    if delta < timedelta(0):
        return f"{at:%-d %B}"       # clock skew; state the date, claim nothing
    if delta < timedelta(minutes=90):
        return "a little earlier"
    if at.date() == ref.date():
        return f"earlier today, around {at:%-I%p}".replace("AM", "am").replace("PM", "pm")
    if at.date() == (ref - timedelta(days=1)).date():
        return f"yesterday, around {at:%-I%p}".replace("AM", "am").replace("PM", "pm")
    if delta < timedelta(days=7):
        return f"on {at:%A}"
    if at.year == ref.year:
        return f"on {at:%-d %B}"
    return f"in {at:%B %Y}"


def search_sources(query, k=3, db_path=None):
    """Search the SOURCE TEXTS — the lookup table, not the memory.

    ⚠️ A SEPARATE ACT FROM REMEMBERING, and deliberately so. What Mati recalls
    is what it knows; what it reads here is what was actually written. A person
    checks a book to confirm a memory rather than recalling the page — and the
    two must not be the same operation, or the memory is just a quotation with
    extra steps.

    Never reaches the prompt on its own. It is invoked to CONFIRM, which is why
    it returns the passage with the sentence it was learned into: the point is
    the comparison, not the text.
    """
    db_path = _db(db_path)
    want = {schedule._stem(w)
            for w in re.findall(r"[a-z0-9'\-]+", (query or "").lower())
            if len(w) > 1 and w not in SEARCH_STOP}
    if not want:
        return []
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT note_id, body, source, verbatim FROM notes "
            "WHERE verbatim IS NOT NULL").fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    idf = _idf(db_path)
    out = []
    unreadable = 0
    for note_id, body, source, raw in rows:
        try:
            text = _open_cached(raw)
            knew = _open_cached(body)
        except Exception:
            # ⚠️ A LOCKED VAULT LOOKED LIKE A DOCUMENT THAT DOES NOT MENTION
            # THE WORD. `search` was given this guard and this
            # function, added later, never got it: every row failed to
            # decrypt, every one was skipped, and `/source tlamatinime`
            # answered "no source text holds 'tlamatinime'" over a book that
            # holds it on many pages. `/recall` said "sealed — /unlock" from
            # the same locked vault, so the two lookups disagreed about
            # whether the store existed.
            #
            # A false negative dressed as an answer is worse than an error,
            # because it is the answer Mati would go on to speak from.
            unreadable += 1
            continue
        have = {schedule._stem(w)
                for w in re.findall(r"[a-z0-9'\-]+", (text or "").lower())}
        hits = want & have
        if not hits:
            continue
        num = sum(idf.get(w, _IDF_MAX) for w in hits)
        den = sum(sorted((idf.get(w, _IDF_MAX) for w in want),
                         reverse=True)[:3]) or 1.0
        out.append({"note_id": note_id, "source": source,
                    "verbatim": text, "learned": knew,
                    "hits": sorted(hits), "score": num / den})
    out.sort(key=lambda r: -r["score"])
    if unreadable and not out:
        raise vault.Locked(
            f"{unreadable} source texts are sealed and this process holds no "
            f"key — that is why nothing was found, not because nothing "
            f"matched")
    return out[:k]


# ─────────────────────────────────────────────────────────────────────────────
# Reading the log back: say that again
# ─────────────────────────────────────────────────────────────────────────────
#
# ⚠️ WHY THIS IS A READ AND NOT A TURN. The user: *"the text in the response
# window is getting cut off. Can you put in a method that will immediately
# repeat the last response when asked. Requesting from the input line always
# triggers a new turn that then fails to answer the question properly. This was
# supposed to be the reason for the log being readable."*
#
# They are right about the mechanism, not just the annoyance. "Say that again"
# sent through the ordinary path is a NEW question: the field addresses it, the
# model generates, and what comes back is a fresh answer to a different
# question — so the one thing the person asked for, the words that were
# already on screen, is the one thing they cannot get. The log holds them
# exactly. This hands them back verbatim, with no model call and no new
# episode.
#
# ⚠️ AND IT LIVES HERE, in the notes module, because the user said so and
# because they are right about that too: this is retrieval of a stored exchange,
# which is what this module is: the notes are the conversation.

_REPEAT = re.compile(
    r"^\s*(?:"
    r"(?:please\s+)?(?:say|repeat|show|print|display|send)\s+"
    r"(?:that|it|this|the\s+(?:last|previous|final)\s+"
    r"(?:one|reply|response|answer|message)?)\s*(?:again|once more|back)?"
    r"|(?:say|repeat)\s+again"
    r"|again,?\s*please"
    r"|what\s+(?:did\s+you\s+(?:just\s+)?say|was\s+that)"
    r"|(?:i\s+)?(?:missed|didn'?t\s+(?:see|get|catch|read))\s+(?:that|it|the)"
    r"|(?:that|it)\s+(?:got\s+)?cut\s*off"
    r"|(?:the\s+)?(?:last|previous)\s+(?:reply|response|answer|message)"
    r")\s*[.!?]*\s*$",
    re.I | re.X,
)


def asks_repeat(text_in):
    """Are they asking for the last reply again, and nothing else?

    ⚠️ ANCHORED AT BOTH ENDS, DELIBERATELY. "Say that again" is a request to
    re-read; "say that again but in Spanish" and "what did you say about the
    calmecac" are new questions that happen to contain the words. Replaying
    the log at those would withhold an answer the person actually wanted,
    which is a worse failure than the one this fixes.
    """
    return bool(_REPEAT.match(text_in or ""))


# The reply Mati last gave, in memory, per session.
#
# ⚠️ THE STORE IS TOO SLOW FOR THIS, AND IT FAILS IN THE EXACT CASE THE
# FEATURE EXISTS FOR. Episodes are written by the WORKER, not by the turn:
# , the count stayed at 677 immediately after a reply and
# reached 678 about forty seconds later. So "say that again", typed straight
# after a reply got cut off, read the store and handed back the PREVIOUS
# answer — confidently, with no sign anything was wrong. Which is worse than
# the problem it was built to fix.
#
# So the live reply is registered here the moment the turn finishes, and the
# store is the fallback for a fresh process.
_LAST_REPLY = {}


def note_last_reply(text, session="default"):
    """Register what Mati just said, so a replay does not wait on the worker."""
    if text and text.strip():
        _LAST_REPLY[session or "default"] = text.strip()


def forget_last_reply(session=None):
    """Drop the register — for tests, and when a session ends."""
    if session is None:
        _LAST_REPLY.clear()
    else:
        _LAST_REPLY.pop(session, None)


def last_reply(db_path=None, skip=0, session="default"):
    """The exact words Mati last said, or None.

    Read from `verbatim`, which holds the whole exchange as
    "They said: …\\nI said: …" — the stored `body` is a 300-character gist and
    is not what was on screen. Falls back to the body only when there is no
    verbatim, which is true of episodes written before that column existed.
    """
    # ⚠️ THE LIVE REGISTER FIRST. Only `skip=0` can use it — asking for the
    # one before that is a question about the log, and the log is the store.
    if not skip:
        live = _LAST_REPLY.get(session or "default")
        if live:
            return live
    db_path = _db(db_path)
    if not db_path or not os.path.exists(db_path):
        return None
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT verbatim, body FROM notes WHERE source='episode' "
            "ORDER BY note_id DESC LIMIT ?", (skip + 1,)).fetchall()
    except sqlite3.Error:
        return None
    finally:
        conn.close()
    if len(rows) <= skip:
        return None
    verb, body = rows[skip]
    for stored in (verb, body):
        if not stored:
            continue
        try:
            text = _open_cached(stored)
        except Exception:
            continue
        if not text:
            continue
        # The "I said:" half is the reply. ⚠️ `rpartition`, not `split`: the
        # person's own words can contain the phrase, and taking the FIRST
        # occurrence would hand back part of their message as Mati's.
        head, sep, said = text.rpartition("I said:")
        out = (said if sep else text).strip()
        if out:
            return out
    return None
