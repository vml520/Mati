"""
One thing Mati may add after it has answered, chosen by what matters most.

The user: "build the segue after generation, filtered by importance to user.
Urgent information — something Mati knows the user would want to know right
away — should trump current context."

AFTER GENERATION, DELIBERATELY. A block sent before generation becomes material
the model writes *from*, and it will work the offer into the answer whether or
not it belongs there. Composed afterwards it is a separate sentence that can be
dropped whole, which is what makes "silence is the default" enforceable rather
than aspirational.

⚠️ URGENCY IS NOT INFERRED, AND TODAY IT CANNOT BE. "Something Mati knows the
user would want to know right away" is only honestly true of things the person
ASKED to be told — a reminder they set. Mati cannot yet look at 324 episodes
and work out what would matter to someone; `predict.may_influence()` is the gate
for ever letting it try, and it is False (0 predictions resolved of the 10 it
needs). So tier 1 is recorded intent, not judgement. When the gate opens, that
is the place to widen it, and widening it is a deliberate act.

THE TIERS, highest first:

  1. URGENT      a reminder that has come due. Already exempt from the topic
                 gate in `schedule.pending`, which is precisely what "trumps
                 current context" means, and it was already true.
  2. WAITING     a finding whose subject the person has just raised.
  3. CONNECTION  something held that this turn touches. The segue proper.

⚠️ COMPOSED, NOT GENERATED. The line is built from the stored text, not written
by the model. A generated segue is an invitation to invent a connection, and the
one invariant this must keep is that every offer can produce the note it came
from. Determinism is cheaper than a guard against confabulation, and stronger.

⚠️ IT JUDGES WHAT SURFACED. IT DOES NOT GO LOOKING. The user, on the first
build: "That is not what humans do. We don't look at every memory to find
relevance. The relevance brings up the memory. Then we choose if the memory is
of any consequence. All of this happens in milliseconds."

They are describing the architecture this whole project rests on, and the first
version violated it. It ran its own `bridge.search` over the store — a scan, to
compute relevance that the field had ALREADY computed earlier in the same turn.
: `bridge.search` costs 27.4s cold and 254ms warm, and
`choose()` added ~965ms on top by re-running the POS tagger once per candidate.
Roughly a second a turn, to redo work that had been done for free.

Surfacing fires once at the top of every turn, before the reply exists. THAT is
"the relevance brings up the memory". What is left for this module is the second
half — CONSEQUENCE: of the few things that came up, is any of them worth saying?
So `choose()` takes what surfaced and never queries. Nothing surfaced means
silence, which is the honest answer.

⚠️ WHICH CHANNEL SURFACES THEM CHANGED, AND THE ARCHITECTURE DID NOT. This first
consumed `bridge.recall`, the FIELD path, because that is where surfacing was
supposed to happen. Measured  on a validated harness: field 4/111 =
0.036 against chance 0.009, lexical 93/111 = 0.838. The field channel is at
chance, so a segue built on it offers near-random connections with a citation
attached — which is worse than none, because the citation makes it look
checked.

It now consumes the lexical surfacing that `turn()` already performs. Still one
surfacing, still before judgement, still never a query of its own. Only the
channel is different. See `ARCHITECTURE.md`.

⚠️ THE STUDY IS BUILT IN. Its design requires the arm to be
assigned and logged AT OFFER TIME — deciding afterwards which segues "were
really apposite" is the freedom preregistration removes. `offer()` therefore
takes an arm and records it, and the control arm is a real, held, topically
DISTANT memory rather than no segue at all: a polite person follows almost any
offer, so take-up without a matched control measures manners, not relevance.
"""

from __future__ import annotations

import hashlib
import random
import re
import sqlite3
import time

from . import bridge, schedule

URGENT = "urgent"
WAITING = "waiting"
CONNECTION = "connection"

APT = "apposite"
DISTANT = "matched-distant"

TABLE = "segue_offers"

# ⚠️ RARITY IS NOT ABOUTNESS, AND HERE IT INVERTS. Written first as "top
# tercile by IDF", measured  on "I was reading about the calmecac
# last night":
#
#     calmecac   idf 5.24     <- the subject, and BELOW the 7.84 tercile floor
#     stuck      idf MAX      <- absent from the store, so it ranked FIRST
#
# A word missing from the store gets `_IDF_MAX` by fallback. For a CONNECTION
# that is self-defeating: the term has to appear in a held note for there to be
# anything to connect to, so ranking by absence selected precisely the words
# that cannot connect. The filter could only ever fire on coincidence.
#
# This is the second time on this codebase — `_focus` picked "tweaked" and
# "let'" over "calmecac" for the same reason.
#
# ⚠️ AND FOR A CONNECTION, RARITY IS BACKWARDS ALTOGETHER. Replacing the
# tercile with "above median IDF" still returned nothing, because the store's
# IDF distribution is dominated by words appearing in ONE note:
#
#     median  7.44        max 7.84 (= ln(3090/1), a word seen once)
#     calmecac    5.24    tlamatinime  4.07
#
# Both real subjects sit BELOW the median. A word is IDF-rare because few notes
# hold it — but a connection needs a topic Mati holds a LOT about, so any IDF
# floor excludes exactly the subjects it should be built on. For RETRIEVAL
# rarity discriminates; for CONNECTION it inverts. Same number, opposite job.
#
# So there is no rarity floor here. Store-presence says the term is connectable
# and the tagger says it is the subject, which is the whole test. IDF is kept
# only to order what survives.
MIN_SHARED_TERMS = 1
# Long enough to be a topic, short enough not to be a second answer.
EXCERPT = 140


def ensure(db_path=None):
    """Where offers are logged, so take-up can be measured against them."""
    conn = sqlite3.connect(db_path or bridge.NOTES_DB_PATH, timeout=10)
    try:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {TABLE} (
                    offer_id  INTEGER PRIMARY KEY AUTOINCREMENT,
                    at        TEXT,
                    tier      TEXT,
                    arm       TEXT,
                    note_id   INTEGER,
                    say_id    INTEGER,
                    terms     TEXT,
                    took_up   INTEGER
                )""")
        conn.commit()
        return True
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def _rare_terms(text, db_path=None):
    """What `text` is about, in terms the store could actually share.

    Two filters, in this order and for different reasons:

      IN THE STORE. A stem with no IDF entry appears in nothing Mati holds,
      so it cannot be the basis of a connection at all.
      AND A NOUN. What a turn is about is a noun, which is what separates
      `calmecac` from `about` — not rarity, which here points the wrong way.
      See the note above `MIN_SHARED_TERMS`.
    """
    idf = bridge._idf(db_path)
    if not idf:
        return []
    try:
        nouns = bridge._nouns(text)
    except Exception:
        nouns = None
    seen, out = set(), []
    for w in re.findall(r"[a-z0-9'\-]+", (text or "").lower()):
        st = schedule._stem(w)
        if st in seen or len(w) < 4 or w in bridge.SEARCH_STOP:
            continue
        seen.add(st)
        if idf.get(st) is None:
            continue                # in nothing Mati holds — cannot connect
        if nouns and st not in nouns and w not in nouns:
            continue                # not what the turn is about
        out.append(st)
    return sorted(out, key=lambda s: -idf.get(s, 0.0))


def _stems(text):
    """Cheap stems. No tagger — candidates do not need parsing, only matching.

    The tagger is run ONCE, on the person's turn, to find what it is about.
    Running it per candidate is what made the first version cost ~965ms.
    """
    return {schedule._stem(w)
            for w in re.findall(r"[a-z0-9'\-]+", (text or "").lower())
            if len(w) > 3 and w not in bridge.SEARCH_STOP}


def _speakable(body):
    """May this note be offered aloud at all?

    ⚠️ A SEGUE IS A NEW DOOR FOR A LEAK ALREADY CLOSED ONCE. Mati's own notes
    include its progress narration — "I have been reading Aztec Thought and
    Culture.txt. I have taken in 1759 passages" — which the matched-distant arm
    selected on its first run. Offered as a connection that is Mati reciting
    its instrumentation at someone, which `admit.py` and `voice.py` block on
    the way in and out and neither sees this path.

    Rejected rather than cleaned. A note that needs editing before it can be
    said is not the note to offer when silence is free.
    """
    low = " ".join((body or "").split()).lower()
    if not low:
        return False
    if any(t in low for t in bridge._INTERNAL_TERMS):
        return False
    if re.search(r"\.(txt|md|pdf|epub)\b", low):
        return False           # a filename in a segue is machinery leaking
    if re.search(r"\b\d{2,}\s+(passages|chunks|notes|episodes|memories)\b", low):
        return False           # counts of its own work
    if body.strip().startswith(bridge._WORK_OPENERS):
        return False           # "It took a while to find the words…"
    return True


def _connection(user_text, reply, surfaced, arm, db_path=None):
    """Is any of what SURFACED worth saying? Consequence, not relevance.

    `surfaced` is what the turn already retrieved — `notes.search` results, as
    dicts with `note_id`, `body` and `source`. This function only decides
    whether one of them is of consequence:

      it can be said at all          `_speakable`
      it is not what was just said   the answer already covered it
      it is on the subject           shares a noun with the turn

    ⚠️ THE CONTROL ARM IS DRAWN FROM THE SAME SURFACED SET. It is still real,
    still held, still something the field brought up — just the one that shares
    least with the subject. Both arms therefore see the same candidates and the
    only variable is which is chosen.
    """
    if not surfaced:
        return None                 # nothing came up; there is nothing to judge
    want = set(_rare_terms(user_text, db_path))
    if not want:
        return None
    said = _stems(reply or "")

    scored = []
    for item in surfaced:
        # Dicts from `notes.search`; 4-tuples from `bridge.recall(with_ids=True)`
        # are still accepted so a field-surfaced caller is not silently dropped.
        if isinstance(item, dict):
            note_id = item.get("note_id"); body = item.get("body")
            source = item.get("source")
        else:
            try:
                # 5-tuple since the read date was threaded through `recall`;
                # 4 is still accepted so an older caller is not dropped.
                _dist, note_id, body, source = item[:4]
            except (TypeError, ValueError):
                continue            # no id, so nothing that could name a source
        if note_id is None or not body:
            continue
        if not _speakable(body):
            continue
        terms = (want & _stems(body)) - said
        if len(terms) >= MIN_SHARED_TERMS:
            scored.append((len(terms), sorted(terms), note_id, body, source))
    if len(scored) < 2:
        return None
    scored.sort(key=lambda t: -t[0])
    n, terms, note_id, body, source = (scored[-1] if arm == DISTANT
                                       else scored[0])
    return {"tier": CONNECTION, "arm": arm, "note_id": note_id,
            "terms": terms, "body": body, "source": source}


def choose(user_text, reply, surfaced=None, now=None, db_path=None,
           arm=None, rng=None):
    """The one thing worth adding, or None. Silence is the default.

    Returns a dict describing the offer; `line()` renders it. Never raises —
    an afterthought must not be able to cost someone their answer.
    """
    now = now or time.time()
    try:
        # ── 1. URGENT ───────────────────────────────────────────────────────
        # `pending` exempts REMINDER from the topic gate already, so this tier
        # trumps the conversation by construction rather than by a rule here.
        # It also owns quiet hours and per-kind consent; asking it rather than
        # re-deciding is what keeps ONE set of interruption rules.
        due = [i for i in schedule.pending(now=now, db_path=db_path)
               if i.get("kind") == schedule.REMINDER]
        if due:
            i = due[0]
            return {"tier": URGENT, "arm": None, "say_id": i["id"],
                    "note_id": None, "terms": [], "body": i["body"]}

        # ── 2. WAITING ──────────────────────────────────────────────────────
        raised = [i for i in schedule.pending(now=now, db_path=db_path,
                                              user_text=user_text)
                  if i.get("kind") != schedule.REMINDER]
        if raised:
            i = raised[0]
            return {"tier": WAITING, "arm": None, "say_id": i["id"],
                    "note_id": None, "terms": [], "body": i["body"]}

        # ── 3. CONNECTION ───────────────────────────────────────────────────
        # ⚠️ NOT WHEN A DIRECT QUESTION WAS JUST ANSWERED. Following someone's
        # question with a change of subject is how an assistant stops being
        # useful and starts being talkative. The urgent tiers above are exempt
        # on purpose: that is what "trumps current context" buys.
        if schedule.user_is_asking(user_text):
            return None
        if arm is None:
            arm = (rng or random).choice((APT, DISTANT))
        return _connection(user_text, reply, surfaced, arm,
                           db_path=db_path)
    except Exception:
        return None


# ⚠️ THE STORAGE FORMAT IS NOT SPEECH. An episode body is written
# "I was asked: <them>\nI said: <Mati>" (see `ollin.py`), and `line` used to
# quote it whole — so a recollection came out as "That connects to something I
# hold: I was asked: Hello Mati I said: I am here." The narration wrapper is
# bookkeeping; Mati's own half is the thing worth saying.
#
# ⚠️ STILL VERBATIM. This selects a span of the stored text, it does not
# rewrite one — which is what keeps the guarantee below true.
_EPISODE = re.compile(r"\bI was asked:\s*(?P<asked>.*?)\s*"
                      r"\bI said:\s*(?P<said>.*)\s*$", re.S)

# ⚠️ THE CONNECTION TIER IS NO LONGER SPOKEN, so there are no frames for it.
#
# Four were added on 29 September because the user said *"the recollections sound
# canned because they start and end the same"* — one fixed opener and a hard
# character cut. That was the right reading of the symptom and the wrong level
# to fix it at. The user:
#
#     "They are supposed to be passive inputs, to maintain integrity of
#      personhood while providing knowledge gained through experience… the
#      way it is implemented now, in the response, is not correct."
#
# A connection read as canned because **announcing it at all was the mistake**.
# A memory is material an argument is built from, not a sentence appended to
# one. No wording makes a recitation into an input.
#
# ⚠️ THE OFFER STILL HAPPENS AND IS STILL LOGGED. `choose` is unchanged, so
# the field's judgement about what connects is recorded — `info['segue']`
# carries the tier and the arm, and `record()` still writes the row. What is
# gone is Mati saying it out loud. That keeps the study's data collectable;
# see the note at the head of this file.


def _excerpt(body):
    """At most EXCERPT characters, cut at a sentence end where there is one.

    ⚠️ THE "…" WAS THE OTHER HALF OF "THEY END THE SAME". A hard character cut
    lands mid-phrase every time. Ending on the last full sentence inside the
    budget reads as a finished thought, and falls back to the ellipsis only
    when there is no sentence end to use.
    """
    if len(body) <= EXCERPT:
        return body
    window = body[:EXCERPT]
    cut = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
    if cut >= EXCERPT // 3:            # far enough in to still say something
        return window[:cut + 1]
    return window.rstrip() + "…"


def line(offer):
    """The offer as one sentence, built from stored text.

    ⚠️ NOT WRITTEN BY THE MODEL. Every offer must be able to produce the note
    it came from, and the cheapest way to guarantee that is to quote it.
    """
    if not offer:
        return None
    body = " ".join((offer.get("body") or "").split())
    if not body:
        return None

    m = _EPISODE.search(body)
    if m:
        said = " ".join(m.group("said").split())
        if said:
            body = said

    body = _excerpt(body)
    # ⚠️ ONLY SCHEDULED DELIVERIES ARE SPOKEN. Both of these come from
    # `schedule.pending` — things the person ASKED to be told. A reminder that
    # is not said is not a reminder, so these are not recitation and they stay.
    if offer["tier"] == URGENT:
        return f"Before you go — you asked me to tell you: {body}"
    if offer["tier"] == WAITING:
        # ⚠️ NO "Since it came up —" PREFIX. The user: *"it is hardly close to
        # the subject of the conversation."* And they are right about the
        # mechanism, not just the phrasing: this tier is a SCHEDULED delivery
        # from `schedule.pending`, selected because it was DUE, not because it
        # relates to anything being discussed. The prefix claimed a connection
        # to the current topic that was never computed — so it was wrong by
        # construction rather than wrong occasionally.
        #
        # ⚠️ Do not restore it, and do not replace it with another connective
        # ("speaking of", "that reminds me") — every one of them asserts the
        # same relevance this tier cannot establish. A thing the person asked
        # to be told can simply be said.
        return body
    # ⚠️ CONNECTION IS SILENT. It is the associative offer — a memory the field
    # brought up on its own — and those are passive inputs. It reaches the
    # model through `state_block`, where it belongs, and reaches the user through
    # the record. Returning None here is what stops it being appended to the
    # reply as a quotation.
    return None


def record(offer, db_path=None, now=None):
    """Log the offer AT OFFER TIME, with its arm. Returns the offer_id."""
    if not offer:
        return None
    ensure(db_path)
    conn = sqlite3.connect(db_path or bridge.NOTES_DB_PATH, timeout=10)
    try:
        cur = conn.execute(
            f"INSERT INTO {TABLE} (at, tier, arm, note_id, say_id, terms) "
            f"VALUES (?,?,?,?,?,?)",
            (bridge._now(), offer["tier"], offer.get("arm"),
             offer.get("note_id"), offer.get("say_id"),
             " ".join(offer.get("terms") or [])))
        conn.commit()
        return cur.lastrowid
    except sqlite3.Error:
        return None
    finally:
        conn.close()


def took_up(offer_id, next_user_text, db_path=None):
    """Did the person pick the topic up? Mechanical, never a model's opinion.

    ⚠️ `reckon.py` is the standing example of why this is not judged: a model
    asked whether something landed rates its own phrasing generously, and the
    question-filter study was stopped at exactly this control. Take-up is a
    rare term from the offer appearing in the next turn — nothing more.
    """
    conn = sqlite3.connect(db_path or bridge.NOTES_DB_PATH, timeout=10)
    try:
        row = conn.execute(
            f"SELECT terms FROM {TABLE} WHERE offer_id=?", (offer_id,)
        ).fetchone()
        if not row or not row[0]:
            return None
        offered = set(row[0].split())
        got = {schedule._stem(w)
               for w in re.findall(r"[a-z0-9'\-]+",
                                   (next_user_text or "").lower())}
        hit = bool(offered & got)
        conn.execute(f"UPDATE {TABLE} SET took_up=? WHERE offer_id=?",
                     (int(hit), offer_id))
        conn.commit()
        return hit
    except sqlite3.Error:
        return None
    finally:
        conn.close()


def tally(db_path=None):
    """Take-up by arm — the study's readout. Reports N, never a verdict."""
    conn = sqlite3.connect(f"file:{db_path or bridge.NOTES_DB_PATH}?mode=ro",
                           uri=True)
    try:
        rows = conn.execute(
            f"SELECT arm, COUNT(*), SUM(COALESCE(took_up,0)) FROM {TABLE} "
            f"WHERE tier=? AND took_up IS NOT NULL GROUP BY arm",
            (CONNECTION,)).fetchall()
    except sqlite3.Error:
        return {}
    finally:
        conn.close()
    out = {}
    for arm, n, hits in rows:
        out[arm or "?"] = {"n": n, "took_up": hits or 0,
                           "rate": (hits or 0) / n if n else None}
    return out
