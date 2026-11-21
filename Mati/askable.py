"""
May Mati ask this? The gate on its own questions, outside the model.

    python3 -m Mati.askable --audit     # score the queue, change nothing
    python3 -m Mati.askable --purge     # retire everything that fails

⚠️ WHY THIS EXISTS. A question for the person is meant to be a sincere inquiry
— something only they can answer — and a generator writing from passages
produces mostly general knowledge the 3B could answer itself, with no
second-person reference at all. The worst are not merely answerable, they are
impossible:

    "Did you live with Hernán Cortés?"
    "Did you live with the Aztec priests on the day of the summer solstice?"
    "Did you live through a medical trial with vitraximab, also known as
     Sareptra?"

⚠️ THOSE TWO FAILURES HAVE ONE CAUSE, which is what makes a single check
possible. A generator that writes with a passage in front of it and applies a
"did you / were you present" template to whatever the passage is about turns
material Mati READ into a question about the person's life. Cortés is in the
store, in a book. `vitraximab` is in the store nowhere at all; it was invented
mid-sentence.

So the test is GROUNDING, not phrasing:

  1. it has to be addressed to the person at all;
  2. what it names has to exist in the store;
  3. if it asks what the person WITNESSED, what it names has to come from a
     conversation they were in — not from something Mati read.

⚠️ AND IT IS NOT ANOTHER BLOCKLIST. Closed lists against open sets fail:
`inquiry._DANGLING` enumerates nouns, so "that change" and "that project" walk
through, and a list of phrasings for denying memory is beaten by changing a
pronoun. Rules 2 and 3 are lookups against what Mati actually holds, which
cannot be out-phrased.

⚠️ THIS IS THE BACKSTOP, NOT THE FIX. The cause is upstream: a question
sourced from `studied:asked about …` — Mati's own 3B answering its own
questions — is a 3B writing a question about a 3B's hallucination. The source
pool is lived episodes only (`bridge._ASK_PRIORITY`), and the prompt asks for
what the other person knew (`inquiry.ASK_USER_SYSTEM`).

⚠️ AND IT LEAKS, KNOWINGLY. A probe asking whether Mati was PRESENT at a
moment in a word's doctrinal history passes when that word genuinely is a topic
the person has discussed. Grounding cannot tell a subject somebody talks about
from a historical event nobody witnessed, when the same word names both; that
needs meaning, not membership. The generator is what stops that question being
written at all. Do not try to close this by adding terms — that is the failure
described above.
"""

from __future__ import annotations

import argparse
import re
import sys

# Asking what someone personally witnessed. ⚠️ THE ONLY PLACE A PHRASE LIST IS
# USED, and it is used to decide which RULE applies rather than to accept or
# reject — a missed phrasing here downgrades the check to "must be grounded
# somewhere" instead of letting a bad question through.
_WITNESSED = re.compile(
    r"\b(?:"
    r"were you (?:there|present|around|at)"
    r"|did you (?:witness|see|observe|experience|live|attend|visit|meet|"
    r"participate|notice|feel|hear)"
    r"|(?:which|what) moment did you"
    r"|did you live (?:with|through|at|in|a)"
    r")\b",
    re.I,
)

_SECOND = re.compile(r"\b(?:you|your|yours|yourself)\b", re.I)

# Words that carry no subject. Everything else in a question is a candidate
# term to look up.
_EMPTY = {
    "the", "a", "an", "of", "to", "in", "on", "it", "its", "and", "or", "that",
    "this", "these", "those", "for", "with", "as", "at", "by", "be", "not",
    "about", "me", "my", "you", "your", "i", "we", "us", "our", "they", "them",
    "their", "what", "when", "where", "which", "who", "whom", "how", "why",
    "did", "does", "do", "was", "were", "is", "are", "am", "been", "being",
    "have", "has", "had", "will", "would", "could", "should", "can", "may",
    "might", "must", "shall", "any", "some", "all", "more", "most", "other",
    "such", "no", "nor", "only", "own", "same", "so", "than", "too", "very",
    "just", "ever", "still", "yet", "also", "there", "here", "then", "now",
    "specific", "particular", "actual", "real", "thing", "things", "moment",
    "moments", "time", "times", "way", "ways", "kind", "sort", "like",
    "mean", "means", "happen", "happened", "happens", "tell", "told", "say",
    "said", "remember", "recall", "know", "think", "feel", "felt", "between",
    "during", "after", "before", "while", "from", "into", "over", "under",
    "person", "people", "someone", "anyone", "something", "anything",
}

# ⚠️ THIS FILE'S OWN PATTERN VOCABULARY, EXCLUDED FROM GROUNDING. `live` is
# itself in the store — it occurs in many notes — so without this "Did you live
# with Hernan Cortes?" would be grounded by its own template word and the
# subject never tested. These words come from `_WITNESSED` above, so this
# excludes what Mati wrote rather than guessing at what the model might write.
_TEMPLATE = {
    "live", "lived", "living", "present", "presence", "witness", "witnessed",
    "see", "saw", "seen", "observe", "observed", "experience", "experienced",
    "attend", "attended", "visit", "visited", "meet", "met", "participate",
    "participated", "notice", "noticed", "hear", "heard", "there", "around",
    "day", "days", "date",
}

# How many of a question's terms must be grounded. ⚠️ NOT ALL OF THEM: a
# question can legitimately introduce an ordinary word Mati has never stored
# ("Did you find the Oaxaca trip tiring?" — "tiring" need not be on record).
# One solid anchor is what separates a question ABOUT something held from a
# question assembled out of nothing.
MIN_GROUNDED = 1


class Verdict:
    """ok, plus why not. ⚠️ The reason is kept so a purge can be AUDITED."""

    def __init__(self, ok, reason="", terms=()):
        self.ok, self.reason, self.terms = ok, reason, tuple(terms)

    def __bool__(self):
        return self.ok

    def __repr__(self):
        return f"Verdict({self.ok}, {self.reason!r})"


def terms(question, db_path=None):
    """The content words of a question, RAREST FIRST.

    ⚠️ RAREST, NOT LONGEST, AND THE DIFFERENCE DECIDES EVERY HARD CASE. Sorted
    by length, "Did you live with the Aztec priests on the day of the summer
    solstice?" offered `priests` and `solstice` alongside `Aztec` — and
    `Aztec` is in real conversation, so it grounded the question and the
    absurd part was never tested. The same way, `medical` and `trial` grounded
    "a medical trial with vitraximab".

    `notes._idf` is the store's own rarity table, and the rule is the one
    `notes.search` follows too: a question is about its rarest words.
    """
    from . import schedule
    # ⚠️ ACCENTS INCLUDED. `[A-Za-z]` split "Hernán" into "Hern" + "n", so the
    # subject of a question about Cortés was a fragment. In a store whose
    # subject matter is Mesoamerican and Spanish, dropping accented letters
    # mangles exactly the names that matter most.
    words = re.findall(r"[^\W\d_][^\W\d_'\-]{2,}", question or "",
                       re.UNICODE)
    out, seen = [], set()
    for w in words:
        low = w.lower()
        if low in _EMPTY or low in _TEMPLATE or low in seen:
            continue
        seen.add(low)
        out.append(w)
    try:
        from . import bridge, notes
        idf = notes._idf(db_path or bridge.NOTES_DB_PATH)
        mx = notes._IDF_MAX
        return sorted(out, key=lambda w: -idf.get(schedule._stem(w.lower()), mx))
    except Exception:
        return sorted(out, key=len, reverse=True)


def build_index(db_path=None):
    """{stem: set of source kinds it appears in}. One pass over the store.

    ⚠️ NOT `notes.search`. Lexical search returns the best k rows whether or
    not anything matched, so "did a search for this term return anything"
    says yes for `vitraximab` (one hit, score 1.0000) and for `Cortés` — an
    invented drug and a name that is nowhere in the store would both read as
    grounded, and every question this gate exists to stop would walk through.

    ⚠️ Search is RANKING, not membership. It has no empty answer to give
    except when a term tokenises to nothing at all ("zzzqqqwww" -> 0 hits),
    which is why the error looks plausible. Membership needs a membership
    test.

    Decryption rides the warm body cache, so this is a few seconds on a store
    whose verbatims the idle worker has already opened.
    """
    import sqlite3
    from . import bridge, notes, schedule, vault
    db_path = bridge.NOTES_DB_PATH if db_path is None else db_path
    index = {}
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT body, source FROM notes").fetchall()
    except sqlite3.Error:
        return index
    finally:
        conn.close()
    for stored, source in rows:
        try:
            body = notes._open_cached(stored)
        except Exception:
            continue
        # Episode = something the person was part of. Everything else —
        # studied passages, self-asked answers, the profile — is material
        # Mati came by on its own, and is exactly what a "were you there"
        # question must NOT be built on.
        kind = "episode" if source == "episode" else "read"
        for w in re.findall(r"[a-z0-9'\-]+", (body or "").lower()):
            if len(w) < 3:
                continue
            index.setdefault(schedule._stem(w), set()).add(kind)
    return index


def _found(term, index, episode_only=False):
    """Is `term` in the store — and if asked, in a LIVED episode?"""
    from . import schedule
    kinds = index.get(schedule._stem(term.lower()))
    if not kinds:
        return False
    return "episode" in kinds if episode_only else True


def check(question, index, db_path=None):
    """May Mati ask this? `Verdict(ok, reason)`.

    ⚠️ ORDER MATTERS, CHEAPEST AND MOST DECISIVE FIRST. The second-person test
    costs nothing and removes four fifths of the queue, so nothing else runs
    on a question that was never for the person in the first place.
    """
    q = (question or "").strip()
    if not q:
        return Verdict(False, "empty")
    if len(q.split()) < 4:
        return Verdict(False, "too short to be a real question")

    # 1. For the person at all? ⚠️ THE 80% CASE. "What is the name of the
    # regulatory framework used by the UK?" is a quiz question; Mati's own
    # model answers it, and the recorded rule for this queue is to log only
    # what the model cannot.
    if not _SECOND.search(q):
        return Verdict(False, "not addressed to the person — the model can "
                              "answer this itself")

    ts = terms(q, db_path)
    if not ts:
        return Verdict(False, "names nothing in particular")

    # 2. Does it ask what the person WITNESSED? Then what it names must come
    # from a conversation, not from Mati's reading. ⚠️ THIS IS THE CORTÉS
    # RULE. "Did you live with Hernán Cortés?" names something real and
    # held — in a book. Lived-experience questions about studied material are
    # a category error, and it is the single largest source of the nonsense.
    if _WITNESSED.search(q):
        # ⚠️ EVERY DISTINCTIVE TERM, NOT JUST ONE. A claim that the person was
        # PRESENT for something has to be wholly about material they were in —
        # one grounded word alongside an invented one is how "a medical trial
        # with vitraximab" passed on the strength of `medical`, and how "the
        # Aztec priests on the day of the summer solstice" passed on `Aztec`.
        #
        # So the three rarest terms are each checked, and any of them being
        # absent from conversation sinks the question. This is the strict side
        # of a deliberate asymmetry: a presence claim is the one shape that
        # turns a passage Mati READ into a false memory of the person's life,
        # so it pays the higher burden. Ordinary questions below do not.
        for t in ts[:3]:
            if _found(t, index, episode_only=True):
                continue
            if _found(t, index):
                return Verdict(False, "asks what they witnessed, about "
                                      f"something only read: {t!r}", ts)
            return Verdict(False, "asks what they witnessed, about nothing "
                                  f"on record: {t!r}", ts)
        return Verdict(True, "grounded in conversation", ts[:3])

    # 3. An ordinary question only has to be ABOUT something Mati holds.
    # ⚠️ ANY TERM, AND DELIBERATELY NOT THE RAREST ONE. Keyed on the rarest,
    # this rejected "What made you want to retell the Bible?" because `retell`
    # has never been stored — a novel VERB outranking the subject noun. The
    # subject was `Bible`, which is on record, and the question is a good one.
    # The fabrication cases this is meant to catch have no grounded term at
    # all, so "at least one" still stops them.
    grounded = [t for t in ts if _found(t, index)]
    if not grounded:
        return Verdict(False, f"nothing it names is on record: {ts[:3]}", ts)

    # 4. The existing dangling-referent check, reused rather than reimplemented.
    try:
        from .inquiry import _DANGLING
        if _DANGLING.search(q):
            return Verdict(False, "refers to something it does not name", ts)
    except Exception:
        pass

    return Verdict(True, "grounded", grounded)


def audit(db_path=None, limit=None):
    """Score every queued question. Changes nothing."""
    from . import schedule, vault
    if not vault.unlocked():
        vault.unlock(reason="audit Mati's queued questions")
    items = schedule.pending(include_held=True, db_path=db_path) or []
    index = build_index(db_path)
    rows = []
    for i in items[: limit or len(items)]:
        body = (i.get("body") or "").strip()
        if not body:
            continue
        rows.append((i.get("id"), body, check(body, index, db_path)))
    return rows


def purge(db_path=None, dry_run=False, limit=None, everything=False):
    """Retire every queued question that fails the gate — or all of them.

    ⚠️ `everything=True` IS FOR A BACKLOG WRITTEN FROM THE WRONG SOURCE.
    Questions written from `studied:asked about` passages — Mati's own model's
    output — can pass the gate, because they contain "you" and name something
    on record, and still be unanswerable:

        "Can you tell me what is the nature of the 'specific, identifiable
         genetic disorders' that CRISPR can..."

    Asking the person to vouch for those is the laundering this exists to
    stop, and without a recorded source note per question there is no way to
    tell which ones they are. The generator refills from lived episodes, so
    such a backlog is cheaper to drop than to sift.
    """

    from . import schedule
    rows = audit(db_path=db_path, limit=limit)
    # ⚠️ MARKED DELIVERED, NOT DELETED, AND THE DISTINCTION IS DELIBERATE.
    # These are Mati's own questions; `forget()` is the one real delete in
    # this system and it is for the person's memories, not for machinery.
    # Marking them delivered takes them out of the queue while leaving the
    # row — so the purge is itself auditable and a mistake in the gate can be
    # read back rather than guessed at.
    bad = rows if everything else [r for r in rows if not r[2].ok]
    if not dry_run and bad:
        schedule.mark_delivered([i for i, _q, _v in bad],
                                session="purged:askable", db_path=db_path)
    return rows, bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--purge", action="store_true",
                    help="retire the questions that fail")
    ap.add_argument("--audit", action="store_true",
                    help="score them and change nothing")
    ap.add_argument("--all", action="store_true",
                    help="retire the whole backlog, not only what fails")
    ap.add_argument("--show", type=int, default=6,
                    help="examples to print per reason")
    a = ap.parse_args(argv)
    if not (a.purge or a.audit):
        a.audit = True

    rows, bad = purge(dry_run=not a.purge, everything=a.all)
    keep = [r for r in rows if r[2].ok]
    import collections
    why = collections.Counter(v.reason.split(":")[0] for _i, _q, v in bad)

    print(f"\n  {len(rows)} queued   {len(keep)} pass   {len(bad)} fail")
    if rows:
        print(f"  {100.0 * len(keep) / len(rows):.0f}% survive the gate\n")
    for reason, n in why.most_common():
        print(f"  {n:4}  {reason}")
        shown = [q for _i, q, v in bad if v.reason.startswith(reason)][:a.show]
        for q in shown:
            print(f"          {q[:92]}")
    print(f"\n  ── kept ({len(keep)}) ──")
    for _i, q, _v in keep[:a.show * 2]:
        print(f"      {q[:92]}")
    if a.purge:
        print(f"\n  ⚠️ retired {len(bad)} questions "
              f"(marked delivered, rows kept for audit)")
    else:
        print("\n  (--audit: nothing changed. Use --purge to retire them.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
