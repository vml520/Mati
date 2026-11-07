"""
Mati looking into something on its own.

    python3 -m Mati.inquiry --topic "Mesoamerican archaeology"
    python3 -m Mati.inquiry --status
    python3 -m Mati.inquiry --step 3        # ask now, instead of on idle

WHAT THIS IS, AND WHAT FREE DREAMING IS NOT. Free dreaming (`bridge.dream`) is
what Mati does when idle with nothing to chew on: it generates from its own
ground state and deepens whatever settles coherently. It is undirected by
construction, and measurement showed it actively ERODES a fresh imprint — after
30 cycles a just-imprinted topic had lost more coherence than a topic never
mentioned (-0.0499 against -0.0245). It is the wrong tool for learning a
subject, and it stays where it belongs: idle time.

This is the directed version. Mati takes a topic it has reason to care about —
the interests given at setup — and works on it:

    ask the language model a question about the topic
      -> compare what came back against what is already in memory
      -> keep the parts that are new
      -> let what came back decide the next question

The comparison is not a gate. Retrieval here is lexical, and it was measured
unable to separate known material from new (known 0.424-0.688 vs new
0.305-0.754, overlapping and inverted). So `recall` is used for what it can
actually do — tell Mati what it already holds nearby, so the NEXT question can
go somewhere else. Exact duplicates are caught by hash, same as reading a file.

⚠️ THE HAZARD THIS LOOP CREATES, stated plainly. The memory loop amplifies
errors: what Mati stores is replayed into later turns, and a flawed answer
gets re-parroted. Observed before, from a single bad reply. This loop runs
UNATTENDED and stores model output by design, so it is that hazard with a pump
attached. A 3B model asked open questions about a subject will produce
confident, wrong, plausible sentences.

Three structural mitigations, none of them an instruction:

  * PROVENANCE. Answers are stored as `studied:asked about <topic>` — the same
    fourth kind as material read in a file, never as an episode. Mati can say
    "I looked into that myself" and can never say "I remember when we discussed
    it", because the source column does not permit it.
  * SANITISED ON WRITE. Bodies go through the same filter as reply excerpts;
    anything mentioning the machinery is dropped rather than stored.
  * ABSORBED, NOT ASSERTED. Inquiry only QUEUES material. The study loop
    imprints it, tests whether it stuck, and reindexes — so the cost bound, the
    test-after and the idle budget all still apply.

It does not make the model correct. It makes what the model said traceable to
the model.
"""

import argparse
import os
import re
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .field import notes as N

from . import bridge
from . import exclusive
from . import schedule
from . import study
from . import vault

_DDL = """
CREATE TABLE IF NOT EXISTS inquiry_topics (
    topic_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    topic      TEXT NOT NULL UNIQUE,
    next_q     TEXT,
    asked      INTEGER NOT NULL DEFAULT 0,
    added      TEXT,
    last_asked TEXT,
    status     TEXT NOT NULL DEFAULT 'open'
);
"""

# What Mati is told when it asks. Deliberately narrow: short, factual, and no
# invitation to speculate — the smaller the model, the more an open prompt
# becomes an invitation to invent.
#
# This is prepended to the USER message, not passed as a system prompt. The
# provider's system prompt is fixed when the client is built (Mati's persona),
# and a second one cannot be layered on it. Passing this as `system` silently
# did nothing: the model answered in Mati's voice and never emitted the NEXT
# line, so every step re-asked the opening question and the loop never advanced.
ASK_SYSTEM = (
    "You are answering a question for your own study notes. Reply with three "
    "or four short factual sentences, no preamble and no lists. If you are not "
    "confident about something, leave it out rather than hedging about it. "
    "Then, on a final line by itself, write:\n"
    "NEXT: <one further question about this topic that your answer did not "
    "already cover>"
)

_NEXT = re.compile(r"^\s*NEXT\s*:\s*(.+?)\s*$", re.I | re.M)


def _opening(topic):
    """First question for a topic. "Tell me about" works for singular and
    plural alike — "What is topological defects?" is what the naive form gives."""
    return f"Tell me about {topic}."


def _conn(db_path=None):
    conn = sqlite3.connect(db_path or bridge.NOTES_DB_PATH, timeout=30)
    conn.executescript(_DDL)
    conn.commit()
    return conn


def add_topic(topic, db_path=None, question=None):
    """Queue something for Mati to look into. Idempotent.

    `question` overrides the default opening. Used when a topic arrives with a
    specific thing to settle — a claim Mati has accepted but not verified —
    rather than as a subject to explore from scratch.
    """
    topic = " ".join(topic.split())
    if not topic:
        return False
    conn = _conn(db_path)
    try:
        conn.execute(
            "INSERT INTO inquiry_topics (topic, next_q, added) VALUES (?,?,?)",
            # ⚠️ SEALED. These are Mati's own questions and the subjects it
            # chose from the person's interests — small in volume, and a direct
            # readout of what they care about.
            (vault.seal_soft(topic),
             vault.seal_soft(question or _opening(topic)), N._now()))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False                       # already on the list
    finally:
        conn.close()


def add_topics_from(text, db_path=None):
    """Split an interests answer into topics. Commas and 'and' are the seams."""
    added = []
    for part in re.split(r",|;|\band\b", text):
        part = " ".join(part.split())
        if len(part) > 3 and add_topic(part, db_path):
            added.append(part)
    return added


def _pick(conn):
    """Least-asked open topic — breadth before depth."""
    row = conn.execute(
        "SELECT topic_id, topic, next_q FROM inquiry_topics "
        "WHERE status='open' ORDER BY asked, topic_id LIMIT 1").fetchone()
    if not row:
        return row
    return (row[0], vault.open_body(row[1]) if row[1] else row[1],
            vault.open_body(row[2]) if row[2] else row[2])


def step(lm, tok, llm, db_path=None, verbose=True):
    """One question, asked and absorbed. Returns a summary, or None if idle.

    The LLM call is the slow part (~60-100s on a 3B), which is why this is one
    question per idle pass and not a loop that runs until it is bored.
    """
    conn = _conn(db_path)
    try:
        row = _pick(conn)
        if row is None:
            return None
        topic_id, topic, question = row
        question = question or _opening(topic)
    finally:
        conn.close()

    if verbose:
        print(f"  [inquiry] {topic}: {question}", file=sys.stderr)

    # What does Mati already hold near this question? Not a gate — retrieval
    # cannot reliably tell known from new — but it is what the model is shown
    # so the answer does not simply repeat what is already stored.
    held = bridge.recall(bridge.address_vector(lm, tok, question),
                         k=3, lm=lm, tok=tok)
    context = ""
    if held:
        context = ("\n\nYou already have these notes nearby; do not repeat "
                   "them, go somewhere they do not cover:\n"
                   + "\n".join(f"- {' '.join(b.split())[:200]}"
                               for _d, b, _s in held))

    try:
        answer, _raw = llm.reply(
            [{"role": "user",
              "content": f"{ASK_SYSTEM}\n\n{question}{context}"}],
            max_tokens=400)
    except Exception as e:
        if verbose:
            print(f"  [inquiry] ask failed: {e}", file=sys.stderr)
        return {"topic": topic, "error": str(e)}

    follow = _NEXT.search(answer or "")
    next_q = follow.group(1) if follow else None
    body = _NEXT.sub("", answer or "").strip()

    # Sanitised on write, exactly as reply excerpts are: material that talks
    # about the machinery must not enter the store, because the store is
    # replayed and pollution there outlives any prompt fix.
    clean = bridge._clean_for_memory(body, limit=10_000)
    queued = 0
    if clean:
        summary = study.enqueue(clean, f"asked about {topic}", tok,
                                db_path=db_path)
        queued = summary["queued"]

    conn = _conn(db_path)
    try:
        conn.execute(
            "UPDATE inquiry_topics SET asked=asked+1, next_q=?, last_asked=? "
            "WHERE topic_id=?",
            (vault.seal_soft(next_q) if next_q else next_q, N._now(), topic_id))
        conn.commit()
    finally:
        conn.close()

    # CLOSING THE LOOP. A topic that arrived because Mati was corrected twice
    # and deferred is not idle curiosity — it went and looked up something it
    # had been told and not verified. When it has an answer, it owes the user
    # that answer. Queued to say, not interrupted with: a finding is
    # interesting, never urgent.
    if clean and queued:
        conn2 = _conn(db_path)
        try:
            row2 = conn2.execute(
                "SELECT next_q FROM inquiry_topics WHERE topic_id=?",
                (topic_id,)).fetchone()
        finally:
            conn2.close()
        row2 = (vault.open_body(row2[0]),) if row2 and row2[0] else row2
        from_deferral = bool(row2 and (row2[0] or "").lower()
                             .startswith("is this correct")) or \
            question.lower().startswith("is this correct")
        if from_deferral:
            schedule.say_later(
                f"You corrected me about \"{topic}\" and I said I would look "
                f"into it. I have: {' '.join(clean.split())[:400]}",
                kind=schedule.FINDING,
                topic=topic,          # waits until the user raises it again
                detail=f"inquiry:{topic}")

    if verbose:
        print(f"  [inquiry] queued {queued} chunk(s)"
              + (f"; next: {next_q}" if next_q else "; no follow-up offered"),
              file=sys.stderr)
    return {"topic": topic, "question": question, "queued": queued,
            "next_q": next_q, "recalled": len(held),
            "dropped_as_machinery": clean is None}


def status(db_path=None):
    conn = _conn(db_path)
    try:
        rows = conn.execute(
            "SELECT topic, asked, next_q, status FROM inquiry_topics "
            "ORDER BY asked, topic_id").fetchall()
    finally:
        conn.close()
    rows = [(vault.open_body(t) if t else t, a,
             vault.open_body(q) if q else q, s) for t, a, q, s in rows]
    return {"topics": [{"topic": t, "asked": a, "next": q, "status": s}
                       for t, a, q, s in rows],
            "open": sum(1 for _t, _a, _q, s in rows if s == "open")}



# ─────────────────────────────────────────────────────────────────────────────
# Asking the person instead of the model
# ─────────────────────────────────────────────────────────────────────────────

# ⚠️ REWRITTEN, AND THE OLD WORDING IS KEPT BELOW BECAUSE IT
# EXPLAINS 862 BAD QUESTIONS. It read:
#
#     "Write ONE short question about it that a person who was there could
#      answer and you cannot work out for yourself — something the passage
#      leaves open, assumes, or does not explain."
#
# Two faults, both visible only once you know what passage it was handed.
#
# 1. "a person who was there". Given a passage about Cortes or the Long Count,
#    nobody alive was there — so the instruction INVITED the anachronism.
#    "Did you live with Hernan Cortes?" is the prompt working as written. It
#    presumed the passage described something the person witnessed, which is
#    true only of a lived episode, and the source pool was 65% not that.
#
# 2. "something the passage leaves open, assumes, or does not explain" is a
#    READING-COMPREHENSION instruction. It asks for gaps in a TEXT, not for
#    things only this person knows — which is why 49.6% of the queue opened
#    with "what is/was/were the..." and 80% contained no "you" at all.
#
# What replaces it asks for the one thing a passage of CONVERSATION genuinely
# cannot contain: what the other person knew and did not say. The source pool
# is now lived episodes only (`bridge._ASK_PRIORITY`), so "you" in the
# instruction below always refers to someone who really was present.
ASK_USER_SYSTEM = (
    "Below is a record of a conversation you had with this person. "
    "Write ONE short question for them.\n\n"
    "IT MUST BE SOMETHING ONLY THEY CAN ANSWER — about their own life, "
    "their reasons, what they meant, what happened to them, what they "
    "decided. Not a fact about the world. If the answer could be looked up, "
    "or if you could work it out yourself, it is the wrong question and you "
    "should ask about them instead.\n\n"
    "DO NOT ask what they witnessed of history, or of anything they could "
    "not personally have been present for. Ask about THEIR part in what was "
    "said.\n\n"
    "Only ask what the conversation actually raised. Do not invent a detail, "
    "a name, a place or an event that is not written below — if you are not "
    "sure something happened, it is not a question.\n\n"
    "Never ask about how you work. One sentence, no preamble, no lists, and "
    "do not answer it yourself.\n\n"
    "THE QUESTION MUST STAND ALONE. They will read it days later with this "
    "record nowhere in sight. Name what you are asking about. Do not write "
    "'those moments', 'that conversation', 'it' or 'they' — say which "
    "moments, which conversation, who."
)


# Referents that only make sense if you are holding the passage.
#
# ⚠️ MEASURED : 59 of 124 queued questions (48%) leaned on one of
# these, and the worst were unanswerable by anyone — "What were those four
# earlier moments?" The person has no way to know which four, and neither does
# anyone reading it later. The referent lived in the episode the model was
# looking at when it wrote the question and never made it into the question.
#
# The generator cannot see this itself: it writes WITH the passage in front of
# it, so "those four moments" feels perfectly specified. Hence a check outside
# the model rather than an instruction to it — the instruction is there too,
# but it is not what is relied on.
_DANGLING = re.compile(
    r"""(?:
        # "those moments" — but NOT "those who…", which is a relative
        # construction and perfectly self-contained. Found by auditing all 33
        # rejections rather than sampling them: "What happens to those who are
        # able to think with their own minds?" was being dropped.
        \b(?:those|these)\b(?!\s+(?:who|whom|which|that)\b)
      | \bthat\s+(?:moment|conversation|topic|passage|note|text|event|
                   time|occasion|subject|discussion|exchange|thing|one)\b
      | \bthis\s+(?:moment|conversation|topic|passage|note|text|event|
                    occasion|subject|discussion|exchange|thing|one)\b
      | \bthe\s+(?:earlier|previous|first|second|last|other)\s+
            (?:moment|conversation|topic|passage|note|text|event|time|
               occasion|subject|discussion|exchange|one)s?\b
      | \bthe\s+(?:two|three|four|five|six|seven|eight|nine|ten|\d+)\s+
            (?:earlier\s+|previous\s+)?
            (?:moment|conversation|topic|event|time|occasion)s?\b
    )""",
    re.I | re.X,
)

# Something specific has to be named. Narration words and pronouns do not
# count — a question made only of those is about nothing in particular.
_NOT_AN_ANCHOR = {
    "what", "when", "where", "which", "who", "whom", "whose", "how", "why",
    "did", "does", "do", "was", "were", "is", "are", "will", "would", "could",
    "should", "have", "has", "had", "been", "being", "the", "and", "for",
    "you", "your", "they", "them", "their", "this", "that", "these", "those",
    "there", "here", "it", "its", "with", "from", "about", "into", "than",
    "then", "some", "any", "more", "most", "such", "said", "told", "asked",
    "mentioned", "remember", "think", "know", "thing", "things", "person",
    "people", "something", "anything", "really", "actually", "specific",
}


def names_its_subject(question):
    """Can this question be understood by someone who has not seen the passage?

    Two conditions, both deterministic:
      - it does not lean on a referent that is not in the question, and
      - it names at least one specific thing.
    """
    q = " ".join((question or "").split())
    if not q or _DANGLING.search(q):
        return False
    anchors = {w for w in re.findall(r"[a-z][a-z'-]{3,}", q.lower())
               if w not in _NOT_AN_ANCHOR}
    return len(anchors) >= 2


def question_for_user(body, llm, topic=None, db_path=None, verbose=False,
                      note_id=None):
    """Turn a memory into a question for the PERSON, and queue it.

    This does not REPLACE `step()` above — the two are one loop. `step()` asks
    Mati's own model and files the answer as `studied:asked about …`, the
    weakest provenance it holds. That weakness is the POINT, not a defect: it
    marks a topic Mati is unsure of. This function then takes the weak topic
    to the one authority on the device that actually knows, and their answer
    comes back through the ordinary turn path as a LIVED episode — the
    strongest. Dreaming finds the soft spot; asking hardens it.

    Which is why `bridge.episode_to_ask_about` orders candidates by WEAKNESS
    rather than recency, and puts `studied:asked about …` first.

    The question is QUEUED, never asked immediately. It waits until the subject
    comes up, exactly as a finding does: raised at the top of a conversation it
    is an interruption; raised when you brought the topic up it is interest.

    Returns the queued question, or None if nothing worth asking came back.
    """
    text = " ".join((body or "").split())
    if len(text) < 40:
        return None                       # too little to have a gap in it
    try:
        asked, _raw = llm.reply(
            [{"role": "user",
              "content": f"{ASK_USER_SYSTEM}\n\nPassage:\n{text[:800]}"}],
                # ⚠️ CAPPED. The first line is taken and a "?" demanded, so
                # everything past it was generated and thrown away.
                max_tokens=60)
    except Exception:
        return None                       # never let curiosity break the loop
    q = " ".join((asked or "").split())
    # A model that returns a paragraph has not written a question; taking the
    # first line and demanding a "?" is cheaper than trusting it to obey.
    q = q.split("\n")[0].strip()
    if "?" not in q or len(q) < 12 or len(q) > 240:
        return None
    if not names_its_subject(q):
        if verbose:
            print(f"  [inquiry] dropped, does not name its subject: {q}",
                  file=sys.stderr)
        return None
    q = q[:q.index("?") + 1]
    # Never let the machinery become the subject — the same rule `admit.py`
    # applies to what may become a memory.
    low = q.lower()
    if any(w in low for w in ("field", "basin", "coherence", "imprint",
                              "theta", "your memory system")):
        return None
    # ⚠️ AND NOT ABOUT MATI'S OWN RECALL EITHER, which the list above missed
    # because those are substrate words and this is a different mistake.
    # , from the first run of the corrected generator:
    #
    #     "What were the four memories that were triggered by your answer to
    #      'How many memories do you have?'"
    #     "What specific moments from your past were triggered by the mention
    #      of the Calmecac?"
    #
    # Both ask the person to account for what MATI remembered. The subject of
    # a sincere question is the person's life, never Mati's retrieval — and a
    # conversation ABOUT Mati's memory is still a conversation, so the
    # episode is legitimate source material while a question about its
    # mechanics is not.
    #
    # ⚠️ This is a list of phrasings and so it will leak; it is on the
    # narrow side of the loop — what Mati may ASK — rather than trying to
    # anticipate everything a model might say.
    if any(w in low for w in ("triggered by", "your answer", "your response",
                              "your reply", "you recalled", "you remembered",
                              "memories that", "memories were", "many memories",
                              "your recollection")):
        return None
    # The topic decides WHEN this is ever raised, so it must be the memory's
    # subject — not its first 60 characters, which are narration. See
    # `bridge.topic_of`; the measurement is in its docstring.
    subject = topic or bridge.topic_of(body)
    if not subject:
        return None                       # no subject, nothing to wait for
    # `detail` carries which memory this came from, so a later conversation
    # that answers it can be attached to the right place. Without it a resolved
    # question can only be retired, not filed.
    schedule.say_later(q, kind=schedule.QUESTION, topic=subject,
                       detail=(f"from note {note_id}" if note_id else None),
                       db_path=db_path)
    if verbose:
        print(f"  [inquiry] will ask when it comes up: {q}", file=sys.stderr)
    return q


# ─────────────────────────────────────────────────────────────────────────────
# Answering a waiting question from what was said later
#
# A queued question has three possible ends: the person raises the subject and
# is asked, it waits forever, or — this — something they said afterwards
# already answers it and nobody needs to be interrupted at all.
#
# WHY THIS IS WORTH HAVING. The person talks about their own life without being
# prompted, so the material that answers Mati's questions arrives on its own.
# Reading it costs a sleep cycle and no interruption, and an answer found this
# way is grounded in a LIVED exchange rather than in the model's own guessing —
# which is the whole provenance ladder, climbed without asking.
#
# ⚠️ THE MODEL IS NOT ASKED "IS THIS ANSWERED?" A 3B judging entailment will say
# yes to almost anything, and a wrong yes retires a real question and records a
# fabricated answer. Instead it is given ONE later episode and told to answer
# STRICTLY from it or return a refusal token, and the reply is then checked
# against the episode's own words. Ungrounded answers are discarded.

ANSWER_FROM_EPISODE = """You are given a QUESTION and one PASSAGE from a later
conversation. Answer the question using ONLY what the passage actually says.

If the passage does not answer it, reply with exactly: NOT HERE

Do not guess, do not use anything you know from elsewhere, and do not restate
the question. One or two sentences."""

REFUSAL = "NOT HERE"


def _grounded(answer, passage, need=2):
    """Does the answer actually come from the passage?

    The cheap check the model cannot fake: content words in the answer have to
    appear in the passage. A fabricated answer draws on the model's own
    vocabulary and fails this; a real one is nearly a paraphrase.
    """
    from . import bridge
    a = {bridge.schedule._stem(w)
         for w in re.findall(r"[a-z0-9']+", (answer or "").lower())
         if len(w) > 3 and w not in bridge.SEARCH_STOP}
    p = {bridge.schedule._stem(w)
         for w in re.findall(r"[a-z0-9']+", (passage or "").lower())}
    if not a:
        return False
    return len(a & p) >= min(need, len(a))



def _their_words(body):
    """The person's half of an episode. Mati's reply is not evidence."""
    text = body or ""
    if "I was asked:" not in text:
        return ""
    after = text.split("I was asked:", 1)[1]
    return " ".join(after.split("I said:", 1)[0].split())[:900]


def _written_after(note_id, when, db_path=None):
    """Was this memory written after `when` (epoch seconds)?"""
    import sqlite3
    from . import bridge
    path = bridge.NOTES_DB_PATH if db_path is None else db_path
    conn = sqlite3.connect("file:" + path + "?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT wall_created FROM notes WHERE note_id=?", (note_id,)
        ).fetchone()
    except sqlite3.Error:
        return False
    finally:
        conn.close()
    if not row or not row[0]:
        return False
    try:
        import datetime as _dt
        return _dt.datetime.fromisoformat(row[0]).timestamp() > when
    except (ValueError, TypeError):
        return False


def resolve_from_later(llm, db_path=None, verbose=False):
    """Try to answer ONE waiting question from something said afterwards.

    Returns (question_id, answer) or None. Retires the question on success —
    it has been answered, so asking it would be asking twice.
    """
    from . import bridge
    waiting = [i for i in schedule.pending(include_held=True, db_path=db_path)
               if i["kind"] == schedule.QUESTION]
    for item in waiting:
        made = item.get("made_at") or 0
        for hit in bridge.search(item["topic"] or item["body"], k=6):
            if hit["source"] != bridge.EPISODE_SOURCE:
                continue
            # ⚠️ ONLY WHAT WAS SAID AFTERWARDS. Earlier material is what the
            # question was asked ABOUT — answering from it is answering the
            # question with the thing that raised it. The first version of this
            # only said so in a comment and did not check.
            if not _written_after(hit["note_id"], made, db_path):
                continue
            # ⚠️ AND ONLY THE PERSON'S HALF. An episode body is "I was asked:
            # <them> / I said: <Mati>", and Mati's half is model output. Left
            # whole, this resolves a question from Mati's own earlier guess —
            # which is `step()`'s circular self-answering with extra steps, and
            # the thing this exists to avoid. Measured on the tlamatinime
            # episode, where the "answer" came straight out of Mati's own
            # reply, priestesses and all.
            passage = _their_words(hit["body"])
            if len(passage) < 40:
                continue                     # nothing of theirs to read
            try:
                said, _raw = llm.reply([{"role": "user", "content":
                                         f"{ANSWER_FROM_EPISODE}\n\n"
                                         f"QUESTION: {item['body']}\n\n"
                                         f"PASSAGE: {passage}"}],
                                          # ⚠️ CAPPED. An answer read out of
                                          # one episode is a sentence or two.
                                          max_tokens=140)
            except Exception:
                return None                  # never let this break the loop
            answer = " ".join((said or "").split())
            if not answer or REFUSAL.lower() in answer.lower():
                continue
            if len(answer) < 12 or len(answer) > 400:
                continue
            if not _grounded(answer, passage):
                continue                     # invented, not read
            schedule.mark_delivered([item["id"]], db_path=db_path)
            if verbose:
                print(f"  [inquiry] answered from a later conversation: "
                      f"{item['body'][:60]} -> {answer[:60]}", file=sys.stderr)
            return item["id"], answer
    return None


def main():
    ap = argparse.ArgumentParser(description="Mati looks into a topic")
    ap.add_argument("--topic", action="append", default=[],
                    help="add a topic (repeatable)")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--step", type=int, default=0,
                    help="ask N questions now rather than waiting for idle")
    ap.add_argument("--force", action="store_true",
                    help="run even if Mati holds the field (unsafe)")
    args = ap.parse_args()

    for t in args.topic:
        print(f"  {'added' if add_topic(t) else 'already listed'}: {t}")

    if args.status or not (args.topic or args.step):
        st = status()
        print(f"  {st['open']} open topic(s)")
        for t in st["topics"]:
            print(f"    {t['topic']:<40} asked {t['asked']}"
                  + (f"  next: {t['next'][:44]}" if t["next"] else ""))
        sq = study.status()
        print(f"  {sq['pending']} chunk(s) waiting to be studied")
        return 0

    if args.step:
        rc = exclusive.guard("inquiry", force=args.force)
        if rc:
            return rc
        from .llm import make_llm
        from .mati import system_prompt
        llm = make_llm(system_prompt())
        if vault.is_set_up() and not vault.unlocked():
            vault.unlock(reason="let Mati study")
        lm, tok = bridge.load_field()
        for _ in range(args.step):
            if step(lm, tok, llm) is None:
                print("  nothing to look into")
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
