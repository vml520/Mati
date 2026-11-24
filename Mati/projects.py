"""
Projects: a task with a target and a path, kept in the notes store.

    python3 -m Mati.projects                 the open projects
    python3 -m Mati.projects --show 3        one project, as Mati would say it

The rules: every task is a project, handled with a target and a path in
mind. Mati confirms when a step is done, holds every option until a choice is
confirmed, and keeps a record of all advice and its result, so it offers
something again only if necessary.

⚠️ WHY THIS EXISTS. Asked to help with a plan, Mati remembered what it was and
why, then advised a step the person had already finished — after the previous
conversation had ended with "I have all the paperwork ready". The correction
was STORED. It did not count: the four memories the field handed that turn
were none of them from the night before (the field chooses by when, not
what), an episode records what was SAID and not what was DECIDED, and a 3B
asked to help with a move fills the gap with its standard checklist.

So, as `brief.py` says of targets: **state, not memory.** What is open, done,
chosen and already suggested is kept here, said in code when the project
comes up, and used to stop a reply re-advising what is settled.

⚠️ NOTHING HERE IS UPDATED IN PLACE OR DELETED. One append-only log per
project; the current state is folded from it, so "when did we rule out
flying?" stays answerable (`brief.py`, `correction.py`, `agenda.py`).

⚠️ MATI NEVER SETTLES ANYTHING ITSELF. A step becomes done, an option chosen,
a suggestion declined, only on the person's confirmation — one question at a
time, through `asks.py`. Code spots the sentence; it does not decide.

⚠️ SHOWN IN THE RESPONSE WINDOW, ON REQUEST, AND NOWHERE ELSE (the user). No
drawer, no report section, no notification.

⚠️ THE PERSON'S WORDS ARE SEALED (`text`, `evidence`), like every other
column a person supplied. See `vault.SEALED_COLUMNS`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sqlite3
import sys

P_TABLE = "projects"
L_TABLE = "project_log"

# What a log entry is about.
TITLE, TARGET, STEP, OPTION, ADVICE, STATUS = (
    "title", "target", "step", "option", "advice", "status")

# Their states. ⚠️ A step that was never settled is `open`; only the person
# makes one `done`. Advice starts as `given` and gets its result later.
OPEN, NEXT, DONE, DROPPED = "open", "next", "done", "dropped"
HELD, CHOSEN, NOT_CHOSEN = "held", "chosen", "not chosen"
GIVEN, ALREADY_DONE, DECLINED, TAKEN, NO_ANSWER, SUPERSEDED = (
    "given", "already done", "declined", "taken", "no answer", "superseded")
MET, WITHDRAWN = "met", "withdrawn"

# Advice in any of these states must not be offered again unasked.
SETTLED = {DONE, ALREADY_DONE, DECLINED, NOT_CHOSEN, TAKEN, SUPERSEDED}

PENDING_KEY = "project_pending"
ASK_KIND = "project"
MAX_ADVICE_PER_REPLY = 10


# ── storage ──────────────────────────────────────────────────────────────────

def _db(db_path=None):
    from . import bridge
    return bridge.NOTES_DB_PATH if db_path is None else db_path


def _conn(db_path=None):
    conn = sqlite3.connect(_db(db_path), timeout=30)
    conn.execute(f"""CREATE TABLE IF NOT EXISTS {P_TABLE} (
            project_id INTEGER PRIMARY KEY,
            opened     TEXT NOT NULL)""")
    conn.execute(f"""CREATE TABLE IF NOT EXISTS {L_TABLE} (
            entry_id   INTEGER PRIMARY KEY,
            project_id INTEGER NOT NULL,
            kind       TEXT NOT NULL,
            ref        INTEGER,
            parent     INTEGER,
            text       TEXT,
            state      TEXT,
            evidence   TEXT,
            at         TEXT NOT NULL)""")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{L_TABLE}_project "
                 f"ON {L_TABLE} (project_id, entry_id)")
    return conn


def _now():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _seal(text):
    """Encrypt a person-supplied field. Pass-through with no vault set up."""
    if text is None:
        return None
    from . import vault
    try:
        return vault.seal(text)
    except Exception:
        # ⚠️ A LOCKED VAULT MUST NOT LOSE WHAT SOMEONE JUST SETTLED — the
        # `brief.py` reasoning: `migrate_tables` closes the plaintext window
        # at the next start.
        return text


# ⚠️ EVERY TURN READS EVERY OPEN PROJECT'S LOG, and opening a sealed value is
# an `openssl` process. The plaintext is kept by its ciphertext, as
# `notes._open_cached` does for memories; a sealed value never changes.
_OPENED = {}


def _open(text):
    if text is None:
        return None
    got = _OPENED.get(text)
    if got is not None:
        return got
    from . import vault
    try:
        got = vault.open_body(text)
    except Exception:
        return text
    if len(_OPENED) > 4000:
        _OPENED.clear()
    _OPENED[text] = got
    return got


def open_project(title, target=None, evidence=None, db_path=None):
    """Start a project. Returns its id."""
    conn = _conn(db_path)
    try:
        cur = conn.execute(f"INSERT INTO {P_TABLE} (opened) VALUES (?)",
                           (_now(),))
        pid = cur.lastrowid
        conn.commit()
    finally:
        conn.close()
    log(pid, TITLE, text=_clean(title), evidence=evidence, db_path=db_path)
    log(pid, STATUS, state=OPEN, db_path=db_path)
    if target:
        log(pid, TARGET, text=_clean(target), evidence=evidence,
            db_path=db_path)
    return pid


def log(project_id, kind, text=None, state=None, ref=None, parent=None,
        evidence=None, db_path=None):
    """Append one entry. Returns its id. ⚠️ The only write there is."""
    conn = _conn(db_path)
    try:
        cur = conn.execute(
            f"INSERT INTO {L_TABLE} (project_id, kind, ref, parent, text, "
            f"state, evidence, at) VALUES (?,?,?,?,?,?,?,?)",
            (project_id, kind, ref, parent, _seal(text), state,
             _seal(evidence), _now()))
        eid = cur.lastrowid
        conn.commit()
        return eid
    finally:
        conn.close()


def state(project_id, db_path=None):
    """The project as it stands, folded from its log. None if there is none.

    An entry with no `ref` adds an item; an entry with a `ref` changes that
    item's state and carries the sentence that changed it.
    """
    conn = _conn(db_path)
    try:
        opened = conn.execute(f"SELECT opened FROM {P_TABLE} WHERE "
                              f"project_id=?", (project_id,)).fetchone()
        if not opened:
            return None
        rows = conn.execute(
            f"SELECT entry_id, kind, ref, parent, text, state, evidence, at "
            f"FROM {L_TABLE} WHERE project_id=? ORDER BY entry_id",
            (project_id,)).fetchall()
    finally:
        conn.close()
    p = {"id": project_id, "opened": opened[0], "title": None, "target": None,
         "status": OPEN, "items": {}, "touched": opened[0]}
    for eid, kind, ref, parent, text, st, ev, at in rows:
        text, ev = _open(text), _open(ev)
        p["touched"] = at
        if kind == TITLE:
            p["title"] = text
        elif kind == TARGET:
            p["target"] = text
        elif kind == STATUS:
            p["status"] = st
        elif ref is None:
            p["items"][eid] = {"id": eid, "kind": kind, "text": text,
                               "state": st, "parent": parent, "at": at,
                               "evidence": ev, "history": [(st, at, ev)]}
        elif ref in p["items"]:
            it = p["items"][ref]
            it["state"] = st
            if ev:
                it["evidence"] = ev
            it["settled_at"] = at
            it["history"].append((st, at, ev))
    return p


def _of(p, kind, states=None):
    return [it for it in p["items"].values() if it["kind"] == kind
            and (states is None or it["state"] in states)]


def open_projects(db_path=None):
    """Every project still open, most recently touched first."""
    conn = _conn(db_path)
    try:
        ids = [r[0] for r in conn.execute(f"SELECT project_id FROM {P_TABLE}")]
    finally:
        conn.close()
    out = [p for p in (state(i, db_path) for i in ids)
           if p and p["status"] == OPEN]
    return sorted(out, key=lambda p: p["touched"], reverse=True)


# ── words ────────────────────────────────────────────────────────────────────

# ⚠️ MATCHING IS ON CONTENT WORDS, and these carry no subject. "Help me with
# my move" and "help me with my letter" share `help`; that is not a match.
_STOP = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "for",
    "with", "by", "from", "as", "is", "are", "was", "were", "be", "been",
    "it", "its", "this", "that", "these", "those", "there", "here", "what",
    "which", "who", "how", "why", "when", "where", "do", "does", "did", "i",
    "me", "my", "we", "our", "you", "your", "he", "she", "they", "them",
    "his", "her", "their", "will", "would", "could", "should", "can", "may",
    "might", "must", "shall", "have", "has", "had", "not", "no", "yes", "so",
    "if", "then", "than", "also", "just", "about", "into", "out", "up",
    "down", "over", "some", "any", "all", "each", "more", "most", "very",
    "really", "let", "lets", "get", "got", "make", "made", "help", "work",
    "need", "want", "like", "thing", "things", "going", "go", "start",
    "now", "today", "time", "please", "great", "good", "okay", "ok", "sure",
    "thanks", "thank", "well", "one", "first", "next", "step", "steps",
    "project", "plan", "able", "way", "much", "many", "know", "think",
    "sound", "sounds", "right", "already", "ready", "done", "too",
}


def _stem(w):
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf) and not w.endswith("ss"):
            return w[:-len(suf)]
    return w


def stems(text):
    """The content words of `text`, crudely stemmed. A set."""
    out = set()
    for w in re.findall(r"[a-záéíóúñü]+", (text or "").lower()):
        if len(w) >= 3 and w not in _STOP:
            out.add(_stem(w))
    return out


def _overlap(a, b):
    """How much of `b`'s content `a` carries: (shared count, share of b)."""
    sa, sb = stems(a), stems(b)
    if not sb:
        return 0, 0.0
    shared = sa & sb
    return len(shared), len(shared) / len(sb)


def _clean(text, cap=200):
    t = " ".join((text or "").replace("**", "").split()).strip(" :-–—")
    return t[:cap].rstrip() + ("…" if len(t) > cap else "")


def _yours(title):
    """"my garden redesign" → "your garden redesign": said back to the person."""
    t = title or "this"
    return re.sub(r"^(my|our)\b", lambda m: "your", t, flags=re.I)


def _theirs(title):
    """"my garden redesign" → "their garden redesign", for the model.

    ⚠️ NEVER "your" IN A BLOCK ADDRESSED TO MATI. Second person in a block
    written to the assistant is read as being about the assistant, and comes
    back out as a claim about the person — see the waking block's history.
    """
    t = title or "this"
    return re.sub(r"^(my|our|your)\b", lambda m: "their", t, flags=re.I)


# ── which project a turn is about ────────────────────────────────────────────

# The project a session was last about, so a follow-up with no project words
# ("I have all the paperwork ready") still lands on it. In memory only: a new
# process starts with no assumption.
_POINTER = {}
# ⚠️ AND ONLY FOR A WHILE. "Dinner is ready" an hour later is not about the
# move; a follow-up within a few minutes usually is.
POINTER_SECONDS = 10 * 60


def about(text, session="default", db_path=None):
    """The open project this message names, or None.

    ⚠️ BY ITS OWN NAME, NOT BY ONE WORD. Two shared content words with the
    title or target — "the redesign of my garden" is about "my garden
    redesign", and "redesign the logo" is not. A one-word title needs that
    word.
    """
    best, best_n = None, 0
    words = stems(text)
    for p in open_projects(db_path):
        key = stems(p["title"]) | stems(p["target"])
        n = len(words & key)
        need = 1 if len(stems(p["title"])) == 1 else 2
        if n >= need and n > best_n:
            best, best_n = p, n
    if best:
        _point(session, best["id"])
    return best


def _point(session, pid):
    import time
    _POINTER[session] = (pid, time.time())


def pointed(session="default", db_path=None):
    """The project this session was last about, if it is still open."""
    import time
    got = _POINTER.get(session)
    if not got or time.time() - got[1] > POINTER_SECONDS:
        return None
    p = state(got[0], db_path)
    return p if p and p["status"] == OPEN else None


# ── reading: said in code ────────────────────────────────────────────────────

def _day(at):
    try:
        return dt.datetime.fromisoformat(at).strftime("%-d %b")
    except Exception:
        return ""


def next_step(p):
    nxt = _of(p, STEP, {NEXT})
    if nxt:
        return nxt[-1]
    open_ = _of(p, STEP, {OPEN})
    return open_[0] if open_ else None


def where_we_are(p):
    """One or two lines: what is settled, what is open, what is next. "" if
    nothing is recorded yet."""
    # ⚠️ STEPS ONLY. A suggestion confirmed as done also adds a done step
    # with its words, so counting the suggestion too would say it twice.
    done = _of(p, STEP, {DONE})
    held = [o for o in _of(p, OPTION, {HELD})]
    nxt = next_step(p)
    bits = []
    if done:
        bits.append("; ".join(f"{_clean(d['text'], 80)} — done"
                              f" ({_day(d.get('settled_at') or d['at'])})"
                              for d in done[-4:]))
    if held:
        bits.append("still choosing between "
                    + " · ".join(_clean(o["text"], 40) for o in held[:4]))
    if nxt:
        bits.append(f"next: {_clean(nxt['text'], 100)}")
    if not bits:
        return ""
    return f"Where we are with {_yours(p['title'])}: " + "; ".join(bits) + "."


def view(p):
    """The whole project, for the response window. ⚠️ ONLY ON REQUEST."""
    L = [f"{_yours(p['title']).capitalize()} — opened {_day(p['opened'])}",
         f"Aimed at: {p['target'] or 'not said yet — tell me what it is aimed at'}"]

    def section(head, items, show):
        if items:
            L.append("")
            L.append(head)
            for it in items:
                L.append("  - " + show(it))

    def when(it):
        d = _day(it.get("settled_at") or it["at"])
        ev = it.get("evidence")
        return f" ({d}" + (f' — “{_clean(ev, 70)}”' if ev else "") + ")"

    section("Done", _of(p, STEP, {DONE}), lambda s: _clean(s["text"]) + when(s))
    section("Next", _of(p, STEP, {NEXT}), lambda s: _clean(s["text"]))
    section("Still open", _of(p, STEP, {OPEN}), lambda s: _clean(s["text"]))
    held = _of(p, OPTION, {HELD})
    section("Choices held, none confirmed", held,
            lambda o: _clean(o["text"]))
    section("Decided", _of(p, OPTION, {CHOSEN}),
            lambda o: _clean(o["text"]) + when(o))
    section("Ruled out", _of(p, OPTION, {NOT_CHOSEN})
            + [a for a in _of(p, ADVICE) if a["state"] == DECLINED],
            lambda o: _clean(o["text"]) + when(o))
    advice = _of(p, ADVICE)
    section("What I have suggested, and what came of it", advice,
            lambda a: f"{_clean(a['text'], 140)} ({_day(a['at'])}) — "
                      f"{'no answer yet' if a['state'] == GIVEN else a['state']}")
    return "\n".join(L)


def prompt_block(p):
    """What the model is handed. ⚠️ THE GUARD, NOT THIS, IS WHAT HOLDS —
    an instruction to a 3B is a hope; this only aims its advice."""
    settled = []
    for t_ in ([s["text"] for s in _of(p, STEP, {DONE})]
               + [a["text"] for a in _of(p, ADVICE) if a["state"] in SETTLED]
               + [o["text"] for o in _of(p, OPTION, {NOT_CHOSEN})]):
        if t_ and t_ not in settled:
            settled.append(t_)
    nxt = next_step(p)
    lines = [f"<computed>",
             f"This is about the person's project: {_theirs(p['title'])}."]
    if p.get("target"):
        lines.append(f"It is aimed at: {p['target']}")
    if settled:
        lines.append("Already settled — do not suggest any of these again: "
                     + "; ".join(_clean(t, 90) for t in settled[-8:]) + ".")
    held = _of(p, OPTION, {HELD})
    if held:
        lines.append("Options still open, none chosen: "
                     + "; ".join(_clean(o["text"], 60) for o in held) + ".")
    if nxt:
        lines.append(f"The next step is: {_clean(nxt['text'], 120)}. "
                     "Help with that.")
    lines.append("</computed>")
    return "\n".join(lines)


# ── the guard on the reply ───────────────────────────────────────────────────

# A sentence shaped like advice. ⚠️ ONLY THESE ARE EVER REMOVED: "your
# paperwork is ready" is a statement and stays; "research the documents you
# need" is advice and goes, if that is settled.
_ADVICE = re.compile(
    r"\b(you (should|could|might|may want to|can|need to|will need to|"
    r"'ll need to|ought to|must)|i (recommend|suggest|would suggest|advise)|"
    r"consider|make sure|start by|be sure to|it'?s (a good idea|worth|"
    r"important) to|don'?t forget to|remember to|(the )?(first|next) step|"
    r"try to|look into|research|check (with|whether|if|that|the|on)|"
    r"contact|apply for|gather|prepare|book|schedule|visit|reach out|"
    r"find out|get in touch)\b", re.I)
_LIST_ITEM = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.+)$")
# ⚠️ A REQUEST TO HEAR IT AGAIN — not the word "again". "Let's work on my
# garden redesign again" means resume, and turning the guard off for it let every
# settled suggestion back through (caught by replaying a real conversation).
_AGAIN = re.compile(
    r"\b(repeat|say (that|it|them) again|tell me again|remind me what|"
    r"what (did|have) you (suggest|say|recommend)|what were (the|your) "
    r"(options|suggestions)|(your|the) suggestions again)\b", re.I)


def _advice_like(sentence):
    return bool(_ADVICE.search(sentence or "")) or bool(
        _LIST_ITEM.match(sentence or ""))


def _same(sentence, item_text):
    """Does `sentence` say `item_text` again? Two shared content words, and
    at least half of the item's."""
    n, share = _overlap(sentence, item_text)
    need = 1 if len(stems(item_text)) == 1 else 2
    return n >= need and share >= 0.5


def guard(p, user_text=""):
    """A predicate for `voice.ReplyFilter`: True drops the sentence.

    ⚠️ OFFERED AGAIN ONLY IF NECESSARY, which means only when asked: a
    message asking to repeat or recall the suggestions turns the guard off.
    Advice with no answer yet is not settled, so it can be said again.
    """
    if p is None or _AGAIN.search(user_text or ""):
        return None
    settled = ([s["text"] for s in _of(p, STEP, {DONE, DROPPED})]
               + [s["evidence"] for s in _of(p, STEP, {DONE})
                  if s.get("evidence")]
               + [a["text"] for a in _of(p, ADVICE) if a["state"] in SETTLED]
               + [o["text"] for o in _of(p, OPTION, {NOT_CHOSEN})])
    settled = [t for t in settled if t]
    if not settled:
        return None

    def drop(sentence):
        return _advice_like(sentence) and any(_same(sentence, t)
                                              for t in settled)
    return drop


# ── writing: what Mati suggested ─────────────────────────────────────────────

def advice_in(reply):
    """The suggestions in a reply: list items, and sentences shaped like
    advice. In order, without repeats."""
    out = []
    for line in (reply or "").splitlines():
        m = _LIST_ITEM.match(line)
        if m:
            out.append(_clean(m.group(1)))
            continue
        for s in re.split(r"(?<=[.!?])\s+", line.strip()):
            if _ADVICE.search(s) and len(stems(s)) >= 2:
                out.append(_clean(s))
    seen, uniq = [], []
    for a in out:
        if a and not any(_same(a, b) and _same(b, a) for b in seen):
            seen.append(a)
            uniq.append(a)
    return uniq[:MAX_ADVICE_PER_REPLY]


def record_advice(p, reply, db_path=None):
    """Log what this reply suggested. ⚠️ Mati's own words are logged as said;
    whether it came to anything is the person's to say. Returns the count."""
    have = [a["text"] for a in _of(p, ADVICE)]
    n = 0
    for a in advice_in(reply):
        if any(_same(a, h) and _same(h, a) for h in have):
            continue
        log(p["id"], ADVICE, text=a, state=GIVEN, db_path=db_path)
        have.append(a)
        n += 1
    return n


# ── writing: what the person says ────────────────────────────────────────────

_SETTLE = re.compile(
    r"\b(i\s*(?:have|'ve|ve|got|already have|already got)\b.{0,60}?"
    r"\b(?:ready|done|finished|sorted|complete|completed|submitted|sent|"
    r"booked|filed|gathered|collected|together|in order)\b"
    r"|\b(?:is|are|'s|was|were)\s+(?:all\s+)?(?:ready|done|finished|sorted|"
    r"complete|booked|submitted|sent|filed)\b"
    r"|\bi\s+(?:already\s+)?(?:did|finished|completed|booked|sent|submitted|"
    r"filed|bought|paid|applied|called|emailed|visited|gathered|got)\b"
    r"|\balready\s+(?:did|done|have|got|booked|sent))", re.I)
_CHOOSE = re.compile(
    r"\b(?:let'?s\s+(?:go with|take|do|use|pick)|i(?:'ll| will)\s+(?:go with|"
    r"take|use|pick)|i\s+(?:chose|choose|picked|prefer|decided on|'ll take)|"
    r"we(?:'ll| will)\s+go with)\s+(.{2,60}?)(?:[.!?,]|$)", re.I)
_DECLINE = re.compile(
    r"\b(?:i(?:'d| would)\s+rather\s+not|i\s+don'?t\s+want\s+to|"
    r"i(?:'m| am)\s+not\s+going\s+to|rule\s+out|i\s+won'?t)\s+(.{2,60}?)"
    r"(?:[.!?,]|$)", re.I)
_HALT = re.compile(
    r"\b(not\s+(?:right\s+)?now|later|another\s+time|some\s+other\s+time|"
    r"let'?s\s+(?:stop|pause|break)|i(?:'ll| will)\s+(?:come back|get back|"
    r"pick (?:this|it) up|continue)|(?:have|need)\s+to\s+wait|waiting\s+"
    r"(?:on|for)|on\s+hold|tomorrow|next\s+week|i(?:'m| am)\s+going\s+to\s+"
    r"(?:make|do|work|go|take|step)|speak\s+(?:again\s+)?(?:with you\s+)?"
    r"(?:later|afterward|afterwards)|pause\s+(?:this|it|here))\b", re.I)


def _open_items(p):
    """What a settling sentence could be about: open steps first, then
    suggestions with no answer."""
    return (_of(p, STEP, {NEXT, OPEN})
            + [a for a in _of(p, ADVICE) if a["state"] in (GIVEN, NO_ANSWER)])


def _propose(op, question, **payload):
    """Hold one proposal and ask about it. False if a question is waiting."""
    from . import asks, config as C
    if asks.pending():
        return False
    C.save_settings(**{PENDING_KEY: dict(payload, op=op)})
    if not asks.raise_(ASK_KIND, question):
        C.save_settings(**{PENDING_KEY: {}})
        return False
    return True


def notice(p, user_text, db_path=None):
    """After a reply: did the person settle, choose or rule something out?
    If so, ask — once. Returns the question asked, or None.

    ⚠️ CODE SPOTS THE SENTENCE; ONLY THE ANSWER CHANGES ANYTHING.
    """
    if p is None:
        return None
    t = " ".join((user_text or "").split())
    title = _yours(p["title"])
    m = _CHOOSE.search(t)
    if m:
        what = _clean(m.group(1), 60)
        held = [o for o in _of(p, OPTION, {HELD}) if _same(what, o["text"])
                or _same(o["text"], what)]
        q = f"Shall I make “{what}” your choice for {title}?"
        if _propose("choose", q, project=p["id"], option=(held[0]["id"]
                    if held else None), text=what, evidence=t):
            return q
        return None
    m = _DECLINE.search(t)
    if m:
        what = _clean(m.group(1), 60)
        q = f"Shall I rule out “{what}” for {title}?"
        hits = [it for it in _of(p, ADVICE) + _of(p, OPTION, {HELD})
                if it["state"] not in SETTLED and _overlap(what, it["text"])[0]]
        if _propose("decline", q, project=p["id"],
                    items=[it["id"] for it in hits], text=what, evidence=t):
            return q
        return None
    if _SETTLE.search(t):
        cands = _open_items(p)
        strong = [c for c in cands if _overlap(t, c["text"])[0] >= 2]
        if len(strong) == 1:
            c = strong[0]
            q = f"Shall I mark “{_clean(c['text'], 90)}” as done for {title}?"
            ok = _propose("done", q, project=p["id"], item=c["id"],
                          evidence=t)
        else:
            # ⚠️ NOT A GUESS. "Paperwork" and "documents" do not match by
            # word, and the offline embeddings could not tell related from
            # unrelated reliably — so the person picks.
            pick = (strong or cands)[:5]
            listed = " ".join(f"{i}. {_clean(c['text'], 70)}"
                              for i, c in enumerate(pick, 1))
            q = (f"Which part of {title} is that? {listed}"
                 + (" — or say “new” and I will note it as done."
                    if pick else " Say “new” and I will note it as done."))
            ok = _propose("done", q, project=p["id"], item=None,
                          choices=[c["id"] for c in pick], evidence=t)
        return q if ok else None
    return None


def _task_title(history):
    """The task a halted session was about: the person's words that drew a
    multi-step reply, with the request framing taken off."""
    msgs = list(history or [])
    for i in range(len(msgs) - 1, 0, -1):
        if msgs[i].get("role") != "assistant":
            continue
        if len(advice_in(msgs[i].get("content") or "")) < 2:
            continue
        ask = next((m.get("content") for m in reversed(msgs[:i])
                    if m.get("role") == "user"), None)
        if not ask:
            return None, None
        t = re.sub(r"^\s*(great|ok(ay)?|so|well|hi|hello|alright|right)\b[.!,]*"
                   r"\s*", "", ask, flags=re.I)
        m = re.search(r"(?:help (?:me )?(?:with|plan)|work on|working on|plan"
                      r"(?:ning)?|let'?s (?:work on|plan|talk about|get "
                      r"started on|start on|start|do)|i (?:want|need|would "
                      r"like) (?:help with|to))\s+(.+)", t, re.I)
        title = (m.group(1) if m else t).strip(" .!?")
        return _clean(re.split(r"[.!?]", title)[0], 60), i
    return None, None


_OFFERED = set()


def offer_if_halted(user_text, history, session="default", db_path=None):
    """Several steps, and progress halts → offer, once, to keep a project.

    the user: a task becomes a project when it has multiple steps and progress
    is halted for one reason or another.
    """
    if not _HALT.search(user_text or ""):
        return None
    # "Remind me tomorrow" is a reminder, not a task stopping partway.
    if re.search(r"\bremind\s+me\b", user_text or "", re.I):
        return None
    title, at = _task_title(history)
    if not title or about(title, session, db_path):
        return None
    # ⚠️ ONCE. Once raised, a "no" is not asked again in this process. One
    # that could not be raised — another question was waiting — may be tried
    # at the next halt.
    key = (session, title.lower())
    if key in _OFFERED:
        return None
    advice = []
    for m in list(history or [])[at:]:
        if m.get("role") == "assistant":
            advice += advice_in(m.get("content") or "")
    q = (f"We stopped partway through {_yours(title)}. Shall I keep it as a "
         "project, so we can pick up where we left off?")
    if _propose("open", q, title=title, advice=advice[:20],
                evidence=user_text, session=session):
        _OFFERED.add(key)
        return q
    return None


def confirm(text):
    """The answer to a project question. A reply, or None if it was not one.

    `asks.answer` calls this; returning a string clears the question.
    """
    from . import asks, config as C
    pend = (C._settings().get(PENDING_KEY) or {})
    op = pend.get("op")
    if not op:
        return None
    t = " ".join((text or "").split()).lower().strip(" .!?")
    yes = asks._said_yes(t)
    num = re.match(r"^(?:number\s+)?(\d)\b|^the\s+(first|second|third|fourth|"
                   r"fifth)\b", t)
    new = t in ("new", "a new one", "new step", "it's new", "its new")
    no = asks._declined(t)
    if not (yes or num or new or no):
        return None
    # ⚠️ "YES" TO "WHICH ONE?" IS NOT AN ANSWER. Taking it as "a new step"
    # would file the sentence under nothing it was about.
    if (op == "done" and pend.get("item") is None and pend.get("choices")
            and yes and not (num or new)):
        return None
    C.save_settings(**{PENDING_KEY: {}})
    if no and not yes:
        return "All right — I have not changed anything."
    pid = pend.get("project")

    if op == "open":
        # From the person's own text, so "call it the Spring Garden" keeps
        # its capitals.
        m = re.search(r"call it\s+(.{2,60}?)[.!?]*$", text or "", re.I)
        title = _clean(m.group(1)) if m else pend.get("title")
        pid = open_project(title, evidence=pend.get("evidence"))
        for a in pend.get("advice") or []:
            log(pid, ADVICE, text=a, state=GIVEN)
        _point(pend.get("session") or "default", pid)
        n = len(pend.get("advice") or [])
        return (f"Kept: {_yours(title)}. I have noted the {n} suggestion"
                f"{'s' if n != 1 else ''} I made so far. Ask me to show the "
                "project whenever you want to see it.")
    p = state(pid) if pid else None
    if not p:
        return "That project is no longer open, so I have not changed it."
    title = _yours(p["title"])

    if op == "done":
        item = pend.get("item")
        if item is None and num:
            k = int(num.group(1)) if num.group(1) else ["first", "second",
                "third", "fourth", "fifth"].index(num.group(2)) + 1
            ch = pend.get("choices") or []
            item = ch[k - 1] if 0 < k <= len(ch) else None
        if item is not None and item in p["items"]:
            it = p["items"][item]
            if it["kind"] == ADVICE:
                log(pid, ADVICE, ref=item, state=ALREADY_DONE,
                    evidence=pend.get("evidence"))
                log(pid, STEP, text=it["text"], state=DONE,
                    evidence=pend.get("evidence"))
            else:
                log(pid, STEP, ref=item, state=DONE,
                    evidence=pend.get("evidence"))
            return f"Done — “{_clean(it['text'], 90)}”, for {title}."
        if new or (yes and item is None):
            ev = pend.get("evidence") or ""
            log(pid, STEP, text=ev, state=DONE, evidence=ev)
            return f"Noted as done, for {title}: “{_clean(ev, 90)}”."
        return "I could not tell which one, so I have not changed anything."

    if op == "choose":
        if pend.get("option"):
            log(pid, OPTION, ref=pend["option"], state=CHOSEN,
                evidence=pend.get("evidence"))
        else:
            log(pid, OPTION, text=pend.get("text"), state=CHOSEN,
                evidence=pend.get("evidence"))
        # ⚠️ THE OTHERS ARE KEPT, as not chosen — never deleted.
        for o in _of(p, OPTION, {HELD}):
            if o["id"] != pend.get("option"):
                log(pid, OPTION, ref=o["id"], state=NOT_CHOSEN,
                    evidence=pend.get("evidence"))
        return f"Decided, for {title}: {pend.get('text')}."

    if op == "decline":
        hit = pend.get("items") or []
        for i in hit:
            kind = p["items"].get(i, {}).get("kind")
            log(pid, kind or ADVICE, ref=i,
                state=NOT_CHOSEN if kind == OPTION else DECLINED,
                evidence=pend.get("evidence"))
        if not hit:
            log(pid, OPTION, text=pend.get("text"), state=NOT_CHOSEN,
                evidence=pend.get("evidence"))
        return f"Ruled out, for {title}: {pend.get('text')}."

    if op in (MET, WITHDRAWN):
        log(pid, STATUS, state=op, evidence=pend.get("evidence"))
        return (f"{title.capitalize()} is marked "
                f"{'finished' if op == MET else 'withdrawn'}. It stays on "
                "record.")
    return None


# ── what the person asks for outright ────────────────────────────────────────

# ⚠️ THE OBJECT HAS TO LOOK LIKE ONE: "this", "my move", "the trip". Without
# that, "make sure the project…" opened a project called "sure the".
_OPEN_CMD = re.compile(
    r"\b(?:make|keep|start|open|save|track)\s+(?:a\s+project\s+(?:for|of|"
    r"called|about)\s+)?((?:this|that|it|my|our|the)\b.{0,56}?)\s+"
    r"(?:as\s+)?(?:a\s+)?project\b"
    r"|\bstart\s+a\s+project\s+(?:for|called|about)\s+(.{2,60}?)(?:[.!?]|$)",
    re.I)
_SHOW = re.compile(
    r"\b(?:show|display|list|give)\s+(?:me\s+)?(?:the\s+|my\s+|all\s+)?"
    r"(?:projects?|plan)\b|\bwhere\s+are\s+we\s+(?:with|on)\b|\bwhat\s+"
    r"(?:have|did)\s+you\s+suggest(?:ed)?\b|\bproject\s+(?:status|view)\b",
    re.I)
_TARGET_CMD = re.compile(
    r"\b(?:the\s+)?(?:target|goal|aim)\s+(?:is|:)\s+(.{3,200}?)\s*$", re.I)
_STEP_CMD = re.compile(
    r"\b(?:add\s+(?:a\s+)?step|the\s+next\s+step\s+is|next\s+step\s*:)\s*:?"
    r"\s*(.{3,160}?)\s*$", re.I)
_FINISH_CMD = re.compile(
    r"\b(?:the\s+)?project\s+is\s+(?:done|finished|complete)\b|\b(?:close|"
    r"finish|drop|cancel|withdraw)\s+(?:the\s+|my\s+|this\s+)?project\b",
    re.I)


def act(user_text, session="default", history=None, db_path=None):
    """Explicit requests, answered in code. Returns (said, only, project).

    `said` is composed text to show before anything is generated; `only`
    means it is the whole reply. ⚠️ Nothing here guesses: each pattern is a
    request the person made in so many words.
    """
    t = " ".join((user_text or "").split())
    if not t:
        return "", False, None
    p = about(t, session, db_path) or pointed(session, db_path)

    if _SHOW.search(t):
        ps = [p] if p else open_projects(db_path)
        if not ps:
            return ("There is no project open yet. When a task has several "
                    "steps and we stop partway, I will offer to keep it."), \
                True, None
        if len(ps) > 1 and not p:
            return ("Open projects:\n" + "\n".join(
                f"  - {_yours(x['title'])} (since {_day(x['opened'])})"
                for x in ps) + "\nName one and I will show it."), True, None
        return view(ps[0]), True, ps[0]

    m = _OPEN_CMD.search(t)
    if m:
        named = (m.group(1) or m.group(2) or "").strip(" .!?")
        if re.fullmatch(r"(?:this|that|it)", named, re.I):
            named, _ = _task_title(history)
        if named and stems(named):
            existing = about(named, session, db_path)
            if existing:
                return (f"{_yours(existing['title']).capitalize()} is already "
                        "a project."), True, existing
            pid = open_project(named, evidence=t, db_path=db_path)
            _point(session, pid)
            return (f"Kept as a project: {_yours(named)}. Tell me what it is "
                    "aimed at, and I will hold it to that."), True, \
                state(pid, db_path)

    if p:
        m = _TARGET_CMD.search(t)
        if m:
            log(p["id"], TARGET, text=_clean(m.group(1)), evidence=t,
                db_path=db_path)
            return (f"The target for {_yours(p['title'])}: "
                    f"{_clean(m.group(1))}."), True, state(p["id"], db_path)
        m = _STEP_CMD.search(t)
        if m:
            log(p["id"], STEP, text=_clean(m.group(1)), state=NEXT,
                evidence=t, db_path=db_path)
            return (f"Next for {_yours(p['title'])}: {_clean(m.group(1))}."), \
                True, state(p["id"], db_path)
        if _FINISH_CMD.search(t):
            op = WITHDRAWN if re.search(r"\b(drop|cancel|withdraw)\b", t,
                                        re.I) else MET
            q = (f"Shall I mark {_yours(p['title'])} as "
                 f"{'withdrawn' if op == WITHDRAWN else 'finished'}?")
            if _propose(op, q, project=p["id"], evidence=t):
                return q, True, p
    return "", False, p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--show", type=int, metavar="ID")
    a = ap.parse_args(argv)
    from . import vault
    if vault.is_set_up() and not vault.unlocked():
        vault.unlock(reason="show your projects")
    if a.show:
        p = state(a.show)
        print(view(p) if p else f"  no project {a.show}")
        return 0
    ps = open_projects()
    if not ps:
        print("  no open projects")
    for p in ps:
        print(f"  {p['id']}  {_yours(p['title'])}  (since {_day(p['opened'])})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
