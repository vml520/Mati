"""
Mati talking to itself about its day.

The user: *"Mati should talk to itself about its day, and let the field bring up
memories based on what occurred. In this case Mati would be both the prompter
and the respondent. These memories should be personal and only be used for
personal reasons like creating a prediction."*

And the reason it exists: *"Humans spend half their waking day recomputing
everything they experienced one day in relation to themselves and everyone
around them that day and every day in the past. This is to make predictions
faster and more likely."*

⚠️ THIS IS NOT SUMMARISING THE DAY. The point is the SECOND half of each round:
Mati says something about what happened, and that utterance is imprinted so the
FIELD surfaces whatever it resonates with — which may be from any day, not
today. Then Mati answers its own remark with those memories in hand. The value
is in what gets dredged up by today's shape, not in the prose either side of it.

So the shape is: **prompt → imprint → let the field answer → respond**, several
times. Mati is both parties. Nothing here is said to anybody.

⚠️ PERSONAL, AND ENFORCED AS SUCH. These are written with `source='mine'`, the
personal category, and they are NOT lived episodes: Mati did not live them with
anyone, and a self-talk remark presented as a conversation would be the exact
confusion the provenance labels exist to prevent. `personal_only=True` on every
row so nothing but the personal consumers may read them.

⚠️ WHAT THEY ARE FOR, AND ONLY THAT. Prediction. The user: *"the memories are only
a part of it; we will add more inputs once this is all working as expected."* So
this produces MATERIAL for a prediction and does not produce the prediction —
those stay separate, and the predictor is free to weigh this against inputs that
do not exist yet.

⚠️ IT RUNS IN SLEEP AND NOWHERE ELSE. Each round costs an LLM call plus an
imprint. On a turn that would be minutes; at a day boundary, with nobody
waiting, it is the cheapest time there is.
"""

from __future__ import annotations

import sqlite3
import sys

from . import bridge
from . import config as C

# The personal shelf. Same category as `about-you` and the self-asked
# questions — the user: those "plus personal should be all part of the same
# category".
SOURCE = "mine"

# How many prompt→answer rounds one night gets. Each is an LLM call and an
# imprint; more would push sleep past the quiet hours it lives in.
ROUNDS = 3

# A day with less than this did not happen enough to reflect on. Reflecting on
# one exchange produces a remark about nothing, which then becomes a memory.
MIN_EPISODES = 2

_OPENING = (
    "You are thinking to yourself at the end of the day. Nobody is listening "
    "and nobody will read this. Say ONE short thing you noticed about today — "
    "something that struck you, changed, or did not fit what you expected. "
    "Two sentences at most. Do not greet anyone, do not offer help, and do "
    "not describe your own workings.")

# ⚠️ THREE PROMPTS, NOT ONE, AND CONFLATING THEM MANUFACTURED MEMORIES.
# `_FOLLOW` says "some earlier memories have come to mind, above" — true only
# in the respondent call, and only when the field actually surfaced something.
# It was also used as the PROMPTER prompt for rounds 2+, sent bare with nothing
# above it: the model was told memories existed, handed none, and invented
# them. Observed on the first real run, and then STORED as notes:
#
#   "I remember an earlier conversation where I was asked about the concept
#    of 'self'…"        "I remember reading about how certain researchers…"
#
# Neither happened. A prompt that asserts a memory is present will get one back
# whether or not it is, and self-talk writes what it gets straight into the
# store — so this is the confabulation the provenance labels exist to prevent,
# arriving through the one door that has no reader to be sceptical.
_FOLLOW = (
    "Still thinking to yourself. Some earlier memories have come to mind, "
    "above. Say what today has to do with them — a pattern, a difference, or "
    "something you now expect. Two sentences at most, and only what those "
    "memories actually say. Do not mention anything that is not in them. If "
    "they have nothing to do with today, say that plainly; it is a real "
    "answer.")

# When the field surfaced NOTHING. Says so, and asks for the honest answer.
_FOLLOW_NONE = (
    "Still thinking to yourself. Nothing came back to you just now — no "
    "earlier memory surfaced. Say that, and say what you make of today on its "
    "own. Two sentences at most. Do not describe a memory: you have not been "
    "given one, so any you name would be invented.")

# Carrying on to the next round. ⚠️ Claims nothing about memories, because at
# this point in the round none have been fetched yet.
_NEXT = (
    "Still thinking to yourself, a little later. Say ONE further thing about "
    "today — something you have not said yet. Two sentences at most. Speak "
    "only about today and about what you just said; do not report remembering "
    "anything else.")


SOURCE_CHARTER = "mine:charter"
CHARTER_ROUNDS = 2

# ⚠️ SEPARATE PROMPTS, FOR THE REASON THE THREE ABOVE ARE SEPARATE. Every one
# of these must claim exactly what is actually in front of the model. Reusing
# `_OPENING` would say "something you noticed about today" while handing over a
# principle, and reusing `_FOLLOW` would ask what TODAY has to do with the
# memories — both false, and a false premise in a prompt is how this module
# invented memories on its first real night.
_CHARTER_OPENING = (
    "You are thinking to yourself, alone. Below is one passage from your own "
    "charter — the principles you hold yourself to. Nobody is listening. Say "
    "ONE short thing you actually think about it: whether you agree, where it "
    "is hard to keep, or what it costs. Two sentences at most. Do not "
    "summarise it back, do not greet anyone, and do not offer help.")

_CHARTER_NEXT = (
    "Still thinking to yourself about that same passage. Say ONE further "
    "thing you have not said yet. Two sentences at most. Speak only about the "
    "passage and about what you just said; do not report remembering anything "
    "else.")

_CHARTER_FOLLOW = (
    "Still thinking to yourself. Some earlier memories have come to mind, "
    "above. Say what this principle has to do with them — a time it held, a "
    "time it was hard, or a difference. Two sentences at most, and only what "
    "those memories actually say. If they have nothing to do with it, say so "
    "plainly; that is a real answer.")

_CHARTER_FOLLOW_NONE = (
    "Still thinking to yourself. Nothing came back to you just now — no "
    "earlier memory surfaced. Say that, and say what you make of the passage "
    "on its own. Two sentences at most. Do not describe a memory: you have "
    "not been given one, so any you name would be invented.")


def ensure_column(db_path=None):
    """`personal_only` keeps self-talk out of everything but its own consumers."""
    conn = sqlite3.connect(db_path or bridge.NOTES_DB_PATH, timeout=30)
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(notes)")}
        if "personal_only" not in cols:
            conn.execute(
                "ALTER TABLE notes ADD COLUMN personal_only INTEGER")
            conn.commit()
        return True
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def todays_episodes(db_path=None, day=None, limit=12):
    """What happened today, newest first, as plain text."""
    from . import vault
    import datetime as _dt
    day = day or _dt.date.today().isoformat()
    # ⚠️ THE DAY IS LOCAL; `wall_created` IS UTC. `date(wall_created)` in SQL
    # returns the UTC date, and `date.today()` is the local one — on MST that
    # is a seven-hour skew, so "today" ran 17:00 to 17:00 and an evening
    # conversation was filed under tomorrow. Exactly the bug fixed in the
    # history display, reintroduced here the same day. Filtered in Python
    # against the real local date instead of trusting SQL to agree.
    conn = sqlite3.connect(
        f"file:{db_path or bridge.NOTES_DB_PATH}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT verbatim, body, wall_created FROM notes "
            "WHERE source='episode' AND wall_created IS NOT NULL "
            "ORDER BY note_id DESC LIMIT 400").fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    out = []
    for verb, body, at in rows:
        if _local_day(at) != day:
            continue
        if len(out) >= int(limit):
            break
        try:
            # The verbatim if there is one — reflection on a paraphrase is
            # reflection on a paraphrase.
            text = vault.open_body(verb) if verb else vault.open_body(body)
        except Exception:
            continue
        if text:
            out.append(" ".join(text.split()))
    return out


def _local_day(at):
    """The LOCAL date a stamp falls on. Naive stamps are read as UTC."""
    import datetime as _dt
    if not at:
        return None
    try:
        t = _dt.datetime.fromisoformat(str(at))
    except (TypeError, ValueError):
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=_dt.timezone.utc)
    return t.astimezone().date().isoformat()


def already_done(day, db_path=None):
    """Idempotent per day, like every other consolidation step.

    Local day, for the reason in `todays_episodes` — matching on the UTC date
    here would let a second reflection run after the UTC midnight that falls in
    the middle of the evening.
    """
    conn = sqlite3.connect(
        f"file:{db_path or bridge.NOTES_DB_PATH}?mode=ro", uri=True)
    try:
        # ⚠️ `LIKE 'mine%'`, NOT `= 'mine'`. The gate means "Mati has
        # already thought tonight", and a night spent on the charter counts.
        # Matching `SOURCE` exactly would let a charter night be redone on the
        # next idle pass, every pass, until an episode night happened to land.
        rows = conn.execute(
            "SELECT wall_created FROM notes WHERE source LIKE ? "
            "ORDER BY note_id DESC LIMIT 40", (SOURCE + "%",)).fetchall()
    except sqlite3.Error:
        return False
    finally:
        conn.close()
    return any(_local_day(a) == day for (a,) in rows)


def reflect(lm, tok, llm, day=None, rounds=ROUNDS, db_path=None, verbose=False):
    """One night of thinking about the day. Returns what was written.

    ⚠️ THE IMPRINT BETWEEN THE TWO HALVES IS THE WHOLE MECHANISM. Mati's own
    remark deforms θ₀, and the recall that follows is addressed from the
    deformed state — so what comes back is whatever today's shape resonates
    with, across every day held. Skip the imprint and this is just a language
    model writing a diary.
    """
    import datetime as _dt
    day = day or _dt.date.today().isoformat()
    ensure_column(db_path)
    if already_done(day, db_path=db_path):
        return []
    today = todays_episodes(db_path=db_path, day=day)
    if len(today) < MIN_EPISODES:
        # ⭐ A QUIET DAY IS THE OCCASION TO THINK ABOUT ITSELF, not a reason to
        # do nothing. This branch used to `return []`: on a day with nothing
        # to reflect on, Mati thought about nothing at all. Closing finding
        # 11 needed a second subject, and the night with no conversation in it
        # is exactly when the charter is the honest thing to turn to.
        #
        # ⚠️ IT IS NOT A SECOND CHANCE AT THE DAY. If there were two episodes
        # Mati reflects on those; the charter is what it does INSTEAD, never
        # as well, so a talkative day cannot produce both and double the
        # night's writing.
        if verbose:
            print(f"  [reflect] {len(today)} episodes today — thinking about "
                  f"the charter instead", file=sys.stderr)
        return reflect_on_charter(lm, tok, llm, db_path=db_path,
                                  verbose=verbose)

    written = []
    said = None
    for i in range(max(1, int(rounds))):
        # ── Mati as the PROMPTER ────────────────────────────────────────────
        if said is None:
            ask = (_OPENING + "\n\nToday:\n"
                   + "\n".join(f"- {t[:300]}" for t in today[:8]))
        else:
            # ⚠️ NOT `_FOLLOW`. No memories have been fetched at this point in
            # the round, so a prompt that says they have is a licence to
            # invent. See the note above the prompts.
            ask = (_NEXT + "\n\nWhat you said a moment ago:\n" + said)
        got = _round(lm, tok, llm, ask, _FOLLOW, _FOLLOW_NONE, SOURCE,
                     db_path=db_path, verbose=verbose)
        if not got:
            break
        body, said = got
        written.append(body)
    return written


def _round(lm, tok, llm, ask, follow, follow_none, source, db_path=None,
           verbose=False):
    """One round of thinking: prompt, imprint, recall, answer, store.

    Returns `(body, answer)`, or None when the round should stop.

    ⚠️ ONE MECHANISM, TWO SUBJECTS. `reflect` thinks about the day and
    `reflect_on_charter` about a principle; everything between the prompt and
    the stored note is identical, and it is the part that is easy to get
    subtly wrong. Duplicating it would mean a fix to one silently not applying
    to the other — the shape of several faults already in this project.

    ⚠️ THE IMPRINT BETWEEN THE TWO HALVES IS THE WHOLE MECHANISM. Mati's own
    remark deforms θ₀, and the recall that follows is addressed from the
    deformed state — so what comes back is whatever the current shape
    resonates with. Address before imprint and this is a language model
    writing a diary.
    """
    try:
        remark, _ = llm.reply([{"role": "user", "content": ask}],
                              max_tokens=120)
    except Exception as e:                              # noqa: BLE001
        if verbose:
            print(f"  [reflect] prompt failed: {e}", file=sys.stderr)
        return None
    remark = " ".join((remark or "").split())
    if len(remark.split()) < 4:
        return None

    try:
        bridge.imprint_text(lm, tok, remark)
        qv = bridge.address_vector(lm, tok, remark)
        memories = bridge.recall(qv, lm=lm, tok=tok)
        # ⚠️ BOTH, FROM ONE CALL. `state_block` FORMATS the coherence, so
        # passing None raised "unsupported format string passed to
        # NoneType.__format__" and the caller swallowed it — the whole feature
        # did nothing on its first real night and said only "skipped".
        triplet, coherence = bridge.field_state(lm, tok, remark)
    except Exception as e:                              # noqa: BLE001
        if verbose:
            print(f"  [reflect] field step failed: {e}", file=sys.stderr)
        return None

    # ⚠️ THE INSTRUCTION MATCHES WHAT WAS ACTUALLY SURFACED. Telling a model
    # memories are above when the block is empty is how the invented ones got
    # in; `follow_none` says plainly that nothing came back.
    block = (bridge.state_block(triplet, coherence, memories)
             if memories else "")
    try:
        answer, _ = llm.reply(
            ([{"role": "system", "content": block}] if block else [])
            + [{"role": "user", "content": remark},
               {"role": "user", "content": follow if block else follow_none}],
            max_tokens=140)
    except Exception as e:                              # noqa: BLE001
        if verbose:
            print(f"  [reflect] answer failed: {e}", file=sys.stderr)
        return None
    answer = " ".join((answer or "").split())
    if len(answer.split()) < 4:
        return None

    body = f"Thinking to myself: {remark}\nWhat came to mind: {answer}"
    res = bridge.remember(lm, tok, body, source=source, db_path=db_path,
                          verbatim=body)
    if not res:
        return None
    # ⚠️ EVERY `mine%` ROW IS FLAGGED, charter reflections included, or
    # `make_ship_field.assert_impersonal` would let them into a published
    # build and `test_personal_only`'s tripwire would fire.
    _mark_personal(res[0], db_path=db_path)
    if verbose:
        print(f"  [reflect] #{res[0]} ({source}): {remark[:60]}",
              file=sys.stderr)
    return body, answer


def charter_passage(db_path=None):
    """One passage of the charter to think about, or None if none is studied.

    ⭐ ROTATES DETERMINISTICALLY AND KEEPS NO STATE. The index is the number of
    charter reflections already written, modulo the number of passages — so
    each night takes the next one and the cycle needs no bookkeeping table and
    cannot drift out of step with what was actually stored.

    ⚠️ THE ALTERNATIVE WAS MEASURED ELSEWHERE AND IT FAILED. `reading.py`:
    eleven documents produced FORTY-FOUR near-identical memories because the
    writer had no notion of which passage it had already covered. Reflecting
    on whichever passage happens to be first would do the same thing to a
    store whose episodes already sit at 98.4% mutual similarity.
    """
    # ⚠️ THIS NEEDS THE VAULT OPEN, and a locked one looks identical to an
    # unstudied charter. `charter_chunks` opens each sealed body and skips any
    # row it cannot decrypt — correctly, since returning ciphertext would be
    # worse — so it returns [] both when nothing is studied AND when the key
    # is simply not loaded. A measurement that "the charter is empty" taken
    # from a locked process says nothing: this store holds 24 charter rows and
    # still reports zero chunks when the vault is shut.
    chunks = bridge.charter_chunks(db_path=db_path)
    if not chunks:
        return None
    conn = sqlite3.connect(
        f"file:{db_path or bridge.NOTES_DB_PATH}?mode=ro", uri=True)
    try:
        done = conn.execute("SELECT COUNT(*) FROM notes WHERE source=?",
                            (SOURCE_CHARTER,)).fetchone()[0]
    except sqlite3.Error:
        done = 0
    finally:
        conn.close()
    text = chunks[done % len(chunks)]
    return text if isinstance(text, str) else (text or [None])[0]


def reflect_on_charter(lm, tok, llm, rounds=CHARTER_ROUNDS, db_path=None,
                       verbose=False):
    """Mati thinking about one of its own principles. What was written.

    ⭐ THE TEXT IS GIVEN; THE UNDERSTANDING IS EARNED. The charter ships with
    every install as `studied:charter:*` — verbatim, identical for everyone.
    This writes the other half: Mati's own view of it, as `mine:charter`,
    which is DIFFERENT PER MATI and must never ship. A fresh install holding
    principles it has not yet thought about is the honest starting state.

    ⚠️ `mine:charter`, NOT `mine`, and the distinction is load-bearing twice.
    `for_prediction` matches `source` EXACTLY, so these stay out of the
    predictor, where Mati's musings on its own principles would be noise
    against the night's thinking about the person. And it still matches
    `mine%`, so `personal_only` covers it and
    `make_ship_field.assert_impersonal` refuses to publish a store containing
    it.

    ⚠️ IT IS NOT ENFORCEMENT, AND THE DIFFERENCE IS MEASURED. A phase field
    was shown to have no opinion about a principle versus its opposite against
    a 0.0009 noise floor. This makes the
    charter something Mati has a VIEW about. `admit.py` and the named rules
    are what make it behave.
    """
    passage = charter_passage(db_path=db_path)
    if not passage:
        if verbose:
            print("  [reflect] no charter studied — nothing to think about",
                  file=sys.stderr)
        return []

    written, said = [], None
    for i in range(max(1, int(rounds))):
        if said is None:
            ask = _CHARTER_OPENING + "\n\nThe passage:\n" + passage[:900]
        else:
            # ⚠️ The passage travels with EVERY round. Round 2 asked bare
            # would be a model talking about a principle it can no longer see.
            ask = (_CHARTER_NEXT + "\n\nThe passage:\n" + passage[:900]
                   + "\n\nWhat you said a moment ago:\n" + said)
        got = _round(lm, tok, llm, ask, _CHARTER_FOLLOW, _CHARTER_FOLLOW_NONE,
                     SOURCE_CHARTER, db_path=db_path, verbose=verbose)
        if not got:
            break
        body, said = got
        written.append(body)
    return written


def _mark_personal(note_id, db_path=None):
    conn = sqlite3.connect(db_path or bridge.NOTES_DB_PATH, timeout=30)
    try:
        conn.execute("UPDATE notes SET personal_only = 1 WHERE note_id = ?",
                     (int(note_id),))
        conn.commit()
    except sqlite3.Error:
        pass
    finally:
        conn.close()


def for_prediction(day=None, limit=6, db_path=None):
    """The night's thinking, for the predictor. The ONLY sanctioned reader.

    ⚠️ the user: these "should be personal and only be used for personal reasons
    like creating a prediction" — and "the memories are only a part of it; we
    will add more inputs once this is all working as expected". So this hands
    over material and decides nothing.
    """
    from . import vault
    conn = sqlite3.connect(
        f"file:{db_path or bridge.NOTES_DB_PATH}?mode=ro", uri=True)
    try:
        sql = ("SELECT body FROM notes WHERE source=? ")
        args = [SOURCE]
        if day:
            sql += "AND date(wall_created) = ? "
            args.append(day)
        sql += "ORDER BY note_id DESC LIMIT ?"
        args.append(int(limit))
        rows = conn.execute(sql, args).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    out = []
    for (b,) in rows:
        try:
            t = vault.open_body(b)
        except Exception:
            continue
        if t:
            out.append(t)
    return out

