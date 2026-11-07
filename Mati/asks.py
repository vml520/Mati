"""
The one question Mati asks at the moment it would change what happens.

    python3 -m Mati.asks            is anything waiting?
    python3 -m Mati.asks --clear    drop it

⚠️ THIS IS THE LAST PHASE OF SETUP, AND THE POINT IS THAT SETUP DOES NOT
ASK. A wizard that collects a location, a voice, a consent and a reading list
before the first word is the failure mode the setup principle names. Each of
these is ONE question, asked the first time it would change an answer, and
remembered afterwards.

⚠️ EMITTED, NEVER PROMPTED. The question is composed here and streamed as
Mati's reply; it is never handed to the model to re-say. That rule is written
in blood in this project: 4/4 headlines once invented from a prompt block, and
a denial of having memory while four memories were held. A 3B asked to
"ask the user where they
are" will invent a place and ask about it.

⚠️ AND IT MUST NOT BECOME A FORM. Exactly one ask may be outstanding; raising
a second while one is pending is dropped on purpose. Two questions in a row is
a wizard arriving by instalments.
"""

from __future__ import annotations

import argparse
import sys

from . import config as C

KEY = "pending_ask"

# kind -> (what to do with the answer, what to say when it worked)
# ⚠️ The handler takes the person's raw words. It returns a confirmation
# string on success, or None if the answer could not be used — in which case
# the ask STAYS pending and nothing is stored.
_HANDLERS = {}


def handler(kind):
    def wrap(fn):
        _HANDLERS[kind] = fn
        return fn
    return wrap


def pending():
    """{'kind':…, 'question':…} or None."""
    try:
        got = C._settings().get(KEY)
        return got if isinstance(got, dict) and got.get("kind") else None
    except Exception:
        return None


def raise_(kind, question):
    """Queue the one question. False if something is already waiting."""
    if kind not in _HANDLERS:
        return False
    if pending():
        return False                 # one at a time, deliberately
    try:
        C.save_settings(**{KEY: {"kind": kind, "question": question}})
        return True
    except Exception:
        return False


def clear():
    try:
        C.save_settings(**{KEY: {}})
    except Exception:
        pass


def answer(text_in):
    """Try to read this message as the answer. Returns a reply, or None.

    ⚠️ RETURNING None MEANS "THIS WAS NOT AN ANSWER", and the turn carries on
    normally. Someone who asks a question instead of answering must not have
    their words filed as a location — so a handler that cannot use the text
    leaves the ask pending rather than storing something wrong.
    """
    got = pending()
    if not got:
        return None
    text = " ".join((text_in or "").split())
    if not text:
        return None
    # ⚠️ A SHORT ANSWER, OR NOT AN ANSWER. a bare city name is a reply to "where are
    # you"; three sentences about something else is a new subject, and
    # treating it as the answer would both lose the subject and store rubbish.
    if len(text.split()) > 8:
        return None
    fn = _HANDLERS.get(got["kind"])
    if fn is None:
        clear()
        return None
    # ⚠️ THE HANDLER SEES A "NO" BEFORE THE GENERIC ONE DOES, and getting this
    # backwards was a real bug: the generic decline cleared the ask and
    # returned "All right", so `_idle("no")` never ran and `idle_work` was
    # never set to False. The person said no, Mati said "all right", and the
    # idle work carried on exactly as before — a refusal that changes nothing
    # is worse than not asking, because they believe it took effect.
    try:
        said = fn(text)
    except Exception:
        said = None
    if said:
        clear()
        return said
    # Only now: a decline the handler had no specific use for.
    if _declined(text):
        clear()
        return "All right — I will not ask again."
    return None                      # stays pending; nothing stored


_NO = ("no", "nope", "not now", "later", "skip", "never", "no thanks",
       "don't", "dont", "rather not", "nah")


def _declined(text):
    t = text.lower().strip(" .!?")
    return t in _NO or t.startswith(("no ", "not now", "rather not", "skip"))


# ── the asks themselves ─────────────────────────────────────────────────────

@handler("place")
def _place(text):
    """⚠️ GEOCODED BEFORE IT IS BELIEVED. `set_place` returns None when the
    name matches nothing, which is what keeps "what do you mean" out of the
    settings file as a location."""
    from . import weather
    label = weather.set_place(text)
    if not label:
        return None
    return (f"{label} — I will remember that. Ask me about the weather "
            "whenever you like.")


_YES = ("yes", "yeah", "yep", "sure", "ok", "okay", "please", "go ahead",
        "do", "fine", "alright", "all right", "y", "yes please")


def exchanges_today():
    """How much has actually been said today. 0 if the log cannot be read.

    ⚠️ THE GUARD AGAINST ASKING ON TURN ONE. "The first time it matters" is
    literally true of the very first reply for several of these — Mati would
    have spoken aloud, so it could ask — but three questions stacked onto a
    first conversation is the wizard this phase exists to avoid, just arriving
    later. A few exchanges first means the person has some idea what they are
    saying yes to.
    """
    try:
        from . import daylog
        return sum(1 for r in daylog.rows() if r.get("who") == daylog.MATI)
    except Exception:
        return 0


# ⚠️ STAGGERED ON PURPOSE, and these are not arbitrary: only ONE ask may be
# pending at a time, so two triggers firing in the same window means the
# second is silently dropped and may never be asked again. Spacing them is
# what keeps each one reachable.
AFTER_VOICE = 3
AFTER_READING = 8


def _said_yes(text):
    t = text.lower().strip(" .!?")
    return t in _YES or t.startswith(("yes", "yeah", "sure", "go ahead", "ok"))


@handler("voice")
def _voice(text):
    """⚠️ A YES/NO, and anything else is NOT an answer — it leaves the ask
    pending rather than guessing. "maybe later" is not consent to talk."""
    if not _said_yes(text):
        return None
    from . import config as _C, speech
    _C.save_settings(voice_on=True)
    got = speech.voices()
    extra = f" I will use {_C.assistant_name()}'s default voice; " \
            f"there are {len(got)} to choose from in settings." if got else ""
    return "I will read my replies out loud from now on." + extra


@handler("idle")
def _idle(text):
    """⚠️ ONE QUESTION FOR FOUR BEHAVIOURS, deliberately. Dreaming, studying,
    self-talk and cache warming are four switches, but the person's actual
    concern is singular — is this spending my battery. Four toggles here is
    the wizard arriving by instalments."""
    from . import config as _C
    yes = _said_yes(text)
    if not yes and not _declined(text):
        return None
    _C.save_settings(idle_work=bool(yes))
    if yes:
        return ("I will use quiet moments to tidy what I know. Nothing "
                "leaves this machine, and you can stop it in settings.")
    return "I will stay idle unless you ask me for something."


@handler("interrupt")
def _interrupt(text):
    """⚠️ CONSENT LIVES IN `schedule`, NOT SETTINGS, because quiet hours and
    the per-kind gates are enforced there. Writing a settings flag instead
    would create a second switch that looks authoritative and does nothing."""
    from . import schedule
    yes = _said_yes(text)
    if not yes and not _declined(text):
        return None
    for kind in ("interrupt.reminder", "interrupt.finding"):
        schedule.set_consent(kind, "on" if yes else "off")
    if yes:
        return ("I will speak up when something is due. Not during quiet "
                "hours — those win over this.")
    return "I will keep things until you ask."


@handler("reading")
def _reading(text):
    """⚠️ NEVER BLOCKS, and the cost is stated before anything starts. ~30s a
    chunk means a book is hours of background work, and someone who says yes
    deserves to know that before it begins rather than after."""
    if _said_yes(text):
        return ("Good — drop a plain text file in the library terminal with "
                "`read <path>`, and I will tell you how long it will take "
                "before I start.")
    return None


@handler("encyclopedia")
def _encyclopedia(text):
    """⚠️ A YES CANNOT INSTALL ANYTHING, so it must not sound as though it
    did. This is the only ask whose answer Mati cannot act on alone — it
    needs a binary and a multi-megabyte file the person fetches themselves.
    So a yes stores the intent and hands back the two concrete steps; it does
    not say "done" or start a download."""
    from . import config as _C
    if _said_yes(text):
        _C.save_settings(encyclopedia_wanted=True)
        return ("Good. Two things, both one-off: install kiwix-serve, and put "
                "a .zim file in my encyclopedias folder — `/encyclopedia` in "
                "the terminal shows you where and what is missing. The "
                "subject builds are small; mathematics is 56 MB. Nothing "
                "about it goes online once it is there.")
    if _declined(text):
        _C.save_settings(encyclopedia_wanted=False)
        return "All right — I will answer from what I know instead."
    return None


def ask_encyclopedia_if_needed():
    """They asked for a reference and there is none. True if asked.

    ⚠️ RAISED AT THE MOMENT IT MATTERS, which here means after a lookup has
    actually failed for want of one — not at setup, where it would be a line
    in a wizard about a feature they have no use for yet. Someone who says
    "look that up" has just demonstrated the need.
    """
    from . import config as _C, encyclopedia
    s = _C._settings()
    if s.get("encyclopedia_asked") or s.get("encyclopedia_wanted") is not None:
        return False
    if encyclopedia.connected():
        return False
    _C.save_settings(encyclopedia_asked=True)
    return raise_("encyclopedia",
                  "I do not have an encyclopedia on this machine, so I "
                  "answered from memory. Would you like me to be able to "
                  "look things up offline?")


def ask_place_if_needed():
    """Called when something needed a location and had none. True if asked."""
    from . import weather
    if weather.where():
        return False
    return raise_("place", "Where are you? I only need it once, and it stays "
                           "on this machine.")


def ask_voice_if_needed():
    """First time Mati would have spoken, had it been allowed to."""
    from . import config as _C, speech
    s = _C._settings()
    # ⚠️ ASKED ONCE, EVER. `voice_asked` is what stops this returning every
    # turn for someone who said no — a question that comes back is nagging,
    # and the spec's whole premise is one question at the right moment.
    if s.get("voice_on") or s.get("voice_asked") or not speech.available():
        return False
    if exchanges_today() < AFTER_VOICE:
        return False
    _C.save_settings(voice_asked=True)
    return raise_("voice", "Would you like me to read my replies out loud?")


def ask_idle_if_needed():
    """First long gap. ⚠️ Not on a short pause — see the caller."""
    from . import config as _C
    s = _C._settings()
    if s.get("idle_work") is not None or s.get("idle_asked"):
        return False
    _C.save_settings(idle_asked=True)
    return raise_("idle", "May I use quiet moments to go over what I know? "
                          "It costs battery, and nothing leaves this machine.")


def ask_interrupt_if_needed():
    """First time something was due and could not be delivered."""
    from . import config as _C, schedule
    if _C._settings().get("interrupt_asked"):
        return False
    c = schedule.consent()
    if any(c.get(k) == "on" for k in
           ("interrupt.reminder", "interrupt.finding")):
        return False
    _C.save_settings(interrupt_asked=True)
    return raise_("interrupt", "Something is due. May I tell you when things "
                               "come up, or should I wait until you ask?")


def ask_reading_if_needed():
    """After the first real conversation, once."""
    from . import config as _C
    if _C._settings().get("reading_asked"):
        return False
    if exchanges_today() < AFTER_READING:
        return False
    _C.save_settings(reading_asked=True)
    return raise_("reading", "Is there anything you would like me to read? "
                             "I can study a text file in the background.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clear", action="store_true")
    a = ap.parse_args(argv)
    if a.clear:
        clear()
        print("  cleared")
        return 0
    got = pending()
    print(f"  waiting: {got['kind']} — {got['question']}" if got
          else "  nothing waiting")
    print(f"  handlers: {', '.join(sorted(_HANDLERS))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
