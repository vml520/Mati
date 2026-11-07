"""
When Mati has grounds to push back.

    (used by `mati.turn` — no CLI)

Being corrected is usually the end of it: you were there, Mati was not, and
most of what it holds came from a 3B model that confabulates. Deferring is not
politeness, it is correct — you have better evidence than it does.

There is exactly one class where that is not true. When Mati COMPUTED an
answer, it can derive it again from nothing, and the derivation is checkable by
both of you. A wrong correction to a computed result is the one case where
agreeing would make the conversation worse.

THE RULE: CHALLENGE ONLY WHAT YOU CAN RECOMPUTE. Everything else, accept.

That falls straight out of the provenance kinds Mati already tracks, which
form an authority ordering against the user:

    computed    deterministic and re-derivable   -> higher. Challenge.
    remembered  you were there too               -> equal.  Accept.
    studied     a document, or the 3B            -> lower.  Accept.
    known       the 3B                           -> lowest. Accept.

This is the same principle as everywhere else in `PRINCIPLES.md`: never claim
more authority than your source warrants. Here it happens to license
disagreement rather than restrain it.

HOW IT DECIDES. The previous user turn is re-routed through `reckon`. If it
routed to a computation, the computation is run again — from scratch, not from
memory — and presented as working. If it does not route, or the engine declines,
there is no challenge and the correction stands.

WHAT IT IS CAREFUL ABOUT. The arithmetic being right does not mean Mati
answered the right question: the router may have misread what was asked. So the
challenge is about the computation, never about the user, and it says so — the
one thing the user is always the authority on is what they meant.

It also challenges ONCE. If you correct it again, that is your call and it
stands; an assistant that keeps relitigating is worse than one that is
occasionally wrong.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from . import reckon


def recheck(prev_user_text):
    """Re-derive the previous turn's computation, if it had one.

    Deliberately re-runs `reckon` from the original question rather than
    trusting anything stored: a recomputation that reads its answer out of
    memory proves nothing, and memory is what is in dispute.
    """
    if not prev_user_text:
        return None
    try:
        return reckon.reckon(prev_user_text)
    except Exception:
        return None                      # engine trouble is not grounds to argue


def block(result, correction_text):
    """The prompt block for a challenge. Shows the working, not a verdict."""
    return (
        "<rechecked>\n"
        f"The user says the previous answer was wrong: "
        f"{' '.join(correction_text.split())[:200]}\n"
        f"You did not remember that answer — you COMPUTED it, so you have just "
        f"computed it again from scratch:\n"
        f"    {result['computation']} = {result['result']}\n"
        f"    {result.get('explanation', '')}\n"
        "The arithmetic is deterministic and it came out the same. Say so, "
        "plainly and without apology, and show the working so they can check "
        "it.\n"
        "BUT: being right about the arithmetic does not mean you answered the "
        "question they meant. If they say you misunderstood what they were "
        "asking, they are the authority on that and you should accept it "
        "immediately. Disagree about the number, never about what they meant.\n"
        "Do not raise this more than once.\n"
        "</rechecked>"
    )


def narrate(correction_text, result):
    """The episode body when Mati stood by a computed answer.

    Recorded as what it was — being told it was wrong, checking, and finding
    the same thing — so a later reader can see that the disagreement happened
    and how it was grounded, rather than only that a correction occurred.
    """
    return "\n".join([
        f"I was told I was wrong: {' '.join(correction_text.split())}",
        f"I had not remembered that answer, I had worked it out, so I worked it "
        f"out again: {result['computation']} = {result['result']}.",
        "It came out the same, so I said so and showed the working. "
        "If I misread the question, that part is theirs to settle.",
    ])
