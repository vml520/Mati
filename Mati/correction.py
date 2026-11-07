"""
Being corrected, and remembering that you were.

When Mati gets something wrong and you tell it so, that is the most valuable
thing that happens in a conversation — and until now it was the most damaging.

WHAT WENT WRONG WITHOUT THIS. An episode records `I said: <the reply>`. If the
reply was wrong, the wrong answer is what gets stored, imprinted into θ₀, and
replayed into later turns as lived memory:

    I was asked: When was Teotihuacan founded?
    I said: Teotihuacan was founded around 1200 CE by the Mexica.

Nothing marked it. Your correction became a separate note with no link to it,
and might not surface at all — so Mati could recall the error cleanly and
repeat it, having also been told the truth. That is the memory loop amplifying
errors, on the path where it matters most.

WHAT THIS DOES. A user turn that reads as a correction is linked to the episode
it corrects. From then on, whenever that episode is recalled, the correction
comes with it — not as a separate memory that may or may not surface, but
attached, in the same line.

CORRECTIONS ANNOTATE. THEY DO NOT ERASE. Mati did say the wrong thing; that
happened, and deleting it would be a lie of a different kind — the field is
supposed to carry what occurred. So the episode stays, and can never surface
unmarked. "I said X, and you told me that was wrong" is a true and useful
memory. "I said X" alone is a trap.

DETECTION IS DELIBERATELY CONSERVATIVE. Two conditions, both required: an
explicit correction marker, and a preceding reply to correct. A missed
correction costs one annotation. A false positive brands a good memory as an
error and teaches Mati to distrust something true — much worse — so the
markers are ones that rarely appear except when correcting.
"""

import re

# Explicit markers only. "Actually" on its own is far too common in ordinary
# speech to be a signal; it counts only when it opens the turn or follows a
# negation, which is how people actually correct each other.
# Openers must be followed by PUNCTUATION, not whitespace. `^no\b\s` fired on
# "No idea, what do you think?" and "No rush — whenever you're ready", and
# `^wrong\b\s` on "Wrong turn on the way here". Requiring the comma is what
# separates a rejection from a word that merely starts a sentence.
_MARKERS = re.compile(
    r"^\s*(?:no|nope|wrong|incorrect)\s*[,.!:;—-]"
    # Bare "actually" opens suggestions as often as corrections — "Actually,
    # let's talk about something else" is not one. Excluded when what follows
    # proposes rather than asserts.
    r"|^\s*actually\b(?!\s*,?\s*(?:let'?s|can we|could we|shall we|maybe|"
    r"i'?d|i would|why don'?t))"
    # The classic correction shape: an assertion, then "not X".
    r"|\b(?:it was|it'?s|that was|that'?s|they were)\b[^.!?]{0,40},\s*not\b"
    r"|\bthat(?:'s| is) (?:not right|wrong|incorrect|false)\b"
    r"|\bnot (?:quite )?(?:right|correct|true)\b"
    r"|\byou(?:'re| are) (?:wrong|mistaken|confusing)\b"
    r"|\bi (?:meant|said)\b.{0,40}\bnot\b"
    r"|\bit(?:'s| was) (?:actually|really)\b"
    r"|\bthat never happened\b"
    r"|\bi never (?:said|told you)\b"
    r"|\bmisremember(?:ed|ing)?\b"
    r"|\bgot that wrong\b",
    re.I)


def looks_like_correction(user_text, had_reply=True):
    """Is this turn correcting something Mati just said?

    `had_reply` guards the other half: with nothing said yet there is nothing
    to correct, and an opening "no, tell me about X" is not a correction.
    """
    if not had_reply or not user_text:
        return False
    return bool(_MARKERS.search(user_text))


def narrate(user_text, wrong_reply=None):
    """The episode body for a correction — what was wrong AND what is right.

    Stored as lived experience, because it is: the correction happened in
    conversation. No provenance question arises, unlike studied material.
    """
    parts = [f"I was corrected: {' '.join(user_text.split())}"]
    if wrong_reply:
        said = " ".join(wrong_reply.split())
        parts.append(f"What I had said was: {said[:300]}")
    parts.append("I was wrong, and I know it because I was told so, "
                 "not because I worked it out.")
    return "\n".join(parts)


def annotate(body, correction_text):
    """Attach a correction to the episode it corrects, for recall.

    Appended to the BODY rather than handed to the caller as a separate field,
    so that no consumer can forget to render it. `recall` does this on the way
    out, which means every path — the state block, the episode list, the
    context an inquiry is given — sees the correction with the error.
    """
    return (f"{body}\n[LATER CORRECTED BY THE USER: "
            f"{' '.join(correction_text.split())[:300]}"
            f" — do not repeat what you said above as if it stood.]")


# Leading correction markers, so the CLAIM can be separated from the rejection.
_STRIP = re.compile(
    # `\b` matters: without it the `no` alternative matches inside "Nope" and
    # leaves "pe, the Olmec came first" as the claim.
    r"^\s*(?:nope|no|wrong|incorrect|actually)\b\s*[,.!:;—-]*\s*"
    r"|^\s*that(?:'s| is) (?:not right|wrong|incorrect|false)\s*[,.!:;—-]*\s*"
    r"|^\s*not (?:quite )?(?:right|correct|true)\s*[,.!:;—-]*\s*"
    r"|^\s*you(?:'re| are) (?:wrong|mistaken)\s*[,.!:;—-]*\s*"
    # Reassertion. Expected on a SECOND correction by definition — people
    # restate when they have already said it once — and it is noise in the
    # claim: "No, I'm telling you again, 91 is prime" should research
    # "91 is prime", not the fact that they repeated themselves.
    r"|^\s*(?:i'?m telling you again|i already (?:told you|said)|"
    r"like i said|as i said|again)\s*[,.!:;—-]*\s*",
    re.I)


def subject(user_text):
    """The claim inside a correction, with the rejection stripped off.

    "No, that's wrong — 91 is prime." -> "91 is prime"

    Used when Mati defers and wants to go and look the claim up: what it needs
    to research is what it was told, not the fact that it was told off.
    """
    text = " ".join((user_text or "").split())
    prev = None
    while text != prev:                    # markers stack: "No, that's wrong,"
        prev = text
        text = _STRIP.sub("", text, count=1)
    return text.strip(" .,;:—-") or " ".join((user_text or "").split())


def defer(user_text, prior_claim=None):
    """The episode body when Mati was corrected twice and stood down.

    Recorded as a change of position with its reason, not as a quiet edit — a
    reader should be able to see that Mati checked, disagreed, was told again,
    and yielded, and that yielding was a decision.
    """
    claim = subject(user_text)
    parts = [f"I was corrected a second time: {' '.join(user_text.split())}"]
    if prior_claim:
        parts.append(f"I had checked and stood by: {prior_claim}")
    parts.append(f"They told me twice, so I accept it: {claim}. "
                 f"I have not verified it myself — I am going to look into it.")
    return "\n".join(parts)
