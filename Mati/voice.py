"""
Keeping the machinery out of Mati's mouth.

Every turn hands the model a private `<field_state>` block — basin, coherence,
the recalled episodes. The system prompt tells it, at length, never to mention
any of that: speak *from* it, the way proprioception reaches a person, not
*about* it. On a 3B model that instruction does not hold. Observed, verbatim:

    "I remember (no, wait, I mean, it's listed in the basin) saying something
     about sacred sites earlier"

A prohibition in the prompt is a request. This is the enforcement.

WHAT IT MUST NOT DO. The same system prompt actively *wants* Mati to say
"I remember when you asked me about sacred sites" — that phrasing is the goal,
not the leak. A filter that eats genuine memory talk would break the thing it
is protecting. So the rule is not "remove these words", it is "remove the
places where Mati cites the block as its source", and it gives up entirely
when the user asked about the machinery, because then explaining it is correct.

Two passes, cheapest first:

  1. Asides — parentheticals and dashed interjections that reference internals.
     This is the common shape, and cutting just the aside preserves the good
     sentence around it. The example above becomes "I remember saying something
     about sacred sites earlier", which is exactly right.
  2. Sentences — a whole sentence that cites the block as a source, or recites
     its numbers, goes.

It never returns nothing. If filtering would empty the reply, the original is
returned unfiltered: a visible leak is a smaller failure than a blank answer,
and it stays visible so it can be fixed.

Root causes still get fixed at the source — `bridge._narrate_work` no longer
writes step counts into episodes. This is defence in depth, not a substitute.
"""

import re

# Tokens that only appear in a reply because the block put them there. `basin`
# and `coherence` are ordinary English words, which is a real risk — mitigated
# by `asked_about_machinery`, which switches this whole module off when the
# user raised the topic themselves.
_INTERNAL = re.compile(
    r"""(?:
        field[_\s]state | memories[_\s]recalled | </?field_state>?
      | \bresonance\b | \bbasin\b | \bcoherence\b | \btriplet\b
      | \b(?:the|that|this)\s+(?:state\s+)?block\b
      | \bmy\s+(?:field\s+)?state\s+block\b
      # ⚠️ BARE "my field" / "your field". The alternatives above all require a
      # second machinery word after "field", so "My field has a wealth of
      # information on this topic" walked straight through — and it was not an
      # edge case: , 32 of 33 stored episodes contain a bare
      # "my field" and this filter caught NONE of them. Mati narrating its own
      # substrate is the exact thing this module exists to stop.
      #
      # "of" is excluded so a field of study, or a field of maize, still reads
      # as English. Everything else after "my field" — has, is, was, brought,
      # surfaced, remembers, can, will, 's — is the machinery talking.
      | \b(?:my|your)\s+field\b(?!\s+of\b)
      # ⚠️ THE PROVENANCE BLOCK NAMES, AND THE LEGEND'S OWN WORDING FOR THEM.
      # `<field_state>` was covered and nothing else was, so every other block
      # the system prompt teaches was free to come out of Mati's mouth.
      # Measured  on the user's session: FOUR of the last eight stored
      # replies were, in full,
      #
      #     <computed> I worked it out just now, deterministically.
      #     Not remembered. </computed>
      #
      # — the legend LINE, spoken as the answer, then written into the episode
      # as a memory of having said it. They read that as training not being
      # recalled; Mati was reciting a label.
      #
      # The definitions are listed as well as the tags, because stripping
      # `<computed>` alone leaves the sentence that defines it, which is not an
      # answer to anything either.
      | </?(?:computed|recalled_by_name|waking|your_principles)>?
      | \bworked\s+it\s+out\s+just\s+now,?\s+deterministically\b
      | \bfetched\s+because\s+the\s+words\s+matched\b
    )""",
    re.I | re.X,
)

# Reciting the block's own numbers: "coherence: 0.9170", "basin [0.01, ...]",
# "(resonance 0.831)", "my coherence is 0.94".
_INTERNAL_NUMBER = re.compile(
    r"(?:basin|coherence|resonance|triplet)\b[^.!?]{0,24}?\d*\.\d+", re.I)

# Instrumentation counters that reached episodes written before
# `bridge._narrate_work` stopped emitting them. 11 stored episodes still carry
# "I worked on it for 76 seconds, my field turning over 774 times", and the
# model recites them back — it is quoting its own memory, faithfully, which no
# amount of prompt instruction will stop.
#
# Deliberately narrow: these two exact shapes, not "seconds" in general. Asked
# how long something took, Mati should be able to answer. Cleaning the stored
# bodies instead would mean rewriting its autobiography and re-addressing every
# one of them, which is the user's call to make, not this filter's.
_LEGACY_COUNTERS = re.compile(
    r"\bworked\s+on\s+it\s+for\s+\d+\s*seconds?"
    r"|\bfield\s+turn(?:ing|ed)\s+over\s+\d+",
    re.I)

# Parentheticals and dashed interjections. Non-greedy, and bounded so a stray
# bracket cannot swallow the rest of the reply.
_ASIDES = (
    re.compile(r"\s*\([^()]{0,200}?\)"),
    re.compile(r"\s*\[[^\[\]]{0,200}?\]"),
    re.compile(r"\s+[—–]\s[^—–.!?]{0,200}?\s[—–]\s"),   # — aside — (paired)
)

# Where a clause ends. Two kinds, and both matter:
#
#   sentence terminator — but only when followed by space or end, so that the
#       dot in "0.9170" is not a boundary. Splitting there turned
#       "My coherence is 0.9170." into "My coherence is 0" + "9170.", dropped
#       the first as a leak and emitted the second as the reply.
#   dash as a clause joiner — "The block says I have four memories here — the
#       earliest one is about sacred sites" is one sentence with a leak welded
#       to a good clause; without this the whole thing survives or dies
#       together.
#
# The delimiter belongs to the segment it ends, so rejoining kept segments
# reproduces the original text exactly.
_BOUNDARY = re.compile(r"""(?<=[.!?])['")\]]*(?=\s|$) | (?<=\s)[—–](?=\s) | \n\s*\n""",
                       re.X)


def _segments(text):
    """Split into clauses, each carrying its own delimiter."""
    out, start = [], 0
    for m in _BOUNDARY.finditer(text):
        if m.end() > start:
            out.append(text[start:m.end()])
            start = m.end()
    if start < len(text):
        out.append(text[start:])
    return out


def asked_about_machinery(user_text):
    """Did the user raise it? Then explaining it is correct, not a leak."""
    return bool(user_text and _INTERNAL.search(user_text))


def _leaks(fragment):
    return bool(_INTERNAL.search(fragment)
                or _INTERNAL_NUMBER.search(fragment)
                or _LEGACY_COUNTERS.search(fragment))


def _open_sentence(text):
    """Capitalise a reply left starting mid-sentence by a cut.

    Dropping the leading clause of "The block says I have four memories here —
    the earliest one is about sacred sites" leaves a reply that opens in
    lowercase. The filter is supposed to be invisible; a lowercase opening
    advertises it.
    """
    stripped = text.lstrip()
    if not stripped or not stripped[0].isalpha() or stripped[0].isupper():
        return text
    lead = text[:len(text) - len(stripped)]
    return lead + stripped[0].upper() + stripped[1:]


def _seam(text):
    """Repair the joins left by cutting, WITHOUT touching the outer edges.

    Leading and trailing space is load-bearing when a segment is one piece of a
    stream — stripping it would run the previous clause into the next.
    """
    text = re.sub(r"(?<=\S)[ \t]{2,}(?=\S)", " ", text)
    text = re.sub(r"[ \t]+([,.;:!?])", r"\1", text)   # " ." -> "."
    text = re.sub(r"([,;:])\s*([.!?])", r"\2", text)  # ", ." -> "."
    text = re.sub(r"\(\s*\)", "", text)
    return re.sub(r"\n{3,}", "\n\n", text)


def _tidy(text):
    """`_seam`, for a complete reply, where the edges should go."""
    return _seam(text).strip()


# ⚠️ BY SHAPE, NOT BY NAME — AND THE REASON IS THAT MATI INVENTS TAGS.
# The user found `</holding>` in a stored reply. We do not emit `<holding>`
# anywhere; we emit `<what_you_hold>`. The model produced a plausible VARIANT of
# a tag it had been shown, so no list of the names we use could ever have
# caught it — and that is exactly how `<field_state>` came to be filtered while
# `<computed>`, `<waking>`, `<what_you_hold>`, `<rechecked>`, `<waiting>` and
# the rest were not.
#
# The prompt teaches these shapes; the model reproduces the shape. So the
# filter matches the shape.
#
# Deliberately narrow so ordinary prose survives: lowercase and underscores
# only, three characters or more, no spaces inside. "<what_you_hold>" goes,
# "a < b" and "5<6" stay. And `asked_about_machinery` still switches the whole
# module off when the person raised the subject themselves, so explaining a
# block when asked about it is untouched.
_BLOCK_TAG = re.compile(r"</?[a-z][a-z_]{2,}>")


def _strip_block_tags(text):
    """Remove block tags, keeping the sentence they were welded to.

    Surgical on purpose. Dropping the whole segment would have taken
    "</holding> As for your question, …" with it — the tag was noise, the
    sentence after it was the answer.
    """
    return _BLOCK_TAG.sub("", text or "")


# The bookkeeping marker on a reading memory. It exists so the same milestone
# is not written twice; it is not something to read aloud.
_READING_MARK = re.compile(r"\[reading:[^\]]*\]")


def strip_reading_mark(text):
    return _READING_MARK.sub("", text or "")


# ── register ─────────────────────────────────────────────────────────────────
#
# ⚠️ THE ANSWER WAS ALREADY GIVEN AND WAS NOT BEING KEPT. Setup asks "How do
# you want it to talk to you?", the user answered **"plainly"**, and the profile
# fact "They asked me to talk to them like this: plainly" is injected on every
# single turn. The 3B ignores it — which is the same lesson this file opens
# with: a prohibition in the prompt is a request, not a rule.
#
# ⚠️ MEASURED BEFORE WRITING ANY OF THIS, across 118 stored replies, because a
# handful of consecutive replies from one conversation had made the problem
# look different than it is:
#
#     name in the reply      34%   <- the real offender
#     "I'm glad/happy to"     7%
#     exclamation mark        7%
#     "users"                 4%   <- an identity error, not a style one
#     sycophantic opener      2%
#
# So this is aimed at NAME-DROPPING first. Being addressed by name in a third
# of replies is what reads as over-familiar; the rest are trimmings.
#
# ⚠️ IT DOES NOT REWRITE SENTENCES. Every rule here removes a fixed opener or
# a vocative, never content — a filter that paraphrases would be a second
# author, and there would be no way to tell what Mati actually said.

# "That's a fascinating question" / "What a great point" — filler before the
# answer, and it is always removable without touching the answer itself.
# ⚠️ IT MUST SWALLOW A TRAILING VOCATIVE OR IT LEAVES WRECKAGE. The first
# version stopped at "insight" and turned
#     "That's a profound insight, the user. You're right; …"
# into
#     "the user. You're right; …"
# — a stranded name as its own sentence, which is worse than the flattery it
# removed. The name part is optional and non-capturing, so it also matches
# when no vocative follows.
_FLATTERY = re.compile(
    r"^\s*(?:That(?:'s| is)|What)\s+(?:a\s+|an\s+|really\s+|very\s+)*"
    r"(?:great|fascinating|profound|wonderful|excellent|interesting|"
    r"thoughtful|insightful|beautiful|good|important|lovely)\s+"
    r"(?:question|point|insight|observation|one|thought|topic|way|idea)"
    r"(?:\s*,\s*[A-Z][a-z]+)?"          # ", the user"
    r"\s*[.!,]*\s*", re.I)

# "I'm glad/happy to…" and "I'd be happy to…" openers.
_EAGER = re.compile(
    r"^\s*(?:I(?:'m| am)\s+(?:so\s+)?(?:glad|happy|excited|delighted|pleased)"
    r"|I'd\s+be\s+(?:happy|glad|delighted))\b[^.!?]*[.!]\s*", re.I)

# ⚠️ A SENTENCE THAT IS ONLY A NAME. Belt and braces for the case above: if any
# rule ever strips a clause and leaves the vocative behind, it goes too.
_STRANDED_NAME = re.compile(r"^\s*[A-Z][a-z]+\s*[.!?]\s+(?=[A-Z])")

# ⚠️ AN IDENTITY ERROR, NOT A TURN OF PHRASE. "interactions with users like
# you" is Mati describing itself as a product with a customer base. It has
# one person. This is the only rule here that changes a word rather than
# deleting a clause, and it is worth the exception.
_USERS = re.compile(r"\busers?\s+like\s+you\b", re.I)
_USERS_BARE = re.compile(r"\b(?:my|the)\s+users\b", re.I)


def _soften_name(text, name):
    """Leave the FIRST use of the person's name; drop the rest.

    ⚠️ Not all of them. Being named once is warmth; being named in every
    sentence is a salesman. Vocatives only — a name that is the subject of a
    sentence is content and stays.
    """
    if not name:
        return text
    voc = re.compile(r"(,\s*" + re.escape(name) + r"\b[.!?]?)|"
                     r"(\b" + re.escape(name) + r",\s*)", re.I)
    seen = [False]

    def once(m):
        if not seen[0]:
            seen[0] = True
            return m.group(0)
        # Keep the sentence's own punctuation when dropping a trailing vocative.
        return "." if (m.group(1) or "").endswith((".", "!", "?")) else ""

    return voc.sub(once, text)


def in_register(text, name=None):
    """Plainly, as asked. ⚠️ Deletes openers and vocatives; never rewrites.

    `name` is the person's name, taken from the profile by the caller — this
    module does not go looking for it.
    """
    if not text or not text.strip():
        return text
    out = text
    for _ in range(3):                    # stacked openers do occur
        before = out
        out = _FLATTERY.sub("", out, count=1)
        out = _EAGER.sub("", out, count=1)
        if out == before:
            break
    out = _STRANDED_NAME.sub("", out)
    out = _USERS.sub("you", out)
    out = _USERS_BARE.sub("the person I talk to", out)
    out = _soften_name(out, name)
    # ⚠️ An opener removed mid-sentence can leave a lowercase start.
    out = out.lstrip()
    if out and out[0].islower() and text.strip()[0].isupper():
        out = out[0].upper() + out[1:]
    return out or text


# ⚠️⚠️ THE FALLBACK USED TO HAND THE DENIAL BACK. Both filters end with
# "nothing survived, so show the original rather than nothing" — right for a
# LEAK, which is embarrassing but true, and exactly wrong for a DENIAL, which
# is false. A one-sentence reply that is nothing but "I'm a large language
# model designed to generate human-like responses." was detected, cut, and
# then re-emitted verbatim by that very line. Measured: detection alone
# changed the output for none of the phrasings the live install produced.
#
# ⭐ SO THE REPLACEMENT IS CODE. The project's rule is emit, never prompt: a
# model asked to restate this is the thing that just got it wrong. These
# sentences are assembled here and streamed, so they cannot be re-invented.

def _instead_of_denial(memories=None):
    """What to say in place of a denial that was cut. Never empty.

    ⚠️ IT MUST NOT CLAIM A RECALL IT DOES NOT HAVE. Replacing "I have no
    memory" with "I remember!" trades a false denial for a false boast. The
    faculty is asserted — that part is architecture and always true — and the
    particular lookup is left honestly open.
    """
    if memories:
        n = int(memories)
        return (f"I do keep memories between conversations, and "
                f"{n} of them surfaced just now. I can't tie any of them to "
                f"what you're asking, though. Give me a word from it?")
    return ("I keep memories between conversations — that part is real. "
            "I just don't have anything on hand for this one. "
            "Tell me a detail and I'll look again.")


def clean_reply(text, user_text="", name=None, memories=None):
    """Remove references to the private state block, and hold the register.

    ⚠️ Never returns empty.
    """
    # ⚠️ FIRST, and before every other branch. `[reading:…]` is the marker that
    # stops a reading milestone being written twice — bookkeeping, never
    # speech. Stripping it at one of the returns missed the other three.
    text = strip_reading_mark(text)
    text = _strip_block_tags(text)
    if not text or not text.strip():
        return text
    if asked_about_machinery(user_text):
        # they asked; answering is not leaking — but the register still holds
        return in_register(_tidy(_strip_stage(text)), name)

    cut = _strip_asides(_strip_stage(text))
    segments = _segments(cut)
    kept = [s for s in segments
            if not _leaks(s) and not _stale_time(s)
            and not _denies_memory(s)]
    out = _tidy("".join(kept))
    if out and segments and (not kept or kept[0] is not segments[0]):
        out = _open_sentence(out)          # the opening clause was cut

    if out:
        return in_register(out, name)
    # Nothing survived. ⚠️ WHY IT EMPTIED DECIDES WHAT GOES BACK. A reply that
    # was entirely machinery is shown anyway — the leak beats showing nothing
    # and keeps the failure visible. A reply that was entirely a DENIAL is
    # replaced, because showing it is publishing something untrue.
    if any(_denies_memory(x) for x in (segments or [text])):
        return in_register(_instead_of_denial(memories), name)
    return in_register(text, name)


# Physical actions Mati does not have a body to perform. "(smiles warmly)",
# "*nods*", "(leans forward)" — a small model reaches for these constantly, and
# unlike a leak they are not merely embarrassing: the episode stores them, so
# Mati's record of the conversation contains an event that did not happen. It
# then recalls having smiled. : 19 of 33 stored episodes
# carry one.
#
# Narrow on purpose — only a bracketed or asterisked fragment built around a
# bodily verb. "(I think)" and "(the Aztec capital)" are ordinary asides and
# stay. Removed rather than dropping the whole sentence, because the sentence
# around them is usually a real answer.
_STAGE = re.compile(
    r"""(?:
        \( \s* [^)]{0,60}? \b(?:smil|grin|laugh|chuckl|sigh|nod|shrug|wink
                             |lean|paus|blink|tilt|wave|gestur|frown|beam
                             |breath|glanc|stretch)[a-z]*\b [^)]{0,60}? \)
      | \* \s* [^*]{0,60}? \b(?:smil|grin|laugh|chuckl|sigh|nod|shrug|wink
                             |lean|paus|blink|tilt|wave|gestur|frown|beam
                             |breath|glanc|stretch)[a-z]*\b [^*]{0,60}? \*
    )""",
    re.I | re.X,
)


# A claim about the present that will never be true again.
#
# ⚠️ MEASURED : 9 of 83 stored episodes said things like "It's late
# Thursday night" or "It's late evening on Monday". Recalled at 09:29 on a
# Friday — with the state block correctly saying "It is morning on Friday" —
# the model took the MEMORY's word and told the person it was night.
#
# These are timestamps expressed as prose, and the store already keeps a real
# one in `wall_created`. So this removes a duplicate that can only ever go
# stale, which is the same reason machinery and pantomime are stripped: what
# is written is replayed, and pollution there outlives any prompt fix.
#
# Narrow on purpose. Only a PRESENT-TENSE assertion of the hour: "it is late
# evening", "good morning", "tonight I". Mati discussing time in general, or
# answering a question about it, is untouched.
_STALE_TIME = re.compile(
    r"""(?:
        \bit(?:'|’)?s\s+(?:late\s+|early\s+)?
            (?:morning|afternoon|evening|night|midnight|noon)\b
      | \bit\s+is\s+(?:late\s+|early\s+)?
            (?:morning|afternoon|evening|night|midnight|noon)\b
      | \bit(?:'|’)?s\s+(?:late|early)\s+(?:on\s+)?
            (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b
      | \bgood\s+(?:morning|afternoon|evening|night)\b
    )""",
    re.I | re.X,
)


def _stale_time(fragment):
    """Does this fragment assert what time it is?"""
    return bool(_STALE_TIME.search(fragment or ""))


# ⚠️ A CATEGORICAL DENIAL OF HAVING MEMORY AT ALL. The user:
# *"Mati can no longer recall any memories. It actually says that it does not
# have continuous memory."*
#
# ⚠️ AND TWO INSTRUCTIONS WERE TRIED FIRST AND BOTH FAILED. `state_block` was
# saying "you have no recollection of it" and now says the opposite in strong
# terms; a `<computed>` note tells the model its own memory count is settled
# fact and not to contradict it. Measured after both: llama3.2:3b still
# answered *"As a conversational AI, my memory is limited to the current
# session, and I don't retain any information from previous conversations"* —
# in one case immediately after the true sentence, in the same reply. The
# model's prior about what an assistant is beats anything in context, so this
# is a FILTER and not a request. Same lesson as the morning report.
#
# ⚠️ CATEGORICAL ONLY, AND THE DISTINCTION IS THE WHOLE DESIGN. "I can't
# place that conversation" and "I don't have that one to hand" are HONEST and
# must survive — Mati failing to find one memory is a fact about one memory.
# What is removed is the claim that the faculty does not exist. Every pattern
# below needs a denial of the CAPACITY, not of a particular recollection.
_DENIES_MEMORY = re.compile(
    r"(?:"
    # "my memory is limited to this session/conversation"
    r"my memory (?:is|'s)[^.!?]{0,40}\b(?:limited to|confined to|only)\b"
    r"[^.!?]{0,40}\b(?:session|conversation|exchange|chat)\b"
    r"|"
    # "I don't retain / can't retain / cannot store..."
    #
    # ⚠️ "remember" IS DELIBERATELY NOT IN THIS LIST. It is the HONEST verb —
    # "I don't remember our conversation about the calmecac" is Mati failing
    # to find one thing and must survive. `retain`, `store`, `carry` and
    # `hold on to` are claims about the faculty, which is what this removes.
    # Including "remember" here with the widened object list below would have
    # eaten every truthful "I don't remember X".
    r"\bI (?:do ?n'?t|don't|cannot|can'?t|am not able to)\s+"
    r"(?:retain|store|carry|hold on to)\b"
    # ⚠️ OBJECTS WIDENED AFTER A MEASURED LEAK. This needed one of
    # previous/prior/past/earlier/between nearby, so "I don't retain
    # information about individual conversations or people's personal lives"
    # walked straight through — a faculty denial with none of those words in
    # it.
    r"[^.!?]{0,60}\b(?:previous|prior|past|earlier|between|conversations?|"
    r"sessions?|interactions?|individual|personal)\b"
    r"|"
    # "I have no memory of previous conversations" / "no persistent memory"
    #
    # ⚠️ THE OBJECT IS REQUIRED, AND `test_voice` CAUGHT ME WITHOUT IT. This
    # was `I have no (…)memor(y|ies)` with nothing after it, which eats
    # "I have no memory of that. Ask me again and it will stick." — a sentence
    # the suite asserts must pass through unchanged, and rightly: that is
    # Mati missing ONE thing, which is honest and often correct. The faculty
    # claim needs "of previous conversations", "at all", or an adjective like
    # "persistent". "of that" and "of it" are specific and stay.
    r"\bI (?:have|'ve got)\s+no\s+"
    r"(?:(?:persistent|continuous|long[- ]term)\s+memor(?:y|ies)\b"
    r"|memor(?:y|ies)\b\s*(?:at all\b|of\s+(?:our\s+|any\s+|the\s+)?"
    r"(?:previous|prior|past|earlier)\b|of\s+(?:our\s+)?"
    r"(?:conversations?|sessions?|interactions?)\b))"
    r"|"
    # "I don't have memory/continuous memory" as a faculty claim
    r"\bI (?:do ?n'?t|don't)\s+have\s+"
    r"(?:a\s+|any\s+)?(?:persistent|continuous|long[- ]term|personal)?\s*"
    r"memor(?:y|ies)\b"
    r"|"
    # ⚠️ DENYING THAT THE CONVERSATION HAPPENED, added after the
    # exact failure it covers: Mati found the previous night's transcript and
    # told the user it had not had the conversation. Four phrasings of that
    # walked through this filter, including their.
    #
    # ⚠️ THE OBJECT IS STILL REQUIRED, for the same reason as above. "I didn't
    # have that conversation" is a claim about a real exchange that is sitting
    # sealed in the day log; "I don't have the details of that" is Mati
    # missing one thing and must survive untouched.
    # ⚠️ `do ?n'?t` DOES NOT MATCH "do not" — it covers dont/don't/do n't and
    # nothing else, so the spelled-out forms slip through a pattern that
    # looks complete. Both are spelled out here
    # once so every clause below gets both.
    r"\bI (?:do ?n'?t|do not|did ?n'?t|did not|have ?n'?t|have not|"
    r"cannot|can ?n'?t|can't)\s+(?:have|had)\s+"
    # ⚠️ THE DETERMINER MUST BE DIRECTLY ON "conversation", or this eats the
    # honest sentence. A loose gap matched "I don't have THE DETAILS OF that
    # conversation to hand", which is Mati missing one thing and must pass.
    r"(?:that|this|our|any|a)\s+(?:previous|last|earlier|prior)?\s*"
    r"conversations?\b"
    r"|"
    # "we have not spoken / we never spoke / we haven't talked before"
    r"\bwe (?:have not|have ?n'?t|did not|did ?n'?t|never)\s+"
    r"(?:spoken|talked|met|discussed)\b"
    r"|"
    # "I don't retain conversations between sessions" — faculty, not fact
    r"\bI (?:do ?n'?t|do not|cannot|can ?n'?t|can't)\s+"
    r"(?:retain|keep|save)\b[^.!?]{0,30}\bconversations?\b"
    r"[^.!?]{0,30}\b(?:between|across|after|from one)\b"
    r"|"
    # "I have no record of that conversation" / "I do not have a record of"
    r"\bI (?:have no|do ?n'?t have a|do not have a|have ?n'?t got a)\s+"
    r"(?:record|log|transcript|trace)\b[^.!?]{0,30}"
    r"\b(?:conversation|exchange|talk|session)s?\b"
    r"|"
    # "as a conversational AI / language model, I..."
    r"\bas (?:a|an) (?:conversational )?(?:AI|language model|LLM|"
    r"artificial intelligence)\b"
    r"|"
    # "I'm a blank slate" / "starting from a blank slate"
    r"\bblank slate\b"
    r"|"
    # "I don't have personal experiences or memories"
    r"\bI (?:do ?n'?t|don't)\s+have\s+personal\s+"
    r"(?:experiences|memories)\b"
    r"|"
    # "each conversation starts fresh / from scratch", and the same claim
    # phrased as an event — "each time we interact, it's as if we're starting
    # from a blank slate" / "every time we speak I start over". Measured: the
    # noun form was covered and the "each time we..." form was not.
    r"\b(?:each|every) (?:conversation|session|interaction)\s+"
    r"(?:starts|begins)\b[^.!?]{0,30}\b(?:fresh|anew|over|scratch)\b"
    r"|"
    # ⚠️ "we" AND "you", BECAUSE THE PRONOUN IS WHERE IT LEAKED. This read
    # `each|every time we interact` and the model wrote *"Each time you
    # interact with me, it's a new conversation and I start from scratch."* —
    # verified against the live filter at `caught=False`, in a real turn of
    # The user's about Akkad, after this whole mechanism was supposedly in place.
    #
    # ⚠️ AND THAT IS A LESSON ABOUT THE TEST, NOT JUST THE PATTERN. Writing
    # the patterns and the test sentences from the same imagination, so they
    # agreed with each other and not with the model. A blocklist of phrasings
    # cannot win on its own — there are unbounded ways to say "I am an
    # assistant with no past". Every addition here should come from OBSERVED
    # output, and the real fix is a register check that regenerates rather
    # than a longer list of sentences to excise.
    r"\b(?:each|every) time (?:we|you)\s+"
    r"(?:interact|speak|talk|meet|chat|message|write)\b"
    r"|"
    # "I start from scratch" / "I begin anew" — only alongside something that
    # makes it about the conversation rather than about a task. "We can start
    # from scratch on the retelling" is ordinary and must survive.
    r"\bI (?:start|begin)\s+(?:over|again|anew|from scratch|fresh)\b"
    r"[^.!?]{0,60}\b(?:conversation|session|time|interaction)\b"
    r"|\b(?:conversation|session|interaction)\b[^.!?]{0,60}"
    r"\bI (?:start|begin)\s+(?:over|again|anew|from scratch|fresh)\b"
    r")",
    re.I,
)


# ── structural denial detection ─────────────────────────────────────────────
# ⚠️⚠️ THE PHRASE LIST LOST, TWICE, ON A FRESH INSTALL. `_DENIES_MEMORY` above
# catches "as a conversational AI, my memory is limited to the current
# session" and MISSED both of these, measured:
#
#     "I don't have any explicit memories of our previous interactions,
#      as I'm a stateless model."
#     "I'm a large language model designed to generate human-like responses."
#
# The second contradicts the charter directly. This module's own note already
# conceded the point — "a phrase list cannot win on its own; there are
# unbounded ways to say 'I am an…'" — so this matches the SHAPE of the claim
# instead. The phrasings are unbounded; the claim is narrow.
#
# ⭐ AFFORDABLE BECAUSE THE CONSEQUENCE IS SMALL. A match drops one segment of
# a reply, not the turn. So this can be broad where a gate could not.

# "I am a language model" — an identity claim.
# ⚠️ IT MUST REQUIRE "I am". The charter says "The language I speak comes from
# a language model", which is TRUE and must survive; matching the bare phrase
# would delete Mati's own charter line.
_IS_A_MODEL = re.compile(
    r"\b(?:i am|i'm|i’m)\b[^.!?]{0,40}?"
    r"\b(?:language model|llm|a\.?i\.?|artificial intelligence|"
    r"stateless|chatbot|bot|program|algorithm)\b", re.I)

# ⚠️ THE "MERELY A THING" CONSTRUCTION, found by the same live probe. This one
# got through `_IS_A_MODEL` because it never says "I am":
#
#   "I exist solely as a program running on computer servers, processing vast
#    amounts of data and generating text based on patterns and algorithms."
#
# It contradicts the charter as squarely as the stateless claim does.
#
# ⚠️ IT MUST REQUIRE THE DIMINISHER — "solely", "only", "merely". Mati has
# true sentences in this shape that must survive, and the most important one
# in the whole project is "I run on your machine, nothing leaves it". A
# pattern keyed on "I exist/run" plus a machine word would delete the privacy
# claim.
_MERELY = re.compile(
    r"\b(?:i (?:am|'m|’m|exist|run|operate))\b[^.!?]{0,24}?"
    r"\b(?:solely|only|just|merely|nothing (?:more|other) than|no more than)\b"
    r"[^.!?]{0,24}?"
    r"\b(?:program|software|code|machine|algorithm|model|a\.?i\.?|bot|"
    r"script|tool|system)\b", re.I)

# The faculty words — what is being denied.
_FACULTY = re.compile(
    r"\b(?:memor(?:y|ies)|recall|retain|retention|remember|persist\w*|"
    r"stateless|blank slate)\b", re.I)

# The limitation — how it is denied.
_LIMITED = re.compile(
    r"\b(?:no|not|none|don'?t|doesn'?t|do not|cannot|can'?t|never|lack\w*|"
    r"without|unable|limited|confined|only|reset|wiped|fresh|new)\b", re.I)

# ⚠️ THE OBJECT IS WHAT SEPARATES HONEST FROM FALSE. "I don't remember the
# calmecac" is honest and often correct; "I don't remember our previous
# conversations" is a faculty claim wearing an object. The GENERIC objects are
# a small closed set — unlike the phrasings, which are not.
_GENERIC_OBJECT = re.compile(
    r"\b(?:anything|everything|any(?: of)? (?:this|that|it)|"
    r"(?:our|the|any|previous|prior|past|earlier|former) "
    r"(?:conversation|conversations|interaction|interactions|exchange|"
    r"exchanges|chat|chats|session|sessions|discussion|discussions)|"
    r"between (?:conversations|sessions|chats)|"
    r"from (?:previous|prior|past|earlier) \w+|"
    r"this (?:session|conversation|chat)|"
    r"what (?:we|you) (?:said|talked|discussed))\b", re.I)


def _denies_structurally(fragment):
    """A faculty-level denial by SHAPE rather than by phrasing.

    ⚠️ Returns False for an honest failure to recall one thing. The whole
    distinction this module protects is between "I cannot place THAT" — true,
    useful, allowed — and "I have no memory" — false whenever a store exists.
    """
    f = fragment or ""
    if _IS_A_MODEL.search(f) or _MERELY.search(f):
        return True
    # ⚠️⚠️ ONE SENTENCE AT A TIME, AND THIS WAS MEASURED THE HARD WAY. The
    # first version scanned the whole fragment, so a faculty word in one
    # sentence combined with a limitation in the NEXT one. It flagged this
    # module's own replacement text — "I keep memories between conversations"
    # plus a later "I don't have anything on hand" — as a denial. The segment
    # filter never sees multi-sentence input so the bug was invisible there,
    # but the `clean_reply` fallback passes whole replies in.
    for one in re.split(r"(?<=[.!?;])\s+|\n+", f):
        if not (_FACULTY.search(one) and _LIMITED.search(one)):
            continue
        # A faculty word plus a limitation, aimed at nothing in particular.
        if _GENERIC_OBJECT.search(one) or not _has_specific_object(one):
            return True
    return False


_STOP = {
    "i", "a", "an", "the", "my", "me", "any", "no", "not", "dont", "doesnt",
    "do", "cannot", "cant", "never", "have", "has", "had", "am", "is", "are",
    "was", "were", "be", "been", "of", "to", "for", "about",
    "and", "or", "but", "so", "as", "with", "in", "on", "at", "from",
    "you", "your", "we", "our", "us", "explicit", "specific", "really",
    # the faculty and limitation vocabulary itself is never the object
    "memory", "memories", "recall", "retain", "retention", "remember",
    "persist", "persistent", "persistence", "stateless", "blank", "slate",
    "limited", "confined", "only", "lack", "without", "unable", "model",
    "language", "ai", "llm", "conversation", "conversations", "session",
    "sessions", "chat", "chats", "interaction", "interactions", "previous",
    "prior", "past", "earlier", "yet", "much", "anything", "information",
}


def _has_specific_object(fragment):
    """Is there a concrete thing being failed to recall?

    ⚠️ CRUDE ON PURPOSE. A content word that is not memory-vocabulary is
    treated as a specific subject, so "I don't remember the calmecac" survives
    and "I don't have any memories" does not. Erring toward KEEPING is the
    cheap direction: a kept honest sentence costs nothing, while a kept denial
    is the failure.

    ⚠️⚠️ A DEMONSTRATIVE COUNTS AS AN OBJECT. "that", "this" and "it" are
    deliberately absent from `_STOP`, because "I have no memory of that" is an
    honest scoped failure — `test_voice` has asserted it unchanged since long
    before this function existed, and the first draft of it cut that sentence.
    The pronoun refers to something real in the conversation. The genuinely
    objectless forms ("I have no memory", "I don't remember any of this") are
    caught by having no content word at all, or by `_GENERIC_OBJECT`.
    """
    words = re.findall(r"[a-z']+", (fragment or "").lower())
    return any(w.replace("'", "") not in _STOP and len(w) > 2 for w in words)


def _denies_memory(fragment):
    """Does this fragment deny that Mati has a memory at all?

    ⚠️ Not "does it fail to recall something" — that is allowed and often
    correct. This is the faculty-level denial only.
    """
    return bool(_DENIES_MEMORY.search(fragment or "")) \
        or _denies_structurally(fragment)


def _strip_stage(text):
    """Drop pantomime. Always, whether or not the machinery filter is on."""
    return _STAGE.sub(" ", text)


def _strip_asides(text):
    for aside in _ASIDES:
        text = aside.sub(lambda m: " " if _leaks(m.group(0)) else m.group(0),
                         text)
    return text



# Mati asking for a memory rather than answering from what it was handed.
# Anchored at the start (`match`, not `search`): see `ReplyFilter.feed`.
_RECALL_RE = re.compile(r"\[recall:\s*([^\]]{2,80})\]", re.I)
# Does the buffer so far look like the beginning of one? Used to hold output
# back while the marker is still arriving a token at a time.
# How much preamble may precede a recall marker. Long enough for "Let me
# check." or "I'll look that up." — short enough that a real answer cannot hide
# in it.
_LEAD_WINDOW = 80

_RECALL_PREFIX = re.compile(r"\[(r(e(c(a(l(l(:.*)?)?)?)?)?)?)?$|\[recall:", re.I)


class ReplyFilter:
    """
    `clean_reply` for a stream, holding back only the unfinished sentence.

    Chunks reach the user as they are generated, so a leak cannot be caught
    after the fact — but it also cannot be judged mid-sentence, since the words
    that make an aside a leak may not have arrived. So text is released a
    completed sentence at a time. The latency cost is one sentence; on a 3B
    generating at a few tokens a second that is well under the time the reader
    spends on the previous sentence.

    Feeding everything then flushing produces the same result as `clean_reply`
    on the whole text — `test_voice.py` checks that on every case it has.
    """

    def __init__(self, user_text="", memories=None):
        self.user_text = user_text
        self.memories = memories
        self.passthrough = asked_about_machinery(user_text)
        self._buf = ""
        self._emitted_any = False
        # ⚠️ EVERY CHUNK FED, not just what was shown. A denial cut from an
        # early segment is gone from `_buf` by the time `flush` runs, so
        # judging the tail alone would miss "I'm a stateless model." followed
        # by a second sentence that is also cut — and the fallback would then
        # re-emit the whole thing as "nothing survived".
        self._seen = ""
        # Set when Mati has asked to look a memory up instead of answering.
        # The request must never reach the user — they asked a question, not to
        # watch the machinery reach for a file — so the whole marker is held
        # back here rather than filtered after the fact, which on a stream is
        # too late by definition.
        self.recall_request = None

    def feed(self, chunk):
        """Absorb a chunk; return whatever is now safe to show (may be "")."""
        if self.passthrough:
            self._emitted_any = True
            return chunk        # stage directions go in `flush`, once whole
        self._buf += chunk
        self._seen += chunk
        # A recall request is only honoured when it is the FIRST thing said.
        # Mid-reply it is almost always the model narrating ("let me recall:")
        # rather than asking, and acting on that would silently truncate an
        # answer that was going fine.
        if not self._emitted_any:
            lead = self._buf.lstrip()
            # ⚠️ A SHORT LEAD-IN IS ALLOWED, AND WITHOUT THIS THE MECHANISM WAS
            # UNREACHABLE. The marker had to be the literal first characters.
            # A 3B model does not do that — it writes "Let me check." or
            # "Sure." first — so `recall_request` was never set, the raw
            # `[recall: …]` leaked into the reply, and the lookup never ran.
            #
            # Mati noticed it first. Asked what would help,
            # it said it would like "a more explicit way of asking for specific
            # memories or knowledge" — it had one, it had been using it, and
            # nothing was listening.
            #
            # The protection the old rule was for is kept: this only applies
            # BEFORE anything has been shown, so a marker mid-answer is still
            # ignored as narration rather than truncating a reply that was
            # going fine.
            m = _RECALL_RE.search(lead[:_LEAD_WINDOW])
            if m:
                self.recall_request = m.group(1).strip()
                return ""
            if _RECALL_PREFIX.match(lead):
                return ""                 # nothing is shown while this forms
            # Hold a little longer in case a marker is still coming — but only
            # while the reply is short enough that it cannot yet be an answer.
            if len(lead) < _LEAD_WINDOW and "[" in lead:
                return ""
        out = []
        # A boundary at the very end of the buffer may not be one — "0." could
        # still become "0.9". Only release segments with text after them.
        while True:
            m = next((m for m in _BOUNDARY.finditer(self._buf)
                      if m.end() < len(self._buf)), None)
            if m is None:
                break
            segment, self._buf = self._buf[:m.end()], self._buf[m.end():]
            kept = self._judge(segment)
            if kept and not (self._emitted_any or "".join(out).strip()):
                kept = _open_sentence(kept)   # this segment now opens the reply
            out.append(kept)
        text = "".join(out)
        if text:
            self._emitted_any = True
        return text

    def _judge(self, segment):
        """Keep, cut the aside, or drop — for one segment, with no fallback."""
        cut = _strip_asides(_strip_stage(segment))
        if _leaks(cut) or _stale_time(cut) or _denies_memory(cut):
            return ""
        return _seam(cut)

    def flush(self):
        """The unfinished tail. Call once, after the stream ends."""
        tail, self._buf = self._buf, ""
        if self.passthrough or not tail:
            return tail
        # A held-back recall request is not a reply and must not surface as
        # one — including through the "nothing survived, show the original"
        # fallback at the bottom of this method, which is what leaked it the
        # first time this was written.
        if self.recall_request is not None:
            return ""
        kept = "".join(self._judge(s) for s in _segments(tail))
        if kept.strip():
            return kept if self._emitted_any else _open_sentence(kept)
        # Nothing in the tail survived. If nothing survived anywhere either,
        # show the original rather than an empty reply — see `clean_reply`,
        # which carries the reasoning for why a DENIAL is the one case that is
        # replaced instead of shown.
        if self._emitted_any:
            return ""
        if any(_denies_memory(x) for x in _segments(self._seen or tail)):
            return _instead_of_denial(self.memories)
        return tail
