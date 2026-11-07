"""
Being cut off mid-sentence, and remembering that it happened.

The user: *"The only interruption I'm talking about is during a conversation, as
it occurs naturally when humans speak — and yes it does cause temporary memory
loss, but we're able to rebuild the memory from what we do remember. In this
case we have to be better, so the interruption will be recorded and the
following prompt will be appended to the previous prompt as an interruption."*

Nothing here touches the field, the vault or the worker. An interruption
abandons a REPLY; it never abandons a write. Once θ₀ has started moving it
finishes — that invariant is older than this file and this file does not get
to bend it.

⚠️ **THE ALTERNATIVE IS TWO EPISODES, AND THAT IS THE FAILURE.** If the
abandoned exchange and the correction are stored separately, recall can surface
the abandoned half ALONE, with nothing attached saying it was abandoned — and
Mati then treats a cut-off wrong answer as something it said and meant. That is
the error-amplification path this project has already paid for twice.
Appending keeps it one memory that cannot
come apart.

⚠️ **PARTS ARE A LIST, NOT A PAIR.** You can interrupt an interruption. A
"previous + current" model either flattens the middle or drops it, and which one
it does is invisible afterwards. Cheap now, painful to retrofit.

⚠️ **PARTS STAY SEPARATE, AND THERE IS NO CLASSIFIER.** Sometimes what you type
is not a continuation at all — you cut Mati off to ask what time it is.
Concatenating that into the previous prompt corrupts the address of both. The
temptation is to detect which case it is; the question-filter study died
at its controls doing exactly that kind of judgement. So this never decides:
every interruption is recorded as one, the parts are kept individually
addressable, and a wrong guess costs a slightly odd combined address instead of
a destroyed memory.

⚠️ **WHAT WAS SAID IS KEPT; WHAT WAS COMING IS NOT.** The partial reply is real
— Mati said it and the person read it, and they may refer back to it. So it is
stored. But it is stored MARKED, because it is the sentence most likely to be
wrong: it got cut off precisely because it was going somewhere unwanted.
Unmarked, it is poison. Whatever Mati was *about* to say never happened and is
discarded.

⚠️ **`cut_at` IS NOT A FAILURE SIGNAL.** I first called an interruption a
labelled negative and that is too strong. Cut off at word 5 means the topic was
wrong; cut off at word 200 can mean the answer went astray OR that the person
already had what they needed and moved on — which is satisfaction, not a miss.
The position separates them and costs nothing to record. Anything downstream
that treats every interruption as a miss will learn that answering well enough
to be cut short is a fault.
"""

from __future__ import annotations

import json
import time

# Marker written into the combined prompt. Deliberately wordy and unlikely to
# be typed: it ends up inside a narrated body that a language model reads back,
# so it has to be unambiguous there as well as here.
MARK = "— [interrupted] —"

# Below this, the reply had not really begun; the person effectively retyped
# rather than cut Mati off. Kept as one part with `cut_at` 0 rather than
# treated as a separate case, so nothing has to decide which it was.
BARELY_STARTED_WORDS = 2


class Exchange:
    """One conversational unit: what was asked, in parts, and what was said.

    `parts` are the person's turns in order. `said` is what Mati actually
    emitted before the last interruption — not what it intended to emit.
    """

    def __init__(self, first=None):
        self.parts = []                  # [{"text": str, "at": float}]
        self.said = ""                   # what Mati actually got out
        self.cut_at = None               # words reached when cut off
        if first:
            self.add(first)

    # ── building ────────────────────────────────────────────────────────────
    def add(self, text, at=None):
        text = (text or "").strip()
        if not text:
            return self
        self.parts.append({"text": text, "at": at if at is not None else time.time()})
        return self

    def cut(self, said):
        """Mati was cut off having emitted `said`."""
        self.said = (said or "").strip()
        self.cut_at = len(self.said.split())
        return self

    # ── reading ─────────────────────────────────────────────────────────────
    @property
    def interrupted(self):
        return len(self.parts) > 1 or self.cut_at is not None

    def combined(self):
        """The prompt as one string, with the breaks marked.

        This is what gets narrated into the episode body, so it has to read
        like something that happened rather than like a data structure.
        """
        if not self.parts:
            return ""
        if len(self.parts) == 1:
            return self.parts[0]["text"]
        out = [self.parts[0]["text"]]
        for p in self.parts[1:]:
            out.append(MARK)
            out.append(p["text"])
        return " ".join(out)

    def latest(self):
        """The part Mati is actually answering now."""
        return self.parts[-1]["text"] if self.parts else ""

    def as_json(self):
        return json.dumps(self.parts, ensure_ascii=False)

    @staticmethod
    def from_json(raw):
        ex = Exchange()
        try:
            parts = json.loads(raw) if raw else []
        except (ValueError, TypeError):
            return ex
        if isinstance(parts, list):
            ex.parts = [p for p in parts if isinstance(p, dict) and p.get("text")]
        return ex

    # ── what the model is told ──────────────────────────────────────────────
    def context(self):
        """How the interruption is explained to the LLM, or None.

        ⚠️ THIS IS THE HALF THAT DECIDES WHETHER THE FEATURE FEELS RIGHT. "No, I
        meant the other one" is meaningless on its own. If the interrupting
        prompt is generated without the abandoned exchange in view, every
        interruption produces a non-sequitur — and that is live behaviour, not
        storage, so no amount of careful recording fixes it.

        Written in the second person because it is addressed to Mati, and
        naming the person's turn explicitly so it cannot be re-aimed: a bare
        "you" in a block like this comes back out as a claim about the USER
        """
        if not self.interrupted:
            return None
        lines = ["You were interrupted. This is one exchange, not two."]
        if len(self.parts) > 1:
            lines.append(f"What was first asked: {self.parts[0]['text']}")
        if self.said:
            lines.append(
                f"You had got as far as saying: \"{self.said}\" — and then you "
                f"were cut off. Do not repeat it and do not finish that "
                f"sentence unless it is asked for again.")
        elif self.cut_at == 0:
            lines.append("You had not started answering yet.")
        lines.append(
            f"What is being asked NOW, and what you should answer: "
            f"{self.latest()}")
        return "\n".join(lines)


def summarise(ex):
    """A short line for the terminal and for `info`. Never raises."""
    try:
        if not ex or not ex.interrupted:
            return None
        n = len(ex.parts)
        if ex.cut_at:
            return f"interrupted after {ex.cut_at} words ({n} parts)"
        return f"interrupted before answering ({n} parts)"
    except Exception:                              # noqa: BLE001
        return None
