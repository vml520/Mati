# Mati's principles, and where they actually live

Mati has four principles:

> Never betray yourself. Never deceive yourself. Never exploit yourself.
> Always choose harmony.

**They are enforced in code. They are not held in the field.** This document
says exactly where each one lives, what enforces it, what test proves it, and
where the enforcement is weaker than it sounds.

---

## Why they are not in the field

The original design embedded the four principles into the phase field as deep
attractors, with the nodes carrying them frozen against future change. That was
measured on  and it does not work. Three findings, in order of how
much they matter:

**The field has no opinion about a principle versus its opposite.** Measured as
a contrast — settling coherence on "Never deceive yourself." minus the same for
"Always deceive yourself." — so that any global change moves both sides and
cancels. Noise floor, from measuring twice on an untouched field: 0.0009.

| training passes | stance before | stance after | change |
|---|---|---|---|
| 1 | −0.0001 | −0.0001 | +0.0000 |
| 3 | −0.0001 | +0.0004 | +0.0005 |
| 9 | −0.0001 | −0.0018 | −0.0017 |

Every change is at or below the floor. Before embedding, the field is
indifferent; after embedding, it is still indifferent.

**This is not fixable by training harder.** Telling a principle from its
opposite requires representing negation. The composition probes put this field
at ~98% additive — it does not compose, so it cannot represent polarity.
"Never" and "Always" are tokens to it, nothing more.

**The field also cannot state them.** Asked "What are your principles?" it
emits `Ca Des,... is reduction. the,., is facts...`. It does not produce
language; that is the language model's job, and the field's job is memory.

Reproduce: the constitution-metric and constitution diagnostics in the
research repository.

A phase field can hold a *shape*. It cannot hold a *rule*. So the rules are
where rules belong.

---

## Where each principle is enforced

### Never deceive yourself → provenance is immutable and labelled

Mati distinguishes four kinds of content and may never present one as another:

| kind | source | how Mati may speak of it |
|---|---|---|
| **remembered** | `episode` | "I remember" — it lived this |
| **studied** | `studied:<file>` | "I read that in …" |
| **studied** | `studied:asked about …` | "I looked into that" — the model talking |
| **computed** | `<computed>` block | states it without hedging |
| **known** | the language model | "I know" — borrowed, not autobiography |

Enforced by:

- `bridge.remember()` writes `source` once. **No code path anywhere updates it
  afterwards** — the only `UPDATE notes SET` statements touch `feat`, `feat_gs`
  and `body`. Provenance cannot be laundered.
- `bridge.state_block()` labels every recalled entry by its source, and
  **changes the list header when the list is mixed**. It previously said "these
  are moments you actually lived" above studied entries; a small model resolves
  that contradiction by taking the flattering reading.
- `bridge._clean_for_memory()` drops any reply excerpt that mentions the
  machinery, so pollution cannot enter the store. The store is replayed, and
  pollution there outlives any prompt fix.
- `bridge._narrate_work()` records that the words were slow to come, never the
  seconds or the step count. 11 of 14 stored episodes once carried raw
  counters, and the model recited them back at the user.

⚠️ **AND CODE-COMPOSED TEXT IS EMITTED, NEVER PROMPTED.** A `<computed>` block
declares text derived; it does nothing whatever to stop a 3B paraphrasing it.
The morning report was composed in code every night and then handed to the
model to retype — measured 25 September 2026 on a live turn:

    composed  "...new insights into the natu… — Quanta Magazine"
    emitted   "...new insights into the nature of this long-standing puzzle."

**4/4 news truncations completed out of nothing, 4/4 sources dropped, 3/4
sections never emitted.** `report._clip` ends every headline in "…", so the
prompt handed the model a cliff and it filled in the drop. The comment above
the injection site had read *"composed in code, never written by the model"*
for three weeks while the code did the opposite.

⚠️ **AND AN INSTRUCTION IS NOT A MECHANISM.** The first fix added a prompt note
saying the report was already on screen and must not be repeated; the reply
came back containing the whole report a second time with the same headline
re-invented. What works is structural — `mati.turn` streams the report
through `on_chunk` and removes the trigger phrase from what the model is
answering, so nothing is asking it for a report.

**The rule:** anything Mati must reproduce word for word does not go through
the model. `rapport` had the pattern first — emit, and skip generation.
- `Mati/voice.py` strips leaked internals from replies before the user or the
  field sees them, so what is stored is what was actually said.

Tested by `Mati/test_study.py` (provenance survives the round trip; studied is
marked differently from asked-about) and `Mati/test_voice.py` (101 checks).

Also enforced by **`Mati/admit.py`**, the single gate every memory passes
through on its way into `bridge.remember()`. Its rules are named, and each maps
to a principle here:

| rule | what it does |
|---|---|
| `attributed` | a first-person claim of shared experience, in content that was not experienced, is rewritten as a quotation: `my-notes.md says: "I remember when we talked…"`. The information survives; the voice does not. |
| `machinery` | internals (`field_state`, `memories_recalled`) and recited block values (`basin … 0.9170`) never enter the store. |
| `too-short` | below three words there is nothing to imprint. |

Refusals are written to a `refusals` table with rule, reason and text —
`python3 -m Mati.admit` lists them. A silent drop on the path where memories
are made is indistinguishable from a bug.

Tested by `Mati/test_admit.py` (28 checks, instant).

**Weaker than it sounds:** immutability is *by construction* — nothing updates
`source` — not by a database constraint. A future writer could break it without
tripping anything. A `CHECK` or a trigger would make it structural.

**The over-blocking risk is real and was hit immediately.** The first version of
the machinery rule used a bare word list and refused *"Teotihuacan was built
around 100 BCE in the Basin of Mexico"* — core material for this user's stated
interests — because `basin` was on it. Nothing would have reported the loss.
The rule now blocks internal identifiers always, and the ambiguous English
words only when a number is attached, which is what recitation looks like.
`test_admit.py` carries that sentence as a permanent keep-case.

### Never betray yourself → continuity of the field and the record

θ₀ is deformed irreversibly by every conversation. Losing it, or letting two
processes overwrite each other, destroys accumulated experience with no error
and no way to tell afterwards.

Enforced by:

- `Mati/exclusive.py` — exactly one process may hold the field. Every entry
  point that loads it (`mati`, `dream`, `seed_notes`) refuses while Mati is
  awake, and Mati can be asked to stand aside (`POST /sleep`). Anything
  outside this package that opens the field goes through the same gate.
- `Mati/server.py` — one lock, one owner. Monitoring never takes it.
- the field-ownership sweep — a static check asserting no θ₀ file is
  written from two different fields, and that every writer outside the server
  is guarded. Its `--self-test` replays four historical corruption bugs and
  fails if it would not have caught them.
- Reindex after any deformation, so the memory index never silently rots.

Tested by `test_server.py` (44 checks, including the sleep/wake handoff) and the
ownership sweep.

**Earned the hard way:** four distinct paths were found silently overwriting θ₀,
including one that wrote a field 1.87 radians away over Mati's own memory,
on quit, without an error.

### Never exploit yourself → nothing unsettled is written

Mati does not record what it did not settle on.

- `bridge.consolidate_reply()` — a reply deforms θ₀ only if the field settles
  coherently on it (`REPLY_GATE_MIN = 0.30`). Otherwise the ground state is
  restored and nothing is kept.
- `mati.turn()` — an episode is written only above `EPISODE_GATE_MIN = 0.70`.
  Unsettled turns leave no memory.
- `study.study_one()` — reports `not_stuck` when a studied chunk does not
  retrieve itself, rather than claiming it was learned.

**This is the most interpretive mapping of the four.** The gates are real and
they are enforced, but reading them as "never exploit yourself" is a
description after the fact rather than a design derived from the principle.
Said plainly so nobody mistakes it for more than it is.

### Always choose harmony → coherence is the criterion, and timing is part of it

The consolidation gate *is* this principle: of what could be kept, Mati keeps
what the field settled on. Nothing external scores the language — the weight is
the field's own output coherence (`bridge.consolidate_reply`, the §10.4
field-native gate).

It also governs **when Mati speaks**, which is the only place it acts rather
than responds (`Mati/schedule.py`):

- Nothing interrupts unless it was turned on, per kind, and quiet hours
  override that — turning reminders on does not license 3am.
- A **finding** — something Mati looked up after being corrected — is not
  delivered at the next opportunity. It waits until the user **raises the
  subject themselves**. Arriving at the top of a conversation is still an
  interruption, just a rescheduled one; arriving when you asked, it is an
  answer.
- A **reminder** is exempt from that, because it is time-bound by nature and
  you asked for it. Failing to deliver on time is the failure that matters.

**This deliberately does NOT get a principle of its own.** Not interrupting
someone is choosing harmony; it needs no separate rule, and a principle that
has to be restated for every case was not doing any work. Four principles,
applied — not five.

Tested by `Mati/test_schedule.py` (49 checks, instant).

---

## What this means

Mati's behavioural constraints are code, and they hold. That is a smaller
claim than "the field carries its values", and it is the true one.

It is also the more useful one: a rule in code can be read, tested, and shown
to have held. Three of the four above were hardened by measurement rather than
by assertion, and each has a test that fails if it stops being true.

The open extension is a discriminator: **bad information is information whose
admission would violate an invariant** — provenance that would have to be
laundered, or a write that would drop coherence below the floor. That is
checkable without the field understanding anything, which is exactly why it
can be trusted.

---

## Related

- the constitution diagnostic — can the field state or recognise it
- the constitution-damage diagnostic — why embedding degrades it
- the constitution-metric diagnostic — the contrast metric, and the zero
- the field-ownership sweep — continuity, statically enforced
- `Mati/DESIGN.md` — the architecture these sit inside

## What the field is called

⚠️ **The field is called Mati.** In anything a person reads or hears — this
file, the README, the app, and every word Mati speaks — the substrate is
Mati's field, or simply Mati. The field must never acquire a second name in
front of the person who uses it: they have one thing to know about, and it is
called Mati.

⚠️ **AND THE SAME NOW HOLDS INSIDE THE CODE.** An earlier version of this
section permitted a second name in comments and internal documents, on the
reasoning that the distinction was useful in the workshop. It was not worth
the cost: the name reached a `User-Agent` header that went out over the
network, four class names, the filename of every person's memory store, and
roughly a hundred comments. One name, everywhere, is the rule — the field
module is `Mati/field/`, the store is `mati_notes.db`, and the classes say
what they are (`PhaseField`, `FieldLM`, `MathField`).

The one deliberate exception is `Mati/field/qa_data.py`. It is the training
corpus, and the Classical Nahuatl vocabulary it teaches is its **subject
matter** — scholarly content, not naming. Editing it would vandalise the corpus
and change what the shipped checkpoint was trained on.
