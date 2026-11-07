# Mati — field-coupled LLM layer

**Status:** scaffold . Not wired into the chat app yet.

## Why this exists

The composition probes  established that the phase field encodes
multi-token context ~99% **additively** — spatial interaction 0.008, temporal
0.022, versus 0.840 for a real tensor code, and unchanged by 40 epochs of
training. Grammar needs conjunctive binding; the field, in its current regime,
does not supply it; the composition-capacity probes are what established that.

That blocks **one** of the three modules — the field as a *language*
faculty. It does not block the other two, and it does not block the vision.
Mati couples the working parts to an LLM that supplies the missing faculty:

| Faculty | Owner | Why |
|---|---|---|
| Composition / grammar | **LLM** | attention *is* a binding mechanism; the field is additive |
| Persistent identity, memory, dreaming | **field** | θ₀ is deformed by every interaction; no LLM has this |
| Ethics / constitution | **field** | locked nodes are not prompt-injectable — it isn't text |
| Logic | `Mati/field/math.py` | unchanged |

## What makes the field non-decorative here

⚠️ **THIS SECTION ONCE LISTED FOUR CLAIMS AND THREE OF THEM HAVE BEEN
REFUTED BY MEASUREMENT.** It was written before the harnesses existed. The
numbers live in `ARCHITECTURE.md`; what follows is what survived, and what
replaced the rest. The refuted versions are named explicitly because a reader
who met them elsewhere needs to know they are dead.

1. **θ₀ is the persistent self — holds.** The LLM is stateless between calls;
   the field is changed by every turn and carries that change forward. This is
   the mind-emulation thesis, intact, and it is the one claim that was never in
   question.
2. ⚠️ **Retrieval is LEXICAL, not field-addressed.** This section used to say
   "memory is retrieved by field resonance, and the field's own dynamics choose
   what surfaces". Measured on a validated harness: lexical **0.838** against
   field **0.036**, with chance at 0.009. The field lost exactly one job —
   being a retrieval index — and lexical search does it instead.
3. ⚠️ **The field's job is IDENTITY, not lookup.** What it contributes to a
   turn is which memory surfaced and the shape Mati is in while answering, not
   the finding of a fact. That is a smaller claim than the original and it is
   the one that is true.
4. ⚠️ **Dreaming does NOT deepen recall.** It used to be listed as "replays and
   deepens basins offline". Preregistered and run: nightly 0/8, per-turn 1/8,
   against a no-imprint control at 8/8 — imprinting a consolidation makes
   retrieval *worse*. Dreaming is kept for continuity of state, not for recall.
5. ⚠️⚠️ **Principles are enforced in CODE, never in the field.** The old claim
   was "ethics live in protected field nodes, so they cannot be argued away by
   text" — and this document already refuted it further down, in *Where the
   principles live*, which is a contradiction that stood in one file. A phase
   field installs no stance: a principle against its own negation moved
   +0.0000 / +0.0005 / -0.0017 against a 0.0009 noise floor.

## Boundary (read this before extending)

Mati **must not** ask the field to produce or score language. That is the
faculty it does not have. The field supplies:

- `triplet` — 3-vector basin coordinates (the Born-rule readout, `w_k = |ψ_k|²`)
- `coherence` — scalar |⟨e^{iθ}⟩| at output nodes; how settled the state is
- `recall` — triplet-addressed notes (text the *user* wrote, not text the field
  generated)

…and consumes the turn (user text and, gated, the reply) as a θ₀ deformation.
Everything that must be grammatical comes from the LLM.

**Mati is allowed to know about its own machinery, and should.** The boundary
above is about what the field can *do*, not about what Mati may understand.
An assistant that can tell you *why* its recall is failing — the address is
stale, the memory is archived, the imprint was shallow, the worker is dead —
is more useful and more honest than one that only says "I don't remember".
That is a goal for a future release (`HANDOFF.md`, "The self-diagnosis goal"),
and it is why Mati's questions about its own workings are kept rather than
filtered out.

The rule that survives it: a diagnosis is **computed**, never guessed. Every
figure it would draw on is already deterministic — `self_knowledge()`,
`tier_counts()`, `shallow_count()`, `/health`. The model phrases the finding;
it does not decide what is wrong. Same shape as `reckon.py`.

## Turn shape

```
user text
  → imprint into θ₀            (field changes; this is the primary training signal)
  → read field state           (triplet + coherence)
  → triplet-addressed recall   (field resonance selects memories)
  → LLM call                   (stable system prompt cached; field state appended last)
  → reply
  → coherence-gated consolidation of θ₀
  → (offline) DreamingLoop replays the notes DB
```

Caching note: the system prompt (constitution + persona) is stable and carries
the `cache_control` breakpoint. Field state and recalled notes are volatile, so
they go **after** it, in `messages` — otherwise every turn would invalidate the
prefix. See `shared/prompt-caching.md`.

## Files

| File | Role |
|---|---|
| `config.py` | model id, generation + coupling tunables, and the measurements behind them |
| `llm.py` | provider clients — Ollama by default, Anthropic optional |
| `bridge.py` | the actual coupling: field → context, reply → θ₀ |
| `mati.py` | one end-to-end turn; the system prompt lives here |
| `server.py` | **the one process that owns θ₀.** Both apps are its clients |
| `client.py` | shared transport + a chat REPL |
| `exclusive.py` | one-owner-of-θ₀ guard for every tool that loads the field |
| `vault.py` | encryption at rest; Touch ID releases the key |
| `setup.py` | first run — vault, then who the user is |
| `admit.py` | the single gate for what becomes a memory |
| `voice.py` | strips leaked internals out of replies |
| `study.py` | reading a file, slowly, on idle |
| `inquiry.py` | the directed ask → compare → ask loop |
| `correction.py` | being corrected, and remembering it |
| `challenge.py` | pushing back, only where it can recompute |
| `schedule.py` | the only place Mati acts rather than responds |
| `reckon.py` | routing arithmetic and logic to a deterministic engine |

⚠️ **The table is the field-coupling CORE, not the whole package.** There are
around seventy-five modules; the twenty above are the ones you must understand
before changing how a turn works. The feature surface — weather, radio, media,
the accountant, the library, drafts, the encyclopedia, speech — sits on top of
this and is documented in `README.md`.

The test suite lives in its own repository rather than beside the code. It runs
against scratch copies of the notes DB and θ₀ and asserts the real ones are
untouched: writing a memory is a real, irreversible deformation of the field,
and a suite that mutates what it is testing is worse than no suite.

## Known gotcha, encoded in `bridge.py`

`imprint_question` builds (context → target) pairs and **early-returns when
there are none** (`field_lm.py:457-459`) — a single token yields zero pairs and
imprints *nothing*, silently. A probe was invalidated by exactly this on
. `imprint_text` refuses inputs shorter than 2 tokens rather than
no-op quietly.


## Before shipping

The release checklist — not a feature list, but the set
of things that would make a shipped Mati *wrong* rather than merely
incomplete: claims that would stop being true, and defects a competent reader
would find first.

## Where the principles live

Not in the field. `embed_constitution` was measured on  and installs
no stance: contrast between a principle and its own negation moved +0.0000 /
+0.0005 / -0.0017 at 1 / 3 / 9 training passes, against a noise floor of 0.0009.
Distinguishing a principle from its opposite needs negation, and the
composition probes put this field at ~98% additive.

The four principles are enforced as code invariants — provenance immutability,
single-owner continuity, and the coherence gates. Each one, what enforces it,
and its test: **[PRINCIPLES.md](PRINCIPLES.md)**.

`imprint_question` now takes `lift_protection`; Mati passes `False` so a
question cannot reshape frozen nodes. Verified: frozen drift 0.000000 against
1.662277 for free nodes. That machinery is sound and currently protects an
empty set, because nothing is embedded and nothing should be until there is a
mechanism that can carry a rule.


## Ordering rules that are not obvious

Learned by measurement, each after getting it wrong:

- **Any θ₀ deformation must be followed by a reindex.** Every stored address was
  measured against the θ₀ of its moment, so imprinting anything — an episode, a
  studied chunk, a profile fact, a constitution — stales every address. It
  surfaced four separate times before being written down. `study`, `dream` and
  `record_episode` all reindex; the server batches it on idle.
- **Recall before imprint, within a turn.** Addressing after the turn has been
  absorbed measures the query against a field that has already changed.
- **Absorb before acquiring — but BOUNDED.** Unimprinted material cannot be
  recalled at all, whereas an unasked question costs nothing, so reading comes
  first. ⚠️ It used to come first *absolutely*, and that was measured to starve
  everything else: a 1,759-chunk document queued overnight stopped self-talk
  dead and left dreams at zero, and would have silenced every other background
  function for about two days. Study now takes one idle pass in two; the rest
  of the chain gets the others. Reading still gets most of the wall clock
  because a study pass is ~90s and a deepen pass is seconds.
- **`CREATE TABLE IF NOT EXISTS` does not add columns.** New columns need an
  explicit `ALTER TABLE` against databases that predate them.
