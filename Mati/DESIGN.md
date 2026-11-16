# Mati — field-coupled LLM layer

## Why this exists

The phase field encodes multi-token context ~99% **additively** — spatial
interaction 0.008, temporal 0.022, against 0.840 for a real tensor code, and
unchanged by training. Grammar needs conjunctive binding, and the field does not
supply it.

That rules out **one** job for the field — being the *language* faculty — and
nothing else. Mati couples the field to a small language model that supplies the
missing faculty:

| Faculty | Owner | Why |
|---|---|---|
| Composition / grammar | **LLM** | attention *is* a binding mechanism; the field is additive |
| Persistent identity | **field** | θ₀ is changed by every interaction and carries it forward; no LLM has this |
| Memory retrieval | `Mati/notes.py` | lexical retrieval is the one that finds the right memory |
| Principles | **code** | a phase field cannot hold a rule; see *Where the principles live* |
| Logic | `Mati/field/math.py` | exact, deterministic |

## What makes the field non-decorative here

1. **θ₀ is the persistent self.** The LLM is stateless between calls; the field
   is changed by every turn and carries that change forward.
2. ⚠️ **Retrieval is LEXICAL, not field-addressed.** On a validated harness
   lexical retrieval finds the right memory 0.838 of the time and field
   addressing 0.036, with chance at 0.009. Being a retrieval index is the one
   job the field does not do.
3. ⚠️ **The field's job is IDENTITY, not lookup.** What it contributes to a
   turn is the shape Mati is in while answering, not the finding of a fact.
4. ⚠️ **Dreaming does NOT deepen recall.** Imprinting a consolidation makes
   retrieval *worse*. Dreaming is kept for continuity of state, not for recall.
5. ⚠️⚠️ **Principles are enforced in CODE, never in the field.** A phase field
   installs no stance: a principle and its own negation are indistinguishable
   against the noise floor.

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
above is about what the field can *do*, not about what Mati may understand. An
assistant that can tell you *why* its recall is failing — the memory is
archived, the imprint was shallow, the vault is locked — is more useful and more
honest than one that only says "I don't remember", which is why Mati's questions
about its own workings are kept rather than filtered out.

The rule that goes with it: a diagnosis is **computed**, never guessed. Every
figure it draws on is deterministic — `self_knowledge()`, `tier_counts()`,
`shallow_count()`, `/health`. The model phrases the finding; it does not decide
what is wrong. Same shape as `reckon.py`.

## Turn shape

```
user text
  → field-addressed recall      (from the frozen address space)
  → lexical search              (the memories the reply is built on)
  → imprint into θ₀             (the field changes; the primary training signal)
  → read field state            (triplet + coherence)
  → LLM call                    (stable system prompt first; field state after it)
  → reply
  → coherence-gated consolidation of θ₀, and the episode, off the turn
```

Overnight, Mati rests: it reflects on the day, makes the day's prediction,
promotes the day's field into the address space, and free-dreams.

Prompt caching: the system prompt (charter + persona) is stable, and Ollama
reuses the KV cache when the prefix is unchanged. Field state and recalled notes
are volatile, so they go **after** it, in `messages` — otherwise every turn
would invalidate the prefix.

## Files

| File | Role |
|---|---|
| `config.py` | model id, generation + coupling tunables, and the reasons behind them |
| `llm.py` | provider clients — Ollama by default, Anthropic optional |
| `bridge.py` | the actual coupling: field → context, reply → θ₀ |
| `mati.py` | one end-to-end turn; the system prompt lives here |
| `server.py` | **the one process that owns θ₀.** Both apps are its clients |
| `client.py` | shared transport + a chat REPL |
| `exclusive.py` | one-owner-of-θ₀ guard for every tool that loads the field |
| `vault.py` | encryption at rest; the key in the keychain or behind a passphrase |
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
around eighty modules; the ones above are those you must understand before
changing how a turn works. The feature surface — weather, radio, media, the
accountant, the library, drafts, the encyclopedia, speech — sits on top of this
and is documented in `README.md`.

The test suite is not part of this package. It runs against scratch copies of
the notes database and θ₀ and asserts the real ones are untouched: writing a
memory is a real, irreversible deformation of the field, and a suite that
mutates what it is testing is worse than no suite.

## Known gotcha, encoded in `bridge.py`

`imprint_question` builds (context → target) pairs and **returns early when
there are none** — a single token yields zero pairs and imprints *nothing*,
silently. `imprint_text` refuses inputs shorter than 2 tokens rather than no-op
quietly.

## Where the principles live

Not in the field. `embed_constitution` installs no stance: the contrast between
a principle and its own negation stays inside the noise floor at every number of
training passes. Telling a principle from its opposite needs negation, and this
field is ~98% additive.

The four principles are enforced as code invariants — provenance immutability,
single-owner continuity, and the coherence gates. Each one, what enforces it,
and how it is checked: **[PRINCIPLES.md](PRINCIPLES.md)**.

`imprint_question` takes `lift_protection`; Mati passes `False`, so a question
cannot reshape frozen nodes. That machinery protects an empty set today, because
nothing is embedded and nothing should be until there is a mechanism that can
carry a rule.

## Ordering rules that are not obvious

- **Addresses are read from a frozen space, and only promotion moves it.**
  Every stored address is a distance measured against a particular θ₀. Recall
  reads a pinned copy, so imprinting does not stale the store; `rest()` promotes
  the day's field into the address space once a night and re-addresses the
  working set immediately after. Promoting without the re-address would leave
  every memory filed in a space that no longer exists.
- **Recall before imprint, within a turn.** Addressing after the turn has been
  absorbed measures the query against a field that has already changed.
- **Absorb before acquiring — but BOUNDED.** Unimprinted material cannot be
  recalled at all, whereas an unasked question costs nothing, so reading comes
  first. ⚠️ Not absolutely: a large document would otherwise silence every other
  background function for days. Study takes one idle pass in `STUDY_EVERY`; the
  rest of the chain gets the others. Reading still gets most of the wall clock,
  because a study pass is long and the others are seconds.
- **`CREATE TABLE IF NOT EXISTS` does not add columns.** New columns need an
  explicit `ALTER TABLE` against databases that predate them.
