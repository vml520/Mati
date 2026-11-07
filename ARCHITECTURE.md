# The framework — one field, three modules

Settled , on measurement. This is the shape the Mati app is built
against; changes to it need evidence, not preference.

---

## The decision

The original design had **four fields** — language, logic, security,
memories. It was scrapped for a single field when the language field took too
long. The question reopened in August: should the others come back?

**Answer: one field, three modules.** Not because fields are wrong, but because
three of the four jobs were measured not to need one.

| | form | why |
|---|---|---|
| **Language** | **FIELD** | It *is* the field. θ₀ shaped by imprinting is the whole project. |
| **Logic** | MODULE | `Mati/field/math.py`, 147 checks, exact to 1e-14. The field was measured **not load-bearing**: factoring 33% with it against a 25.5% control. |
| **Memory** | MODULE | Lexical **0.838** against field **0.036** on a validated harness, chance 0.009. |
| **Security / health** | MODULE | The real failures are process-level and not field-structured. |

## What the field keeps, and what it loses

⚠️ **This is not a demotion of the field.** It remains the language substrate,
the thing that gets trained, and the identity — the field IS Mati. It loses exactly
one job: **being a retrieval index**, which it was measured at chance on.

⚠️ **And one question stays open in the field's favour.** Every retrieval arm was
CUED. Whether the right memory arrives **unasked** is untested, and it is the one
thing lexical cannot do at all. If the field earns a retrieval role back, it
earns it there — see the segue.

> ⚠️ **PARTLY CLOSED.**
> Unasked arrival has two possible forms and only one survives. A memory that
> **surfaces** does so as text, by address, top-k — that is the segue, and it
> is still the open case. A memory that does **not** surface can now be shown
> to reach the model through nothing but `triplet` + `coherence`, and that
> channel was measured to carry **no information about content**: two different
> meanings separated by 2.59e-02 against an arrangement floor of 3.52e-02 —
> less than two shufflings of the same words. There is a tail and it is static.

## ⭐ How a reply is produced, and where the field can touch it

Established by reading `mati.turn` before measuring anything, and the reason
the study above had a design at all.

```
1  address_vector(user_text)          150-D address (the `feat` column)
2  recall(addr)            → memories FIELD-addressed, top-k
3  imprint_text(user_text)            query settles into θ₀ — AFTER recall
4  field_state(user_text)  → triplet, coherence   (snapshots + restores θ₀)
5  parts[0] = state_block(triplet, coherence, memories)
6  named = bridge.search(...)         LEXICAL — a separate block
7  … report / world / weather / charter / draft / reference blocks …
8  messages = history + system(context) + user
9  think_while(stream_reply(messages))
```

⚠️ **`think_while` "yields chunks unchanged".** The field steps between chunks
so the substrate is genuinely running while Mati talks — but it **cannot
alter a token**. There is no non-textual path from field to reply. Everything
the field contributes is text in the prompt, through exactly two doors:

| path | width | can carry a tail? |
|---|---|---|
| surfaced memories | top-k items | **no** — in the block or absent |
| `triplet` + `coherence` | **4 floats** | yes, and it is static |

⚠️ **That second door also decides what is REMEMBERED**, because an episode is
written only when `coherence >= EPISODE_GATE_MIN`. It is a narrow channel with
a wide consequence.

⚠️ **`imprint_text` writes NOTHING to the notes DB** — it deforms θ₀ only,
while `recall` searches the DB. So an imprinted text can never surface. Any
future study arm that checks "did my imprint stay off-address?" is vacuously
true, and a positive control built that way is **impossible, not weak**. That
cost one full run.

## Memory: the module boundary, and how it went wrong

The design was already right and the code put the pieces in the wrong boxes:

- `field/notes.py` — the notes module, **field-addressed since the initial
  commit**. Its docstring: *"The database is a pure index of physical
  addresses. Retrieval is nearest-neighbor search in triplet space R³."*
- `bridge.py` — 2,644 lines of field↔LLM coupling, which is where lexical
  `search` landed when it arrived on ****.

So the lexical retrieval that measures 0.838 lives in the coupling layer, and
the addressing that measures 0.036 is the notes module. **Lexical was not lost;
it arrived late and went into the wrong file.**

**DONE .** `Mati/notes.py`, 604 lines; `bridge.py` 2,645 → 2,164.
What moved:

| moves to the notes module | stays in `bridge.py` |
|---|---|
| `search`, `search_sources` | `recall` (field addressing, secondary) |
| `_idf`, `_focus`, `_nouns`, `asking_part` | `imprint_text`, `address_vector` |
| `_BODY_CACHE`, `when_said`, `record_hits` | `consolidate_reply`, `state_block` |
| `library.py` (catalogue) sits beside it | the field↔LLM coupling and nothing else |

Result: **`field/` is the substrate, `notes` is retrieval, `math` is logic,
`bridge` is coupling.** Each nameable in one sentence.

`bridge` re-exports the moved names as a **compatibility surface, not
ownership** — the server, terminal, segue and suites all call `bridge.search`,
and renaming every call site inside the same change would have made a
mechanical move impossible to review. New callers import from `Mati.notes`.

⚠️ **Two real defects surfaced during the move, and both were the kind that
pass quietly.** `_IDF`, `_TAGGER` and `_NOUNISH` were left behind — that one
announced itself as a NameError. The second did not: `NOTES_DB_PATH` was
imported into the new module as a module-level copy, which broke every caller
that REDIRECTS the store by assigning `bridge.NOTES_DB_PATH`. Five do —
`test_tiers`, `test_server`, `test_coldstart`, `test_transfer` and two
constitution diagnostics. `test_tiers` happened to raise `vault.Locked`; **the
others would have silently read the live store and reported a pass.** The path
is now resolved at call time, one source of truth.

## Security / health, corrected

⚠️ An earlier assessment here was **wrong and is withdrawn**. It conflated
*encoding rules into a field* — genuinely refuted, since a phase field cannot
hold "never" against "always" and composition is ~98% additive — with **monitoring
and repairing failing nodes**, which is a different proposal and a needed role.

The evidence that it is needed is this project's own August:

- a stale process served 11.5 days of old code while `/health` said ready
- the reindex loop cannot converge, and reported success for weeks
- the worker thread dies silently and `/health` still reports ready
- three separate θ₀-corruption paths, found by accident

Split by what the signal actually is:

- **Process and state integrity → MODULE.** Stale code, dead workers, corrupt
  θ₀, backup validity. None of this is field-structured; it is bookkeeping, and
  it failed because nothing owned it.
- **Node health → stays field-native**, inside the field's own maintenance.
  Per-node coherence and decoherence genuinely are field quantities, and that
  half of the original security-field proposal is sound.

## ⚠️ Correction — "security as a field" was dismissed too broadly

Twice I wrote that field-native security was refuted. That was true of ONE
proposal and I generalised it to all three. The original framing, recalled:
*"the agent who first designed the architecture may have called the security
modules 'brain blood barrier', because that is how I originally described
it."* It is recorded in the project's own planning notes, and it is a THIRD
thing:

| proposal | verdict |
|---|---|
| **encode rules into the field** | **refuted** — a phase field cannot hold "never" against "always"; composition is ~98% additive |
| **monitor and repair failing nodes** | needed; node coherence genuinely is a field quantity, process integrity is not |
| **decoherence barrier (blood-brain barrier)** | a REQUIREMENT, never tested — implement however is fastest |

The barrier is *"a separate preprocessing field, structurally distinct from the
main θ, that uses TFT decoherence dynamics (`eta_decohere` on isolated modes) to
filter incoming information… adversarial, unstable, or high-frequency-junk input
patterns decohere at the barrier field and never reach the main field's θ₀.
Coherent (learned/expected) patterns pass through."*

⚠️ **The refutation does not reach it, and that is the point.** The constitution
work showed a field cannot hold a RULE. The barrier holds no rule — it exposes
input to the substrate's own dynamics and lets incoherent patterns fall apart on
their own. That is a mechanism, not a stored proposition, so the ~98%
composition cap says nothing about it either.

⚠️ **And it GATES rather than monitors.** I argued a health module could only
observe, never block. The barrier blocks, by construction, which is the property
I said was missing.

⚠️ **AND IT IS A JOB, NOT A SUBSTRATE — I over-corrected.** The correction:
*"the point is not to recreate the barrier as a field if a security module can
do the same faster."* Having wrongly filed the barrier under "refuted", I
then filed it under "field", which is the same mistake pointing the other
way. The name
describes what it must DO: stop adversarial, unstable or junk input from
deforming θ₀. How it does that is open, and on this project's own evidence a
module is the default.

The cost argument is decisive and not close. A barrier field means evolving a
second field on every input, when field work is already what makes a turn slow —
field work is the expensive part of a turn. ⚠️ The `/health` figure once
cited here — 22s from GIL starvation — was a misdiagnosis of mine: it was
`schedule.pending` decrypting 690 sealed rows, fixed in f13f673, and /health
now answers in 0.05s. The cost argument does not depend on it.
Paying that twice per turn to filter input would be the most expensive possible
way to do it.

**And most of the mechanism already exists.** `consolidate_reply` imprints a
reply ONLY if the field settled coherently on it — a coherence gate on the way
in, using a number the turn has already computed. The barrier's job on the input
side is the same shape, against the same number, for free. That is the thing to
try first: if a threshold on the coherence already in hand does the work, no
barrier field is needed and none should be built.

A barrier field earns its place only if it is MEASURED to filter something the
cheap gate cannot. Until then it stays Stage 4/5, and the requirement — not the
implementation — is what belongs in the record.

## What this framework does NOT fix

- ⚠️ **The ~98% composition cap.** Conjunctive recall — "the thing about X *and*
  Y" — is out of reach on any field here, and no module boundary touches it.
- ⚠️ **Unbidden surfacing**, which is untested in either direction.
- The two-tier archive: 1,744 book passages sit outside the working set, so the
  *field* path cannot reach them at all. Lexical can, which is part of why it
  wins.

## Consequence for what is already built

`segue.choose` consumes `bridge.recall` — the field path — because that is where
surfacing happens. On this evidence that channel is at chance, so the segue
would offer near-random connections.

The original architecture survives and only the mechanism changes: *the
relevance brings up the memory, then consequence is judged.* Surfacing becomes
lexical.
This is **not** a return to the scan that was removed — surfacing still happens
once, before judgement, and the segue still never queries on its own behalf.

## Provenance of every number here

⚠️ Replacing the citation column left every row reading "measured", which is
a column that says nothing. What a reader needs is HOW each was established:

| claim | how it was established |
|---|---|
| field 0.036 / lexical 0.838 / chance 0.009 | 111 cued retrievals on a validated harness, against a shuffled-cue floor |
| drift is the learning signal, four negatives | four separate arms, all negative, on a fresh field per run |
| frozen field holds, live collapses in one turn | self-retrieval over six turns, frozen 6/6 against live 1/6 |
| field not load-bearing for factoring | the solver scored the same with the field disabled |
| ~98% additive composition | composition probes against a conjunctive target |

Every one is a preregistered run with its controls, not an impression formed
while building. Where a number here disagrees with a comment in the code, the
comment is the one to re-measure — several were written before the harness
that later contradicted them.
