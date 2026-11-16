# The framework — one field, three modules

This is the shape Mati is built against. Changing it needs evidence, not
preference.

---

## The decision

Mati has four jobs — language, logic, security, memory — and one phase field.
The field does the one job only it can do, and modules do the rest, because
the other three jobs were shown not to need a field.

| | form | why |
|---|---|---|
| **Language** | **FIELD** | It *is* the field. θ₀ shaped by imprinting is what Mati is. |
| **Logic** | MODULE | `Mati/field/math.py` is exact to 1e-14. The field is **not load-bearing** for it: factoring scores 33% with the field against a 25.5% control. |
| **Memory** | MODULE | Lexical retrieval finds the right memory **0.838** of the time on a validated harness; field addressing finds it **0.036**, against chance at 0.009. |
| **Security / health** | MODULE | The failures that matter are process-level, not field-structured. |

## What the field keeps, and what it does not do

⚠️ **This is not a demotion of the field.** It is the language substrate, the
thing that gets trained, and the identity — the field IS Mati. The one job it
does not do is **being a retrieval index**, where it performs at chance.

⚠️ **One question stays open in the field's favour.** Every retrieval test was
CUED. Whether the right memory arrives **unasked** is untested, and it is the
one thing lexical retrieval cannot do at all.

A memory reaches a reply in only two ways. One that **surfaces** does so as
text, by address, top-k — that is where unasked arrival would have to happen.
One that does **not** surface can reach the model only through `triplet` +
`coherence`, and that channel carries **no information about content**: two
different meanings separate by 2.59e-02, against 3.52e-02 between two
shufflings of the same words. There is a tail, and it is static.

## ⭐ How a reply is produced, and where the field can touch it

```
1  address_vector(user_text)          150-D address (the `feat` column)
2  recall(addr)            → memories FIELD-addressed, top-k
3  notes.search(...)                  LEXICAL — a separate block
4  imprint_text(user_text)            the message settles into θ₀
5  field_state(user_text)  → triplet, coherence   (snapshots + restores θ₀)
6  parts[0] = state_block(triplet, coherence, memories)
7  … report / world / weather / charter / draft / reference blocks …
8  messages = history + system(context) + user
9  think_while(stream_reply(messages))
```

⚠️ **`think_while` yields chunks unchanged.** The field steps between chunks,
so the substrate is genuinely running while Mati talks — but it **cannot alter
a token**. There is no non-textual path from field to reply. Everything the
field contributes is text in the prompt, through exactly two doors:

| path | width | can carry a tail? |
|---|---|---|
| surfaced memories | top-k items | **no** — in the block or absent |
| `triplet` + `coherence` | **4 floats** | yes, and it is static |

⚠️ **That second door also decides what is REMEMBERED.** An episode is written
only when `coherence >= EPISODE_GATE_MIN`, which is why step 4 cannot be moved
after the reply: without the imprint, the field has not settled on the
message, the gate does not clear, and nothing is remembered. A narrow channel
with a wide consequence.

⚠️ **`imprint_text` writes NOTHING to the notes database** — it deforms θ₀
only, while `recall` searches the database. An imprinted text can never
surface, so any test that imprints and then checks recall is testing nothing.

## Memory: where each piece lives

**`field/` is the substrate, `notes` is retrieval, `math` is logic, `bridge` is
coupling.** Each can be named in one sentence.

| the notes module (`Mati/notes.py`) | `bridge.py` |
|---|---|
| `search`, `search_sources` | `recall` (field addressing, secondary) |
| `_idf`, `_focus`, `_nouns`, `asking_part` | `imprint_text`, `address_vector` |
| `_BODY_CACHE`, `when_said`, `record_hits` | `consolidate_reply`, `state_block` |
| `library.py` (the catalogue) sits beside it | the field↔LLM coupling and nothing else |

`bridge` re-exports the notes names as a **compatibility surface, not
ownership**. New callers import from `Mati.notes`.

⚠️ **The store's path is read at call time, never copied.** Anything that
points Mati at a different store does it by assigning
`bridge.NOTES_DB_PATH`; a module holding its own copy of the path would go on
reading the real store and report that all was well.

Memories beyond the working set (`WORKING_SET` episodes) are archived. The
field path cannot reach them at all; lexical search can, which is part of why
lexical retrieval is the primary one.

## Security and health

Three different proposals go by the name "security", and they have different
answers:

| proposal | answer |
|---|---|
| **encode rules into the field** | **does not work** — a phase field cannot hold "never" against "always"; composition is ~98% additive. The principles are enforced in code instead (`PRINCIPLES.md`). |
| **monitor and repair failing nodes** | needed. Per-node coherence genuinely is a field quantity and stays field-native; process and state integrity is not, and is a module. |
| **a decoherence barrier** | a requirement: adversarial, unstable or junk input must not deform θ₀. How it is met is open, and a module is the default. |

**Process and state integrity is a module** because none of it is
field-structured: stale code still answering, a dead worker, a corrupt θ₀, a
backup that cannot be restored. These fail quietly unless something owns them,
which is why `/health` reports `code_stale`, `worker_alive` and
`memories_locked` rather than a single "ready".

**The barrier gates; it does not monitor.** It holds no rule — it lets input
meet the substrate's own dynamics so that incoherent patterns fall apart — so
the result that a field cannot hold a rule does not reach it. Today the job is
done by two cheap gates rather than a second field:

- `admit.check`, at the single point where anything becomes a memory
  (`bridge.remember`), against named rules;
- `consolidate_reply`, which imprints a reply only if the field settled
  coherently on it, using a number the turn has already computed.

A barrier *field* would mean evolving a second field on every input, when
field work is already the expensive part of a turn. It earns a place only if
it is shown to filter something these gates cannot.

## What this framework does NOT fix

- ⚠️ **The ~98% composition cap.** Conjunctive recall — "the thing about X
  *and* Y" — is out of reach for any field here, and no module boundary
  touches it.
- ⚠️ **Unbidden surfacing** is untested in either direction.
- ⚠️ **Archived memories** are reachable only by their words.

## Consequence for the segue

The segue — Mati bringing up something related on its own — works from the
memories the turn already retrieved by their words, not from the field path,
because the field path performs at chance and would offer near-random
connections. Surfacing still happens once, before judgement, and the segue
never runs a search of its own.

## How every number here was established

| claim | how it was established |
|---|---|
| field 0.036 / lexical 0.838 / chance 0.009 | 111 cued retrievals on a validated harness, against a shuffled-cue floor |
| the static tail | two meanings against a floor of shufflings of the same words |
| field not load-bearing for factoring | the solver scored the same with the field disabled |
| ~98% additive composition | composition probes against a conjunctive target |

Each is a preregistered run with its controls, not an impression formed while
building.
