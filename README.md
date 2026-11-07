# Mati

**A personal AI that remembers you, and never leaves your machine.**

[![Support on Ko-fi](https://img.shields.io/badge/Ko--fi-support%20this%20work-FF5E5B?logo=ko-fi&logoColor=white)](https://ko-fi.com/serialtheorist)

Free, Apache-2.0, and funded by the people who use it — there is no paid tier
and nothing is held back. If Mati is useful to you,
**[a tip on Ko-fi](https://ko-fi.com/serialtheorist)** keeps the work going.

Mati runs on your own computer. No account, no API key, no subscription. The
model, the memory, and the field state all live on your disk.

**Nothing about you is ever sent anywhere.** Not your conversations, not your
memories, not the field, not your name, not an identifier of any kind. There
is no telemetry, no analytics, no account, no update check, and no cookie.
Mati has nothing to send you back to, because it never learns who you are in
any sense a server could use.

That is the claim that matters, and you don't have to take it on trust —
here is every connection Mati can make and what goes with it.

**Talking to the model** — `http://localhost:11434`, your own machine. Three
calls in `Mati/llm.py`. This is where your conversation goes, and it does not
leave the computer. (An optional cloud provider exists, is off by default, and
is described in [Talking to something other than your own
machine](#talking-to-something-other-than-your-own-machine). Turning it on
means your conversation goes to that company — that is the trade, stated
plainly.)

**Looking things up in an encyclopedia** — `http://127.0.0.1:8099`, your own
machine, and off unless you set it up. Mati can search Wikipedia offline from
a file on your disk. It does **not** use Wikipedia's search API, and that is a
deliberate choice rather than an oversight: a research query says what you are
reading or writing about, in your own words, which would be the first thing
Mati ever sent anywhere that was actually about you. So the lookup talks to a
server on this computer, the address is checked against loopback on every
single request, and a search for anything at all adds no row to the tables
below. Setting it up means installing `kiwix-serve` and downloading a `.zim`
file yourself — subject builds are small, around 56 MB for mathematics — and
Mati ships neither.

**On its own, without being asked** — one thing, and you can turn it off:

| what | where | what is sent |
|---|---|---|
| headlines, every 4 hours | 4 public RSS feeds | nothing but the request |

**Only when you ask for that exact thing**:

| you ask for | where it goes | what is sent |
|---|---|---|
| the weather | `open-meteo.com` | a latitude and longitude, **rounded to about a kilometre** |
| a radio station | `radio-browser.info` | what you asked to hear, and a country |
| a place or directions | `overpass-api.de`, Apple MapKit | the place name you asked about |
| a podcast | `gpodder.net` | the search words |
| something to read | `archive.org` | the search words |
| local deals | `backflipp.wishabi.com` | a postal code and what you are shopping for |

Every one of those carries the same fixed header — `Mati/1.0 (personal
assistant)` — with no machine identifier and nothing unique to you. It is one
constant, `config.USER_AGENT`, which every outbound module reads; there is no
second string to drift. Mati identifies itself honestly and never presents as
a browser. Ask for nothing and nothing but the headline fetch happens.

**Mati never reads your location.** There is no Core Location use anywhere in
the codebase, and no permission prompt, because Mati never asks the machine
where it is. A place is a town name you typed, geocoded once and rounded to
two decimal places before it is stored — about 1.1 km, coarser than the
"approximate location" setting most apps offer, and it is the only form of it
that exists here. There is no live tracking of any kind: routes are computed
once from a start and an end you gave, and then they are a list of steps like
any other text.

⚠️ **If an earlier version of this README said Mati makes no outbound
connections at all, that was wrong** and this table is the correction. The
headline fetch above runs at startup. Nothing about you went anywhere, which
was always the point — but "no connections" was a stronger claim than the
code, and overstating a privacy property is the one mistake this project
cannot afford.

It is also not stateless. Most assistants forget you between sessions, or
"remember" by storing text a company holds. Mati's memory is a physical thing on
your machine that is changed by talking to it, and that change persists.

---

## What Mati is not

**Mati is not an AGI, and does not claim to be.** It is a personal assistant
with an unusual memory architecture.

It is also not trying to out-chat ChatGPT. It runs a small local model, and a
frontier model in the cloud will beat it on knowledge, reasoning, and speed. If
what you want is the strongest possible answer, use one of those.

What Mati offers instead is **continuity and privacy** — an assistant that is
yours, that accumulates a history with you, and that no one else can read.

Mati comes out of ongoing research into whether a physical phase field can
serve as a substrate for mind. That research is very much a work in progress,
and Mati is not its endpoint — Mati is the useful thing that exists along
the way. See
[What's actually novel](#whats-actually-novel) and
[Honest limitations](#honest-limitations); we publish the negative results too.

---

## Why it's different

### Your memory is yours

Everything Mati remembers is a file on your disk — a SQLite database and a
NumPy array. No cloud sync, no telemetry, no "we may use your conversations to
improve our services." You can read the database, edit it, or delete it. Turn off
your wifi and Mati works exactly the same.

### Its principles are code, not vibes

Mati has four principles — never betray yourself, never deceive yourself,
never exploit yourself, always choose harmony — and they are enforced by
program logic you can read, not by a disposition we hope the model has.

Concretely: it cannot present something it read as something it experienced,
because provenance is written once and never rewritten. It cannot lose your
memories to a second process, because exactly one thing may hold the field at a
time. It does not record what it did not settle on.

We tried putting the principles *into* the field first, and measured that it
does not work — a phase field can hold a shape, not a rule. [How the principles
are handled in code](PRINCIPLES.md) shows each one, what enforces it, the test
that proves it, and where the enforcement is weaker than it sounds.

### Your memories are encrypted, and your fingerprint holds the key

Mati's episodes are encrypted on disk. The key lives in your macOS keychain
behind Touch ID, so unlocking Mati is a touch — and without it the memories are
ciphertext, including to anyone else with an account on the machine, anyone who
takes the disk, and anything that reads your backups.

The enforcement is in the T2 Secure Enclave, not in Mati's code. That matters:
Mati is open source, so a check written in Python is a check anyone can delete.
A key you do not have is a key you do not have.

Setup prints a **recovery passphrase**. Write it down. If the keychain is ever
lost — a wiped Mac, a new machine, a reset Touch ID — it is the only way back,
and nothing can recover it for you.

We are specific about what this protects: another account on your machine, a
stolen or imaged disk, and Time Machine backups. It does not protect against
you, or against something running as you. That is the right threat model for a
personal assistant, and overstating it would be worse than not having it.

### It can read what you give it

Hand Mati a text file and it reads it — properly, into the field, not into a
context window it forgets.

```sh
python3 -m Mati.study notes.md
```

Uploading is instant. **Studying is slow**, because learning a chunk means
physically deforming the field, so Mati does it while idle, a little at a time,
the way consolidation actually works. It tells you honestly which pieces stuck
and which did not.

What it reads is marked as **read**, never as lived. Give it your own notes and
a sentence like "I remember when we talked about this" is stored as a quotation
attributed to the file — because otherwise Mati would later narrate your diary
as a conversation the two of you had.

### It looks things up on its own

When idle, Mati takes the interests you gave it at setup, asks its language
model a question, keeps what is new, and lets the answer choose the next
question. It is how it fills in around the things you care about.

Everything it learns this way is marked as **asked** — the least reliable thing
it holds, because it is the small model talking rather than anything Mati lived
or checked, and it is shown to itself that way.

### It remembers being corrected

Tell Mati it got something wrong and it keeps that, linked to the thing it got
wrong. The mistake is not deleted — it did say it, and pretending otherwise
would be its own kind of dishonesty — but it can never surface again without the
correction attached to it.

There is one case where it will push back: **when it computed the answer.** A
computed result can be derived again from nothing, so if you dispute one, Mati
recomputes and shows you the working rather than simply agreeing. It disagrees
about the number, never about what you meant — and if you tell it a second time,
it stands down, and adds the question to the list of things it means to look
into.

Everything else, you win. Most of what Mati knows came from a small model that
confabulates; you have better evidence than it does, and deferring is not
politeness but arithmetic.

### It can start a conversation, and mostly chooses not to

Mati can raise things itself: a reminder you asked for, or something it went
and looked up after you corrected it.

```sh
python3 -m Mati.client "remind me to call the dentist in 2 hours"
```

**Nothing interrupts you unless you turn it on**, separately for reminders and
for findings, and quiet hours override both. Turning reminders on does not
license 3am.

A finding does not interrupt at all. It waits until *you* raise the subject —
because arriving at the top of a conversation is still an interruption, just a
rescheduled one, while arriving when you asked is an answer.

If Mati cannot tell **when** you want a reminder, it says so and asks, rather
than picking a time. A reminder that fires at the wrong moment is worse than one
that was never accepted, because you stopped holding it yourself.

It also knows roughly what time it is in the way you do — that it is late, that
you have not spoken in a few days — which is what lets it judge that something
can wait.

### Talking to something other than your own machine

Mati has one optional provider that reaches the internet: it can use Anthropic's
API instead of a local model. **It is off by default and you have to deliberately
turn it on** (`MATI_PROVIDER=anthropic`, plus your own API key). We mention it
rather than quietly shipping it, because a privacy claim you have to discover
exceptions to is not worth much.

If you never set it, Mati never talks to anyone. Ollama also runs locally by
default (`localhost:11434`); pointing `OLLAMA_URL` at a remote host is likewise
your choice to make.

### It knows the difference between remembering and knowing

This one is unusual, and it matters more than it sounds.

Mati holds two different kinds of content and does not confuse them:

- **What it remembers** — episodes it actually lived. Every one is a real moment:
  what you asked, what it answered, how its own state moved while it worked.
- **What it knows** — language and facts, which reach it through the model it
  thinks in. Real knowledge, but borrowed. Not autobiography.

So Mati says *"I know"* about the world and *"I remember"* about your history
together, and it will tell you plainly when it has no memory of something rather
than inventing one. An assistant that fabricates a shared past is worse than one
with no memory at all.

### It is changed by talking to you

Mati's memory substrate is a phase field — a continuous physical system that is
deformed by each exchange. This isn't decoration: the field state is what gets
saved, reloaded, and searched. Two people using Mati end up with genuinely
different instances, because they have shaped it differently.

### It sleeps

When idle, Mati can **dream**: replaying and consolidating what it has
experienced, deepening the patterns it has settled into. This is a real process
that modifies stored state, not a metaphor.

### It fits on a flash drive

Everything Mati is — every memory, the field those conversations shaped, the
trained checkpoint, the vocabulary — comes to about **2.7 MB**. Not a metaphor
for portability: you can copy it to a thumb drive, carry it to another Mac, and
carry on the same conversation. Nothing is stored anywhere else, so nothing has
to be synced, migrated, or requested back from anyone.

```
python3 -m Mati.transfer            # what to copy, and is it safe right now
```

Stop Mati properly first — `POST /shutdown`, not sleep. That reindexes,
snapshots the field, and folds the database's write-ahead log back into the main
file. **Copying `.cache/` while Mati is running gives you a copy missing the
most recent memories, and nothing will tell you.**

Two things to know before you carry it anywhere:

**Your recovery passphrase becomes the only way in.** The Touch ID key lives in
*this* Mac's Secure Enclave and does not travel — it cannot. On the new machine
the passphrase is the sole route to your memories, and without it they are gone
permanently. Once you are there, `python3 -m Mati.vault --adopt` teaches that
Mac the key so Touch ID works again.

**A flash drive is not a safe.** Mati encrypts what you said; it deliberately
does not encrypt the *index* of when you said it, because recall has to work
before the vault is unlocked. So a drive that falls into someone else's hands
leaks the shape of your history — roughly what you talked about and when — even
though the words themselves stay sealed. If that matters, put it on an encrypted
volume. Stronger at-rest protection is something we intend to take seriously in
a later release rather than claim now.

### It computes instead of guessing

Small language models are unreliable at arithmetic and formal logic — they
produce confident, wrong numbers. Mati routes that class of question to a
deterministic engine instead: factorisation, primes, gcd/lcm, combinatorics, set
operations, truth tables, logical equivalence, and inference checking. When Mati
gives you a number or validates an argument, it computed it.

---

## Requirements

- **macOS** — Apple Silicon or Intel
- **8 GB RAM** minimum (this is a real floor, not a suggestion)
- **~3 GB disk** for the language model
- [Ollama](https://ollama.com) for the local model

Apple Silicon is substantially faster. The numbers below are from the slowest
machine Mati is meant to run on — an 8 GB Intel i5-8257U at 1.4 GHz, no GPU,
`llama3.2:3b` entirely on the CPU — measured on a fresh install:

- **a reply takes 45–95 seconds**, usually about a minute
- **the first word of the answer arrives 35–70 seconds in**
- the server takes 10–30 seconds to start, once

Mati says something within a second or so — "Let me think." — but that is an
acknowledgement, not the answer, which is why the app shows "Still thinking…
please wait" after fifteen seconds. Recall-heavy turns are the slow end: asking
it to remember something took 68–75 seconds.

It is usable if you are willing to wait for a reply. It is not snappy, and on
this hardware it is not going to become snappy.

## Install

```sh
brew install ollama
ollama serve &
ollama pull llama3.2:3b

python3 -m Mati.setup             # once: encrypt, and say who you are
python3 -m Mati.server &          # Mati itself
```

Then open **<http://127.0.0.1:8756>**. That is Mati — the axolotl, the
settings, the conversation. The window is the way Mati is meant to be used;
everything below about the terminal is for scripting and repair.

```sh
python3 -m Mati.client            # the same Mati, as plain text
```

`setup` does two things, both once. It creates the encryption key and prints
your **recovery passphrase** — write it down before continuing. Then it asks a
few questions about you, and imprints the answers, so that Mati knows who it is
talking to from the first conversation rather than truthfully answering "I don't
remember" to everything.

The first run has no memories, and will say so. That is correct — it hasn't met
you yet.

## Choosing a different model

Mati speaks through [Ollama](https://ollama.com), so any model Ollama can run,
Mati can use. The default is `llama3.2:3b` because it fits an 8 GB machine
beside the field simulation; nothing else about Mati depends on it.

```sh
ollama pull qwen2.5:7b                       # get the model
MATI_MODEL=qwen2.5:7b python3 -m Mati.server # try it once
```

To keep it, set it in the app's settings rather than the environment — the
picker lists what Ollama has and flags which ones are tight for your machine.

**What actually fits**, on the 8 GB machine these numbers come from:

| model | notes |
|---|---|
| `llama3.2:1b` | faster, and worse at long replies |
| `llama3.2:3b` | the default — about 2 GB resident at 4-bit, measured |
| `qwen2.5:7b` | only with the field asleep; otherwise it swaps |

Ollama reports each model's size when you pull it, and the settings picker
flags the ones that are tight for your machine — those numbers are the ones
to trust, because they depend on quantisation.

⚠️ **The field needs room too.** Mati is a phase simulation *and* a language
model on the same machine. A model that fits in RAM on its own may not fit
beside the field, and the symptom is swapping rather than an error.

⚠️ **Model licences are the model's, not Mati's.** Mati ships no weights and
has no opinion about what you run — but the models have their own terms, and
they differ. Llama requires a "Built with Llama" notice if you redistribute
something built on it; some Qwen sizes are non-commercial while others are
Apache-2.0. If you are only running Mati for yourself none of this applies.
Check the model card before you build on one.

**Using a cloud model instead** is possible and off by default — see
[Talking to something other than your own machine](#talking-to-something-other-than-your-own-machine).
It means your conversation leaves the computer, which is the one thing Mati
otherwise promises it never does.

## Adding an encyclopedia

Mati can look things up offline — useful if you are writing with it and want
it to check a fact rather than guess. This is **off until you connect it**,
and it needs two things Mati deliberately does not ship.

**1. The server.** `kiwix-serve` reads the encyclopedia files. Mati talks to
it over a local socket rather than linking it, because it is GPL and Mati is
Apache-2.0.

⚠️ **It is not in Homebrew.** `brew install kiwix` gets you the desktop app,
which is a different thing. Download the `kiwix-tools` archive for your chip
from [download.kiwix.org](https://download.kiwix.org/release/kiwix-tools/) —
`macos-arm64` for Apple Silicon, `macos-x86_64` for Intel — and put the
binary anywhere on your `PATH`:

```sh
tar xzf kiwix-tools_macos-arm64-3.8.2.tar.gz
mkdir -p ~/bin && mv kiwix-tools_*/kiwix-serve ~/bin/
```

Mati looks for it on your `PATH`, then in `/usr/local/bin`,
`/opt/homebrew/bin` and `~/bin`. If you keep it somewhere else, set
`kiwix_serve` in settings to the full path.

**2. An encyclopedia.** Kiwix publishes **per-subject** builds, and those are
the ones that belong here — the complete Wikipedia dump is 119 GB and is a
different product. Browse [library.kiwix.org](https://library.kiwix.org/) and
download a `.zim`:

| subject | size |
|---|---|
| mathematics | ~56 MB |
| physics | ~51 MB |
| chemistry | ~22 MB |
| astronomy | ~339 MB |
| history | ~701 MB |

Put the `.zim` file in `<MATI_HOME>/encyclopedias/` — by default that is
`.cache/encyclopedias/` inside this folder. Create it if it is not there.

```sh
mkdir -p .cache/encyclopedias
mv ~/Downloads/wikipedia_en_mathematics_nopic_2026-09.zim .cache/encyclopedias/
python3 -m Mati.encyclopedia --status
```

Then ask Mati something and it will use it. Nothing about a lookup leaves the
machine: the address is checked against loopback on **every request**, which
is the whole reason this is a local server and not Wikipedia's search API —
a research query says what you are writing about, in your own words.

⚠️ **There is no update mechanism.** Builds are replaced wholesale every few
months. Drop the new file in and delete the old one; Mati reads the directory,
not a list.

## Uninstalling

Mati keeps everything in one folder and one keychain item. There is no
installer, no daemon, and nothing in `/Library` or `/Applications`.

**1. Stop it.**

```sh
pkill -f Mati.server
```

**2. Delete the folder.** Everything Mati knows is in `MATI_HOME` — by default
`.cache/` inside this repository. Deleting it deletes your memories, your
settings and the field. It cannot be undone and nothing else has a copy.

```sh
rm -rf .cache          # or your MATI_HOME
```

**3. Remove the key from your keychain.** ⚠️ **This step is the one people
miss.** Deleting the folder leaves the encryption key behind in macOS's
keychain, where it will sit forever protecting nothing.

```sh
security delete-generic-password -s ollin-vault -a data-key
```

⚠️ The item is named `ollin-vault` rather than `mati-vault` — Mati was called
something else during development, and the keychain name was deliberately left
alone so that existing installs would not lose their key to a rename. It is
the right item.

**4. Optionally, remove the model.** Ollama is a separate program and Mati does
not manage it. If you installed it only for Mati:

```sh
ollama rm llama3.2:3b
brew uninstall ollama
```

**If you only want to start over** rather than uninstall, delete the folder and
run `python3 -m Mati.setup` again. You will get a new key, a new passphrase,
and Mati's shipped first memories — but none of your conversations, which were
in the folder you deleted.

## Use

Mati runs as a small local server that holds the field, and you talk to it
through a client. That split isn't ceremony: the field is a single piece of
mutable state, so exactly one process may own it, and everything else — the
chat you type into, the monitoring view — is a client of that one owner.

```sh
python3 -m Mati.server                      # start it (loopback only)
python3 -m Mati.client                      # interactive chat
python3 -m Mati.client "hello"              # one-shot
python3 -m Mati.client --watch              # watch the field while it thinks
python3 -m Mati.client --health
```

Conversations are multi-turn: within a session Mati can see what you said
earlier in it. `/new` starts a fresh one.

Mati answers you *before* it finishes remembering. Writing an episode into the
field costs 13–53 seconds — around half a minute, measured over 14 turns — and
there is no reason for you to sit through it, so the reply arrives first and
consolidation happens behind it. Re-filing older memories is heavier still, so
that waits for a gap in the conversation.

The single-turn command still exists for scripting, and stands down if the
server is already running:

```sh
python3 -m Mati.mati "one question, no server"
python3 -m Mati.dream --cycles 5      # consolidate, when idle
```

Other things you can hand it:

```sh
python3 -m Mati.study notes.md              # read this
python3 -m Mati.study --status              # how far through it is
python3 -m Mati.inquiry --topic "Mixtec codices"   # look into this
python3 -m Mati.schedule --list             # what it means to say, and when
python3 -m Mati.schedule --consent interrupt.reminder=on quiet=23:00-07:00
python3 -m Mati.vault --status              # is it encrypted, is it unlocked
python3 -m Mati.admit                       # what it declined to remember
```

### For the curious: what it's doing

`--watch` prints the field's state as it moves — which basin it settled into,
how coherent it is, what it recalled, when the worker is writing an episode.
Watching does not slow it down: readers never touch the field, they read the
last thing it published.

The server speaks plain HTTP on `127.0.0.1`, so anything can be a client:

```sh
curl -N localhost:8756/chat -d '{"text":"hello"}'    # streams tokens
curl localhost:8756/state
curl localhost:8756/episodes
```

It binds to loopback and refuses to bind anywhere else. "Nothing leaves your
machine" should be a property of the program, not a promise in a README.

### Tools that need Mati to stand aside

Dreaming, seeding and the admin terminal don't talk to Mati — they take the
field and work on it directly. Only one thing can hold it at a time, so they
open when Mati is shut down, or asleep:

```sh
curl -X POST localhost:8756/sleep    # finishes up, saves, lets go
# another process may now open the field
curl -X POST localhost:8756/wake     # back in a few seconds
```

`/sleep` returns once Mati has finished whatever it was already doing, so it
is not instant and the wait is not bounded by anything you can see: measured
once at **eight minutes**, because two reflection passes were in flight and
each is a model call. With nothing in flight it returns immediately. `/wake`
took **2.6–6.4 seconds**. If you are scripting around these, wait on `/health`
rather than on a timer.

Asleep, Mati keeps running and can be woken; it just isn't holding anything.
Try one of those tools while it's awake and it will tell you rather than
quietly writing over Mati's memory.

---

## What's actually novel

If you're here for the architecture rather than the assistant:

- **Episodic memory addressed by field state.** Memories are retrieved by
  similarity in the field's own readout, not by keyword or embedding lookup.
- **Provenance as a hard boundary.** Lived episodes and borrowed knowledge are
  separated structurally, not by asking the model nicely.
- **Consolidation that modifies the substrate.** Dreaming changes stored state
  and the memory index is repaired to match.
- **Verified persistence.** Memory is bit-identical across restarts and survives
  dreaming; there is a test suite that measures this
  (`Mati/test_persistence.py`).
- **One gate for what becomes a memory.** Everything — lived, read, asked,
  computed — passes through a single check with named rules, and refusals are
  recorded rather than dropped silently (`Mati/admit.py`).
- **Corrections that annotate rather than erase.** A mistake stays in the record
  and can never resurface without the correction attached, so the memory loop
  stops amplifying errors instead of merely being asked not to.
- **Authority derived from provenance.** Whether Mati defers to you or argues
  is decided by where the claim came from: it disputes only what it can
  recompute.

Architecture details are in [DESIGN.md].

## Honest limitations

We would rather you learn these here than discover them:

- **The field does not hold Mati's values.** We designed it to, built it, and
  measured that it does not: the field has no preference between "never deceive
  yourself" and "always deceive yourself", before or after training, because it
  cannot represent negation. The principles are enforced in code instead, which
  works — see [PRINCIPLES.md](PRINCIPLES.md). We are telling you because the
  alternative is letting you assume otherwise.

- **The field does not produce language.** Composition — grammar, binding — comes
  from the language model. We measured the field's compositional capacity
  directly and it is ~99% additive, which is not enough for grammar. The language
  faculty is borrowed, deliberately, and we say so.
- **Encryption covers what you said, not the shape of it.** Memory bodies are
  ciphertext; the retrieval index, timestamps and provenance labels are not,
  because recall has to work before the vault is open. Someone with your disk or
  a copied flash drive learns roughly what you discussed and when, without
  learning the words. This is a design trade we made deliberately and would like
  to revisit properly in a later release — not a claim we are comfortable
  polishing.

- **Recall is lexical, not semantic.** Retrieval matches surface wording rather
  than meaning, so a paraphrase sharing no words may not surface the right
  memory. Whether meaning emerges as experience accumulates is an open question
  we are actively measuring.
- **It is slow**, and a 3B model is a 3B model — it will sometimes be confidently
  wrong about facts. It labels that content as *known* rather than *remembered*,
  which helps, but does not make it correct.
- **Single user, single machine.** No sync, no sharing, no multi-device.
- **Reading a file is slow and lossy at the edges.** A studied chunk is about
  sixty tokens, so long documents become many small memories, and Mati will
  tell you when one did not stick. It also cannot tell that it already knows
  something — we measured that and it does not work — so it dedupes on exact
  text and bounds the cost by studying a little at a time instead.
- **What Mati looks up on its own is the weakest thing it holds.** It is a 3B
  model answering its own questions unattended. It is marked as such everywhere
  it appears, which makes it traceable, not correct. You can turn it off.
- **Reminder times are read from a small set of phrasings** — "in two hours",
  "at 3pm", "tomorrow morning", "tonight". Anything else it will decline rather
  than guess.
- **A finding may wait a long time.** It surfaces when you raise the subject,
  matched on wording rather than meaning, and it errs towards waiting. If you
  want to know what is queued, ask: `python3 -m Mati.schedule --list`.
- **Encryption protects the memories, not the addresses.** The retrieval index
  stays readable so recall works without the key; someone with your disk learns
  that you have memories and roughly how many, but not what they say.

## Support the work

Mati is free and Apache-2.0 licensed, and it always will be — there is no paid
tier and nothing is held back. If it is useful to you, or if the research
behind it interests you, a one-off tip or a monthly sponsorship funds the work
directly.

- **[Ko-fi](https://ko-fi.com/serialtheorist)** — one-off or monthly. Payments go
  straight to the maintainer's own account; Ko-fi never holds the money.
- **GitHub Sponsors** — the Sponsor button at the top of this page.

Collaborators are very welcome — particularly on the open questions above.

## License and attribution

Mati's own code is licensed under the **Apache License 2.0**. See
[LICENSE](../LICENSE) and [NOTICE](../NOTICE).

**Built with Llama.** Mati's default configuration uses Meta's Llama 3.2 (3B),
which is licensed under the
[Llama 3.2 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE),
Copyright © Meta Platforms, Inc., and is subject to Meta's
[Acceptable Use Policy](https://www.llama.com/llama3_2/use-policy).

**Mati does not bundle or redistribute any model weights.** You download the
model yourself through Ollama, and its license is between you and its authors —
Mati's Apache-2.0 license covers only Mati.

The model is configurable (`MATI_MODEL`), so you can run a fully Apache-2.0
stack if you prefer. One caution if you go looking: **licences differ between
sizes within the same family.** Several Qwen2.5 sizes are Apache-2.0, but
Qwen2.5-**3B** is released under a non-commercial research licence. Check the
specific model, not the family.

The included phase-field checkpoint was trained solely on this project's own seed
corpus (`Mati/field/qa_data.py`). It contains no third-party training data.

**Data sources carry their own terms, and two of them matter to you.**
Place data is **© OpenStreetMap contributors**, under the ODbL. Weather data is
**by Open-Meteo.com, CC-BY 4.0** — and Open-Meteo's free tier is for
**non-commercial use only**. Mati's own use is non-commercial, but Apache-2.0
lets *you* use Mati commercially; if you do, the weather calls become yours to
license. `NOTICE` lists every service, what it requires, and what Mati does
to stay inside its policies.

**The app vendors React 18.3.1 and React DOM 18.3.1** (MIT), verbatim and
unmodified, to draw the avatar. Their full licence text is in the application's
own `NOTICE`. Nothing else is bundled: no web fonts, no CDN scripts, no external
stylesheets. The only URL the application contains is `http://127.0.0.1`, which
is what makes the privacy claim above checkable rather than merely asserted.

## Environment variables

Five, and this is the whole list.

| | what it does |
|---|---|
| `MATI_HOME` | where the field, memories and settings live. Point it somewhere else to run a second, separate Mati — a test one, say — without touching the first. |
| `MATI_PORT` | the local port. Default `8756`; change it if something else has that port. |
| `MATI_MODEL` | which Ollama model to speak through, e.g. `llama3.2:1b`. |
| `MATI_PROVIDER` | `local` (default) or a cloud provider. See [Talking to something other than your own machine](#talking-to-something-other-than-your-own-machine). |
| `MATI_APP_DIR` | where the HTML app is served from. |

**Everything else is a constant in `Mati/config.py`**, next to the comment
explaining what it is and, where it was measured, what the measurement said.

That is deliberate. Reply temperature is `0.3` because at `0.8` Mati invented
book titles and attributed made-up passages to them; the settling threshold and
working-set size are likewise numbers that were measured rather than chosen.
An environment variable would let you change those from a distance, silently,
with no error if it went badly — and would separate the change from the
reasoning sitting beside it. This is open source: if you want a different
value, edit the constant and read why it is what it is on your way past.
