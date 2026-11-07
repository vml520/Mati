"""Mati configuration — model, paths, and coupling tunables."""

import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.dirname(_HERE)


# ── Where a user's own choices live ───────────────────────────────────────────
# Setup asks which language model to speak through; that answer has to outlive
# the shell that ran setup, so it is written here rather than left as an env
# var the user would have to remember to re-export.
#
# Precedence is env > settings file > default. The env var still wins, so a
# one-off `MATI_MODEL=... python3 -m Mati.server` works for trying something
# without committing to it.
# ── Where Mati's field lives ─────────────────────────────────────────────────
# ONE definition, and everything derives from it. `MATI_HOME` repoints the
# whole install at a different directory — which is how you get a TEST field
# without touching the real one (`python3 -m Mati.sandbox`).
#
# This was a real bug, not a hypothetical: the vendored `field/data.py` computed
# its own cache path from __file__, and vendoring moved it a directory deeper,
# so it silently resolved to `Mati/.cache` — a store that does not exist. A
# path computed in two places is a path that will disagree in one of them.
CACHE_DIR = os.environ.get("MATI_HOME") or os.path.join(ROOT, '.cache')

SETTINGS_PATH = os.path.join(CACHE_DIR, 'mati_settings.json')

# Where the HTML app lives. Served by the server so the page can reach the
# API at all: a file:// page is a different origin and every fetch is
# blocked, which means no auto-wake and no conversation.
# ⚠️⚠️ IT USED TO BE A POINTER AT A SIBLING FOLDER, AND THAT SHIPPED A
# TERMINAL-ONLY MATI. The old default was `../Mati App`, which resolves on
# the machine where both folders sit side by side and resolves NOWHERE in a
# download. Measured on a real clone: the server came up, the field loaded,
# and `GET /` returned 915 bytes of status JSON — no window, no avatar, no
# settings. Mati is GUI-centred; a person who downloads it and gets a text
# client has not been given Mati.
#
# ⭐ SO: THE BUNDLED COPY FIRST, THE SIBLING SECOND. A release carries `app/`
# and uses it. A development checkout that has the app repo next door keeps
# using that, so editing the real source still shows up immediately — which
# is what the old comment was protecting and is worth keeping.
def _app_dir():
    env = os.environ.get("MATI_APP_DIR")
    if env:
        return env
    bundled = os.path.join(ROOT, "app")
    if os.path.isfile(os.path.join(bundled, "index.html")):
        return bundled
    # ⚠️ Checked for `index.html`, not just the directory: an empty `app/`
    # left behind by a partial copy would otherwise win over a working
    # sibling and serve nothing.
    # ⚠️ BOTH SPELLINGS. The product became Mati ; the app repo
    # on this machine is still the folder it always was. Accepting either
    # means the rename does not quietly disconnect a development checkout
    # from the window it is meant to be editing.
    for name in ("Mati App", "Ollin App"):
        p = os.path.join(os.path.dirname(ROOT), name)
        if os.path.isfile(os.path.join(p, "index.html")):
            return p
    return os.path.join(os.path.dirname(ROOT), "Mati App")


APP_DIR = _app_dir()


def _settings():
    try:
        with open(SETTINGS_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _choice(env_var, key, default):
    v = os.environ.get(env_var)
    if v:
        return v
    return _settings().get(key, default)


def save_settings(**kw):
    """Persist a user choice. Merges, so one key never clobbers another."""
    cur = _settings()
    cur.update({k: v for k, v in kw.items() if v is not None})
    os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
    tmp = SETTINGS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cur, f, indent=2)
    os.replace(tmp, SETTINGS_PATH)          # atomic; never a half-written file
    return cur


# ── LLM provider ──────────────────────────────────────────────────────────────
# 'local'     — Ollama on localhost. Free, offline, no account, no key.
#               Stdlib-only client (urllib); nothing to pip install.
# 'anthropic' — Claude API. Costs money and needs a key. Optional.
#
# Local is the default on purpose: Mati's LLM layer supplies COMPOSITION only
# (identity/memory/ethics live in the field), and small models do grammar well.
# It also keeps Mati self-contained — it runs offline, on your machine, with no
# external dependency.
PROVIDER = _choice("MATI_PROVIDER", "provider", "local")

# --- local (Ollama) ---
# Sized for 8GB Intel: a 3B at 4-bit is ~2GB resident and leaves room for the
# field sim. Swap to 'llama3.2:1b' if you want speed, 'qwen2.5:7b' only if the
# field isn't running.
LOCAL_MODEL   = _choice("MATI_MODEL", "local_model", "llama3.2:3b")
OLLAMA_URL    = _choice("OLLAMA_URL", "ollama_url", "http://localhost:11434")
LOCAL_NUM_CTX = 4096         # keep modest — context length costs RAM
# ⚠️ 0.3, AND THE CREATIVE-WRITING END OF THE RANGE IS WRONG HERE. 0.8 is a
# creative-writing temperature; Mati's job is to answer from stored memory
# without inventing. At 0.8 it wandered, elaborated, and reached for
# plausible-sounding detail — "Epiphanies, the Aztecs, and Stoicism" for three
# schools it holds 844 passages on, an invented book title presented as
# something it had read, and a fabricated passage attributed to the book.
#
# Lowered to 0.3, which is the retrieval-grounded range. It costs variety in
# phrasing, and that is the trade: a narrower voice that invents less.
#
# ⚠️ CHANGE THIS ALONE. The gate log records reply length now, so the effect is
# measurable against real conversation — but only if nothing else moves at the
# same time.
LOCAL_TEMP    = 0.3

# Generation runs at ~7.2 tok/s on this machine (, 3B
# CPU), so every token costs ~139ms and REPLY LENGTH IS THE LATENCY. An
# unprompted 192-token answer to a one-line question is 27 seconds of waiting.
#
# ⚠️ THE CEILING IS GONE. It was 220 tokens, with a comment saying
# it sat "well above the target so it effectively never truncates mid-sentence".
# It did truncate, repeatedly, and the user hit it in ordinary use — a cap set
# by counting tokens rather than by watching replies is a guess, and this one
# was wrong.
#
# -2 means "fill the context window": no length imposed by us, bounded only by
# what actually fits alongside the prompt and the memories. The brevity
# instruction in the system prompt is now the ONLY brake on length, which is
# the right place for it — a rule the model can follow naturally beats a knife
# that cuts mid-word.
#
# ⚠️ AND REPLY LENGTH IS LATENCY. ~103 ms/token measured  (9.7
# tok/s), so a reply that fills the remaining window is minutes.
#
# ⚠️ THE ADVICE ABOVE — "fix the PROMPT, not this number" — WAS TESTED AND
# REFUTED. Three prompt configurations were measured on  (1,745 /
# 606 / 695 tokens, with and without the anti-padding lines) and NONE of them
# moved verbosity: replies stayed +35% over baseline throughout. `-2` means
# "fill the context window" — an explicit instruction to keep going, against
# which BE SHORT is a suggestion.
#
# Measured on the same question, same prompt, same temperature:
#     num_predict = -2   375 tok   54.1s   272 words
#     num_predict = 220  220 tok   31.6s   139 words
#     num_predict = 160  160 tok   22.5s   118 words
#
# ⚠️ AND THE ORIGINAL OBJECTION IS ANSWERED RATHER THAN IGNORED. A cap did cut
# mid-sentence — but `voice.ReplyFilter` already releases only COMPLETED
# sentences and holds the fragment back until `flush()`. Ollama reports
# `done_reason: "length"` when the cap bit, so the fragment is DROPPED in that
# case. The ceiling stops at the last whole sentence; it is not a knife.
LOCAL_MAX_TOKENS = 260

# ── Reply length, as a person's choice rather than a constant ────────────────
#
# The user: *"Wire reply length. We should keep in mind that the shipping app
# will run on any model, so we should make efficiency changes like that
# everywhere."*
#
# ⚠️ MEASURED, AND THE NUMBERS ARE THE REASON THIS IS A SETTING AT ALL. Same
# question, same prompt, same temperature:
#
#     160 tokens   22.5s   118 words
#     220 tokens   31.6s   139 words
#     260 tokens   54.1s   272 words
#
# Reply length IS latency on a small model. This is the one control a person
# can turn that changes the wait, which is why it is in the app's own settings
# and shows its own cost.
#
# ⚠️ A TOKEN CEILING IS THE ONE KNOB EVERY PROVIDER HAS, which is what makes
# it the right shape for a build that ships against any model. Ollama calls it
# `num_predict`, Anthropic calls it `max_tokens`, and both mean "stop after
# this much". Nothing here assumes a tokenizer, a speed, or a family.
#
# ⚠️ AND THE CEILING IS NOT A KNIFE. `voice.ReplyFilter` releases only
# COMPLETED sentences and holds the remainder; when Ollama reports
# `done_reason == "length"` the fragment is DROPPED. Ending one sentence early
# is not a defect — ending mid-word is, and that is what the old uncapped
# setting was avoiding by spending a minute instead.
REPLY_LENGTHS = {"brief": 160, "normal": 220, "thorough": 320}


def reply_tokens(default=None):
    """The ceiling this person chose, or the configured default.

    ⚠️ READ AT CALL TIME. The setting is written by the app while the server
    is running; a value captured at import would not change until restart,
    and a preference that needs a restart is not a preference.

    ⚠️ BRIEF IS THE DEFAULT (the user), not `normal`. Length IS
    latency on a 3B — 160 tokens is 22.5s against 220's 31.6s — so the
    out-of-the-box experience should be the fast one. Someone who wants longer
    answers can say so; someone who never opens settings should not be waiting
    nine extra seconds a turn for a default they did not choose.
    """
    try:
        choice = _settings().get("reply_length")
        if choice in REPLY_LENGTHS:
            return REPLY_LENGTHS[choice]
    except Exception:
        pass
    return default or REPLY_LENGTHS["brief"]


# ── how Mati identifies itself on the network ──────────────────────────────
# ⭐ ONE STRING, USED BY EVERY OUTBOUND MODULE, because the README promises
# exactly that: "every one of those carries the same fixed header". It did not.
# There were EIGHT different User-Agents — `Mati/1.0`, `(personal reader)`,
# `(personal podcast client)`, `(single-user; low volume)` and others — and two
# of them were worse than merely inconsistent:
#
#   ⚠️ `radio.py` sent `https://github.com/<account>`, a real person's account,
#      to radio-browser.info. The README's "nothing unique to you" was false.
#   ⚠️⚠️ `flipp.py` sent a full Safari string. Its own module header argued
#      "nothing here is being bypassed" — but presenting as a browser IS the
#      mechanism for not being seen as automation, so those two claims could
#      not both be true. MEASURED: the honest header below returns HTTP 200
#      and 67 items from the same endpoint, identical to the spoof, and so
#      does sending no User-Agent at all. The disguise bought nothing.
#
# ⚠️ KEEP IT DESCRIPTIVE. Radio Browser and the Overpass usage policy both ask
# that a client say what it is so operators can see who is calling; "Mati/1.0"
# alone does not. What must stay out is anything unique to the person.
USER_AGENT = "Mati/1.0 (personal assistant)"


# ── what the assistant is called ────────────────────────────────────────────
# ⚠️ THE ASSISTANT IS RENAMEABLE; THE PRODUCT IS NOT. The user:
# *"the name must be choosable; Mati is a personal AI."* A personal assistant
# that cannot be called what its person wants to call it is not personal.
#
# ⚠️ AND THE SPLIT IS THE WHOLE DESIGN. "Mati" stays the project, the repo,
# the app, the window title and the lock name — those are the PRODUCT. This
# function is only for the places where the assistant refers to ITSELF: the
# system prompt, the charter it speaks in first person, the provenance labels
# it reads aloud, and the greeting recogniser. Renaming the product too would
# make the docs, the commands and the support answers all wrong at once; a
# renamed assistant inside an app called Mati is coherent, a half-renamed one
# is not.
DEFAULT_ASSISTANT_NAME = "Mati"


def assistant_name():
    """What this person calls their assistant. Read at call time."""
    try:
        got = (_settings().get("assistant_name") or "").strip()
        # ⚠️ A ceiling, because this lands in every prompt and in `smalltalk`'s
        # recogniser. An essay here would be a prompt-injection surface and a
        # token cost on every single turn.
        if got and len(got) <= 32:
            return got
    except Exception:
        pass
    return DEFAULT_ASSISTANT_NAME

# Ollama unloads the model after 5 minutes idle by default. Reloading cost a
# measured 7.0s plus a 27.3s uncached prompt pass — ~34s added to the first
# turn after a pause, which is exactly when someone is most likely to be
# waiting. Holding it resident trades ~2GB of RAM for that.
LOCAL_KEEP_ALIVE = _choice("MATI_KEEP_ALIVE", "keep_alive", "30m")

# --- anthropic (optional) ---
# Sampling params are rejected on Opus 5 — steer via system prompt.
MODEL       = "claude-opus-5"
MAX_TOKENS  = 16000          # non-streaming; raise + stream for long outputs
THINKING    = {"type": "adaptive"}
EFFORT      = "high"         # low | medium | high | xhigh | max


# ── Field ─────────────────────────────────────────────────────────────────────
VOCAB_PATH   = os.path.join(CACHE_DIR, 'qa_vocab.json')
CKPT_PATH    = os.path.join(CACHE_DIR, 'diag_pattern8d_shuf_final.npz')
CKPT_THETA0  = os.path.join(CACHE_DIR, 'diag_pattern8d_shuf_final_theta0.npz')
NOTES_DB     = os.path.join(CACHE_DIR, 'mati_notes.db')


# ⚠️ AT IMPORT, because the rename has to happen before ANYTHING opens the
# database — `bridge.load_field()` runs in `FieldService.__init__`, long
# before the startup migrations at `server.py`'s vault step. Every entry point
# (server, terminal, tests, the ship build) imports this module first, which
# is the only place that is true of.
#
# ⚠️⚠️ OPTIONAL BY DESIGN. `legacy_names.py` is the one module that still has
# to name the pre-1.0 store, so the public build does not contain it — a first
# release has no earlier install to migrate. The check is a FILE TEST rather
# than `except ImportError`, because swallowing the exception would also
# swallow a genuine error raised inside that module and turn "the migration is
# broken" into "there is no migration", silently, on the one code path whose
# job is not to lose somebody's memories.
if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "legacy_names.py")):
    from . import legacy_names as _legacy_names
    _legacy_names.adopt(CACHE_DIR)

# ⚠️ AFTER the legacy rename and BEFORE anything opens the store. The order
# matters: an install with a pre-1.0 store must be renamed first, or seeding
# would see no `mati_notes.db`, place the shipped memories, and leave the
# person's real store sitting beside them under the old name.
#
# ⚠️ UNLIKE `legacy_names`, THIS ONE SHIPS — it is how a fresh install gets
# anything at all to recall, so it is imported directly rather than file-tested.
from . import first_run as _first_run
_first_run.seed(CACHE_DIR)

FIELD_SEED   = 42
ENCODING_DIM = 8
CONTEXT_LEN  = 8
ANSWER_STEPS = 40
MAX_RHO      = 0.05
ETA_DECOHERE = 0.005


# ── Coupling ──────────────────────────────────────────────────────────────────
# User turns imprint at full strength: user interaction is the PRIMARY training
# signal. The reply imprints only if the field
# settled coherently — the §10.4 field-native gate, not an external scorer.
IMPRINT_PASSES   = 3
IMPRINT_STRENGTH = 5.0
IMPRINT_STEPS    = 80

REPLY_GATE_MIN   = 0.30   # min coherence before a reply deforms θ₀ at all
REPLY_ALPHA      = 0.05   # lm_alpha_gs used when consolidating a reply

RECALL_K         = 4      # notes surfaced per turn by triplet resonance

# Token cap for triplet measurement. `imprint_question` does one forward pass
# PER TOKEN, so cost is linear in tokens (~1s/token on this machine) — full
# 250-token answers make seeding take many hours.
#
# CONSISTENCY REQUIREMENT: this cap must be applied identically when measuring a
# stored note address AND when measuring a query, or the two land in different
# spaces and every distance is meaningless. Both paths read this constant.
# NOTE: mati_notes.py's own CLI measures on FULL text — notes added through it
# will not be comparable with notes seeded here unless it gets the same cap.
#
# RAISED 32 -> 64 (the user: "raise the cap and pay the imprint
# cost"). At 32 a chunk was ~26 words — smaller than a prose sentence — so
# stored notes were fragments: "Mexico and Central America) until its collapse
# around 400-500 CE." was a real memory, unmatched bracket and all.
#
# Measured over 67 sentences of text Mati actually stores (median 16 tokens,
# mean 20, max 41):
#
#     cap  32 -> 82.1% of sentences fit whole
#     cap  48 -> 100%
#     cap  64 -> 100%, and holds 3-4 median sentences
#
# 64 rather than 48 because the unit that matters is a whole short answer, not
# a whole sentence — `inquiry.ASK_SYSTEM` asks the model for "three or four
# short factual sentences", and that should land as ONE note.
#
# THE COST IS REAL AND WAS ACCEPTED: imprinting is ~1s/token, so an episode
# write goes ~27s -> ~55s and a studied chunk ~30s -> ~60s. Both are deferred
# to the worker, so nobody waits for them. Reindex roughly doubles too
# (~1.9s -> ~3.8s per note), which is why it batches on idle.
#
# ⚠️ CHANGING THIS INVALIDATES EVERY STORED ADDRESS. `address_vector` caps here
# too, so old `feat` vectors were computed over a different span and land in a
# different space. A FULL reindex is mandatory after any change.
TRIPLET_CAP      = 64

# Episodic memory: record an episode only when the field actually settled.
# Consolidation is selective — unsettled turns leave no memory. Also keeps the
# ~1s/token imprint cost off the average turn.
EPISODE_GATE_MIN = 0.70

# Retrieval: minimum cosine similarity for an episode to surface at all.
# Trajectory features discriminate token surface, so unrelated text still scores
# fairly high — this floor keeps weak matches out rather than always returning
# the top-K of whatever exists.
RECALL_MIN_SIM = 0.15   # on CENTERED cosine (see bridge.recall), not raw

# Field steps taken between streamed reply chunks. The field runs WHILE the
# words form instead of idling through the call; costs time already being spent.
THINK_STEPS_PER_CHUNK = 3

# Ceiling on time SPENT stepping during a reply (not elapsed wall clock — that
# distinction matters, see bridge.think_while). : a step over
# three modes is 4.8ms and a typical reply spends ~1s stepping against ~26s of
# generation, so this never binds today. It is a guard against a pathologically
# long reply, not a latency fix. 0 disables it.
THINK_BUDGET_S = 4.0

# Max episodes re-addressed per reindex call. Bounds cost (reindex is O(n)
# forwards, ~1.9s/note measured). Selection is by STALENESS not recency, so
# everything rotates through — capping by recency would strand old memories in
# an obsolete address space permanently. 0 = no cap.
REINDEX_LIMIT = 32

# How far an address may drift from the current θ₀ before it counts as stale.
# Checked on startup and on wake, which is when θ₀ is most likely to have moved
# under the store — sleep hands the field to the pro terminal, and the terminal
# exists to train it.
#
# ⚠️ THIS IS NOT A "RECALL IS FINE BELOW HERE" THRESHOLD, and must not be
# described as one. Nothing has measured what drift actually costs retrieval.
# It is set just above float noise so that a genuinely untouched store reports
# clean (: 14/14 episodes at exactly 0.0) rather than
# queueing a pointless ~29s reindex on every restart.
STALE_EPS = 1e-9


# ── Server ────────────────────────────────────────────────────────────────────
# One process owns θ₀. Both apps — the chat app and the pro monitoring/training
# terminal — are clients of it, because θ₀ is mutable state backed by a file and
# two owners would corrupt each other.
#
# Loopback only, and deliberately not configurable to a public interface: the
# privacy claim ("nothing leaves your machine") is a property of the binding.
SERVER_HOST = "127.0.0.1"
SERVER_PORT = int(os.environ.get("MATI_PORT", "8756"))

# Conversation history kept per session, in user+reply pairs. Local context is
# only LOCAL_NUM_CTX tokens and the system prompt already eats a chunk of it, so
# this is a real ceiling rather than a tidiness measure.
SESSION_MAX_TURNS = 8

# Measured turn profile (, no LLM call): 71.6s total, of which
# `remember` is 27.5s and `reindex` 28.9s — 78.7%. Both are POST-reply work, so
# they run on the server's worker and the user never waits for them.
#
# Reindex additionally does not run per-turn at all. It re-addresses stored
# notes after θ₀ moves; batching it into idle time costs some address freshness
# between windows and saves ~29s of machine time on every single turn.
# ⚠️ 45 SECONDS WAS SHORTER THAN 96% OF THE GAPS IN A REAL CONVERSATION.
# Measured over 333 consecutive-turn gaps: median 3.0 min, p90 10.6, p95 32.8.
# So idle work started after almost every turn — the user: "idle is coming on too
# soon. Mati is not really idle unless it hasn't been spoken to in more than 45
# minutes at least, DEPENDING ON THE AGE OF THE PERSON."
#
# The last clause is the design. A fixed number is wrong for everyone except
# whoever it was tuned for, so the live value is LEARNED from the person's own
# record — `rhythm.idle_after()`, their p95 gap, floored at 45 minutes. This
# constant survives only as the fallback for a store with nothing in it yet.
IDLE_SECONDS = float(45 * 60)

# How long since the PERSON last said anything before Mati treats itself as
# sleeping. Idle is a gap in a conversation; sleep is being left alone.
#
# The split exists because the two want different work. Idle gets what is quick
# and what recall cannot do without — reindex, which must run because a single
# turn stales every stored address. Sleep gets what is expensive and can wait:
# studying a document (~60s a chunk), finishing an imprint that was cut short,
# asking anything.
#
# Keyed off the last USER TURN, deliberately, not `last_activity` — the worker
# resets that itself, so sleep would never arrive.
# ⚠️ 600s sat INSIDE p90 (10.6 min), so the expensive path fired during
# ordinary pauses. Raised past p99 (110 min): going quiet for two hours is a
# person who has left, not one who is thinking.
SLEEP_AFTER_SECONDS = float(2 * 3600)

# Mati MANUFACTURING a question for the person, offline, while it sleeps.
#
# ⚠️ OFF, AND THE REASON IS A DESIGN CORRECTION RATHER THAN A
# BUG. The user, after reading 863 of them: *"Maybe there really shouldn't be a
# generator. My initial understanding was that these questions would emerge
# naturally from the field during a turn — otherwise they will always sound
# forced."*
#
# They are right, and it explains why every fix upstream of this still left them
# sounding forced. A question written days earlier, from a passage, by a
# template, and then parked until its topic happens to come up, has no
# relation to the moment it arrives in. No amount of filtering repairs that:
# the problem is not which pre-fabricated question gets delivered, it is that
# it was pre-fabricated. Measured symptoms, all downstream of this one fact:
# 80% of the queue never addressed the person, 49.6% opened with the same
# three words, ~6% were sincere.
#
# ⚠️ THE TRAINING HALF STAYS ON, and it is a different function. The user:
# *"Maybe just keep the question generator for training."*
#
#     inquiry.step()            Mati asks its OWN model and imprints what
#                               comes back. Self-talk, DESIRED, untouched.
#     inquiry.question_for_user()  manufactures a question FOR THE USER and
#                               parks it. This is what is switched off.
#
# ⚠️ AND THE REPLACEMENT IS NOT WRITTEN YET, so turning this off means Mati
# asks nothing unprompted for now. That is the right way round: silence is
# better than 863 forced questions, and `inquiry.question_for_user` plus
# `bridge.episode_to_ask_about` are kept rather than deleted so the in-turn
# design can be measured against them.
ASK_USER_ON_SLEEP = False

# Asking instead of answering, when the field surfaces something Mati has
# already been corrected about. See `Mati/clarify.py`. This is what replaces
# the manufactured questions above.
CLARIFY_ON_CORRECTED = True

# ⚠️ "AFTER MANY MEMORIES OF ASSUMING INCORRECTLY" — the user's condition, and
# the dial that makes this self-limiting. Below this many corrections on
# record, Mati has not been wrong often enough to have learned to check, and
# asking would be a mannerism rather than caution.
#
# ⚠️ SET TO 5 AGAINST 11 ON RECORD (, of 4,372 notes), so
# it is live now rather than dormant — but barely, and that is honest: a fresh
# install has 0 and will not clarify at all until it has actually been wrong.
# The number is deliberately low because the OTHER gate is the strong one: a
# clarification also requires the surfaced memory to carry a correction, so
# this cannot fire on material Mati has never got wrong.
CLARIFY_AFTER_CORRECTIONS = 5

# Asking the 3B what an unrecognised turn MEANT. See `Mati/intent.py`.
#
# ⚠️ ON, AND THE USER CHOSE IT KNOWING THE COST: *"I would rather use the 3B
# than constantly have to add new terms to the listener."* The cost is not
# permanent — whatever the model answers is written to `aliases`, so an
# unfamiliar phrasing costs one classification and is a string lookup
# thereafter.
INTENT_FROM_MODEL = True

# Longer than this and a message is prose, not a command. ⚠️ MEASURED: the
# failure that started this was a sixty-word budget paragraph answered by
# stopping a webradio. Asking a model about prose invites the same error with
# more steps, so the cap is applied before the call.
INTENT_MAX_WORDS = 12

# No cap on how many questions may be waiting. A cap would be arbitrary — the
# thing that stops questions being intrusive is not how many exist but WHEN
# they are raised: never at the top of a conversation, only once the person
# brings the subject up themselves. An unraised question costs nothing.
#
# And the person can turn them off by saying so (`schedule.reads_ask_control`).

# Notes per reindex batch. The worker holds the field lock for a batch at a
# time and checks between batches for a waiting turn, so an arriving message
# waits for one batch (~8s) rather than a whole reindex (~60s).
REINDEX_BATCH = 4
# ⚠️ FIELD REINDEXING IS RETIRED BY MEASUREMENT, not switched off in a hurry.
# On a harness validated at lexical 0.84 (two earlier probe sets were rejected
# at 0.15 and 0.29), field addressing scored 4/111 = 0.036 against chance 0.009
# while lexical scored 93/111 = 0.838 on identical cues and pool — and 0.00
# after a single turn of drift. See ARCHITECTURE.md.
#
# It also could not terminate. `check_addresses` queues a reindex whenever
# anything is stale; the worker reindexes REINDEX_BATCH=4 notes and calls
# `check_addresses("after reindex")`, which finds everything stale again —
# because staleness is |gs_now - feat_gs| against a single GLOBAL scalar that
# moves whenever the field learns, so 4 repaired per pass loses to 62
# invalidated per turn. Its own log read "67/67 addresses stale after after
# reindex", repeatedly. The user's server burned CPU on that loop for 10.5 hours
# and stopped answering /health, which is what made the terminal call a working
# field unreachable.
#
# The staleness MEASUREMENT stays — it is a cheap SELECT and an honest
# diagnostic. What stops is automatically acting on it.
REINDEX_ON_IDLE = False

# ⚠️ ADDRESSING READS A FROZEN θ₀ AND WRITES THE LIVE ONE. There is no
# setting for this and there must not be one — see
# `bridge.pin_address_space`. It was `MATI_FROZEN_ADDRESSING` while the
# question was open; it is not open (6/6 against live's 1/6, and the
# reindex terminates, which it did not before). A switch between two memory
# architectures invalidates every stored address whichever way it is thrown.

# How many of the newest self-talk notes stay field-addressed.
# ⚠️ BOUNDED BECAUSE IT GROWS. Reflection writes ~3 a night; pinned
# outright it would be the entire recall pool within a year and every
# slot would go to Mati talking to itself rather than to anything that
# happened with the person. Ten nights is enough to resonate against.
PINNED_SELFTALK = 30

# Chunks of an uploaded document studied per idle pass. Studying is an imprint
# (~1s/token, so ~30s a chunk at TRIPLET_CAP) and it competes with the next
# turn for the field, so the queue is drained a little at a time rather than in
# one sitting. This is what bounds the cost of reading a file — see
# `Mati/study.py` on why the planned "skip what I already know" test could not.
# ⚠️ RAISED FROM 2 TO 8 at the user's instruction: *"Be sure to
# increase the dose chunks for the uploaded text files."* And the charter
# measurement is why it is safe to ask for a bigger dose on a BOOK while the
# charter needed a smaller one — they are different problems.
#
# Measured: 24 chunks of the charter put settling
# coherence at 0.587-0.682 against a 0.700 gate, so every turn was rejected;
# 6 chunks put 7 of 12 back above it. That is a ONE-VOICE store on a COLD
# field — a few thousand words of Mati's own text with nothing else in the
# store to settle against. A book is the opposite case: varied material
# arriving into a store that already holds conversation, which is what the
# dose limit was protecting against in the first place.
#
# ⚠️ AND THE OLD REASON FOR 2 IS WEAKER THAN IT WAS. The comment below says
# studying competes with the next turn for the field — true, and `study()`
# yields between chunks via `should_stop`, so an arriving turn interrupts a
# batch rather than waiting it out. 8 chunks is ~4 minutes of idle work and
# still interruptible at every chunk boundary.
#
# ⚠️ WHAT IT COSTS, STATED: every chunk is an imprint, and imprinting is what
# destroys addressability (measured: no-imprint 8/8,
# imprinted 1/8). Frozen addressing is what makes this affordable now —
# addresses are read from a space that holds still until `promote()`, so a
# burst of study no longer strands the notes written before it.
STUDY_BATCH = 8

# Skip a chunk already recallable above this similarity. None = off, and off is
# correct: , known material scores 0.424–0.688 and unseen
# material 0.305–0.754 — overlapping, and inverted on the mean, so no threshold
# separates them. Left here so it can be re-measured if retrieval ever becomes
# semantic rather than lexical.
STUDY_SIMILARITY_GATE = None


# Dream cycles at the end of first-run setup. ZERO — the premise was refuted.
#
# The idea was that dreaming right after imprinting the user's interests would
# deepen those basins. , paired against an untouched control
# topic, on a fresh field per run:
#
#     cycles   Δ seed    Δ control   seed − control
#          3  -0.0011     -0.0028         +0.0017
#         10  -0.0090     -0.0038         -0.0052
#         30  -0.0499     -0.0245         -0.0254
#
# The seeded topic loses coherence FASTER than a topic never mentioned, and the
# gap grows with cycles. Free-dreaming relaxes θ₀ toward the field's own
# intrinsic attractors, so it washes out a fresh imprint — and the seeded topic
# erodes most because it is the thing that was just imprinted. Dreaming here
# undoes what setup just did.
#
# The timing is what is wrong, not the mechanism. `DreamingLoop`'s REPLAY mode
# consolidates real turns; at setup there are none, so free-dream has nothing
# to work with but its own attractors. Dream once there is something to dream
# about — the server already does this on idle.
#
# `setup.first_dream()` is kept so this can be re-measured, not because it helps.
SETUP_DREAM_CYCLES = 0


# Should Mati look into the user's interests on its own while idle?
# (`Mati/inquiry.py` — ask the model a question, keep what is new, let the
# answer choose the next question.) This is the DIRECTED counterpart to free
# dreaming, which stays undirected and stays on idle too.
#
# ⚠️ On by default, but understand what it does: it stores model output
# unattended, and the memory loop amplifies errors. Everything it keeps is
# tagged `studied:asked about...` and rendered to Mati as "the model talking,
# not something you lived or verified". Turn it off if that trade is not worth
# it for your topics.
# ⚠️ OFF AT SHIP (the user), and the reason is what it produces, not
# what it costs. Self-talk is the WEAKEST content in the store — it is the
# model's own guesses, tagged `studied:asked about...` precisely so Mati does
# not mistake them for things it lived. A fresh install filling its memory with
# its own speculation before the person has said much is the wrong first week.
# ⚠️ The training half is a separate switch; this one is about what gets
# WRITTEN. Turn it on deliberately, per machine.
INQUIRE_ON_IDLE = False

# Longest inquiry may stay silent while a study queue is pending.
#
# ⚠️ IT USED TO BE FOREVER. `_maybe_inquire` returned early on ANY pending
# chunk — "absorbing beats acquiring", written when a queue was a handful of
# chunks. A 1,759-chunk book is not a backlog, it is a standing workload, and
# the gate silenced self-talk with months still to run. Same
# shape as the four other starvations found in this project: one bulk producer
# quietly turning off everything downstream of it.
#
# Absorbing still comes first — this only guarantees a floor, so a long read
# cannot switch off the subconscious for a season.
INQUIRE_MAX_SILENCE_H = 6.0

# How many memories a by-name lookup returns. Small: these go into every turn's
# context alongside the field's own recall, and the local model's window is
# LOCAL_NUM_CTX. Three is enough to cover a topic mentioned in passing without
# crowding out the conversation itself.
NAMED_RECALL_K = 3

# ── the working set ─────────────────────────────────────────────────────────
#
# How many memories stay FIELD-ADDRESSED. Everything older is archived and
# found lexically instead.
#
# ⚠️ THIS EXISTS BECAUSE ADDRESSING DOES NOT SCALE, and the numbers are
# measured, not guessed. One turn stales every stored address (100% -> 0%
# self-retrieval), and re-addressing costs ~3.48 s/note — 9.7 HOURS for 10,000
# notes, per turn. Correcting instead of recomputing was tested and refuted:
# the drift is 98.8–99.7% one shared shape, but the best possible shared
# correction reaches 0–17% against 100% for a real reindex, because the shared
# part is already free (centring removes it) and the residual is exactly what a
# shared correction cannot carry.
#
# So the only lever left is how MANY notes owe re-addressing. 40 costs about
# 140 s of background work to refresh completely, which fits in the gaps
# between turns; 400 would not.
#
# What archiving costs: an archived memory can still be found by its words
# (measured 20/20 where the field path got 0/20) but no longer surfaces
# associatively, by the field drifting near it. That is the trade, and it is
# the right way round — the expensive channel is the one that was failing.
# ⚠️ SET TO 62 — the exact size of the store when this was switched on, so
# that nothing already remembered was archived by enabling it. It is a
# deliberate placeholder, not a tuned value: the moment a 63rd memory is
# written the oldest drops to the archive, and from then on the working set
# behaves as the bound it is meant to be. Raise it again to hold more close;
# `bridge.rebalance_tiers` restores as readily as it archives.
WORKING_SET = 62

# ── dreaming ────────────────────────────────────────────────────────────────
#
# ⚠️ IT WAS NEVER WIRED. `bridge.dream` existed, `/dream` existed as an HTTP
# endpoint, and the worker never called either — so an install left running for
# hours never dreamed once. Found by leaving the terminal open and
# watching for it.
#
# It runs LAST in the sleep cycle, after study, deepening, question-resolving
# and inquiry, because those repay debts and this one creates them: dreaming
# deforms θ₀ and every stored address is measured against θ₀, so a dream is
# followed by a reindex. Doing it before the debts are paid would mean
# reindexing twice.
DREAM_ON_IDLE = True

# Cycles per dream. Small on purpose — each one moves θ₀ and buys a reindex,
# and the measured value of free dreaming is "consolidates near where the field
# already is", not "learns something". One is a rest, not a training run.
DREAM_CYCLES = 1

# How long after the last dream before another is worth doing. Dreaming on
# every idle pass would mean the field never settles between them.
DREAM_EVERY_SECONDS = 3600.0

# How often the feeds are read. ⚠️ Four hours, so a night covers it several
# times over and a morning request never waits — `world.fetch_all` is seconds
# of network and touches no field.
# ⚠️ IT NEEDS A CALLER TO BE WORTH ANYTHING. With none, the store silently
# served news 15 days old — a fetch nothing runs is not a feature.
WORLD_EVERY_SECONDS = 14400.0

# ⚠️ HOW OLD A NEWS ITEM MAY BE AND STILL BE CALLED NEWS. The morning report
# had no age filter at all: it took the newest rows whatever their date and
# printed them under "My report for Saturday 3 October" — so it reported
# two-week-old items as today's, with no date attached. Being empty is allowed;
# being wrong about the day is not.
WORLD_FRESH_HOURS = 36.0

# ── how much of the sleep cycle study may claim ─────────────────────────────
#
# ⚠️ IT USED TO CLAIM ALL OF IT. `_idle_pass` runs study first and returns, so
# while any chunk was pending, nothing below it ever ran — no deepening, no
# question-resolving, no question-forming, no inquiry, no dreaming.
#
# : a 1,759-chunk document was queued at ~22:45 and by the
# next morning self-talk had stopped dead at 22:43 and `dreams` was still 0.
# Absorption ran at 39.6 chunks/hour, so that document alone would have
# silenced every other background function for about two days.
#
# So study now gets 1 pass in STUDY_EVERY. The rest of the chain gets the
# others. 2 is an even split of PASSES, not of time — a study pass is ~90s and
# a deepen pass a few seconds, so study still gets most of the wall clock. That
# is deliberate: reading is the slow thing, and starving it would be the same
# mistake pointed the other way.
STUDY_EVERY = 2

# How long "read this now" lasts when the person asks for it directly. Bounded
# because full speed is what the budget exists to prevent: a queued book
# silenced dreaming, deepening and inquiry for a whole night. Asking for focus
# should cost an evening, not a week.
STUDY_PRIORITY_MINUTES = 60.0

# Chunks per batch DURING a priority window, against STUDY_BATCH's 2 outside
# one. Reindexing is part of studying — every imprint moves θ₀, so studying
# chunk N stales chunk N-1 — and it runs once per BATCH, not once per chunk.
# At ~89s a reindex against ~10s an imprint, a batch of 2 spends roughly 80% of
# its time repairing the index rather than reading: measured 152s per chunk
# absorbed across 19.5h of actual reading, or 24 chunks/hour.
#
# Amortising one reindex over 20 chunks instead of 2 is where the time is. It
# is safe to raise only here, because a priority window is exactly the case
# where nobody is waiting: `should_stop` still breaks the loop between chunks
# when a turn arrives, and the reindex still runs afterwards, so no chunk is
# left unaddressed either way. Outside the window STUDY_BATCH stays small so
# dreaming, deepening and inquiry keep getting their passes.
STUDY_PRIORITY_BATCH = 20

# ── leaning on lexical while the field settles ──────────────────────────────
#
# Every imprint moves θ₀ and every stored address is measured against θ₀, so
# after a wake — or a night of reading — the field path is filed in an older
# frame and degrades for minutes. Lexical search does not depend on θ₀ at all:
# it is exactly as good then as ever. , fetching a memory by
# its rarest words: lexical 20/20, the field path 0/20.
#
# So during the settling window Mati leans on the channel that needs no
# warm-up, rather than the one that is still warming. Above this fraction of
# the working set being stale, by-name recall is widened and the turn is told
# the field's own suggestions are less trustworthy than usual.
SETTLING_THRESHOLD = 0.25

# How much wider by-name recall goes while settling. Bounded by the context
# window — these lines go into every turn alongside everything else.
SETTLING_RECALL_BOOST = 2

# ── the live-field heartbeat ────────────────────────────────────────────────
#
# How often the server publishes the live phases for the terminal to draw.
# 4 Hz is fast enough to read as motion and slow enough to be free: the read
# costs 0.24 ms including coherence, and it is skipped entirely whenever the
# worker holds the field.
PULSE_SECONDS = 0.25

# How many bins the 200 nodes are folded into for display. 24 fits a narrow
# panel and still shows structure moving across the field.
PULSE_BINS = 24
