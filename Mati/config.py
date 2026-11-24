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
# ⚠️ NOTHING ELSE MAY COMPUTE ITS OWN CACHE PATH. A path computed in two places
# is a path that will disagree in one of them: a module deriving it from
# `__file__` resolves somewhere else the moment the module moves, and silently
# opens a store that does not exist.
CACHE_DIR = os.environ.get("MATI_HOME") or os.path.join(ROOT, '.cache')

SETTINGS_PATH = os.path.join(CACHE_DIR, 'mati_settings.json')

# Where the HTML app lives. Served by the server so the page can reach the
# API at all: a file:// page is a different origin and every fetch is
# blocked, which means no auto-wake and no conversation.
# ⚠️⚠️ THE BUNDLED COPY FIRST. A release carries `app/` at its root and must
# use it, or a download comes up as a terminal-only Mati — the field loaded,
# and `GET /` answering with status JSON instead of a window, an avatar and
# settings. Mati is GUI-centred; a person who downloads it and gets a text
# client has not been given Mati.
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
    # A development checkout may instead keep the window in a sibling folder,
    # under either name; accepting both keeps a renamed folder from quietly
    # disconnecting the window from the code it serves.
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

# Generation on the slowest supported machine (an 8 GB Intel Mac, 3B on the
# CPU) runs at roughly 7–10 tokens a second, so every token costs ~100–140 ms
# and REPLY LENGTH IS THE LATENCY.
#
# ⚠️ NOT WHAT A TURN USES. The length a reply may reach is the person's choice,
# `reply_tokens()` below. This fixed ceiling is not read by anything.
#
# ⚠️ AN UNCAPPED REPLY (`num_predict = -2`, "fill the context window") IS AN
# INSTRUCTION TO KEEP GOING, against which "be short" in the system prompt is
# a suggestion. Changing the prompt does not move verbosity; the ceiling does.
# Same question, same prompt, same temperature:
#
#     num_predict = -2   375 tok   54.1s   272 words
#     num_predict = 220  220 tok   31.6s   139 words
#     num_predict = 160  160 tok   22.5s   118 words
#
# ⚠️ AND A CEILING IS NOT A KNIFE. `voice.ReplyFilter` releases only COMPLETED
# sentences and holds the fragment back until `flush()`. Ollama reports
# `done_reason: "length"` when the cap bit, so the fragment is DROPPED in that
# case. The ceiling stops at the last whole sentence; it is not a knife.
LOCAL_MAX_TOKENS = 260

# ── Reply length, as a person's choice rather than a constant ────────────────
#
# The shipping app will run on any model, so the efficiency controls have to
# work everywhere.
#
# ⚠️ THE NUMBERS ARE THE REASON THIS IS A SETTING AT ALL (the table above):
# reply length IS latency on a small model. This is the one control a person
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
# is not a defect — ending mid-word is.
REPLY_LENGTHS = {"brief": 160, "normal": 220, "thorough": 320}


def reply_tokens(default=None):
    """The ceiling this person chose, or the configured default.

    ⚠️ READ AT CALL TIME. The setting is written by the app while the server
    is running; a value captured at import would not change until restart,
    and a preference that needs a restart is not a preference.

    ⚠️ BRIEF IS THE DEFAULT, not `normal`. Length IS latency on a 3B, so the
    out-of-the-box experience should be the fast one. Someone who wants longer
    answers can say so; someone who never opens settings should not be waiting
    extra seconds every turn for a default they did not choose.
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
# exactly that: "every one of those carries the same fixed header".
#
#   ⚠️ NOTHING UNIQUE TO THE PERSON — no account URL, no name. The README says
#      nothing unique to you leaves the machine, and a header is part of that.
#   ⚠️⚠️ AND NO BROWSER DISGUISE. Presenting as a browser IS the mechanism for
#      not being seen as automation, so it cannot sit beside a claim that
#      nothing is being bypassed. It also buys nothing: the honest header below
#      gets the same answer from the same endpoint as a full Safari string, and
#      so does sending no User-Agent at all.
#
# ⚠️ KEEP IT DESCRIPTIVE. Radio Browser and the Overpass usage policy both ask
# that a client say what it is so operators can see who is calling; "Mati/0.1"
# alone does not. What must stay out is anything unique to the person.
USER_AGENT = "Mati/0.1 (personal assistant)"


# ── what the assistant is called ────────────────────────────────────────────
# ⚠️ THE ASSISTANT IS RENAMEABLE; THE PRODUCT IS NOT. A personal assistant
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

# Ollama unloads the model after 5 minutes idle by default. Reloading costs
# several seconds plus an uncached prompt pass — over half a minute added to
# the first turn after a pause, which is exactly when someone is most likely to
# be waiting. Holding it resident trades ~2GB of RAM for that.
LOCAL_KEEP_ALIVE = _choice("MATI_KEEP_ALIVE", "keep_alive", "30m")

# --- anthropic (optional) ---
# Sampling params are rejected on Opus 5 — steer via system prompt.
MODEL       = "claude-opus-5"
MAX_TOKENS  = 16000          # non-streaming; raise + stream for long outputs
THINKING    = {"type": "adaptive"}
EFFORT      = "high"         # low | medium | high | xhigh | max


# ── Which model speaks — the person's choice, read at call time ──────────────
#
# Mati connects to the model the person chooses — most popular models.
#
#   model_choice        "local:<ollama name>" or "conn:<id>"
#   model_connections   [{id, kind, label, url, model}], kind "openai" (any
#                       OpenAI-compatible service or local server) or
#                       "anthropic"
#
# ⚠️ THE DEFAULT IS UNCHANGED: with no `model_choice`, the local Ollama model
# (`local_model`, else LOCAL_MODEL) speaks, exactly as before.
# ⚠️ KEYS ARE NOT HERE. A connection's key lives sealed in `credentials`,
# under `model:<id>`; settings hold only what is safe to read unlocked.
CONNECTION_KINDS = ("openai", "anthropic")
ANTHROPIC_URL = "https://api.anthropic.com"


def model_connections():
    """The connections the person has added, in the order they added them."""
    got = _settings().get("model_connections") or []
    return [c for c in got if isinstance(c, dict) and c.get("id")
            and c.get("kind") in CONNECTION_KINDS and c.get("model")]


def connection(conn_id):
    return next((c for c in model_connections() if c["id"] == conn_id), None)


def model_choice():
    """What speaks now: {"choice", "kind", "model", "url", "label", "conn"}.

    ⚠️ A CHOICE THAT NO LONGER RESOLVES FALLS BACK TO THE LOCAL MODEL, and
    says so in `missing`. A removed connection must not leave Mati mute.
    """
    s = _settings()
    # ⚠️ `MATI_MODEL` STILL WINS, for one run, over every saved choice —
    # setup tells people they can do exactly that.
    env = os.environ.get("MATI_MODEL")
    if env:
        return {"choice": f"local:{env}", "kind": "local", "model": env,
                "url": OLLAMA_URL, "label": env, "conn": None}
    local = s.get("local_model") or LOCAL_MODEL
    ch = s.get("model_choice") or ""
    if ch.startswith("conn:"):
        c = connection(ch[len("conn:"):])
        if c:
            url = (ANTHROPIC_URL if c["kind"] == "anthropic"
                   else (c.get("url") or "")).rstrip("/")
            return {"choice": ch, "kind": c["kind"], "model": c["model"],
                    "url": url, "label": c.get("label") or c["model"],
                    "conn": c["id"]}
        return {"choice": f"local:{local}", "kind": "local", "model": local,
                "url": OLLAMA_URL, "label": local, "conn": None,
                "missing": ch}
    if ch.startswith("local:") and len(ch) > len("local:"):
        local = ch[len("local:"):]
    return {"choice": f"local:{local}", "kind": "local", "model": local,
            "url": OLLAMA_URL, "label": local, "conn": None}


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
# PER TOKEN, so cost is linear in tokens (~1s/token on the slowest machine) —
# full 250-token answers would make seeding take many hours.
#
# CONSISTENCY REQUIREMENT: this cap must be applied identically when measuring a
# stored note address AND when measuring a query, or the two land in different
# spaces and every distance is meaningless. Both paths read this constant.
#
# WHY 64. Smaller and a chunk is smaller than a prose sentence, so stored notes
# become fragments — "Mexico and Central America) until its collapse around
# 400-500 CE.", unmatched bracket and all. Over sentences of text Mati actually
# stores (median 16 tokens, mean 20, max 41):
#
#     cap  32 -> 82.1% of sentences fit whole
#     cap  48 -> 100%
#     cap  64 -> 100%, and holds 3-4 median sentences
#
# 64 rather than 48 because the unit that matters is a whole short answer, not
# a whole sentence — `inquiry.ASK_SYSTEM` asks the model for "three or four
# short factual sentences", and that should land as ONE note.
#
# THE COST IS REAL AND ACCEPTED: imprinting is ~1s/token, so an episode write
# is about a minute and a studied chunk the same. Both run on the worker, so
# nobody waits for them, and re-addressing a note takes a few seconds.
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
# distinction matters, see bridge.think_while). A typical reply spends about a
# second stepping against tens of seconds of generation, so this never binds
# today. It is a guard against a pathologically long reply, not a latency fix.
# 0 disables it.
THINK_BUDGET_S = 4.0

# Max episodes re-addressed per reindex call. Bounds cost (reindex is O(n)
# forwards, seconds per note). Selection is by STALENESS not recency, so
# everything rotates through — capping by recency would strand old memories in
# an obsolete address space permanently. 0 = no cap.
REINDEX_LIMIT = 32

# How far an address may drift from the current address space before it counts
# as stale. Checked on startup and on wake, which is when θ₀ is most likely to
# have moved under the store — the field is handed to the pro terminal so the
# terminal can train it.
#
# ⚠️ THIS IS NOT A "RECALL IS FINE BELOW HERE" THRESHOLD, and must not be
# described as one. Nothing has measured what drift actually costs retrieval.
# It is set just above float noise so that a genuinely untouched store reports
# clean — every address at exactly 0.0 — rather than re-addressing on every
# restart.
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

# The episode write is post-reply work: it runs on the server's worker and the
# person never waits for it. Re-addressing does not run per turn at all.
# ⚠️ IDLE MEANS GONE QUIET, NOT PAUSED. Most gaps in a real conversation are a
# few minutes, and some are half an hour; idle work starting after every one of
# them would be under the person's feet. How long someone pauses depends on the
# person.
#
# So the live value is LEARNED from the person's own record —
# `rhythm.idle_after()`, their p95 gap, floored at 45 minutes. This constant
# survives only as the fallback for a store with nothing in it yet.
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
# ⚠️ OFF, AND THE REASON IS DESIGN, NOT A BUG. A question written days earlier,
# from a passage, by a template, and then parked until its topic happens to
# come up has no relation to the moment it arrives in — it will always sound
# forced. No amount of filtering repairs that: the problem is not which
# pre-fabricated question gets delivered, it is that it was pre-fabricated.
# Questions should emerge from the field during a turn (`clarify.py`).
#
# ⚠️ THE TRAINING HALF STAYS ON, and it is a different function:
#
#     inquiry.step()            Mati asks its OWN model and imprints what
#                               comes back. Self-talk, DESIRED, untouched.
#     inquiry.question_for_user()  manufactures a question FOR THE USER and
#                               parks it. This is what is switched off.
#
# `inquiry.question_for_user` and `bridge.episode_to_ask_about` are kept rather
# than deleted so the in-turn design can be measured against them.
ASK_USER_ON_SLEEP = False

# Asking instead of answering, when the field surfaces something Mati has
# already been corrected about. See `Mati/clarify.py`. This is what replaces
# the manufactured questions above.
CLARIFY_ON_CORRECTED = True

# ⚠️ AFTER MANY MEMORIES OF ASSUMING INCORRECTLY — the dial that makes this
# self-limiting. Below this many corrections on record, Mati has not been
# wrong often enough to have learned to check, and asking would be a mannerism
# rather than caution.
#
# ⚠️ DELIBERATELY LOW, because the OTHER gate is the strong one: a
# clarification also requires the surfaced memory to carry a correction, so
# this cannot fire on material Mati has never got wrong. A fresh install has
# no corrections and will not clarify at all until it has actually been wrong.
CLARIFY_AFTER_CORRECTIONS = 5

# Asking the 3B what an unrecognised turn MEANT. See `Mati/intent.py`.
#
# ⚠️ ON, BECAUSE THE ALTERNATIVE IS A LISTENER THAT NEEDS NEW TERMS ADDED FOR
# EVER. The cost is not permanent — whatever the model answers is written to
# `aliases`, so an unfamiliar phrasing costs one classification and is a string
# lookup thereafter.
INTENT_FROM_MODEL = True

# Longer than this and a message is prose, not a command. ⚠️ A sixty-word
# budget paragraph must not be answered by stopping a webradio, and asking a
# model about prose invites the same error with more steps — so the cap is
# applied before the call.
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
# ⚠️ FIELD REINDEXING ON IDLE IS RETIRED, by measurement. Field addressing
# scores 4/111 = 0.036 against chance 0.009 while lexical scores 93/111 = 0.838
# on identical cues and pool — see ARCHITECTURE.md.
#
# It also cannot terminate against a moving target. Staleness is
# |gs_now - feat_gs| against a single scalar that moves whenever the field
# learns, so a few addresses repaired per pass lose to many invalidated per
# turn, and a worker re-checking after each batch finds everything stale again
# — a loop that burns the CPU and stops `/health` answering.
#
# The staleness MEASUREMENT stays — it is a cheap SELECT and an honest
# diagnostic. Repair happens once a night, after promotion.
REINDEX_ON_IDLE = False

# ⚠️ ADDRESSING READS A FROZEN θ₀ AND WRITES THE LIVE ONE. There is no
# setting for this and there must not be one — see
# `bridge.pin_address_space`. A switch between two memory architectures
# invalidates every stored address whichever way it is thrown.

# How many of the newest self-talk notes stay field-addressed.
# ⚠️ BOUNDED BECAUSE IT GROWS. Reflection writes ~3 a night; pinned
# outright it would be the entire recall pool within a year and every
# slot would go to Mati talking to itself rather than to anything that
# happened with the person. Ten nights is enough to resonate against.
PINNED_SELFTALK = 30

# Chunks of an uploaded document studied per idle pass. Studying is an imprint
# (~1s/token, so most of a minute a chunk at TRIPLET_CAP) and it competes with
# the next turn for the field, so the queue is drained a little at a time
# rather than in one sitting. This is what bounds the cost of reading a file —
# see `Mati/study.py` on why a "skip what I already know" test cannot.
#
# ⚠️ A BOOK AND THE CHARTER ARE DIFFERENT PROBLEMS. The charter needed a small
# dose: a few thousand words of Mati's own text arriving into a cold, empty
# field pushed settling coherence below the 0.700 gate on every turn. A book is
# the opposite case — varied material arriving into a store that already holds
# conversation — and takes a bigger one.
#
# ⚠️ IT STAYS INTERRUPTIBLE. `study()` yields between chunks via
# `should_stop`, so an arriving turn interrupts a batch rather than waiting it
# out.
#
# ⚠️ WHAT IT COSTS, STATED: every chunk is an imprint, and imprinting is what
# destroys addressability read from a live field. Frozen addressing is what
# makes this affordable — addresses are read from a space that holds still
# until `promote()`, so a burst of study does not strand the notes written
# before it.
STUDY_BATCH = 8

# Skip a chunk already recallable above this similarity. None = off, and off is
# correct: known material scores 0.424–0.688 and unseen material 0.305–0.754 —
# overlapping, and inverted on the mean, so no threshold separates them. Left
# here so it can be re-measured if retrieval ever becomes semantic rather than
# lexical.
STUDY_SIMILARITY_GATE = None


# Dream cycles at the end of first-run setup. ZERO — dreaming here undoes what
# setup just did.
#
# Dreaming right after imprinting the user's interests looks as if it should
# deepen those basins. Paired against an untouched control topic, on a fresh
# field per run:
#
#     cycles   Δ seed    Δ control   seed − control
#          3  -0.0011     -0.0028         +0.0017
#         10  -0.0090     -0.0038         -0.0052
#         30  -0.0499     -0.0245         -0.0254
#
# The seeded topic loses coherence FASTER than a topic never mentioned, and the
# gap grows with cycles. Free-dreaming relaxes θ₀ toward the field's own
# intrinsic attractors, so it washes out a fresh imprint — and the seeded topic
# erodes most because it is the thing that was just imprinted.
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
# ⚠️ OFF BY DEFAULT, and the reason is what it produces, not what it costs. It
# stores model output unattended, and the memory loop amplifies errors.
# Self-talk is the WEAKEST content in the store — the model's own guesses,
# tagged `studied:asked about...` and rendered to Mati as "the model talking,
# not something you lived or verified". A fresh install filling its memory with
# its own speculation before the person has said much is the wrong first week.
# ⚠️ The training half is a separate switch; this one is about what gets
# WRITTEN. Turn it on deliberately, per machine.
INQUIRE_ON_IDLE = False

# Longest inquiry may stay silent while a study queue is pending.
#
# ⚠️ A FLOOR, NOT FOREVER. "Absorbing beats acquiring" — but a large book is
# not a backlog, it is a standing workload, and an inquiry that waited for an
# empty queue would be silent for months: one bulk producer quietly turning
# off everything downstream of it.
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
# ⚠️ THIS EXISTS BECAUSE ADDRESSING DOES NOT SCALE. Every promotion stales
# every stored address, and re-addressing costs seconds a note — hours for
# ten thousand notes. Correcting instead of recomputing does not work: the
# drift is 98.8–99.7% one shared shape, but the best possible shared
# correction reaches 0–17% against 100% for a real re-address, because the
# shared part is already free (centring removes it) and the residual is
# exactly what a shared correction cannot carry.
#
# So the only lever left is how MANY notes owe re-addressing. A few dozen
# can be refreshed completely in minutes of background work; hundreds cannot.
#
# What archiving costs: an archived memory can still be found by its words
# but no longer surfaces associatively, by the field drifting near it. That
# is the trade, and it is the right way round — the expensive channel is the
# one that fails.
# ⚠️ A PLACEHOLDER, NOT A TUNED VALUE. Raise it to hold more close;
# `bridge.rebalance_tiers` restores as readily as it archives.
WORKING_SET = 62

# ── dreaming ────────────────────────────────────────────────────────────────
#
# Free dreaming on idle. It comes after the debts — study, deepening and
# question-resolving — because those repay work and this one can create it:
# dreaming deforms θ₀, and once the address space moves, a dream that ran
# before the repair means repairing twice.
DREAM_ON_IDLE = True

# Cycles per dream. Small on purpose — each one moves θ₀, and the value of
# free dreaming is "consolidates near where the field already is", not "learns
# something". One is a rest, not a training run.
DREAM_CYCLES = 1

# How long after the last dream before another is worth doing. Dreaming on
# every idle pass would mean the field never settles between them.
DREAM_EVERY_SECONDS = 3600.0

# How often the feeds are read. ⚠️ Four hours, so a night covers it several
# times over and a morning request never waits — `world.fetch_all` is seconds
# of network and touches no field.
# ⚠️ IT NEEDS A CALLER TO BE WORTH ANYTHING. A fetch nothing runs leaves the
# store serving whatever it last read, however old.
WORLD_EVERY_SECONDS = 14400.0

# ⚠️ HOW OLD A NEWS ITEM MAY BE AND STILL BE CALLED NEWS. Without an age
# filter the morning report would take the newest rows whatever their date and
# print two-week-old items under today's heading, with no date attached. Being
# empty is allowed; being wrong about the day is not.
WORLD_FRESH_HOURS = 36.0

# ── how much of the sleep cycle study may claim ─────────────────────────────
#
# ⚠️ NOT ALL OF IT. `_idle_pass` runs study first and returns, so while any
# chunk is pending nothing below it would run — no deepening, no
# question-resolving, no question-forming, no inquiry, no dreaming. A large
# document would silence every other background function for days.
#
# So study gets 1 pass in STUDY_EVERY and the rest of the chain gets the
# others. 2 is an even split of PASSES, not of time — a study pass is long and
# a deepen pass a few seconds, so study still gets most of the wall clock. That
# is deliberate: reading is the slow thing, and starving it would be the same
# mistake pointed the other way.
STUDY_EVERY = 2

# How long "read this now" lasts when the person asks for it directly. Bounded
# because full speed is what the budget exists to prevent: a queued book
# silenced dreaming, deepening and inquiry for a whole night. Asking for focus
# should cost an evening, not a week.
STUDY_PRIORITY_MINUTES = 60.0

# Chunks per batch DURING a priority window, against STUDY_BATCH outside one.
# A bigger bite is safe only here, because a priority window is exactly the
# case where nobody is waiting: `should_stop` still breaks the loop between
# chunks when a turn arrives. Outside the window STUDY_BATCH stays small so
# dreaming, deepening and inquiry keep getting their passes.
STUDY_PRIORITY_BATCH = 20

# ── leaning on lexical while the field settles ──────────────────────────────
#
# After a promotion, until the working set is re-addressed, the field path is
# filed in an older frame. Lexical search does not depend on θ₀ at all: it is
# exactly as good then as ever.
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
