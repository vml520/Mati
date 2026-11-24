"""
Each model's settings and safeguards, in one place.

    python3 -m Mati.model_settings                 what applies to the model in use
    python3 -m Mati.model_settings qwen2.5:3b      what would apply to that one

the user: *"We need to create a model setting module that holds the setting and
safeguards for each model. Modify the code so that it runs regardless of which
model settings and safeguards apply. We will let Beta testers give us the
feedback we need to create separate settings; let's keep it Ollama centered
for now."*

⚠️ ONE DEFAULT FOR EVERY MODEL, FOR NOW. `DEFAULT` is what Mati was tuned on:
llama3.2:3b through Ollama, on an 8 GB Mac. Every model gets it until Beta
feedback earns a model its own entry in `PROFILES` — which starts EMPTY on
purpose, because a setting nobody measured is a guess presented as tuning.

⚠️ TWO THINGS ARE NOT TUNING, and so are here already:
  - `KIND_FACTS`: what a kind of connection REQUIRES. Claude rejects sampling
    parameters on some models; warming a remote model is a billed request.
  - `LICENCES`: a model whose licence must be agreed to before it is used.

⚠️ EVERY CONSUMER READS FROM HERE, AND MUST WORK WHATEVER IT SAYS. A
safeguard switched off is skipped, not crashed on; a different context
window or temperature is simply passed on. The turn's filter is
built to run with every safeguard on and with every one off.
"""

from __future__ import annotations

import copy
import re
import sys
import threading

from . import config as C


# ── The default: Ollama-centred, what Mati was tuned on ──────────────────────

DEFAULT = {
    # Generation. ⚠️ The values that were config constants, unchanged — see
    # config.py for the measurements behind each.
    "num_ctx": C.LOCAL_NUM_CTX,        # the context window asked for
    "temperature": C.LOCAL_TEMP,       # 0.3: the retrieval-grounded range
    "keep_alive": C.LOCAL_KEEP_ALIVE,  # how long Ollama keeps it resident
    "send_temperature": True,          # some APIs reject it — see KIND_FACTS
    "prewarm": True,                   # pay the first prefill at start-up
    # Safeguards on what the model says. Each one is a check in
    # `voice.ReplyFilter` or the turn, written against llama3.2:3b's habits.
    "safeguards": {
        "stage_directions": True,  # "(smiles)", "*thinks*" — cut
        "asides": True,            # bracketed notes to itself — cut
        "leaks": True,             # machinery named: "field state", "<world>"
        "stale_time": True,        # a clock time it was never told
        "memory_denial": True,     # "I have no memory of past sessions"
        "recall_marker": True,     # [recall: …] honoured as a lookup
        "drop_cut_fragment": True, # the half-sentence a length cap leaves
    },
    "licence": None,
}

# What a kind of connection REQUIRES. Not preferences: a request that breaks
# these fails, or costs money for nothing.
KIND_FACTS = {
    "local": {},
    # ⚠️ Sampling parameters are rejected on some Claude models; and a
    # warm-up is a billed generation.
    "anthropic": {"send_temperature": False, "prewarm": False},
    # A remote OpenAI-compatible service: warming it is billed too. A server
    # on this Mac or network (LM Studio) keeps `prewarm`.
    "openai_remote": {"prewarm": False},
    "openai_local": {},
}

# Per-model overrides, from Beta feedback. ⚠️ EMPTY UNTIL MEASURED.
#   (pattern on the model name, {overrides}) — the first match wins.
PROFILES = []


# ── Licences that must be agreed to ──────────────────────────────────────────
#
# ⚠️ LICENCES DIFFER BY SIZE WITHIN ONE FAMILY. Qwen2.5 at 0.5B, 1.5B, 7B,
# 14B and 32B is Apache-2.0; at 3B it is the Qwen Research Licence — non-
# commercial only — and at 72B the Qwen Licence, with its own terms. So the
# match is on family AND size, never on "qwen".
# ⚠️ AGREED EVERY TIME THE MODEL IS USED (the user): each time it is chosen,
# and again in every run of Mati before it speaks or works with it. The
# agreement is held in memory only, never saved.

_QWEN25 = re.compile(r"qwen[-_ ]?2\.5", re.I)

LICENCES = [
    {"id": "qwen-research",
     "size": re.compile(r"(?<![\d.])3b\b", re.I),
     "name": "Qwen Research Licence",
     "text": ("This model is licensed by Alibaba Cloud under the Qwen Research "
              "Licence: for non-commercial purposes only. Using it for "
              "anything commercial needs a separate licence from them.")},
    {"id": "qwen",
     "size": re.compile(r"(?<![\d.])72b\b", re.I),
     "name": "Qwen Licence",
     "text": ("This model is licensed by Alibaba Cloud under the Qwen "
              "Licence, which is not an open-source licence and has its own "
              "terms of use.")},
]


def licence_for(model):
    """The licence that must be agreed to before `model` is used, or None."""
    name = str(model or "")
    if not _QWEN25.search(name):
        return None
    for lic in LICENCES:
        if lic["size"].search(name):
            return {"id": lic["id"], "name": lic["name"], "model": name,
                    "text": lic["text"],
                    "read": f"ollama show {name} --license"}
    return None


_AGREED = set()
_LOCK = threading.Lock()


def agreed(model):
    """True if `model` needs nothing agreed, or was agreed to in this run."""
    if licence_for(model) is None:
        return True
    with _LOCK:
        return str(model) in _AGREED


def agree(model):
    """Record that the person agreed to `model`'s licence — for this run."""
    if licence_for(model) is not None:
        with _LOCK:
            _AGREED.add(str(model))


def forget_agreements():
    """Every agreement ends (a restart does this by itself)."""
    with _LOCK:
        _AGREED.clear()


def require_agreed(model):
    """⚠️ CALLED BY EVERY CLIENT BEFORE IT GENERATES — the conversation, the
    night's reflection, the work done while the person is away. Raises
    `llm.LLMUnavailable` (via `LicenceNotAgreed`) so every caller already
    handles it: the window shows the sentence, idle work simply skips."""
    if agreed(model):
        return
    lic = licence_for(model)
    from . import llm
    raise llm.LicenceNotAgreed(
        f"Before I use {model}, its licence has to be agreed to: "
        f"{lic['name']}. Choose it again with the arrow beside its name, or "
        f"read it with `{lic['read']}`.")


# ── What applies to a model ──────────────────────────────────────────────────

def _kind_key(kind, url=None):
    if kind == "openai":
        from . import llm
        return "openai_local" if llm.is_local_url(url) else "openai_remote"
    return kind or "local"


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


def for_model(model, kind="local", url=None):
    """The settings and safeguards for one model: the default, then what its
    kind of connection requires, then its own profile, then its licence."""
    s = _merge(DEFAULT, KIND_FACTS.get(_kind_key(kind, url), {}))
    for pattern, over in PROFILES:
        if re.search(pattern, str(model or ""), re.I):
            s = _merge(s, over)
            break
    s["licence"] = licence_for(model)
    s["model"], s["kind"] = model, kind
    return s


def current():
    """What applies to the model the person has chosen."""
    ch = C.model_choice()
    return for_model(ch["model"], ch["kind"], ch.get("url"))


def safeguards(settings=None):
    """The safeguard switches, always a full dict — a missing key is ON, so a
    profile written before a safeguard existed does not silently drop it."""
    on = dict(DEFAULT["safeguards"])
    on.update((settings or {}).get("safeguards") or {})
    return on


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    s = for_model(argv[0]) if argv else current()
    print(f"  {s['model']}  ({s['kind']})")
    for k in ("num_ctx", "temperature", "keep_alive", "send_temperature",
              "prewarm"):
        print(f"    {k:<17} {s[k]}")
    print("    safeguards       " + ", ".join(
        f"{k}={'on' if v else 'off'}" for k, v in safeguards(s).items()))
    lic = s["licence"]
    print(f"    licence          {lic['name'] + ' — must be agreed to' if lic else 'none to agree to'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
