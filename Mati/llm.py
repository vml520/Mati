"""
Language faculty for Mati.

The LLM supplies the ONE thing the field cannot: composition (grammar, binding).
It does not own identity, memory, or ethics — those stay in the field. Because
the job is composition rather than deep reasoning, a small local model is a
correct fit, not a compromise.

Two providers, same interface (`.reply(messages) -> (text, raw)`):

    LocalLLM     — Ollama on localhost. Free, offline, no key, stdlib only.
    AnthropicLLM — Claude API. Costs money, needs a key. Optional.

Use `make_llm(system_prompt)` and let config.PROVIDER decide.
"""

import json
import urllib.error
import urllib.request

from . import config as C


class Refusal(Exception):
    """Provider declined to answer (Anthropic safety classifiers)."""

    def __init__(self, category=None, explanation=None):
        self.category = category
        self.explanation = explanation
        super().__init__(f"refused (category={category}): {explanation}")


class LLMUnavailable(Exception):
    """The backend isn't reachable (server down, model not pulled, no key)."""


# ── Local: Ollama ─────────────────────────────────────────────────────────────

class LocalLLM:
    """
    Ollama client over plain HTTP. No SDK, no account, no network egress.

    Ollama keeps the system prompt as messages[0], so there's no separate
    cached-prefix concept — but keeping the system prompt stable still helps:
    Ollama reuses the KV cache across calls when the prefix is unchanged.
    """

    def __init__(self, system_prompt, model=None, url=None):
        self.model  = model or C.LOCAL_MODEL
        self.url    = (url or C.OLLAMA_URL).rstrip("/")
        self.system = system_prompt
        # Set by `stream_reply`: "stop" or "length".
        self.last_done_reason = None

    def _post(self, path, payload, timeout=600):
        req = urllib.request.Request(
            f"{self.url}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.URLError as e:
            raise LLMUnavailable(
                f"Can't reach Ollama at {self.url} ({e.reason}).\n"
                f"  Start it:   ollama serve\n"
                f"  Pull model: ollama pull {self.model}"
            ) from e

    def reply(self, messages, max_tokens=None, **_ignored):
        """`messages` is the Anthropic-shaped list; system is prepended here."""
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": self.system}] + messages,
            "stream": False,
            "keep_alive": C.LOCAL_KEEP_ALIVE,
            "options": {
                "num_ctx": C.LOCAL_NUM_CTX,
                "temperature": C.LOCAL_TEMP,
                # ⚠️ `reply_tokens()`, NOT the constant. It reads the person's
                # choice at call time — see config.REPLY_LENGTHS. An explicit
                # `max_tokens` still wins, which is how `intent.classify` asks
                # for 8 and gets 8.
                "num_predict": max_tokens or C.reply_tokens(),
            },
        }
        data = self._post("/api/chat", payload)
        return data.get("message", {}).get("content", ""), data

    def stream_reply(self, messages, max_tokens=None, **_ignored):
        """Yields text chunks. Worth using locally — 3B on CPU is slow."""
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": self.system}] + messages,
            "stream": True,
            "keep_alive": C.LOCAL_KEEP_ALIVE,
            "options": {
                "num_ctx": C.LOCAL_NUM_CTX,
                "temperature": C.LOCAL_TEMP,
                # ⚠️ `reply_tokens()`, NOT the constant. It reads the person's
                # choice at call time — see config.REPLY_LENGTHS. An explicit
                # `max_tokens` still wins, which is how `intent.classify` asks
                # for 8 and gets 8.
                "num_predict": max_tokens or C.reply_tokens(),
            },
        }
        req = urllib.request.Request(
            f"{self.url}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                for line in r:
                    if not line.strip():
                        continue
                    chunk = json.loads(line.decode())
                    piece = chunk.get("message", {}).get("content", "")
                    if piece:
                        yield piece
                    if chunk.get("done"):
                        # ⚠️ WHY IT STOPPED, NOT JUST THAT IT DID. "length"
                        # means the cap cut it mid-sentence; "stop" means it
                        # finished. The caller drops the trailing fragment in
                        # the first case, which is what makes a token ceiling
                        # safe — it stops at the last COMPLETE sentence rather
                        # than mid-word.
                        self.last_done_reason = chunk.get("done_reason")
                        break
        except urllib.error.URLError as e:
            raise LLMUnavailable(f"Can't reach Ollama at {self.url}: {e.reason}") from e

    def available(self):
        """(ok, detail) — is the server up and is the model pulled?"""
        try:
            with urllib.request.urlopen(f"{self.url}/api/tags", timeout=5) as r:
                tags = json.loads(r.read().decode())
        except urllib.error.URLError as e:
            return False, f"Ollama not reachable at {self.url} ({e.reason})"
        names = [m.get("name", "") for m in tags.get("models", [])]
        if not any(n == self.model or n.startswith(self.model.split(":")[0])
                   for n in names):
            return False, (f"model '{self.model}' not pulled. "
                           f"Have: {names or 'none'}. Run: ollama pull {self.model}")
        return True, f"{self.model} ready"


# ── Anthropic (optional) ──────────────────────────────────────────────────────

class AnthropicLLM:
    def __init__(self, system_prompt, model=None):
        self.model = model or C.MODEL
        # Stable prefix carries the cache breakpoint; volatile field state goes
        # in `messages`, after it.
        self.system = [{
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }]
        try:
            import anthropic
        except ImportError as e:
            raise LLMUnavailable(
                "Anthropic provider needs the SDK. On this machine `pip` is not "
                "on PATH:\n    python3 -m pip install --user anthropic\n"
                "Or use the free local provider: MATI_PROVIDER=local"
            ) from e
        self.client = anthropic.Anthropic()

    def reply(self, messages, max_tokens=C.MAX_TOKENS, effort=C.EFFORT):
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            thinking=C.THINKING,
            output_config={"effort": effort},
            system=self.system,
            messages=messages,
        )
        # Check stop_reason BEFORE reading content — a refusal has empty or
        # partial content and indexing content[0] would raise.
        if resp.stop_reason == "refusal":
            det = resp.stop_details
            raise Refusal(getattr(det, "category", None),
                          getattr(det, "explanation", None))
        return next((b.text for b in resp.content if b.type == "text"), ""), resp


# ── Factory ───────────────────────────────────────────────────────────────────

def make_llm(system_prompt, provider=None):
    provider = (provider or C.PROVIDER).lower()
    if provider == "local":
        return LocalLLM(system_prompt)
    if provider == "anthropic":
        return AnthropicLLM(system_prompt)
    raise ValueError(f"unknown provider {provider!r} (use 'local' or 'anthropic')")


# Alias: `LLM` names the Anthropic client.
LLM = AnthropicLLM
