"""
Language faculty for Mati.

The LLM supplies the ONE thing the field cannot: composition (grammar, binding).
It does not own identity, memory, or ethics — those stay in the field. Because
the job is composition rather than deep reasoning, a small local model is a
correct fit, not a compromise.

Providers, one interface (`.reply(messages) -> (text, raw)`, `.stream_reply`):

    LocalLLM         — Ollama on this machine. The default: free, offline, no
                       key, stdlib only.
    OpenAICompatLLM  — any OpenAI-compatible service or local server, if the
                       person adds one.
    ClaudeLLM        — Anthropic's Messages API, streamed, if the person adds
                       a key.
    AnthropicLLM     — the older SDK client, kept for `--provider anthropic`.

Use `make_llm(system_prompt)`; `config.model_choice()` decides, at call time.
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


class LicenceNotAgreed(LLMUnavailable):
    """⚠️ The model's licence has not been agreed to in this run of Mati.

    A kind of "unavailable" on purpose: every caller already handles that —
    the window shows the sentence, work done while the person is away skips.
    See `model_settings.require_agreed`."""


def _settings_for(model, kind, url=None):
    from . import model_settings
    return model_settings.for_model(model, kind, url)


def _require(model):
    from . import model_settings
    model_settings.require_agreed(model)


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
        self.label = self.model
        # ⚠️ OLLAMA ON THIS MACHINE (or a URL the person set). Nothing leaves.
        self.remote = False
        # ⚠️ READ FROM `model_settings`, never from constants here: what this
        # model gets — window, temperature, residency, safeguards, licence.
        self.settings = _settings_for(self.model, "local")
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
        _require(self.model)
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": self.system}] + messages,
            "stream": False,
            "keep_alive": self.settings["keep_alive"],
            "options": {
                "num_ctx": self.settings["num_ctx"],
                "temperature": self.settings["temperature"],
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
        _require(self.model)
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": self.system}] + messages,
            "stream": True,
            "keep_alive": self.settings["keep_alive"],
            "options": {
                "num_ctx": self.settings["num_ctx"],
                "temperature": self.settings["temperature"],
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


# ── Remote, or any OpenAI-compatible server ──────────────────────────────────
#
# Mati connects to the model the person chooses — most popular models.
#
# ⚠️ STDLIB ONLY, LIKE `LocalLLM`. No SDK to install, so a connection works on
# a fresh Mac the moment a key is added.
# ⚠️ THE SAME INTERFACE: `stream_reply` yields text, `reply` returns
# (text, raw), `last_done_reason` is "stop" or "length", `available()` is
# (ok, detail). `ollin.turn` drops a cut fragment on "length" for every
# provider alike, so a token ceiling is never a knife.

_UA = "Mati"


def is_local_url(url):
    """Is this address on this machine or its network? (A person-named local
    server, such as LM Studio on 127.0.0.1:1234.)"""
    import urllib.parse
    from . import outbound
    host = urllib.parse.urlparse(url or "").hostname or ""
    return outbound.host_refused(host) is not None


def _check_url(url, keyed):
    """⚠️ A KEY IS NEVER SENT IN THE CLEAR OVER THE INTERNET. http:// is fine
    for a server on this machine or network; to anything else it would hand
    the key to every hop on the way."""
    import urllib.parse
    u = urllib.parse.urlparse(url or "")
    if u.scheme not in ("http", "https") or not u.hostname:
        raise LLMUnavailable(f"'{url}' is not an http or https address")
    if keyed and u.scheme == "http" and not is_local_url(url):
        raise LLMUnavailable("that address is plain http on the internet, and "
                             "the key would travel unencrypted — use https")


def _http_error(e, who):
    """A readable failure. ⚠️ The key is never in it — only the status and
    the service's own message."""
    try:
        body = e.read().decode("utf-8", "replace")
        msg = json.loads(body).get("error", body)
        if isinstance(msg, dict):
            msg = msg.get("message") or json.dumps(msg)
    except Exception:
        msg = ""
    hint = {401: "the key was refused", 403: "the key may not use this model",
            404: "no such model or address", 429: "the service is rate-limiting "
            "or the account is out of credit"}.get(e.code, f"HTTP {e.code}")
    return LLMUnavailable(f"{who}: {hint}" + (f" — {str(msg)[:160]}" if msg
                                              else ""))


def _sse(resp):
    """The `data:` payloads of a server-sent-event stream, as text."""
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if line.startswith("data:"):
            yield line[5:].strip()


class OpenAICompatLLM:
    """Any service or server that speaks the OpenAI chat-completions format:
    OpenAI, OpenRouter, Groq, Mistral, Together, Alibaba Cloud's Qwen service,
    and local servers such as LM Studio and llama.cpp."""

    def __init__(self, system_prompt, model, url, key=None, label=None):
        self.model, self.system = model, system_prompt
        self.url = (url or "").rstrip("/")
        self.key, self.label = key, label or model
        _check_url(self.url, keyed=bool(key))
        self.remote = not is_local_url(self.url)
        self.settings = _settings_for(model, "openai", self.url)
        self.last_done_reason = None

    def _headers(self):
        h = {"Content-Type": "application/json", "User-Agent": _UA}
        if self.key:
            h["Authorization"] = f"Bearer {self.key}"
        return h

    def stream_reply(self, messages, max_tokens=None, **_ignored):
        _require(self.model)
        payload = {
            "model": self.model, "stream": True,
            "messages": [{"role": "system", "content": self.system}] + messages,
            "max_tokens": max_tokens or C.reply_tokens(),
        }
        if self.settings["send_temperature"]:
            payload["temperature"] = self.settings["temperature"]
        req = urllib.request.Request(
            f"{self.url}/chat/completions", data=json.dumps(payload).encode(),
            headers=self._headers(), method="POST")
        self.last_done_reason = None
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                for data in _sse(r):
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except ValueError:
                        continue
                    for ch in chunk.get("choices") or []:
                        piece = (ch.get("delta") or {}).get("content")
                        if piece:
                            yield piece
                        if ch.get("finish_reason"):
                            # "length" when the ceiling cut it, as Ollama says.
                            self.last_done_reason = ch["finish_reason"]
        except urllib.error.HTTPError as e:
            raise _http_error(e, self.label) from e
        except urllib.error.URLError as e:
            raise LLMUnavailable(f"Can't reach {self.label} at {self.url} "
                                 f"({e.reason})") from e

    def reply(self, messages, max_tokens=None, **_ignored):
        text = "".join(self.stream_reply(messages, max_tokens=max_tokens))
        return text, {"done_reason": self.last_done_reason}

    def available(self):
        """(ok, detail). ⚠️ A FREE CALL — the model list, never a generation."""
        req = urllib.request.Request(f"{self.url}/models",
                                     headers=self._headers())
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                r.read(1)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                return False, str(_http_error(e, self.label))
            # Some servers do not list models; the chat route is what matters.
        except urllib.error.URLError as e:
            return False, f"{self.label} not reachable at {self.url} ({e.reason})"
        where = "remote" if self.remote else "on this network"
        return True, f"{self.model} via {self.label} ({where})"


class ClaudeLLM:
    """Anthropic's Messages API, streamed, over stdlib HTTP.

    ⚠️ NO TEMPERATURE: sampling parameters are rejected on some Claude models
    (see config), and the ceiling plus the system prompt steer it instead.
    ⚠️ THE SYSTEM PROMPT IS MARKED FOR CACHING. It is the prompt plus the
    waking block, stable for a whole session, so it is billed at the cache
    rate after the first reply instead of in full every turn.
    """

    VERSION = ""

    def __init__(self, system_prompt, model, key, url=None, label=None):
        if not key:
            raise LLMUnavailable("the Claude connection has no key yet")
        self.model, self.key = model, key
        self.url = (url or C.ANTHROPIC_URL).rstrip("/")
        self.label = label or "Claude"
        _check_url(self.url, keyed=True)
        self.remote = True
        self.settings = _settings_for(model, "anthropic")
        self.system = [{"type": "text", "text": system_prompt,
                        "cache_control": {"type": "ephemeral"}}]
        self.last_done_reason = None

    def _headers(self):
        return {"Content-Type": "application/json", "User-Agent": _UA,
                "x-api-key": self.key, "anthropic-version": self.VERSION}

    @staticmethod
    def _alternate(messages):
        """The Messages API needs user/assistant turns that alternate and open
        with the user; the turn can hand over two in a row."""
        out = []
        for m in messages:
            role = "assistant" if m.get("role") == "assistant" else "user"
            text = m.get("content") or ""
            if out and out[-1]["role"] == role:
                out[-1]["content"] += "\n\n" + text
            elif out or role == "user":
                out.append({"role": role, "content": text})
        return out or [{"role": "user", "content": "."}]

    def stream_reply(self, messages, max_tokens=None, **_ignored):
        _require(self.model)
        payload = {"model": self.model, "stream": True, "system": self.system,
                   "max_tokens": max_tokens or C.reply_tokens(),
                   "messages": self._alternate(messages)}
        # Off for Claude in `model_settings.KIND_FACTS`; a profile may say
        # otherwise for a model that accepts it.
        if self.settings["send_temperature"]:
            payload["temperature"] = self.settings["temperature"]
        req = urllib.request.Request(
            f"{self.url}/v1/messages", data=json.dumps(payload).encode(),
            headers=self._headers(), method="POST")
        self.last_done_reason = None
        said = False
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                for data in _sse(r):
                    try:
                        ev = json.loads(data)
                    except ValueError:
                        continue
                    kind = ev.get("type")
                    if kind == "content_block_delta":
                        d = ev.get("delta") or {}
                        if d.get("type") == "text_delta" and d.get("text"):
                            said = True
                            yield d["text"]
                    elif kind == "message_delta":
                        stop = (ev.get("delta") or {}).get("stop_reason")
                        if stop:
                            # ⚠️ "max_tokens" IS "length" in the turn's words.
                            self.last_done_reason = ("length"
                                                     if stop == "max_tokens"
                                                     else stop)
                    elif kind == "error":
                        err = ev.get("error") or {}
                        raise LLMUnavailable(f"{self.label}: "
                                             f"{err.get('message') or err}")
        except urllib.error.HTTPError as e:
            raise _http_error(e, self.label) from e
        except urllib.error.URLError as e:
            raise LLMUnavailable(f"Can't reach {self.label} ({e.reason})") from e
        if self.last_done_reason == "refusal" and not said:
            raise Refusal("refusal", "the model declined to answer")

    def reply(self, messages, max_tokens=None, **_ignored):
        text = "".join(self.stream_reply(messages, max_tokens=max_tokens))
        return text, {"done_reason": self.last_done_reason}

    def available(self):
        """(ok, detail). ⚠️ A FREE CALL — the model list, never a generation."""
        req = urllib.request.Request(f"{self.url}/v1/models",
                                     headers=self._headers())
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                r.read(1)
        except urllib.error.HTTPError as e:
            return False, str(_http_error(e, self.label))
        except urllib.error.URLError as e:
            return False, f"{self.label} not reachable ({e.reason})"
        return True, f"{self.model} via {self.label} (remote)"


def for_choice(system_prompt, choice=None):
    """The client for the person's choice (`config.model_choice`).

    ⚠️ A CONNECTION'S KEY IS READ HERE, at the moment of use, from the sealed
    store — never kept in settings, never logged.
    """
    choice = choice or C.model_choice()
    if choice["kind"] == "local":
        return LocalLLM(system_prompt, model=choice["model"])
    from . import credentials
    key = None
    if choice.get("conn"):
        key = credentials.get(credentials.MODEL_PREFIX + choice["conn"])
    if choice["kind"] == "anthropic":
        return ClaudeLLM(system_prompt, choice["model"], key,
                         label=choice.get("label"))
    return OpenAICompatLLM(system_prompt, choice["model"], choice["url"],
                           key=key, label=choice.get("label"))


def is_remote(llm):
    """Does this client send the conversation off this machine's network?"""
    return bool(getattr(llm, "remote", False))


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
    """The client every caller uses — the turn, the night, the idle work.

    ⚠️ ONE CHOICE FOR ALL OF THEM (the user: a remote model sees the same as a
    local one; it is the person's prerogative). With no explicit `provider`,
    this is `config.model_choice()`, read now, so a switch reaches the next
    caller without a restart.
    """
    if provider:                     # explicit, e.g. the CLI's --provider
        p = provider.lower()
        if p == "local":
            return LocalLLM(system_prompt)
        if p == "anthropic":
            return AnthropicLLM(system_prompt)
        raise ValueError(f"unknown provider {provider!r} "
                         "(use 'local' or 'anthropic')")
    # The old environment switch, still honoured where nothing was chosen.
    if C.PROVIDER.lower() == "anthropic" and not C._settings().get(
            "model_choice"):
        return AnthropicLLM(system_prompt)
    return for_choice(system_prompt)


# Alias: `LLM` names the Anthropic client.
LLM = AnthropicLLM
