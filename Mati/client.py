"""
Talking to the field owner.

    python3 -m Mati.client                 # interactive chat (multi-turn)
    python3 -m Mati.client --watch         # follow telemetry, like the pro app
    python3 -m Mati.client --health

Both Mati apps are clients of `Mati.server` — the chat app and the pro
monitoring/training terminal — so the transport lives here once rather than
twice. Stdlib only (urllib); nothing to install, and nothing that could reach
off the machine.

The chat REPL below is deliberately plain. It exists to prove the server does
what a chat app needs — a session that remembers the previous turn, tokens that
appear as they are generated, and a reply that returns before consolidation —
not to be the shipped interface.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

from . import config as C


class MatiUnavailable(Exception):
    """The server isn't running, or isn't reachable at that address."""


class MatiClient:
    def __init__(self, host=None, port=None, timeout=900):
        host = host or C.SERVER_HOST
        port = C.SERVER_PORT if port is None else port
        self.base = f"http://{host}:{port}"
        self.timeout = timeout

    # ── plumbing ─────────────────────────────────────────────────────────────

    def _open(self, path, payload=None, method=None, timeout=None):
        url = f"{self.base}{path}"
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(
            url, data=data, method=method or ("POST" if data else "GET"),
            headers={"Content-Type": "application/json"} if data else {})
        try:
            return urllib.request.urlopen(req, timeout=timeout or self.timeout)
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            try:
                raise MatiUnavailable(json.loads(body).get("error", body))
            except (json.JSONDecodeError, AttributeError):
                raise MatiUnavailable(f"HTTP {e.code}: {body}") from e
        except urllib.error.URLError as e:
            raise MatiUnavailable(
                f"Can't reach Mati at {self.base} ({e.reason}).\n"
                f"  Start it:  python3 -m Mati.server") from e

    def _json(self, path, payload=None, method=None, timeout=None):
        with self._open(path, payload, method, timeout) as r:
            return json.loads(r.read().decode())

    @staticmethod
    def _sse(response):
        """Yield (event, data) pairs from a text/event-stream response."""
        event, data = None, []
        for raw in response:
            line = raw.decode(errors="replace").rstrip("\n")
            if not line:
                if event is not None:
                    try:
                        yield event, json.loads("\n".join(data) or "{}")
                    except json.JSONDecodeError:
                        pass
                event, data = None, []
                continue
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].strip())

    # ── API ──────────────────────────────────────────────────────────────────

    def health(self):
        # ⚠️ 10s DECLARED A BUSY SERVER DEAD, and the terminal showed "Mati
        # is not reachable" over a server that was awake and working.
        #
        # ⚠️ AN EARLIER NOTE HERE BLAMED THE WRONG CAUSE — GIL starvation —
        # the idle loop holding the interpreter and choking the HTTP thread.
        # The evidence against it was already on the page: /state answered in
        # 0.03s throughout, on the same thread in the same server. A starved
        # thread does not serve one route instantly and another never.
        #
        # It was `schedule.pending(include_held=True)` inside /health,
        # decrypting ~690 sealed rows at 59ms of openssl each. Fixed at source
        # in f13f673; /health now answers in 0.05s.
        #
        # The generous timeout STAYS. It costs nothing when the server is well,
        # and the next slow thing should be reported as slow rather than dead.
        return self._json("/health", timeout=45)

    def state(self):
        return self._json("/state", timeout=10)

    def episodes(self):
        return self._json("/episodes", timeout=30)["episodes"]

    def history(self, session="default"):
        return self._json(f"/sessions/{session}", timeout=10)["history"]

    def clear(self, session="default"):
        return self._json(f"/sessions/{session}", method="DELETE", timeout=10)

    def pending(self):
        """What Mati is holding to say, and whether it may interrupt."""
        # ⚠️ 10s TIMED OUT AGAINST A 41s CALL. Sealing `to_say` made
        # `schedule.pending` decrypt every candidate row, one openssl
        # subprocess each. Fixed at the source (cache + lazy body open, now
        # ~7s cold and 0.02s warm) — this is headroom for the FIRST call in a
        # fresh process, which is the one that cannot use the cache.
        return self._json("/pending", timeout=30)

    def remind(self, text, when=None):
        return self._json("/remind", {"text": text, "when": when})

    def consent(self, **kw):
        return self._json("/consent", kw or {})

    def shutdown(self):
        """Stop for real, leaving an install that can be carried away.

        Not `/sleep`, which hands the field to the pro terminal and expects to
        wake. This reindexes to completion and folds the write-ahead log into
        the database, so the directory is safe to copy to a drive.
        """
        # ⚠️ Kept for callers that still say "shutdown". It PACKS; it does
        # not stop the process. `quit()` is the stop.
        return self._json("/pack", {}, timeout=900)

    def pack(self):
        """Make the install carryable. Does not stop the process."""
        return self._json("/pack", {}, timeout=900)

    def stop(self):
        """Pack, then stop the process."""
        return self._json("/stop", {}, timeout=900)

    def imprint(self, text, **kw):
        return self._json("/imprint", {"text": text, **kw})

    def dream(self, cycles=1):
        return self._json("/dream", {"cycles": cycles})

    # ── the library ──────────────────────────────────────────────────────
    # ⚠️ HERE, NOT IN A NEW MODULE. The user asked for a "new terminal"; the
    # transport, the REPL and the routes all live here already, and both
    # existing apps are clients of `Mati.server`. A second terminal would
    # duplicate the transport, which is the mistake this file's own docstring
    # says it exists to avoid.
    def library(self):
        return self._json("/library")

    def library_read(self, path):
        return self._json("/library/read", {"path": path})

    def library_shelve(self, title):
        return self._json("/library/shelve", {"title": title})

    def reindex(self, limit=None):
        return self._json("/reindex", {"limit": limit})

    def chat(self, text, session="default", on_chunk=None, remember=True):
        """Stream a turn. Returns (reply, info)."""
        with self._open("/chat", {"text": text, "session": session,
                                  "remember": remember, "stream": True}) as r:
            pieces, info = [], {}
            for event, data in self._sse(r):
                if event == "chunk":
                    pieces.append(data["text"])
                    if on_chunk:
                        on_chunk(data["text"])
                elif event == "done":
                    info = data.get("info", {})
                    return data.get("reply", "".join(pieces)), info
                elif event == "error":
                    raise MatiUnavailable(
                        data.get("detail") or data.get("error", "unknown error"))
        return "".join(pieces), info

    def interrupt(self, said, session="default"):
        """Tell Mati the person typed over the reply. Never raises.

        Best-effort on purpose: the interruption has already happened on
        screen, and failing to record it must not also lose the new message.
        """
        try:
            return self._json("/interrupt",
                              {"said": said, "session": session}, timeout=20)
        except Exception:
            return {}

    def watch(self):
        """Yield telemetry events until interrupted. Never blocks the field."""
        with self._open("/events", timeout=None) as r:
            for event, data in self._sse(r):
                if event != "ping":
                    yield data


# ─────────────────────────────────────────────────────────────────────────────
# Interactive chat — the multi-turn proof
# ─────────────────────────────────────────────────────────────────────────────

def repl(client, session, show_info):
    try:
        h = client.health()
    except MatiUnavailable as e:
        print(e, file=sys.stderr)
        return 1
    print(f"Mati — {h['llm']}, {h['episodes']} episodes, session {session!r}")
    print("Ctrl-D to leave.  /new clears the session, /state shows the field.\n")

    while True:
        try:
            text = input("you  ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            continue
        if text == "/new":
            client.clear(session)
            print("  [session cleared]\n")
            continue
        # ── library commands. A CATALOGUE, NOT A READER: these never print
        # studied verbatim. Mati may quote a line
        # in a reply, but browsing the text of books it has read turns this
        # into an ebook reader carrying someone's library.
        if text == "/library" or text.startswith("/library "):
            want = text[9:].strip()
            d = c.library()
            works = d.get("works") or []
            if d.get("error"):
                print(f"     ! {d['error']}")
            elif not works:
                print("     nothing read yet — /read <path> to add a book")
            else:
                for w in works:
                    t = w.get("title") or "?"
                    if want and want.lower() not in t.lower():
                        continue
                    bits = [f"{w.get('chunks', 0)} passages"]
                    if w.get("author"):
                        bits.insert(0, w["author"])
                    if w.get("status"):
                        bits.append(w["status"])
                    print(f"     {t}  —  {', '.join(str(b) for b in bits)}")
            continue
        if text == "/reading":
            st = (c.library().get("reading") or {})
            pend = st.get("pending", 0)
            print(f"     {pend} passage(s) still to read"
                  + (f"  (~{round(pend * 30 / 60)} min)" if pend else ""))
            for doc in (st.get("documents") or []):
                if doc.get("pending"):
                    print(f"       {doc['origin']}: {doc['pending']} left "
                          f"of {doc['chunks']}")
            continue
        if text.startswith("/read "):
            got = c.library_read(text[6:].strip())
            if got.get("error"):
                print(f"     ! {got['error']}")
            else:
                # ⚠️ THE COST FIRST. ~30s a passage; a book is hours.
                print(f"     {got['chunks']} passages queued "
                      f"(~{got.get('minutes', '?')} min of idle reading)"
                      + (f", {got['already_had']} already known"
                         if got.get("already_had") else ""))
            continue
        if text.startswith("/shelve "):
            title = text[8:].strip()
            got = c.library_shelve(title)
            if got.get("error"):
                print(f"     ! {got['error']}")
            else:
                # ⚠️ ARCHIVED, NOT DELETED, and the wording says so — an
                # archived work is still findable by its words.
                print(f"     shelved {got['archived']} passage(s) of "
                      f"{got['title']} — archived, still findable by name")
            continue
        if text == "/state":
            print("  " + json.dumps(client.state(), indent=2, default=str) + "\n")
            continue

        t0 = time.time()
        print("mati  ", end="", flush=True)
        try:
            reply, info = client.chat(
                text, session=session,
                on_chunk=lambda p: (sys.stdout.write(p), sys.stdout.flush()))
        except MatiUnavailable as e:
            print(f"\n  [error] {e}\n", file=sys.stderr)
            continue
        print()
        if show_info:
            trip = info.get("triplet") or []
            basin = ", ".join(f"{x:.3f}" for x in trip) if trip else "-"
            print(f"  [{time.time()-t0:.1f}s  basin=({basin})  "
                  f"coherence={info.get('coherence', 0):.3f}  "
                  f"recalled={info.get('n_memories', 0)}"
                  + ("  episode queued" if info.get("episode_deferred") else "")
                  + "]")
        print()


def main():
    ap = argparse.ArgumentParser(description="Mati client")
    ap.add_argument("text", nargs="*", help="one-shot turn; omit for a REPL")
    ap.add_argument("--session", default="default")
    ap.add_argument("--host", default=C.SERVER_HOST)
    ap.add_argument("--port", type=int, default=C.SERVER_PORT)
    ap.add_argument("--health", action="store_true")
    ap.add_argument("--watch", action="store_true",
                    help="follow telemetry (what the pro terminal does)")
    ap.add_argument("--quiet", action="store_true", help="reply only")
    args = ap.parse_args()

    client = MatiClient(args.host, args.port)

    if args.health:
        print(json.dumps(client.health(), indent=2, default=str))
        return 0

    if args.watch:
        print("watching field telemetry — Ctrl-C to stop\n", file=sys.stderr)
        try:
            for ev in client.watch():
                stage = ev.get("stage", "")
                extra = " ".join(f"{k}={v}" for k, v in ev.items()
                                 if k not in ("t", "stage", "since"))
                print(f"  {time.strftime('%H:%M:%S')}  {stage:<24} {extra}")
        except (KeyboardInterrupt, MatiUnavailable) as e:
            print(f"\n{e if isinstance(e, MatiUnavailable) else ''}",
                  file=sys.stderr)
        return 0

    if args.text:
        reply, info = client.chat(
            " ".join(args.text), session=args.session,
            on_chunk=None if args.quiet
            else (lambda p: (sys.stdout.write(p), sys.stdout.flush())))
        if args.quiet:
            print(reply)
        else:
            print()
        return 0

    return repl(client, args.session, show_info=not args.quiet)


if __name__ == "__main__":
    raise SystemExit(main())
