"""
Mati out loud. Off by default, and never in the way.

    python3 -m Mati.speech --voices        # what this Mac has
    python3 -m Mati.speech "hello"         # try one

⚠️ `/usr/bin/say` IS THE WHOLE DEPENDENCY. Built into macOS, 177 voices on
this machine, no key, no account, no network — which is the only kind of
service this project will take.

⚠️ OFF UNTIL ASKED FOR, like everything here that can interrupt. The setting
is `voice_on`, written by the app's own settings sheet. A machine that starts
talking because software was installed is the failure this default exists to
prevent.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import threading

_SAY = "/usr/bin/say"
_proc = None

# ── ducking the radio ───────────────────────────────────────────────────────
# ⚠️ WITHOUT THIS THE FEATURE IS UNUSABLE AND NOT MERELY RUDE. The radio plays
# through the same speaker, and `say` has no mixing of its own — a station at
# full volume does not make Mati quieter, it makes Mati inaudible. A voice
# you cannot hear over the music is the same as no voice.

DUCK_TO = 0.22          # fraction of the station's own level, not an absolute
_duck_lock = threading.Lock()
_ducked = None          # (was, set_to) while ducked, else None
_gen = 0                # ⚠️ see `_unduck` — a stale watcher must not restore


def available():
    """Is there a voice on this machine at all?"""
    import os
    return os.path.exists(_SAY)


# ⚠️ BUBBLES BY DEFAULT (the user) — when this Mac has it. macOS ships a
# different set of voices per install, so a default that is not installed
# falls back to the system's own voice rather than to silence.
DEFAULT_VOICE = "Bubbles"

_TABLE = {"at": 0.0, "rows": []}


def _voice_table():
    """[(name, language, demo sentence)] from `say -v ?`, kept for a minute —
    the list changes only when someone installs a voice."""
    import time
    if _TABLE["rows"] and time.time() - _TABLE["at"] < 60:
        return _TABLE["rows"]
    if not available():
        return []
    try:
        out = subprocess.run([_SAY, "-v", "?"], capture_output=True,
                             text=True, timeout=10).stdout
    except Exception:
        return []
    rows = []
    for line in out.splitlines():
        m = re.match(r"(\S+(?: \S+)?)\s+([a-z]{2})_\S*\s*#\s*(.*)$", line)
        if m:
            rows.append((m.group(1), m.group(2), m.group(3).strip()))
    _TABLE.update(at=time.time(), rows=rows)
    return rows


def voices(lang="en"):
    """The installed voices, names only."""
    return [n for n, lg, _d in _voice_table() if not lang or lg == lang]


def voice_name():
    """The voice Mati speaks with: the person's choice, else Bubbles if this
    Mac has it, else None — which is the system's own voice."""
    try:
        from . import config as C
        chosen = C._settings().get("voice_name")
    except Exception:
        chosen = None
    if chosen:
        return chosen
    return DEFAULT_VOICE if DEFAULT_VOICE in voices(lang=None) else None


def sample(name):
    """Say this voice's own demo sentence, now. True if it started.

    ⚠️ ON REQUEST ONLY — the person picked the voice in the selector — so it
    plays whether or not replies are spoken; hearing a voice is how it is
    chosen. ⚠️ ONLY AN INSTALLED VOICE, by exact name: the name goes to `say`
    as an argument and nothing else is accepted.
    """
    global _proc
    row = next((r for r in _voice_table() if r[0] == name), None)
    if row is None:
        return False
    demo = row[2] or f"Hello! My name is {name}."
    try:
        stop()
        gen = _duck()
        _proc = subprocess.Popen([_SAY, "-v", name, "--", demo],
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
        threading.Thread(target=_watch, args=(_proc, gen), daemon=True).start()
        return True
    except Exception:
        return False


def _duck():
    """Turn the radio down. Returns True if it did.

    ⚠️ NEVER DUCKS TWICE. A second reply arriving while the first is still
    being spoken would read the ALREADY-DUCKED level as the station's own and
    take another 22% of it — 0.8 → 0.18 → 0.04 — then "restore" to 0.18. Two
    interruptions and the radio is off for good, with nothing to restore from.
    `_ducked` holds the ORIGINAL level until it is given back.
    """
    global _ducked, _gen
    with _duck_lock:
        _gen += 1
        if _ducked is not None:
            return _gen                  # already down; leave the original be
        try:
            from . import radio
            if radio.playing() is None:
                return _gen
            was = radio.volume()
            if was is None or was <= 0.0:
                return _gen
            lvl = radio.volume(was * DUCK_TO)
            if lvl is not None:
                _ducked = (was, lvl)
        except Exception:
            pass
        return _gen


def _unduck(gen=None):
    """Give the radio back its volume.

    ⚠️ A STALE WATCHER MUST NOT RESTORE. When a second reply interrupts the
    first, the first's watcher wakes up and calls this — by which time the new
    speech is under way and the radio should still be down. `gen` is the duck
    this caller is responsible for; anything older is ignored. The level check
    below cannot catch this on its own, because both ducks set the same value.

    ⚠️ ONLY IF IT IS STILL WHERE WE PUT IT. If the person reached for the
    volume while Mati was talking, that is their decision and restoring would
    silently undo it. A duck that can overwrite a deliberate setting is worse
    than one that occasionally leaves the radio quiet — they can turn it up.
    """
    global _ducked
    with _duck_lock:
        if _ducked is None or (gen is not None and gen != _gen):
            return False
        was, set_to = _ducked
        _ducked = None
        try:
            from . import radio
            now = radio.volume()
            if now is None:
                return False
            if abs(now - set_to) > 0.02:
                return False             # they changed it — theirs now
            radio.volume(was)
            return True
        except Exception:
            return False


def _watch(proc, gen):
    """Wait out the speech, then restore. ⚠️ Daemon: never holds a shutdown.

    ⚠️ AND IT HAS A TIMEOUT. `say` on a stuck audio device can hang, and a
    duck with no path back would leave the radio at a fifth of its volume with
    no visible cause. Ten minutes is far past the 220-word ceiling.
    """
    try:
        proc.wait(timeout=600)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass
    _unduck(gen)


def stop():
    """Silence whatever is being said. Returns True if something was.

    ⚠️ RESTORES THE RADIO ON THE WAY OUT. `terminate` makes the watcher's
    `wait` return, so it would unduck anyway — but `stop()` is also what the
    app's "be quiet" calls, and leaving the music low until a thread notices
    reads as a bug.
    """
    global _proc
    spoke = False
    if _proc is not None and _proc.poll() is None:
        try:
            _proc.terminate()
        except Exception:
            pass
        spoke = True
    _proc = None
    _unduck()
    return spoke


def speaking():
    return _proc is not None and _proc.poll() is None


def _speakable(text):
    """What is worth saying aloud out of a reply.

    ⚠️ A REPLY IS WRITTEN TO BE READ, and some of it does not survive being
    spoken. The money confirmation and the budget readback are tables of
    figures; a bulleted list read aloud is a wall. This keeps the prose and
    drops the scaffolding rather than narrating punctuation.
    """
    out = []
    fenced = False
    for line in (text or "").splitlines():
        s = line.strip()
        # ⚠️ A FENCE IS A BLOCK, NOT A LINE. Skipping the ``` lines alone
        # would leave the code between them to be read out character by
        # character.
        if s.startswith("```"):
            fenced = not fenced
            continue
        if fenced or not s:
            continue
        # Table and list scaffolding — spoken, these are noise.
        if s.startswith(("|", "---", "===")):
            continue
        s = re.sub(r"^[-*•]\s+", "", s)
        s = re.sub(r"\s{2,}", " ", s)
        out.append(s)
    said = " ".join(out)
    # ⚠️ A CEILING, for the same reason replies have one: `say` reads at about
    # 175 words a minute, so an unbounded reply is minutes of talking a person
    # cannot skim past. They have the text on screen; this is an accompaniment.
    words = said.split()
    if len(words) > 220:
        said = " ".join(words[:220]) + "…"
    return said.strip()


_queue = []
_queue_lock = threading.Lock()
_pump_thread = None


def enqueue(text, on_start=None):
    """Speak this after whatever is already speaking. True if queued.

    ⚠️ `say()` INTERRUPTS, WHICH IS RIGHT FOR A WHOLE REPLY AND WRONG FOR A
    SENTENCE. Speaking a streamed reply means several calls in a row, and with
    `say()` each one would cut off the one before — the person would hear the
    first words of every sentence and the whole of none.

    ⚠️ AND THE DUCK SPANS THE WHOLE REPLY, not each sentence. Ducking and
    restoring per sentence would pump the radio volume up and down through an
    answer, which is worse than not ducking at all.
    """
    global _pump_thread
    try:
        from . import config as C
        if not C._settings().get("voice_on") or not available():
            return False
        said = _speakable(text)
        if not said:
            return False
        with _queue_lock:
            _queue.append((said, on_start))
            if _pump_thread is not None and _pump_thread.is_alive():
                return True
            _pump_thread = threading.Thread(target=_pump, daemon=True)
            _pump_thread.start()
        return True
    except Exception:
        return False


def _pump():
    """Speak the queue in order, ducking once for the whole run."""
    global _proc
    gen = _duck()
    try:
        while True:
            with _queue_lock:
                if not _queue:
                    return
                said, on_start = _queue.pop(0)
            voice = voice_name()
            # ⚠️ RATE IS A SETTING because 175 wpm is a default, not a law.
            # Someone reading along wants it faster; someone listening while
            # doing something else wants it slower.
            try:
                from . import config as C
                rate = C._settings().get("voice_rate")
            except Exception:
                rate = None
            cmd = [_SAY] + (["-v", str(voice)] if voice else [])
            if rate:
                cmd += ["-r", str(int(rate))]
            # ⚠️ "--" BEFORE THE TEXT. Without it a sentence starting "-f…"
            # is read as `say -f`, which reads a FILE aloud — measured.
            try:
                p = subprocess.Popen(cmd + ["--", said], stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
            except Exception:
                return
            _proc = p
            # ⚠️ THE TEXT APPEARS WHEN THE VOICE REACHES IT, not when the
            # model produced it. `say` reads at about 175 words a minute and
            # the stream arrives as fast as the model writes, so a screen fed
            # straight from the stream runs ahead and the two drift apart
            # over a reply. Changing the RATE only changes how fast they
            # drift; pacing the display to the speech is what keeps them
            # together.
            if on_start is not None:
                try:
                    on_start()
                except Exception:
                    pass
            try:
                p.wait(timeout=600)
            except Exception:
                try:
                    p.terminate()
                except Exception:
                    pass
                return
    finally:
        _proc = None
        _unduck(gen)


def drain():
    """Drop anything queued and silence what is speaking."""
    # ⚠️ A DRAINED SENTENCE MUST STILL BE SHOWN. Its `on_start` is what puts
    # it on screen, so dropping it silently would lose text the person never
    # saw — turning the voice off would delete the rest of the answer.
    with _queue_lock:
        pending = list(_queue)
        _queue.clear()
    for _said, on_start in pending:
        if on_start is not None:
            try:
                on_start()
            except Exception:
                pass
    return stop()


def say(text, db_path=None):
    """Speak it, if the person asked for that. Returns True if it started.

    ⚠️ NEVER BLOCKS AND NEVER RAISES. This runs on the turn path: a voice that
    can delay a reply, or break one, is worse than no voice. Detached, and
    every failure is silent-by-design rather than silent-by-accident — the
    caller gets False.

    ⚠️ AND IT INTERRUPTS ITSELF. A new reply stops the old one mid-sentence;
    two answers talking over each other is worse than missing the first.
    """
    global _proc
    try:
        from . import config as C
        st = C._settings()
        if not st.get("voice_on") or not available():
            return False
        said = _speakable(text)
        if not said:
            return False
        stop()
        cmd = [_SAY]
        voice = voice_name()
        if voice:
            cmd += ["-v", str(voice)]
        gen = _duck()                  # before the first syllable, not after
        _proc = subprocess.Popen(cmd + ["--", said],
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
        threading.Thread(target=_watch, args=(_proc, gen),
                         daemon=True).start()
        return True
    except Exception:
        _unduck()                      # ⚠️ never leave the radio down on error
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("text", nargs="*")
    ap.add_argument("--voices", action="store_true")
    a = ap.parse_args(argv)
    if a.voices:
        got = voices()
        print(f"  {len(got)} English voices")
        for v in got:
            print(f"    {v}")
        return 0
    if not a.text:
        print("  nothing to say")
        return 1
    from . import config as C
    was = C._settings().get("voice_on")
    C.save_settings(voice_on=True)          # a try is an explicit request
    ok = say(" ".join(a.text))
    if was is not None:
        C.save_settings(voice_on=was)
    print("  spoke" if ok else "  silent — no voice, or it failed")
    if ok and _proc:
        _proc.wait()
    return 0


if __name__ == "__main__":
    sys.exit(main())
