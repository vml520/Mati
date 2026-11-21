"""
Only one thing may hold the field.

θ₀ is mutable state backed by a file. Two processes that load it, deform it and
save it will overwrite each other — silently, because neither can see the
other. There is no merge, and no way to tell afterwards which writes survived.

So the rule: **the utility tools open only when Mati has handed the field
over (`/release`) or is shut down.** They are not clients of the server the
way the chat app is; they take the field, work on it directly, and give it
back. Sleeping does NOT free the field — a sleeping Mati still holds it.

Three states, and only two of them allow a tool to run:

    Mati not running       → take the field
    Mati released it       → take the field (it let go on purpose)
    Mati holds it          → refuse, and say how to hand it over

This is not hypothetical. A profiling script is enough to cause it, or a test
that isolates the notes DB but not θ₀ — and a process that TRAINS the field
would do something worse than either, overwriting Mati's notes θ₀ with a field
1.87 radians away.

Every entry point that calls `bridge.load_field()` goes through `require()`.

⚠️ THE CHECK GOES BOTH WAYS. Asking the SERVER's `/health` lets a tool tell
when Mati has the field — but a starting server cannot tell that way when a
TOOL has it, which would leave two owners and no warning. So whoever holds θ₀
writes a lock file naming themselves, and everyone checks it: the server takes
it on start and on wake, drops it on sleep and on shutdown; the terminal takes
it on `/load` and drops it on save or quit.

A lock whose owner has died is not a lock. Every read checks the recorded pid
is still alive and clears the file if it is not — otherwise one crash would
lock the field until somebody deleted a file they did not know about.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request

from . import config as C


LOCK_PATH = os.path.join(C.CACHE_DIR, "field.lock")


def _alive(pid):
    """Is that process still running? Signal 0 asks without sending anything."""
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, TypeError):
        return False


def held_by():
    """Who holds θ₀ right now — {holder, pid, since} — or None.

    Clears the lock as a side effect if its owner is gone, because a stale
    lock is indistinguishable from a real one to every caller.
    """
    try:
        with open(LOCK_PATH) as fh:
            doc = json.load(fh)
    except (OSError, ValueError):
        return None
    if not _alive(doc.get("pid")):
        try:
            os.remove(LOCK_PATH)
        except OSError:
            pass
        return None
    return doc


def take(holder, force=False):
    """Claim θ₀ for `holder`. Returns True, or False if someone else has it.

    Re-taking a lock this process already holds is fine and is not an error —
    the server takes it on start and again on every wake.
    """
    cur = held_by()
    if cur and not force and cur.get("pid") != os.getpid():
        return False
    try:
        os.makedirs(os.path.dirname(LOCK_PATH), exist_ok=True)
        tmp = LOCK_PATH + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({"holder": holder, "pid": os.getpid(),
                       "since": time.time()}, fh)
        os.replace(tmp, LOCK_PATH)
        return True
    except OSError:
        return False            # a lock we cannot write must not block anyone


def release():
    """Give θ₀ back, if this process is the one holding it."""
    cur = held_by()
    if not cur or cur.get("pid") != os.getpid():
        return False
    try:
        os.remove(LOCK_PATH)
        return True
    except OSError:
        return False


class FieldBusy(RuntimeError):
    """Mati is awake and holding the field."""


def server_health(timeout=0.5):
    """Mati's health dict, or None if nothing is listening."""
    try:
        url = f"http://{C.SERVER_HOST}:{C.SERVER_PORT}/health"
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
        return None


def holds_field():
    """Is ANYTHING holding the field right now — Mati or a tool?

    Both are asked, because they fail differently: the server can be up and
    awake without having written a lock (an older build, or a crash between
    taking the field and recording it), and a tool holding the field has no
    HTTP surface to ask at all.
    """
    cur = held_by()
    if cur and cur.get("pid") != os.getpid():
        return True
    health = server_health()
    if health is None:
        return False                        # not running
    return bool(health.get("holds_field", not health.get("asleep", False)))


def put_to_sleep(timeout=180):
    """Ask Mati to finish up and hand the field over. True if it did.

    ⚠️ `/release`, NEVER `/sleep`. Sleeping keeps the field; only a release
    lets a tool take it.
    """
    try:
        req = urllib.request.Request(
            f"http://{C.SERVER_HOST}:{C.SERVER_PORT}/release",
            data=b"{}", method="POST",
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return bool(json.loads(r.read().decode()).get("asleep"))
    except (urllib.error.URLError, OSError, ValueError):
        return False


def require(tool, force=False, auto_sleep=False):
    """
    Claim the field for `tool`, or raise FieldBusy explaining why not.

    force=True skips the check — for when the operator knows better than this
    function does. auto_sleep=True asks Mati to stand down first, which is
    polite but not free: waking it again costs a reload, and any queued episode
    writes have to finish before it can let go.
    """
    if force or not holds_field():
        return True
    if auto_sleep and put_to_sleep():
        return True
    cur = held_by()
    if cur and cur.get("pid") != os.getpid():
        raise FieldBusy(
            f"{cur.get('holder', 'something')} (pid {cur.get('pid')}) is "
            f"holding the field, so {tool} cannot open.\n"
            f"  Two processes writing θ₀ would overwrite each other.\n"
            f"  Close it, or re-run with --force if you know it is gone.")
    raise FieldBusy(
        f"Mati is awake and holding the field, so {tool} cannot open.\n"
        f"  Two processes writing θ₀ would overwrite each other.\n"
        f"  Hand it over:  curl -X POST localhost:{C.SERVER_PORT}/release\n"
        f"    (with the terminal switched on in the app)\n"
        f"  Or stop it, or re-run with --force if you know it is idle.")


def guard(tool, force=False, auto_sleep=False):
    """`require`, but prints and returns an exit code instead of raising.

    For `main()` functions — every utility here is a CLI, and a traceback is a
    worse explanation than the sentence above.
    """
    try:
        require(tool, force=force, auto_sleep=auto_sleep)
        return 0
    except FieldBusy as e:
        print(f"[refused] {e}", file=sys.stderr)
        return 2
