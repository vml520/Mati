"""
A throwaway copy of Mati's field, for testing against.

    python3 -m Mati.sandbox                  # make one, print how to use it
    python3 -m Mati.sandbox --run            # make one and open a shell in it
    python3 -m Mati.sandbox --list           # what sandboxes exist
    python3 -m Mati.sandbox --clean          # delete them all

WHY THIS EXISTS. Benchmarking a turn needs a real running server, and the
easy way to get one is to point at the real install — so that is what got
done, repeatedly. Every one of those turns wrote an episode
and moved θ₀. Mati ended up holding 43 machine-generated "memories" of the
same probe question, and `self_knowledge` then reported them back as lived
experience. The store had to be cleaned by hand and reindexed.

The suites already knew better — each one copies the store to a temp dir and
monkeypatches module globals. That worked and nobody reached for it outside a
suite, because it was six lines of setup rather than one command. So: one
command.

WHAT A SANDBOX IS. Exactly what `transfer.manifest()` says a portable install
is — the notes DB, the notes θ₀, the checkpoint, its θ₀, and the vocabulary.
Copied, not linked. `MATI_HOME` repoints every path in the package at it
(`Mati/config.py`), so nothing needs patching and nothing can reach the real
store by accident.

WHAT IT IS NOT. Not isolated from Ollama — a sandbox talks to the same local
model on the same port. And if the real server is running it still owns port
8756, so a sandboxed server needs `--port`.
"""

import argparse
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from . import transfer


PREFIX = "mati-sandbox-"


def _real_cache():
    """The REAL cache, regardless of any MATI_HOME already in force.

    Copying a sandbox from inside a sandbox would compound drift, so the
    source is always resolved from the package location, never the env.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, ".cache")


def create(dest=None, verbose=True):
    """Copy the field into a fresh directory. Returns its path."""
    src = _real_cache()
    if dest is None:
        dest = tempfile.mkdtemp(prefix=PREFIX)
    os.makedirs(dest, exist_ok=True)

    copied, missing, total = [], [], 0
    for item in transfer.manifest(root=os.path.dirname(src)):
        name = item["name"]
        s = os.path.join(src, name)
        if not os.path.exists(s):
            missing.append(name)
            continue
        shutil.copy2(s, os.path.join(dest, name))
        copied.append(name)
        total += os.path.getsize(s)
    # The write-ahead log holds committed rows that the.db alone does not.
    # Copying the.db without it is the "looks fine, silently older" failure
    # `transfer.py` was written to prevent.
    for side in ("mati_notes.db-wal", "mati_notes.db-shm"):
        s = os.path.join(src, side)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(dest, side))
            copied.append(side)

    if verbose:
        print(f"\n  sandbox: {dest}")
        print(f"  copied {len(copied)} file(s), {total/1e6:.1f} MB")
        for m in missing:
            print(f"    (absent in the real install, skipped: {m})")
        print("\n  Use it — every path in the package follows MATI_HOME:\n")
        print(f"      export MATI_HOME={dest}")
        print("      python3 -m Mati.server --port 8757")
        print(f"      MATI_HOME={dest} python3 -m Mati.mati \"hello\"\n")
        print("  The real store is untouched. Delete with "
              "`python3 -m Mati.sandbox --clean`.\n")
    return dest


def existing():
    tmp = tempfile.gettempdir()
    try:
        names = os.listdir(tmp)
    except OSError:
        return []
    return sorted(os.path.join(tmp, n) for n in names if n.startswith(PREFIX))


def main():
    ap = argparse.ArgumentParser(
        description="Make a throwaway copy of Mati's field to test against")
    ap.add_argument("--dest", help="where to put it (default: a temp dir)")
    ap.add_argument("--list", action="store_true", help="show existing sandboxes")
    ap.add_argument("--clean", action="store_true", help="delete every sandbox")
    ap.add_argument("--run", action="store_true",
                    help="create one and open a shell with MATI_HOME set")
    args = ap.parse_args()

    if args.list:
        found = existing()
        print(f"\n  {len(found)} sandbox(es)")
        for f in found:
            age = (time.time() - os.path.getmtime(f)) / 3600
            size = sum(os.path.getsize(os.path.join(f, x))
                       for x in os.listdir(f)) / 1e6
            print(f"    {f}   {size:5.1f} MB   {age:.1f}h old")
        print()
        return 0

    if args.clean:
        found = existing()
        for f in found:
            shutil.rmtree(f, ignore_errors=True)
        print(f"  removed {len(found)} sandbox(es)")
        return 0

    dest = create(args.dest)
    if args.run:
        env = dict(os.environ, MATI_HOME=dest,
                   PS1="(mati-sandbox) $ ")
        print("  opening a shell with MATI_HOME set — exit to leave it\n")
        os.execvpe(os.environ.get("SHELL", "/bin/sh"),
                   [os.environ.get("SHELL", "/bin/sh")], env)
    return 0


if __name__ == "__main__":
    sys.exit(main())
