"""
Snapshots of Mati's whole memory, and the way back from one.

    python3 -m Mati.backup                    # take one
    python3 -m Mati.backup --list             # what exists
    python3 -m Mati.backup --restore <name>   # go back to one
    python3 -m Mati.backup --prune 10         # keep the newest 10

WHY THIS IS A FEATURE AND NOT A HOLE. Mati's store is a plain SQLite file and
a plain .npz beside it, so anything running as you can read or rewrite them —
that is how the machinery-and-pantomime cleanup on  was done, by a
script, from outside. Rather than pretend that is preventable on a machine its
owner controls, the deliberate operations get a front door here and in the pro
terminal, and the rest of the defence is at-rest encryption plus a signed app.

The thing that made that cleanup safe to attempt was a backup taken first.
That was a throwaway script; this is the same thing, repeatable, so no future
edit has to be brave.

WHAT A SNAPSHOT IS. Exactly `transfer.manifest()` — the notes DB, the notes θ₀,
the checkpoint and its θ₀, the vocabulary, and the vault descriptor. The set is
consistent as a set: addresses live in the space θ₀ describes, so restoring the
database without its field would leave every address pointing into a field that
no longer exists. They move together or not at all.

⚠️ THREE WAYS A COPY SILENTLY BREAKS, all handled here:

  - THE WRITE-AHEAD LOG. A live SQLite database keeps committed rows in
    `-wal` that are not yet in the `.db`. Copying the file alone yields a
    store that looks fine and is quietly older. Snapshots go through sqlite's
    own `backup()` API, which is consistent against a database in use; on
    restore the destination's `-wal` and `-shm` are REMOVED, because leaving
    a newer log beside an older database is the same bug pointing the other
    way.

  - STALE ADDRESSES. Restoring an old θ₀ means the running field is no longer
    the one the addresses were filed against. `feat_gs` is cleared on restore,
    which `bridge.reindex` treats as maximally stale, so Mati re-addresses
    everything itself on next idle — against its own field, not a second copy
    loaded here.

  - THE KEYCHAIN. `mati_vault.json` is a descriptor, not a key. Restoring it
    onto a Mac whose keychain never held the key gives encrypted bodies and no
    way in; the recovery passphrase is the only route. Said plainly at restore
    rather than discovered later.

⚠️ NOTHING HERE MAY RUN WHILE MATI HOLDS THE FIELD. Two writers is one of the
three θ₀-corruption paths this project has already been bitten by, so every
mutating path goes through `exclusive.require`.
"""

import argparse
import json
import os
import shutil
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from . import config as C
from . import exclusive
from . import transfer


PREFIX = "backup-"
NOTE = "WHAT_THIS_IS.txt"
SIDECARS = ("mati_notes.db-wal", "mati_notes.db-shm")


def _root():
    return os.path.join(C.CACHE_DIR, "backups")


def _db(path):
    return os.path.join(path, "mati_notes.db")


def _snapshot_db(dest):
    """A consistent copy of the notes DB, even while it is being written to.

    `shutil.copy2` on a live SQLite file can catch a torn write or miss the
    write-ahead log entirely. `Connection.backup` is sqlite's own answer and
    takes a proper snapshot; it is also why this does not require the database
    to be idle, only that nothing is about to write θ₀ underneath us.
    """
    src_path = os.path.join(C.CACHE_DIR, "mati_notes.db")
    if not os.path.exists(src_path):
        return 0
    src = sqlite3.connect("file:" + src_path + "?mode=ro", uri=True)
    dst = sqlite3.connect(dest)
    try:
        with dst:
            src.backup(dst)
        return dst.execute("SELECT count(*) FROM notes").fetchone()[0]
    finally:
        src.close()
        dst.close()


def create(label=None, why=None, verbose=True):
    """Take a snapshot. Returns its path.

    Safe to call with Mati awake — nothing is written to the live store, and
    the database copy is consistent by construction.
    """
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = f"{PREFIX}{stamp}" + (f"-{_slug(label)}" if label else "")
    dest = os.path.join(_root(), name)
    os.makedirs(dest, exist_ok=True)

    notes = _snapshot_db(_db(dest))
    copied = ["mati_notes.db"] if notes else []
    missing = []
    for item in transfer.manifest():
        fname = item["name"]
        if fname == "mati_notes.db":
            continue
        src = os.path.join(C.CACHE_DIR, fname)
        if not os.path.exists(src):
            missing.append(fname)
            continue
        shutil.copy2(src, os.path.join(dest, fname))
        copied.append(fname)

    total = sum(os.path.getsize(os.path.join(dest, f)) for f in copied)
    meta = {"taken": time.time(), "when": time.strftime("%Y-%m-%d %H:%M:%S"),
            "notes": notes, "files": copied, "absent": missing,
            "bytes": total, "why": why or "manual"}
    with open(os.path.join(dest, NOTE), "w") as fh:
        fh.write(
            "A snapshot of Mati's whole memory.\n\n"
            f"taken   {meta['when']}\n"
            f"notes   {notes}\n"
            f"reason  {meta['why']}\n\n"
            "Restore with:  python3 -m Mati.backup --restore "
            f"{name}\n"
            "Mati must be asleep first. The database and the field are one\n"
            "set — restoring either without the other leaves every address\n"
            "pointing into a field that no longer exists.\n")
        fh.write("\n" + json.dumps(meta, indent=2) + "\n")

    if verbose:
        print(f"\n  snapshot: {name}")
        print(f"  {len(copied)} file(s), {total/1e6:.1f} MB, {notes} notes")
        for m in missing:
            print(f"    (absent in this install, skipped: {m})")
        print()
    return dest


def _slug(text):
    keep = [c if (c.isalnum() or c in "-_") else "-" for c in str(text).lower()]
    return "".join(keep).strip("-")[:32]


def read_meta(path):
    """The metadata block, or a best guess for a snapshot made by hand."""
    note = os.path.join(path, NOTE)
    if os.path.exists(note):
        text = open(note).read()
        brace = text.find("{")
        if brace >= 0:
            try:
                return json.loads(text[brace:])
            except json.JSONDecodeError:
                pass
    notes = None
    if os.path.exists(_db(path)):
        try:
            conn = sqlite3.connect("file:" + _db(path) + "?mode=ro", uri=True)
            notes = conn.execute("SELECT count(*) FROM notes").fetchone()[0]
            conn.close()
        except sqlite3.Error:
            pass
    return {"taken": os.path.getmtime(path), "notes": notes, "why": "unknown",
            "when": time.strftime("%Y-%m-%d %H:%M:%S",
                                  time.localtime(os.path.getmtime(path)))}


def existing():
    """Every snapshot, newest first. Includes ones made by hand."""
    root = _root()
    if not os.path.isdir(root):
        return []
    out = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if not os.path.isdir(path) or not os.path.exists(_db(path)):
            continue
        meta = read_meta(path)
        meta.update(name=name, path=path)
        out.append(meta)
    return sorted(out, key=lambda m: m.get("taken") or 0, reverse=True)


def find(name):
    """Resolve a snapshot by exact name, or by unique prefix."""
    have = existing()
    for m in have:
        if m["name"] == name:
            return m
    hits = [m for m in have if m["name"].startswith(name)]
    return hits[0] if len(hits) == 1 else None


def restore(name, force=False, verbose=True):
    """Put a snapshot back. Returns the path of the safety copy taken first.

    Raises FieldBusy if Mati still holds the field, KeyError if there is no
    such snapshot.
    """
    exclusive.require("restoring a backup", force=force)

    found = find(name)
    if not found:
        raise KeyError(name)
    src = found["path"]

    # The current state is itself worth keeping: a restore is the one command
    # here that destroys something, and "I restored the wrong one" must not be
    # a dead end.
    safety = create(label="before-restore", why=f"replaced by {found['name']}",
                    verbose=False)

    restored = []
    for item in transfer.manifest():
        fname = item["name"]
        s = os.path.join(src, fname)
        if not os.path.exists(s):
            continue
        shutil.copy2(s, os.path.join(C.CACHE_DIR, fname))
        restored.append(fname)

    # ⚠️ A write-ahead log left over from the running Mati describes the
    # database we just replaced. Leaving it would let sqlite replay newer
    # pages onto older content — the same "looks fine, quietly wrong" failure
    # the snapshot itself is taken to avoid, pointing the other way.
    for side in SIDECARS:
        stale = os.path.join(C.CACHE_DIR, side)
        if os.path.exists(stale):
            os.remove(stale)

    # Addresses were filed against the field that was here a moment ago. The
    # restored θ₀ is a different one, so every address is stale by definition.
    # Clearing feat_gs makes each row maximally stale in `bridge.reindex`'s
    # ranking, and Mati re-addresses them against its OWN field when it wakes.
    marked = 0
    db = os.path.join(C.CACHE_DIR, "mati_notes.db")
    if os.path.exists(db):
        conn = sqlite3.connect(db)
        try:
            conn.execute("UPDATE notes SET feat_gs=NULL")
            conn.commit()
            marked = conn.execute("SELECT count(*) FROM notes").fetchone()[0]
            # This write creates a NEW write-ahead log — a legitimate one
            # describing the restored database, but the goal is to leave
            # nothing beside the file that has to be replayed to be correct.
            # TRUNCATE folds it back in and empties it, so what is on disk
            # after a restore is one self-contained database.
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error:
            pass
        finally:
            conn.close()
    for side in SIDECARS:
        leftover = os.path.join(C.CACHE_DIR, side)
        if os.path.exists(leftover) and os.path.getsize(leftover) == 0:
            os.remove(leftover)

    if verbose:
        print(f"\n  restored {found['name']}  "
              f"({found.get('notes', '?')} notes, {len(restored)} file(s))")
        print(f"  the state you just replaced is kept: "
              f"{os.path.basename(safety)}")
        print(f"  {marked} address(es) marked stale — Mati re-files them "
              f"when it next idles")
        if os.path.exists(os.path.join(src, "mati_vault.json")):
            print("\n  NOTE: this snapshot carries an encrypted vault. If this "
                  "Mac's\n  keychain never held its key, the recovery "
                  "passphrase is the only way in.")
        print()
    return safety


def prune(keep=10, verbose=True):
    """Delete all but the newest `keep` snapshots. Returns how many went."""
    have = existing()
    doomed = have[keep:]
    for m in doomed:
        shutil.rmtree(m["path"], ignore_errors=True)
    if verbose:
        print(f"  removed {len(doomed)}, kept {len(have) - len(doomed)}")
    return len(doomed)


def main():
    ap = argparse.ArgumentParser(
        description="Snapshot Mati's memory, and go back to one")
    ap.add_argument("--label", help="a word to remember this one by")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--restore", metavar="NAME")
    ap.add_argument("--prune", type=int, metavar="KEEP")
    ap.add_argument("--force", action="store_true",
                    help="restore even if Mati holds the field (unsafe)")
    args = ap.parse_args()

    if args.list:
        have = existing()
        print(f"\n  {len(have)} snapshot(s) in {_root()}\n")
        for m in have:
            age = (time.time() - (m.get("taken") or 0)) / 3600
            print(f"    {m['name']:<44} {str(m.get('notes', '?')):>4} notes  "
                  f"{age:5.1f}h  {m.get('why', '')}")
        print()
        return 0

    if args.prune is not None:
        prune(args.prune)
        return 0

    if args.restore:
        try:
            restore(args.restore, force=args.force)
        except exclusive.FieldBusy as e:
            print(f"[refused] {e}", file=sys.stderr)
            return 2
        except KeyError:
            print(f"  no snapshot called {args.restore!r} — "
                  f"see --list", file=sys.stderr)
            return 1
        return 0

    create(label=args.label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
