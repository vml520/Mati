"""
Mati is a thing you can carry.

    python3 -m Mati.transfer --check      # is this install portable right now?
    python3 -m Mati.transfer --manifest   # exactly what to copy

Everything Mati knows lives on this machine, in a handful of files totalling a
few megabytes. That is the whole point of the architecture and the reason for a
real shutdown: **a clean stop should leave a directory you can drag onto a
thumb drive and open somewhere else.** Sleep does not do that — it hands the
field to the pro terminal and the process stays up.

WHAT MAKES A COPY SAFE RATHER THAN MERELY COMPLETE. Three things, none obvious,
all of which produce a copy that *looks* fine:

  1. ⚠️ **THE DATABASE IS IN WAL MODE.** Committed data can sit in
     `mati_notes.db-wal` rather than in `mati_notes.db`. Copying the `.db`
     alone — which is exactly what dragging one file to a drive does — loses
     it, with no error at either end. `prepare()` runs a TRUNCATE checkpoint so
     the main file is self-contained and the sidecars are empty.

  2. ⚠️ **ADDRESSES GO STALE FASTER THAN THEY ARE REPAIRED.** Measured
: after roughly nine turns, 22 of 23 episodes were addressed
     against an older θ₀. Repair is bounded and runs only on idle, so an
     install handed over mid-session carries a store whose recall has quietly
     collapsed toward recency. `prepare()` reindexes to completion first —
     unbounded, because a transfer is exactly the moment worth paying for.

  3. ⚠️ **THE KEYCHAIN DOES NOT TRAVEL.** The vault key is wrapped twice: into
     this Mac's Secure Enclave behind Touch ID, and into the recovery
     passphrase. Only the second one is in the file. On the destination machine
     Touch ID cannot help and **the passphrase is the only way in** — so a
     transfer without it is a permanent loss of every memory, not an
     inconvenience. `vault.adopt()` re-wraps into the new machine's keychain
     once, after which Touch ID works there.

⚠️ **A DRIVE IS NOT A SAFE.** The vault encrypts memory *bodies*; addresses and
metadata stay in plaintext so recall can work without the key. That is a
deliberate trade documented in `vault.py`, and it means a lost thumb drive
leaks the retrieval index — roughly, what you talked about and when — even
though the text is sealed. Encrypt the volume if that matters.

WHAT DOES NOT TRAVEL, ON PURPOSE. Nothing here is machine-specific except the
keychain wrap, and no absolute paths are stored anywhere (verified: `ROOT` is
derived from this file's location and the database holds zero). So the same
directory works from `/Users/…`, from `/Volumes/…`, or from anywhere else.
"""

import argparse
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from . import bridge
from . import config as C

# ── the manifest ────────────────────────────────────────────────────────────
#
# Split by what losing it costs. The trained field can be downloaded again;
# the memories cannot be reconstructed by anyone, including us.
IRREPLACEABLE = [
    ("mati_notes.db", "every memory, correction and reminder"),
    ("mati_notes_theta0.npz", "θ₀ — the field shaped by those conversations"),
    # ⚠️ IRREPLACEABLE DESPITE BEING DERIVABLE-LOOKING. Every `feat` in the
    # database was computed in THIS space; arriving without it, the field falls
    # back to addressing against the live θ₀ and all of them become distances
    # between two different spaces — not worse matches, meaningless ones. It is
    # 9.6 KB and it cannot be recovered from the other files, only rebuilt by
    # re-addressing the whole working set at ~3.8s a note.
    ("mati_notes_theta0_address.npz", "the space those memories are filed in"),
    ("mati_vault.json", "the wrapped key; without it the memories stay sealed"),
]

REPLACEABLE = [
    ("diag_pattern8d_shuf_final.npz", "the trained checkpoint"),
    ("diag_pattern8d_shuf_final_theta0.npz", "the baseline θ₀"),
    ("qa_vocab.json", "the vocabulary"),
]

# Copying these is harmless but pointless, and copying the first two WITHOUT
# the main database is actively dangerous — they are meaningless alone.
SIDECARS = ["mati_notes.db-wal", "mati_notes.db-shm"]


def _cache(root=None):
    return os.path.join(root or C.ROOT, ".cache")


def manifest(root=None):
    """What travels, what it is, and whether it is here."""
    cache = _cache(root)
    out = []
    for group, items in (("irreplaceable", IRREPLACEABLE),
                         ("replaceable", REPLACEABLE)):
        for name, why in items:
            p = os.path.join(cache, name)
            out.append({"name": name, "why": why, "group": group,
                        "present": os.path.exists(p),
                        "bytes": os.path.getsize(p) if os.path.exists(p) else 0})
    return out


def checkpoint(db_path=None):
    """Fold the write-ahead log back into the main database file.

    Returns (busy, log_pages, checkpointed_pages). `busy` non-zero means a
    reader blocked the truncation and the sidecar still holds data — the copy
    would be incomplete, so this is reported rather than swallowed.
    """
    db_path = db_path or bridge.NOTES_DB_PATH
    if not os.path.exists(db_path):
        return None
    conn = sqlite3.connect(db_path)
    try:
        return tuple(conn.execute(
            "PRAGMA wal_checkpoint(TRUNCATE)").fetchone())
    finally:
        conn.close()


def prepare(lm=None, tok=None, reindex=True, verbose=True):
    """Make this install safe to copy. Returns what it did.

    Call with the field if you have it — a full reindex needs it. Without one
    the checkpoint still runs, which is the part that silently loses data.
    """
    did = {}

    if reindex and lm is not None and tok is not None:
        # Unbounded on purpose. Everywhere else reindex is capped to keep
        # turns responsive; a transfer is the one moment where finishing
        # matters more than latency.
        did["reindexed"] = bridge.reindex(lm, tok, limit=0)
        bridge.save_notes_field(lm)
        did["theta0_saved"] = True
    elif reindex:
        did["reindexed"] = None          # no field, so it could not be done
        did["warning"] = ("no field supplied — addresses NOT refreshed; the "
                          "copy may carry stale recall")

    ck = checkpoint()
    did["checkpoint"] = ck
    if ck is not None:
        # busy != 0 means a reader held it open and the -wal still has content.
        did["wal_clean"] = (ck[0] == 0)
        for s in SIDECARS:
            p = os.path.join(_cache(), s)
            if os.path.exists(p) and os.path.getsize(p) and s.endswith("-wal"):
                did["wal_clean"] = False

    if verbose:
        for line in describe(did):
            print(line, file=sys.stderr)
    return did


def describe(did):
    out = []
    if did.get("reindexed") is not None:
        out.append(f"  [transfer] re-addressed {did['reindexed']} episodes")
    if did.get("warning"):
        out.append(f"  [transfer] ⚠️  {did['warning']}")
    if "wal_clean" in did:
        out.append("  [transfer] write-ahead log folded in"
                   if did["wal_clean"] else
                   "  [transfer] ⚠️  the -wal still holds data — copy the "
                   "sidecars too, or stop everything and retry")
    return out


def check(root=None):
    """Would a copy of this directory work elsewhere? Returns (ok, findings)."""
    findings = []
    cache = _cache(root)

    missing = [m for m in manifest(root) if not m["present"]]
    for m in missing:
        # A missing vault is normal — it only exists if encryption was set up.
        if m["name"] == "mati_vault.json":
            findings.append(("info", "no vault — memories travel unencrypted"))
        elif m["group"] == "irreplaceable":
            findings.append(("fail", f"missing and unrecoverable: {m['name']}"))
        else:
            findings.append(("warn", f"missing: {m['name']} ({m['why']})"))

    wal = os.path.join(cache, "mati_notes.db-wal")
    if os.path.exists(wal) and os.path.getsize(wal):
        findings.append(("fail", f"the write-ahead log holds "
                                 f"{os.path.getsize(wal)} bytes not yet in the "
                                 f"database — run --prepare or copy it too"))

    if os.path.exists(os.path.join(cache, "mati_vault.json")):
        findings.append(("warn", "the keychain does NOT travel — on the new "
                                 "machine the recovery passphrase is the ONLY "
                                 "way in. Confirm it is written down first"))

    # Absolute paths inside the store would break relocation silently.
    db = os.path.join(cache, "mati_notes.db")
    if os.path.exists(db):
        conn = sqlite3.connect(db)
        try:
            n = 0
            for (t,) in conn.execute("SELECT name FROM sqlite_master "
                                     "WHERE type='table'"):
                for row in conn.execute(f"SELECT * FROM {t}"):
                    n += sum(1 for v in row if isinstance(v, str)
                             and ("/Users/" in v or "/Volumes/" in v))
            if n:
                findings.append(("warn", f"{n} absolute path(s) stored in the "
                                         f"database — informational, but they "
                                         f"will not resolve elsewhere"))
        finally:
            conn.close()

    ok = not any(level == "fail" for level, _ in findings)
    return ok, findings


def total_bytes(root=None):
    return sum(m["bytes"] for m in manifest(root))


def main():
    ap = argparse.ArgumentParser(description="Carry Mati somewhere else")
    ap.add_argument("--check", action="store_true",
                    help="is this install safe to copy right now?")
    ap.add_argument("--manifest", action="store_true", help="what to copy")
    ap.add_argument("--prepare", action="store_true",
                    help="checkpoint the database (does NOT reindex — the "
                         "server does that on /shutdown, where it has the field)")
    args = ap.parse_args()

    if args.prepare:
        prepare(reindex=False)
        print()

    if args.manifest or not (args.check or args.prepare):
        print(f"  what travels — {total_bytes()/1e6:.1f} MB\n")
        for group in ("irreplaceable", "replaceable"):
            print(f"  {group}:")
            for m in manifest():
                if m["group"] != group:
                    continue
                mark = "  " if m["present"] else "??"
                print(f"   {mark} {m['name']:<40}{m['bytes']/1e3:>8.1f} kB  "
                      f"{m['why']}")
            print()

    if args.check or not (args.manifest or args.prepare):
        ok, findings = check()
        print("  portability check\n")
        for level, msg in findings:
            print(f"   {'FAIL' if level=='fail' else level:>4}  {msg}")
        if not findings:
            print("    ok   nothing in the way")
        print(f"\n  {'safe to copy' if ok else 'NOT safe to copy'}")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
