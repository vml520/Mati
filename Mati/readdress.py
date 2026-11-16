"""
Re-file every working-set address into the space Mati is currently reading.

    python3 -m Mati.readdress            # against the real store
    python3 -m Mati.readdress --dry-run  # just say what is owed

⚠️ NOT A USER COMMAND, AND NOT PART OF NORMAL OPERATION. The night already
does this: `rest()` promotes the live field and queues a bounded reindex
against it. Nothing here needs running on a healthy install.

⚠️ WHAT IT IS FOR IS A FIELD THAT ARRIVED FROM SOMEWHERE ELSE — a restored
backup, or a store copied to another machine. A stored `feat` is only
meaningful in the space it was computed in, so addresses written against
another machine's θ₀ are not worse matches, they are distances measured
between two different spaces. `C.REINDEX_ON_IDLE` is off (field addressing
scores 4/111 against lexical's 93/111), so nothing repairs that in the
background, and `rest()` re-files only after a promotion of its own.

⚠️ THE SERVER MUST NOT BE HOLDING THE FIELD. It takes the exclusive lock; a
running Mati owns it, and two processes writing θ₀ is how a field gets lost.
Stop Mati first, or use `POST /reindex`.
"""

from __future__ import annotations

import sys
import time


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    dry = "--dry-run" in argv

    from . import bridge, vault
    from . import config as C

    if not vault.unlocked():
        vault.unlock(reason="re-file memory addresses")

    lm, tok = bridge.load_field()
    # ⚠️ There is one mode now, so this reports WHICH frozen space is pinned
    # rather than which of two architectures is running.
    print(f"  addressing against frozen space: "
          f"{getattr(lm, '_address_space', None)}")

    stale, total = bridge.settling(lm)
    print(f"  working set: {total}    owed: {stale}")
    if not total:
        print("  nothing to do — empty working set")
        return 0
    if not stale:
        print("  nothing to do — every address is already in this space")
        return 0
    if dry:
        print(f"  --dry-run: would re-file {stale} at ~3.8s each "
              f"(~{stale * 3.8 / 60:.0f} min)")
        return 0

    # ⚠️ A LOOP, FOR THE SAME REASON `_reindex_after_promote` IS ONE.
    # `bridge.reindex` commits what it finished and can return with a
    # remainder; one call is not a completed job. It terminates because the
    # space it repairs toward holds still while this runs.
    t0, done, passes = time.time(), 0, 0
    while passes < 50:
        passes += 1
        n = bridge.reindex(lm, tok, limit=0)
        done += n
        if not n:
            break
        print(f"    pass {passes}: {done}/{stale}  "
              f"[{time.time() - t0:.0f}s]", flush=True)

    bridge.save_notes_field(lm)
    left, total = bridge.settling(lm)
    print(f"  re-filed {done} in {passes} pass(es), {time.time() - t0:.0f}s")
    print(f"  owed now: {left} of {total}")
    # ⚠️ REPORT THE RESIDUAL RATHER THAN CLAIMING SUCCESS. A row that cannot
    # be addressed at all would otherwise leave this printing a tidy total and
    # exiting 0 while recall stayed broken for it.
    if left:
        print(f"  ⚠️ {left} STILL OWED — those rows could not be addressed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
