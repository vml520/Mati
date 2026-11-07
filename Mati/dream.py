"""
Mati dreams — offline consolidation.

    python3 -m Mati.dream                 # 5 cycles, reindex, persist
    python3 -m Mati.dream --cycles 20
    python3 -m Mati.dream --dry-run       # report state, change nothing

Free-dream cycles deform θ₀ at `dream_alpha`, deepening basins the field has
settled into. Because every episode address is computed against θ₀, dreaming
ages the index — so `bridge.dream()` reindexes afterwards and persists the
result. Measured: 3 cycles → 0.031 address drift unrepaired; left to accumulate
it degrades retrieval into pure recency.

Run it when the machine is idle: it is cheap (~3s/cycle) but it competes with
the field's own work.
"""

import argparse
import sys

from . import bridge
from . import exclusive


def main():
    ap = argparse.ArgumentParser(description="Run Mati's dreaming consolidation")
    ap.add_argument("--cycles", type=int, default=5)
    ap.add_argument("--dream-length", type=int, default=12,
                    help="tokens generated per dream cycle")
    ap.add_argument("--dream-alpha", type=float, default=0.001,
                    help="θ₀ deformation per dream — keep small")
    ap.add_argument("--no-reindex", action="store_true",
                    help="skip the post-dream reindex (diagnostic only — leaves "
                         "the index stale)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report state and exit without dreaming")
    ap.add_argument("--force", action="store_true",
                    help="run even if Mati holds the field (unsafe)")
    args = ap.parse_args()

    # Exclusive access: it deforms θ₀ and persists it, and Mati must not be
    # holding it at the same time. See `exclusive`.
    rc = exclusive.guard("dreaming", force=args.force)
    if rc:
        return rc


    lm, tok = bridge.load_field()

    print(f"θ₀ source: {lm._theta_source.split('/')[-1]}")

    if args.dry_run:
        print("dry run — nothing changed")
        return 0

    print(f"dreaming {args.cycles} cycles (alpha={args.dream_alpha})...")
    res = bridge.dream(lm, tok,
                       cycles=args.cycles,
                       dream_length=args.dream_length,
                       dream_alpha=args.dream_alpha,
                       reindex_after=not args.no_reindex,
                       persist=True, verbose=True)

    print(f"\n  gs_std {res['gs_std_before']:.6f} → {res['gs_std_after']:.6f} "
          f"in {res['seconds']:.0f}s")
    if args.no_reindex:
        print("  !! index NOT repaired — addresses are now stale relative to θ₀",
              file=sys.stderr)
    else:
        print(f"  index repaired: {res['reindexed']} episodes re-addressed")
    print(f"  θ₀ persisted: {res['persisted']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
