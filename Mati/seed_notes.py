"""
Build Mati's notes DB from the seed corpus (`Mati/field/qa_data.py`).

    python3 -m Mati.seed_notes --limit 60          # quick, usable subset
    python3 -m Mati.seed_notes                     # all 285 (hours)
    python3 -m Mati.seed_notes --status            # progress, no work

Each note gets a triplet address = `coherence × |Δθ₀_std|` per mode, measured by
the notes module's own `_measure_triplet`, then is permanently imprinted into a
notes-specific θ₀ (`mati_notes_theta0.npz`) — the field is genuinely changed by
what it knows.

Cost: `imprint_question` runs one forward pass PER TOKEN, so this is linear in
tokens — roughly 1s/token on an 8GB Intel. That's why bodies are measured on the
first `config.TRIPLET_CAP` tokens while the FULL text is stored and recalled.
The same cap is applied to queries in `bridge.field_state` — if you change it,
existing addresses become incomparable and the DB must be rebuilt.

Resumable: notes are tagged `qa_data:<index>`, and an interrupted run skips what
it already wrote. θ₀ is checkpointed every `--save-every` notes.
"""

import argparse
import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .field import notes as N
from .field.qa_data import QA_PAIRS

from . import bridge
from . import exclusive
from . import config as C


def seedable_pairs():
    """Seed pairs, minus the '!' note-commands the training scripts exclude."""
    return [(q, a) for q, a in QA_PAIRS if not q.startswith('!')]


def open_db(path):
    conn = sqlite3.connect(path)
    conn.executescript(N._DDL)          # CREATE TABLE IF NOT EXISTS
    conn.commit()
    return conn


def already_seeded(conn):
    rows = conn.execute(
        "SELECT source FROM notes WHERE source LIKE 'qa_data:%'").fetchall()
    out = set()
    for (src,) in rows:
        try:
            out.add(int(src.split(':', 1)[1]))
        except (IndexError, ValueError):
            pass
    return out


def seed_one(conn, lm, tok, idx, question, answer, cap):
    """Measure address on the capped slice; store the FULL body."""
    body = f"{question}\n{answer}"
    ids  = tok.encode(body)[:cap]
    if len(ids) < 2:                    # imprint_question no-ops below 2 tokens
        return None

    w0, w1, w2 = N._measure_triplet(lm, ids)
    N._imprint_permanent(lm, ids)       # the field is changed by knowing this

    cur = conn.execute(
        """INSERT INTO notes
           (body, source, triplet_w0, triplet_w1, triplet_w2, gs_std, wall_created)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (body, f"qa_data:{idx}", w0, w1, w2, N._gs_std(lm), N._now()),
    )
    conn.commit()
    return cur.lastrowid, (w0, w1, w2)


def main():
    ap = argparse.ArgumentParser(description="Seed Mati's notes DB from QA_PAIRS")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N new notes (default: all)")
    ap.add_argument("--cap", type=int, default=C.TRIPLET_CAP,
                    help=f"tokens used to measure the address (default {C.TRIPLET_CAP}); "
                         "must match bridge.field_state's cap")
    ap.add_argument("--save-every", type=int, default=10,
                    help="checkpoint notes θ₀ every N notes")
    ap.add_argument("--status", action="store_true", help="report progress and exit")
    ap.add_argument("--force", action="store_true",
                    help="run even if Mati holds the field (unsafe)")
    args = ap.parse_args()

    # Exclusive access: it imprints episodes into θ₀, and Mati must not be
    # holding it at the same time. See `exclusive`.
    rc = exclusive.guard("seeding", force=args.force)
    if rc:
        return rc


    pairs = seedable_pairs()
    conn  = open_db(N.NOTES_DB_PATH)
    done  = already_seeded(conn)

    if args.status:
        total = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
        print(f"notes DB : {N.NOTES_DB_PATH}")
        print(f"rows     : {total}  (from qa_data: {len(done)}/{len(pairs)})")
        print(f"notes θ₀ : {N.NOTES_THETA0_PATH} "
              f"({'present' if os.path.exists(N.NOTES_THETA0_PATH) else 'not written yet'})")
        return 0

    if args.cap != C.TRIPLET_CAP:
        print(f"!! --cap {args.cap} != config.TRIPLET_CAP {C.TRIPLET_CAP}. Queries are "
              f"measured with the config value, so addresses written now would be "
              f"in a different space. Update config.TRIPLET_CAP too, or rebuild.",
              file=sys.stderr)

    todo = [(i, q, a) for i, (q, a) in enumerate(pairs) if i not in done]
    if args.limit:
        todo = todo[:args.limit]
    if not todo:
        print(f"nothing to do — {len(done)}/{len(pairs)} already seeded")
        return 0

    print(f"seeding {len(todo)} notes (cap={args.cap} tokens, "
          f"{len(done)} already done)")
    print(f"  est ~{args.cap * len(todo) / 60:.0f} min at ~1s/token")

    lm, tok = bridge.load_field()
    print(f"  field θ₀ from {os.path.basename(getattr(lm, '_theta_source', '?'))}")

    t0 = time.time()
    written = 0
    try:
        for n, (idx, q, a) in enumerate(todo, 1):
            res = seed_one(conn, lm, tok, idx, q, a, args.cap)
            if res is None:
                print(f"  [{n}/{len(todo)}] pair {idx}: too short, skipped")
                continue
            note_id, trip = res
            written += 1
            rate = (time.time() - t0) / n
            print(f"  [{n}/{len(todo)}] note #{note_id} (pair {idx}) "
                  f"triplet=({trip[0]:.5f}, {trip[1]:.5f}, {trip[2]:.5f}) "
                  f"{rate:.0f}s/note  eta {rate*(len(todo)-n)/60:.0f}min",
                  flush=True)
            if written % args.save_every == 0:
                N.save_field(lm)                 # crash-safe checkpoint
    except KeyboardInterrupt:
        print("\ninterrupted — progress is saved, rerun to resume", file=sys.stderr)
    finally:
        N.save_field(lm)
        conn.close()

    print(f"\nwrote {written} notes in {(time.time()-t0)/60:.1f} min")
    print(f"  db  → {N.NOTES_DB_PATH}")
    print(f"  θ₀  → {N.NOTES_THETA0_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
