# Part of Mati's field — the phase-field substrate its memory is built on.
#
# ⚠️ THIS IS THE AUTHORITATIVE COPY. Nothing outside Mati imports these
# modules, so the field can change without anything else moving under it.
#
# ⚠️ KEEP THIS A COMMENT, NOT A DOCSTRING. A docstring here would sit
# before `from __future__`, which is a SyntaxError that `ast.parse`
# does not report.
"""
Phase-native personal knowledge base.

Notes are stored by their physical address in the field's topological
space — a triplet signature (w0, w1, w2) measured at the three attractor
basins of the phase field at the moment of imprinting.

There is no scoring, no decay, no dormancy, no background computation.

The database is a pure index of physical addresses.  Retrieval is
nearest-neighbor search in triplet space R³.

'list'  — ledger view, insertion order, no field computation
'find'  — imprint query, rank all notes by triplet distance
'like'  — use a note's stored triplet as query point

Every note permanently reshapes θ₀.  The field accumulates the full
history of what has been written into it — even notes that sink to the
bottom of a search are still present in the ground state.

Uses the trained checkpoint when there is one, and the vocabulary alone
otherwise.

Usage:
    python3 -m Mati.field.notes

Commands:
    note            — enter a new note (blank line to finish)
    load <path>     — import all .txt files from a folder
    list            — ledger: all notes in insertion order
    find <query>    — rank notes by resonance with query
    like <id>       — rank notes by resonance with note #id
    stats           — field and library statistics
    save            — save ground state now
    quit            — save and exit
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sqlite3
import datetime as dt
import numpy as np

from .modes         import MultiModeField
from .field_lm      import MultiModeFieldLM
from .tokenizer_bpe import BPESubsetTokenizer
from .trainer       import embed_constitution
from .defects       import precompute_triangles, update_defect_registry
from .checkpoint    import load_checkpoint
from .data          import CACHE_DIR

# ── Paths ─────────────────────────────────────────────────────────────────────
VOCAB_PATH        = os.path.join(CACHE_DIR, 'qa_vocab.json')
# The FINAL checkpoint, matching Mati (`Mati/config.CKPT_PATH`), never the
# rolling one. This CLI writes the same NOTES_THETA0_PATH that Mati lives in,
# and the moment training resumes the rolling checkpoint diverges — notes would
# be addressed with a different encoder while sharing Mati's θ₀. Addresses are
# only comparable within one field, so two checkpoints writing one θ₀ corrupts
# it. Only this module's own CLI reads this — `Mati/bridge.py` imports paths
# from `Mati/config.py`, not from here.
CKPT_PATH         = os.path.join(CACHE_DIR, 'diag_pattern8d_shuf_final.npz')
NOTES_THETA0_PATH = os.path.join(CACHE_DIR, 'mati_notes_theta0.npz')
NOTES_DB_PATH     = os.path.join(CACHE_DIR, 'mati_notes.db')

# ── Field hyperparameters ─────────────────────────────────────────────────────
MAX_RHO          = 0.05
ETA_DECOHERE     = 0.005
DEFECT_THR       = 0.4
INFER_STEPS      = 80
TEMPERATURE      = 0.85
IMPRINT_PASSES   = 3
IMPRINT_STRENGTH = 5.0

CONSTITUTION = [
    "Never betray yourself.",
    "Never deceive yourself.",
    "Never exploit yourself.",
    "Always choose harmony.",
]

# ── Schema ────────────────────────────────────────────────────────────────────
_DDL = """
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS notes (
    note_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    body         TEXT    NOT NULL,
    source       TEXT,               -- 'cli' | filename if loaded from folder
    -- Physical address in triplet space R³.
    -- Each coordinate is the per-mode θ₀ std change induced by this note,
    -- weighted by coherence.  Three independent spatial-scale responses:
    --   w0: fine mode  (ξ=2.0) — character/local activation
    --   w1: medium mode (ξ=3.0) — word/semantic activation
    --   w2: coarse mode (ξ=4.5) — phrase/global activation
    triplet_w0   REAL    NOT NULL,
    triplet_w1   REAL    NOT NULL,
    triplet_w2   REAL    NOT NULL,
    -- Field state at imprint time — provenance record.
    gs_std       REAL,               -- mean θ₀ std when note was imprinted
    wall_created TEXT                -- ISO-8601, metadata only
);

CREATE INDEX IF NOT EXISTS idx_notes_insertion
    ON notes(note_id ASC);
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')


def _gs_std(lm) -> float:
    """
    Mean θ₀ standard deviation across all modes.
    Scalar summary of the field's current memory depth.
    """
    stds = [th.std() for th in lm.mf.theta_0s if th is not None]
    return float(np.mean(stds)) if stds else 0.0


def _imprint_permanent(lm, text_ids):
    """
    Permanently deform θ₀ toward the attractor basin this text drives.
    The field is measurably different after this call.
    """
    lm.imprint_question(
        text_ids,
        n_passes         = IMPRINT_PASSES,
        imprint_strength = IMPRINT_STRENGTH,
        steps            = 20,
    )


def _measure_triplet(lm, text_ids) -> tuple:
    """
    Measure the triplet signature of a text against the current field.

    Uses a temporary imprint (snapshot → imprint → measure → restore)
    so θ₀ is not permanently modified by this call.

    The triplet (w0, w1, w2) is the note's physical address in the
    field's topological space.  Each coordinate is the per-mode θ₀
    standard deviation CHANGE induced by this text, weighted by output
    coherence.  Three independent dimensions:
        w0 — fine mode  (ξ=2.0): character/local response
        w1 — medium mode (ξ=3.0): word/semantic response
        w2 — coarse mode (ξ=4.5): phrase/global response

    Notes that activate the same spatial scales cluster together in R³,
    regardless of date or shared keywords.
    """
    snap = lm.snapshot_ground_state()
    stds_before = [float(th.std()) for th in snap[0]]

    lm.imprint_question(text_ids, n_passes=1,
                        imprint_strength=IMPRINT_STRENGTH, steps=10)

    seed = text_ids[:1] or [1]
    _, _, states, _ = lm.forward(seed, steps=INFER_STEPS)
    coherence = float(lm.output_coherence(states))

    stds_after = [float(th.std()) for th in lm.mf.theta_0s]
    w0 = coherence * abs(stds_after[0] - stds_before[0])
    w1 = coherence * abs(stds_after[1] - stds_before[1]) if len(stds_after) > 1 else 0.0
    w2 = coherence * abs(stds_after[2] - stds_before[2]) if len(stds_after) > 2 else 0.0

    lm.restore_ground_state(snap)
    return (w0, w1, w2)


def _triplet_distance(a: tuple, b: tuple) -> float:
    """Euclidean distance between two triplet addresses in R³."""
    return float(np.sqrt(sum((x - y) ** 2 for x, y in zip(a, b))))


def _resonance(distance: float) -> float:
    """
    Convert triplet distance to a resonance score in (0, 1].
    Closer in triplet space → higher resonance.
    """
    return 1.0 / (1.0 + distance * 10.0)


# ── Display ───────────────────────────────────────────────────────────────────

def _local_date(stored) -> str:
    """The stored UTC stamp as a LOCAL date. Naive stamps are treated as UTC."""
    import datetime as _dt
    if not stored:
        return ''
    try:
        t = _dt.datetime.fromisoformat(str(stored))
    except (TypeError, ValueError):
        return str(stored)[:10]          # unreadable: show it rather than blank
    if t.tzinfo is None:
        t = t.replace(tzinfo=_dt.timezone.utc)
    return t.astimezone().strftime('%Y-%m-%d')


def _bar(value: float, width: int = 12) -> str:
    filled = max(0, min(width, int(round(value * width))))
    return '█' * filled + '░' * (width - filled)


def _display_note(row: dict, resonance: float = None):
    """
    Render one note to stdout.

    If resonance is provided (find/like mode) it is shown as a phase bar.
    In ledger mode (list) no bar is shown — insertion order only.
    """
    # ⚠️ LOCAL. `wall_created` is UTC with an offset, so slicing it printed the
    # UTC date — anything after ~17:00 local was filed under tomorrow. Done
    # inline rather than via `activity.local_parts` on purpose: this module is
    # the vendored field and does not import the rest of Mati.
    date = _local_date(row['wall_created'])
    src     = f"  [{row['source']}]" \
              if row['source'] and row['source'] != 'cli' else ''
    preview = row['body'][:120].replace('\n', ' ')
    if len(row['body']) > 120:
        preview += '…'

    if resonance is not None:
        bar = _bar(resonance)
        print(f"\n  #{row['note_id']:4d}  {bar}  {resonance:.3f}  {date}{src}")
    else:
        print(f"\n  #{row['note_id']:4d}  {date}{src}")

    print(f"        {preview}")
    print(f"        ↳ ({row['triplet_w0']:.4f}, "
          f"{row['triplet_w1']:.4f}, {row['triplet_w2']:.4f})")


def _header(title: str):
    width = 54
    print(f"\n  ┌{'─' * (width - 2)}┐")
    print(f"  │  {title:<{width - 4}}│")
    print(f"  └{'─' * (width - 2)}┘")


# ── Core operations ───────────────────────────────────────────────────────────

def save_note(conn, lm, tok, body: str, source: str = 'cli'):
    """
    Measure the note's triplet address, imprint it permanently into θ₀,
    and record it in the database.

    The triplet is measured before the permanent imprint so it reflects
    the note's address in the field as it was when the note arrived —
    not the field after it has already been changed by the note.
    """
    text_ids = tok.encode(body)
    if not text_ids:
        print("  [empty — not saved]")
        return None

    gs_before     = _gs_std(lm)
    w0, w1, w2    = _measure_triplet(lm, text_ids)
    _imprint_permanent(lm, text_ids)
    gs_after      = _gs_std(lm)
    imprint_depth = gs_after - gs_before

    cursor = conn.execute(
        """INSERT INTO notes
           (body, source, triplet_w0, triplet_w1, triplet_w2,
            gs_std, wall_created)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (body, source, w0, w1, w2, gs_after, _now())
    )
    conn.commit()
    note_id = cursor.lastrowid

    print(f"  Note #{note_id} saved  "
          f"[depth={imprint_depth:+.4f}  "
          f"triplet=({w0:.4f}, {w1:.4f}, {w2:.4f})]")
    return note_id


def load_folder(conn, lm, tok, folder_path: str):
    """
    Import all .txt files from a folder as notes.

    Each file is imprinted sequentially, permanently accumulating into θ₀.
    After loading, the field's ground state encodes the full corpus.
    """
    if not os.path.isdir(folder_path):
        print(f"  Not a directory: {folder_path}")
        return

    files = sorted(f for f in os.listdir(folder_path) if f.endswith('.txt'))
    if not files:
        print(f"  No .txt files in {folder_path}")
        return

    print(f"  Loading {len(files)} files...")
    saved = 0
    for fname in files:
        fpath = os.path.join(folder_path, fname)
        try:
            with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                body = f.read().strip()
            if body:
                note_id = save_note(conn, lm, tok, body, source=fname)
                if note_id:
                    saved += 1
        except OSError as e:
            print(f"    ERROR {fname}: {e}")

    print(f"  {saved}/{len(files)} notes saved from {folder_path}")
    save_field(lm)


def cmd_list(conn):
    """
    Display all notes in insertion order — pure ledger, no field computation.

    Insertion order is the only sort here.  All resonance-based ordering
    belongs to find and like.  The ledger is stable and cheap.
    """
    rows = conn.execute(
        "SELECT * FROM notes ORDER BY note_id ASC"
    ).fetchall()

    if not rows:
        print("\n  No notes saved yet.")
        return

    _header(f"ALL NOTES  ({len(rows)})  —  insertion order")
    for row in rows:
        _display_note(dict(row))
    print()


def cmd_find(conn, lm, tok, query: str):
    """
    Rank all notes by phase resonance with a query.

    The query is temporarily imprinted (θ₀ unchanged) and its triplet
    address measured.  All notes are then ranked by Euclidean distance
    from that address in triplet space R³.

    Notes that drove the field toward the same attractor basin as the
    query surface first — regardless of date, source, or shared keywords.
    """
    text_ids = tok.encode(query)
    if not text_ids:
        print("  [empty query]")
        return

    print("  [measuring query triplet...]", flush=True)
    q_triplet = _measure_triplet(lm, text_ids)

    rows = conn.execute("SELECT * FROM notes").fetchall()
    if not rows:
        print("\n  No notes to search.")
        return

    ranked = sorted(
        rows,
        key=lambda r: _triplet_distance(
            q_triplet,
            (r['triplet_w0'], r['triplet_w1'], r['triplet_w2'])
        )
    )

    _header(f"FIND: '{query[:42]}'")
    print(f"\n  Query triplet: ({q_triplet[0]:.4f}, "
          f"{q_triplet[1]:.4f}, {q_triplet[2]:.4f})\n")

    for row in ranked:
        dist = _triplet_distance(
            q_triplet,
            (row['triplet_w0'], row['triplet_w1'], row['triplet_w2'])
        )
        _display_note(dict(row), resonance=_resonance(dist))
    print()


def cmd_like(conn, note_id: int):
    """
    Rank all other notes by phase resonance with note #note_id.

    Uses the stored triplet of the seed note as the query point —
    no field computation needed at all.  Pure nearest-neighbor search
    in the stored triplet index.  The field already did its work when
    the note was saved.
    """
    seed = conn.execute(
        "SELECT * FROM notes WHERE note_id = ?", (note_id,)
    ).fetchone()

    if seed is None:
        print(f"  Note #{note_id} not found.")
        return

    seed_triplet = (seed['triplet_w0'], seed['triplet_w1'], seed['triplet_w2'])

    rows = conn.execute(
        "SELECT * FROM notes WHERE note_id != ?", (note_id,)
    ).fetchall()

    if not rows:
        print("\n  No other notes to compare.")
        return

    ranked = sorted(
        rows,
        key=lambda r: _triplet_distance(
            seed_triplet,
            (r['triplet_w0'], r['triplet_w1'], r['triplet_w2'])
        )
    )

    _header(f"SIMILAR TO #{note_id}")
    print(f"\n  Seed: {seed['body'][:80].replace(chr(10), ' ')}")
    print(f"  Triplet: ({seed_triplet[0]:.4f}, "
          f"{seed_triplet[1]:.4f}, {seed_triplet[2]:.4f})\n")

    for row in ranked:
        dist = _triplet_distance(
            seed_triplet,
            (row['triplet_w0'], row['triplet_w1'], row['triplet_w2'])
        )
        _display_note(dict(row), resonance=_resonance(dist))
    print()


def cmd_stats(conn, lm):
    """Field and library statistics."""
    n_notes = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
    gs_std  = _gs_std(lm)

    print(f"\n  Notes   : {n_notes}")
    print(f"  GS std  : {gs_std:.4f}  (θ₀ memory depth)")
    print(f"  Defects : {len(lm.mf.defect_nodes)}")
    print(f"  Const.  : {len(lm.mf.constitution_nodes)} nodes frozen")

    if n_notes > 0:
        rows  = conn.execute(
            "SELECT triplet_w0, triplet_w1, triplet_w2 FROM notes"
        ).fetchall()
        trips    = np.array([(r[0], r[1], r[2]) for r in rows])
        centroid  = trips.mean(axis=0)
        distances = np.linalg.norm(trips - centroid, axis=1)
        print("\n  Triplet space:")
        print(f"    Centroid  : ({centroid[0]:.4f}, "
              f"{centroid[1]:.4f}, {centroid[2]:.4f})")
        print(f"    Mean dist : {distances.mean():.4f}  "
              f"(spread from centroid)")
        print(f"    Max dist  : {distances.max():.4f}  "
              f"(most isolated note)")
    print()


# ── Ground state persistence ──────────────────────────────────────────────────

def save_field(lm):
    """
    Atomically persist the notes ground state θ₀.

    Uses os.replace() for atomic write on POSIX — the live file is
    never in a partially-written state.  The notes field accumulates
    a distinct experiential history from the chat field.
    """
    arrays = {}
    for i in range(lm.mf.n_modes):
        arrays[f'theta_{i}'] = lm.mf.theta_0s[i]
        arrays[f'mass_{i}']  = lm.mf.mass_0s[i]
    arrays['omegas'] = lm.mf.get_omegas()

    tmp = NOTES_THETA0_PATH.replace('.npz', '_saving.npz')
    np.savez(tmp, **arrays)
    # ⚠️ OWNER-ONLY ON EVERY WRITE, NOT ONCE AT SETUP. `harden_permissions()`
    # runs during first-run setup, but this file is REWRITTEN on every save and
    # would come back at the umask default — 0644, world-readable, beside a
    # notes DB and vault that are 0600, on a machine where the whole point is
    # that the field lives on one disk and goes nowhere.
    #
    # Set on the temp file BEFORE the rename, so the live path is never briefly
    # readable: os.replace is atomic and carries the mode with it.
    os.chmod(tmp, 0o600)
    os.replace(tmp, NOTES_THETA0_PATH)
    print(f"  Notes θ₀ saved → {NOTES_THETA0_PATH}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("╔══════════════════════════════════════════════════╗")
    print("║  Mati Notes  —  Phase-native knowledge base    ║")
    print("║  Notes stored by attractor address, not date    ║")
    print("║  find / like rank by triplet resonance in R³    ║")
    print("╚══════════════════════════════════════════════════╝\n")

    # EXCLUSIVE ACCESS. This CLI writes NOTES_THETA0_PATH — the same θ₀ Mati
    # lives in — on quit, on save, and after importing a folder. Mati must not
    # be holding it, or one of the two sets of writes disappears with no error
    # and no way to tell which. See `Mati/exclusive.py`.
    try:
        from Mati import exclusive as _exclusive
        _exclusive.require("mati_notes")
    except ImportError:
        pass                                    # Mati not installed alongside
    except Exception as _e:                     # FieldBusy
        print(f"[refused] {_e}")
        sys.exit(2)

    # ── Build field ───────────────────────────────────────────────────────────
    print("--- Building field ---")
    mf = MultiModeField(n_modes=3, xi_base=2.0, sigma_base=1.0,
                        scale=1.5, seed=42)
    mf.compute_ground_states(steps=200)
    for mode in mf.modes:
        mode.max_rho      = MAX_RHO
        mode.eta_decohere = ETA_DECOHERE

    # ── Load checkpoint ───────────────────────────────────────────────────────
    if os.path.exists(CKPT_PATH):
        print("\n--- Loading checkpoint ---")
        lm, tok, _ = load_checkpoint(
            CKPT_PATH, mf,
            context_len = 8,
            steps       = INFER_STEPS,
            temperature = TEMPERATURE,
            lm_alpha_gs = 0.05,
            proj_lr     = 0.1,
            enc_lr      = 0.01,
        )
    else:
        if not os.path.exists(VOCAB_PATH):
            print("ERROR: no checkpoint or vocabulary found.")
            print("  No vocabulary yet — imprint the QA set first.")
            sys.exit(1)
        print("\n--- Loading vocabulary ---")
        tok = BPESubsetTokenizer.load(VOCAB_PATH)
        lm  = MultiModeFieldLM(
            mf,
            vocab_size  = tok.vocab_size,
            context_len = 8,
            steps       = INFER_STEPS,
            temperature = TEMPERATURE,
            lm_alpha_gs = 0.05,
            proj_lr     = 0.1,
            enc_lr      = 0.01,
        )

    # ── Constitution + defects ────────────────────────────────────────────────
    print("\n--- Embedding constitution ---")
    const_nodes = embed_constitution(
        lm, CONSTITUTION, tokenizer=tok,
        weight=3.0, steps_per_fwd=20, n_passes=2,
    )
    mf.constitution_nodes = const_nodes
    triangles = precompute_triangles(mf)
    update_defect_registry(mf, triangles, threshold=DEFECT_THR)
    print(f"  Defects: {len(mf.defect_nodes)}  "
          f"Constitution: {len(mf.constitution_nodes)} nodes frozen")

    # ── Load notes θ₀ ─────────────────────────────────────────────────────────
    if os.path.exists(NOTES_THETA0_PATH):
        print("\n--- Loading notes ground state ---")
        data = np.load(NOTES_THETA0_PATH)
        for i in range(lm.mf.n_modes):
            if f'theta_{i}' in data:
                lm.mf.theta_0s[i] = data[f'theta_{i}']
                lm.mf.mass_0s[i]  = data[f'mass_{i}']
        if 'omegas' in data:
            lm.mf.set_omegas(data['omegas'])
        print(f"  Notes field resumed ← {NOTES_THETA0_PATH}")
    else:
        print("\n  No prior notes field — starting fresh.")

    # ── Open notes DB ─────────────────────────────────────────────────────────
    conn = sqlite3.connect(NOTES_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(_DDL)
    conn.commit()

    n_notes = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
    print(f"\n  Library: {n_notes} notes  |  GS std: {_gs_std(lm):.4f}")

    # ── REPL ──────────────────────────────────────────────────────────────────
    print("\n╔══════════════════════════════════════════════════╗")
    print("║  note · load <path> · list · find <query>       ║")
    print("║  like <id> · stats · save · quit                ║")
    print("╚══════════════════════════════════════════════════╝\n")

    while True:
        try:
            raw = input("notes> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  [saving...]")
            save_field(lm)
            conn.close()
            print("Goodbye.")
            break

        if not raw:
            continue

        parts = raw.split(None, 1)
        cmd   = parts[0].lower()
        arg   = parts[1].strip() if len(parts) > 1 else ''

        if cmd == 'quit':
            save_field(lm)
            conn.close()
            print("Goodbye.")
            break

        elif cmd == 'note':
            print("  Enter note (blank line to finish):")
            lines = []
            while True:
                try:
                    line = input("  > ")
                except (EOFError, KeyboardInterrupt):
                    break
                if line == '':
                    break
                lines.append(line)
            body = '\n'.join(lines).strip()
            if body:
                save_note(conn, lm, tok, body)
            else:
                print("  [empty — not saved]")

        elif cmd == 'load':
            if not arg:
                print("  Usage: load <folder_path>")
            else:
                load_folder(conn, lm, tok, arg)

        elif cmd == 'list':
            cmd_list(conn)

        elif cmd == 'find':
            if not arg:
                print("  Usage: find <query text>")
            else:
                cmd_find(conn, lm, tok, arg)

        elif cmd == 'like':
            try:
                cmd_like(conn, int(arg))
            except ValueError:
                print("  Usage: like <note_id>")

        elif cmd == 'stats':
            cmd_stats(conn, lm)

        elif cmd == 'save':
            save_field(lm)

        else:
            print(f"  Unknown: '{cmd}'")
            print("  note · load <path> · list · find <query> · "
                  "like <id> · stats · save · quit")
