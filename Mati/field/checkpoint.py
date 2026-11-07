# Part of Mati's field — the phase-field substrate its memory is built on.
#
# ⚠️ THIS IS THE AUTHORITATIVE COPY. These modules began as a shared study
# codebase and were split off so the field could change without anything
# else moving under it. Nothing outside Mati imports them.
#
# ⚠️ KEEP THIS A COMMENT, NOT A DOCSTRING. A docstring here would sit
# before `from __future__`, which is a SyntaxError that `ast.parse`
# does not report.
"""
Field checkpoint: self-contained bundle of W + phi + vocab.

The field carries its own vocabulary — loading a checkpoint gives you a
complete, runnable LM without any separate JSON or weight files.

Format: .npz with keys
    W           — projection weights (vocab_size × n_feat)
    phi         — learned boundary phases (vocab_size,), or zeros if fixed
    vocab_json  — BPESubsetTokenizer serialisation as a single string item
    use_gs      — 1 if gs_features active, 0 otherwise

Usage:
    from .checkpoint import save_checkpoint, load_checkpoint

    # Save after training
    save_checkpoint('.cache/field.npz', lm, tokenizer)

    # Load in chat or inference script
    lm, tok, _ = load_checkpoint('.cache/field.npz', mf)
"""

import numpy as np
import json
import os
import sys
import signal
import uuid
# (vendored: no sys.path hack — imports are package-relative)
def atomic_savez(path, **arrays):
    """Write npz atomically via temp+rename — old or new file always intact."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    # Temp name ends in .npz so np.savez won't append another .npz suffix
    tmp = os.path.join(d, f'.tmp_{uuid.uuid4().hex}.npz')
    try:
        np.savez(tmp, **arrays)
        with open(tmp, 'rb') as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


class InterruptGuard:
    """Write a final checkpoint on SIGTERM/SIGINT, then exit cleanly."""
    def __init__(self, save_fn):
        self.save_fn   = save_fn
        self.triggered = False
        signal.signal(signal.SIGTERM, self._handler)
        signal.signal(signal.SIGINT,  self._handler)

    def _handler(self, signum, frame):
        if self.triggered:
            os._exit(1)
        self.triggered = True
        print(f"\n[InterruptGuard] signal {signum} — saving final checkpoint...",
              flush=True)
        try:
            self.save_fn()
            print("[InterruptGuard] checkpoint saved. Safe to exit.", flush=True)
        except BaseException as e:
            print(f"[InterruptGuard] save failed: {e}", flush=True)
        finally:
            os._exit(0)


def save_checkpoint(path, lm, tokenizer, field_seed=None):
    """
    Bundle W + phi + vocab + field_seed into a single .npz file.

    Args:
        path        : output path (should end in .npz)
        lm          : MultiModeFieldLM instance
        tokenizer   : BPESubsetTokenizer instance
        field_seed  : integer seed used to build the MultiModeField geometry.
                      Required for reproducible cloning. If None, attempts to
                      read from lm.mf.seed — falls back to 0 (unknown).
    """
    phi = lm.encoder.phi if lm.encoder is not None \
          else np.zeros(lm.vocab_size)

    # Serialise tokenizer to JSON bytes
    vocab_data = {
        'max_vocab':  tokenizer.max_vocab,
        'vocab_size': tokenizer.vocab_size,
        'id_map':     {str(k): v for k, v in tokenizer._id_map.items()},
    }
    vocab_json = json.dumps(vocab_data)

    # Learned node frequencies (mode 0 — same across all modes)
    omegas = lm.mf.get_omegas() if hasattr(lm, 'mf') else np.zeros(0)

    # Field seed — the single integer that reproducibly generates geometry
    if field_seed is None:
        field_seed = getattr(getattr(lm, 'mf', None), 'seed', 0) or 0

    traj_steps = list(getattr(lm, 'traj_sample_steps', [5, 10, 20, 30, 40]))

    arrays = dict(
        W           = lm.proj.W,
        phi         = phi,
        vocab_json  = np.array([vocab_json]),
        use_gs      = np.array([1 if lm.use_gs_features else 0]),
        use_traj    = np.array([1 if getattr(lm, 'use_traj_features', False) else 0]),
        traj_steps  = np.array(traj_steps, dtype=np.int32),
        omegas      = omegas,
        field_seed  = np.array([int(field_seed)]),
    )
    if hasattr(lm, 'mf') and lm.mf.hebbian:
        for i, J in enumerate(lm.mf.get_j_arrays()):
            arrays[f'J_{i}'] = J

    atomic_savez(path, **arrays)
    print(f"  Checkpoint saved → {path}")
    print(f"    W={lm.proj.W.shape}  phi={phi.shape}"
          f"  vocab={tokenizer.vocab_size}"
          f"  gs={'on' if lm.use_gs_features else 'off'}"
          f"  seed={field_seed}")


def load_checkpoint(path, mf, context_len=8, steps=20,
                    temperature=1.0, lm_alpha_gs=0.05,
                    proj_lr=0.1, enc_lr=0.01):
    """
    Restore LM + tokenizer from a checkpoint file.

    Args:
        path        : .npz checkpoint path
        mf          : pre-built MultiModeField (same geometry as when saved)
        context_len : context window length
        steps       : field evolution steps
        temperature : sampling temperature
        lm_alpha_gs : ground state deformation rate
        proj_lr     : projection learning rate
        enc_lr      : encoder learning rate (> 0 enables learned encoding)

    Returns:
        (lm, tokenizer)
    """
    from .field_lm import MultiModeFieldLM
    from .tokenizer_bpe import BPESubsetTokenizer

    data      = np.load(path, allow_pickle=True)
    W         = data['W']
    phi       = data['phi']
    use_gs    = bool(data['use_gs'][0])
    vocab_str = str(data['vocab_json'][0])

    # Restore traj config if stored; otherwise infer from W shape
    if 'use_traj' in data:
        use_traj   = bool(data['use_traj'][0])
        traj_steps = list(data['traj_steps'].tolist()) if 'traj_steps' in data \
                     else [5, 10, 20, 30, 40]
    else:
        # Legacy checkpoint: infer from W columns vs default traj_steps
        _default_traj = [5, 10, 20, 30, 40]
        from .decoder import find_output_nodes
        from .base import BOUNDARY_IDS
        _n_out = len(find_output_nodes(mf.modes[0], set(BOUNDARY_IDS)))
        use_traj   = (W.shape[1] == len(_default_traj) * _n_out)
        traj_steps = _default_traj if use_traj else [5, 10, 20, 30, 40]

    vocab_data = json.loads(vocab_str)

    # Rebuild tokenizer using the same path as BPESubsetTokenizer.load()
    tok = BPESubsetTokenizer(max_vocab=vocab_data['max_vocab'])
    tok._id_map   = {int(k): v for k, v in vocab_data['id_map'].items()}
    tok._rev_map  = {0: None, 1: None}
    tok._rev_map.update({v: int(k) for k, v in tok._id_map.items()})
    tok.vocab_size = vocab_data['vocab_size']
    vocab_size = tok.vocab_size

    # Build LM
    lm = MultiModeFieldLM(
        mf,
        vocab_size        = vocab_size,
        context_len       = context_len,
        steps             = steps,
        temperature       = temperature,
        lm_alpha_gs       = lm_alpha_gs,
        proj_lr           = proj_lr,
        enc_lr            = enc_lr,
        use_gs_features   = use_gs,
        use_traj_features = use_traj,
        traj_sample_steps = traj_steps,
    )

    # Load weights — warn only if still mismatched after correct feature config
    if W.shape == lm.proj.W.shape:
        lm.proj.W = W
    else:
        print(f"  WARNING: checkpoint W shape {W.shape} ≠ expected"
              f" {lm.proj.W.shape} — forcing W shape")
        lm.proj.W      = W
        lm.proj.n_feat = W.shape[1]

    if lm.encoder is not None and phi is not None:
        lm.encoder.phi = phi
        # Sync encoding_dim to loaded phi shape so encode_context takes the
        # correct branch (2D phi → pattern-per-node, 1D → shared scalar).
        lm.encoder.encoding_dim = 1 if phi.ndim == 1 else phi.shape[1]

    # Restore learned node frequencies if present
    if 'omegas' in data and data['omegas'].size > 0:
        mf.set_omegas(data['omegas'])

    # Restore Hebbian J arrays if present
    j_arrays = []
    for i in range(mf.n_modes):
        key = f'J_{i}'
        if key in data:
            j_arrays.append(data[key].astype(np.float32))
    if j_arrays:
        mf.set_j_arrays(j_arrays)

    # Restore field seed — 0 means unknown (checkpoint predates this feature)
    field_seed = int(data['field_seed'][0]) if 'field_seed' in data else 0

    print(f"  Checkpoint loaded ← {path}")
    print(f"    W={W.shape}  vocab={vocab_size}"
          f"  gs={'on' if use_gs else 'off'}"
          f"  seed={field_seed if field_seed else 'unknown'}")
    return lm, tok, field_seed
