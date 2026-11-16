# Part of Mati's field — the phase-field substrate its memory is built on.
#
# ⚠️ THIS IS THE AUTHORITATIVE COPY. Nothing outside Mati imports these
# modules, so the field can change without anything else moving under it.
#
# ⚠️ KEEP THIS A COMMENT, NOT A DOCSTRING. A docstring here would sit
# before `from __future__`, which is a SyntaxError that `ast.parse`
# does not report.
"""
Context encoder: token IDs → phase boundary conditions on the field.

Layout (reuses v20.3 8×8 boundary grid at z=0):
    The 64 boundary nodes are arranged as 8 rows × 8 columns in x-y.
    Each row l (0=oldest, 7=most recent) is locked to the phase of
    context token at position l.

    e_k = 2π · k / V    (reference phase for token k on S¹)

This is structurally identical to apply_boundary() in v20.3, extended
to L=8 spatial rows instead of a single pattern function.
"""

import numpy as np
import sys, os
# (vendored: no sys.path hack — imports are package-relative)
from .base import N_BOUNDARY, BOUNDARY_IDS

CONTEXT_LEN  = 8               # matches 8 rows of 8×8 boundary grid
NODES_PER_ROW = N_BOUNDARY // CONTEXT_LEN   # 8 nodes per context row


def token_phase(token_id, vocab_size):
    """Reference phase for token k: e_k = 2π·k/V  ∈ [0, 2π)."""
    return 2.0 * np.pi * float(token_id) / float(vocab_size)


def encode_context(field, context_ids, vocab_size, context_len=CONTEXT_LEN,
                   phases=None):
    """
    Apply context tokens as phase boundary conditions on the input face.

    context_ids : sequence of token IDs, most recent last.
    phases      : optional list (same length as context_ids) of pre-computed
                  phases.  When provided, UNK tokens (id=0) with a non-None
                  phase are encoded rather than skipped — compositional phase
                  encoding for unknown words.

    Pads with BOS (id=1) on the left if shorter than context_len.
    Returns: set of locked node indices.
    """
    ctx = list(context_ids[-(context_len):])
    while len(ctx) < context_len:
        ctx = [1] + ctx

    # Align phases with the padded context window
    if phases is not None:
        ph = list(phases[-(context_len):])
        while len(ph) < context_len:
            ph = [None] + ph
    else:
        ph = None

    locked = set()
    for row, tok_id in enumerate(ctx):
        if tok_id == 0:
            # UNK: use compositional phase if available, else skip
            if ph is not None and ph[row] is not None:
                phase = ph[row]
            else:
                continue
        else:
            phase = token_phase(tok_id, vocab_size)

        start = row * NODES_PER_ROW
        end   = start + NODES_PER_ROW
        for bid in BOUNDARY_IDS[start:end]:
            field.nodes[bid].theta = float(phase)
            locked.add(bid)

    return locked


class LearnedBoundaryEncoder:
    """
    Learned phase embedding: replaces the fixed e_k = 2π·k/V encoding.

    Each token k has a learned phase phi[k] ∈ [0, 2π) on S¹.
    Initialized to the uniform 2π·k/V schedule (same as fixed encoder),
    then refined during training via S¹ gradient descent.

    Update rule (circular gradient, field-native):
        For each context token k, given true target t and output features:
            circ_grad = Σ_mode Σ_node [ -sin(θ_out) · ∂L/∂cos_out
                                       + cos(θ_out) · ∂L/∂sin_out ]
            phi[k] ← (phi[k] − lr · circ_grad / (n_out · n_modes)) mod 2π

    The gradient flows backward from the projection loss through the
    output phase structure, estimating the direction on S¹ that would
    have improved the prediction — no backprop through field dynamics.
    """

    def __init__(self, vocab_size, lr=0.01, encoding_dim=1):
        """
        encoding_dim=1  (default, backward compat) — scalar phi[k] shared
                       across the row's 8 boundary nodes (identity-in-phase).
        encoding_dim=NODES_PER_ROW  (=8) — pattern-8D: phi[k] is an 8-vector,
                       node j in the row locks at θ₀ + phi[k, j]. The one to
                       use: capacity probes show baseline-1D resolves only
                       ~15-25 tokens per row.

        For encoding_dim>1: per-dimension stride-permuted init, each dim
        using a distinct coprime stride so token IDs decorrelate across
        codeword coordinates. Update: shared scalar S¹ gradient tiled
        across all encoding_dim entries per token (rigid rotation on
        S¹^encoding_dim — preserves the random-codebook separability
        proven at N=vocab_size while still following gradient direction).
        """
        from math import gcd
        self.vocab_size   = vocab_size
        self.lr           = lr
        self.encoding_dim = int(encoding_dim)

        def _coprime_stride(seed_val):
            stride = seed_val
            delta = 0
            while True:
                if gcd(stride + delta, vocab_size) == 1:
                    return stride + delta
                if gcd(stride - delta, vocab_size) == 1:
                    return stride - delta
                delta += 1

        if self.encoding_dim == 1:
            # golden-ratio stride (V/φ), scanned to coprimality
            stride = _coprime_stride(int(round(vocab_size / 1.6180339887)))
            assert gcd(stride, vocab_size) == 1
            self.phi = np.array(
                [2.0 * np.pi * ((k * stride) % vocab_size) / vocab_size
                 for k in range(vocab_size)],
                dtype=np.float64,
            )
        else:
            # per-dimension distinct coprime strides — spread codewords in
            # each of the encoding_dim coordinates independently.
            #
            # Seed strides at fractions of V (φ, √2, √3, √5, π, e, γ+1, ln2+1)
            # then scan for coprimality. Each yields a distinct permutation.
            seeds = [1.6180339887, 1.4142135624, 1.7320508076,
                     2.2360679775, 3.1415926536, 2.7182818285,
                     1.5772156649, 1.6931471806]
            self.phi = np.zeros((vocab_size, self.encoding_dim),
                                dtype=np.float64)
            for j in range(self.encoding_dim):
                s = seeds[j % len(seeds)]
                stride_j = _coprime_stride(int(round(vocab_size / s)))
                assert gcd(stride_j, vocab_size) == 1
                self.phi[:, j] = np.array(
                    [2.0 * np.pi * ((k * stride_j) % vocab_size) / vocab_size
                     for k in range(vocab_size)],
                    dtype=np.float64,
                )

    def encode_context(self, field, context_ids, context_len=CONTEXT_LEN,
                       phases=None):
        """
        Apply learned boundary phases to field.
        Drop-in replacement for encode_context() — no vocab_size arg needed.

        phases : optional list (same length as context_ids) of pre-computed
                 phases from encode_with_phases().  UNK tokens with a non-None
                 phase are encoded rather than skipped.
        """
        ctx = list(context_ids[-context_len:])
        while len(ctx) < context_len:
            ctx = [1] + ctx

        if phases is not None:
            ph = list(phases[-context_len:])
            while len(ph) < context_len:
                ph = [None] + ph
        else:
            ph = None

        locked = set()
        for row, tok_id in enumerate(ctx):
            if tok_id == 0:
                if ph is not None and ph[row] is not None:
                    phase = ph[row]
                else:
                    continue
                per_node_phase = None
            elif self.encoding_dim == 1:
                phase = float(self.phi[tok_id])
                per_node_phase = None
            else:
                phase = None
                per_node_phase = self.phi[tok_id]   # (encoding_dim,)

            start = row * NODES_PER_ROW
            end   = start + NODES_PER_ROW
            row_bids = BOUNDARY_IDS[start:end]
            for k, bid in enumerate(row_bids):
                if per_node_phase is not None:
                    field.nodes[bid].theta = float(per_node_phase[k])
                else:
                    field.nodes[bid].theta = phase
                locked.add(bid)
        return locked

    def update(self, context_ids, target_id, feat, proj_W, probs, n_out, n_modes):
        """
        S¹ gradient step: pull boundary phases toward better target prediction.

        Args:
            context_ids : token IDs used as context (last = most recent)
            target_id   : correct next-token ID
            feat        : output feature vector from forward pass, shape (n_modes·3·n_out,)
            proj_W      : projection weight matrix, shape (vocab_size, n_modes·3·n_out)
            probs       : softmax output, shape (vocab_size,)
            n_out       : number of output nodes
            n_modes     : number of field modes
        """
        # Feature-space gradient: ∂L/∂feat  (n_feat,)
        residual  = probs.copy()
        residual[target_id] -= 1.0
        grad_feat = proj_W.T @ residual   # (n_feat,)

        # Circular gradient: ∂L/∂phi = Σ [ -sin(θ_out)·∂L/∂cos + cos(θ_out)·∂L/∂sin ]
        circ_grad = 0.0
        for m in range(n_modes):
            off     = m * 3 * n_out
            cos_out = feat[off          : off + n_out]
            sin_out = feat[off + n_out  : off + 2 * n_out]
            gf_cos  = grad_feat[off          : off + n_out]
            gf_sin  = grad_feat[off + n_out  : off + 2 * n_out]
            circ_grad += float(np.sum(-sin_out * gf_cos + cos_out * gf_sin))

        # Normalise by output node count to keep lr scale-invariant
        circ_grad /= max(n_out * n_modes, 1)

        # Update all unique context tokens with the same gradient signal.
        # UNK (id=0) is never updated — it has no learned phase.
        # For encoding_dim > 1 the scalar circ_grad is tiled across all dims
        # (rigid rotation on S¹^encoding_dim — preserves random-codebook
        # separability while following gradient direction).
        step = self.lr * circ_grad
        for tok_id in set(context_ids):
            if tok_id == 0:
                continue
            self.phi[tok_id] = (self.phi[tok_id] - step) % (2.0 * np.pi)

    def save(self, path):
        np.save(path, self.phi)

    @classmethod
    def load(cls, path, vocab_size, lr=0.01):
        phi = np.load(path)
        encoding_dim = 1 if phi.ndim == 1 else phi.shape[1]
        enc = cls(vocab_size, lr=lr, encoding_dim=encoding_dim)
        enc.phi = phi
        return enc
