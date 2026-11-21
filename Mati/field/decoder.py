# Part of Mati's field — the phase-field substrate its memory is built on.
#
# ⚠️ THIS IS THE AUTHORITATIVE COPY. Nothing outside Mati imports these
# modules, so the field can change without anything else moving under it.
#
# ⚠️ KEEP THIS A COMMENT, NOT A DOCSTRING. A docstring here would sit
# before `from __future__`, which is a SyntaxError that `ast.parse`
# does not report.
"""
Field state → token probability decoder.

Phase readout (Fix A+B+C):
    ctx_phase = arg( mean(e^{iθ}) )              order parameter phase of output nodes
    psi_s     = Σ_h (1/h) Re[e^{-ih·ctx} · z_h_s]   h-th harmonic resonance of sink s
    sink_w    = softmax(psi_s / T)               soft sink assignment (Fix C)
    ψ_k       = Σ_s sink_w_s · Σ_h (1/h) Re[e^{-ih·e_k} · z_h_s]
    p_k       = softmax(ψ / T)

Harmonics h ∈ {1,3,5,7} odd only (odd-k Fourier invariant).
Cap: include h only when 2h+1 ≤ N_out_per_sink.

T = T0·(1 − |⟨e^{iθ}⟩|) + T_floor   (sharp field → low T → decisive readout)
"""

import numpy as np
import sys, os
# (vendored: no sys.path hack — imports are package-relative)
from .base import SPACE_L


def find_output_nodes(field, boundary_ids):
    """
    Identify output face nodes: top quarter of z-positions, excluding
    boundary nodes (which sit at z=0).

    Returns sorted list of node indices.
    """
    candidates = [
        (i, field.nodes[i].position[2])
        for i in range(field.N)
        if i not in set(boundary_ids)
    ]
    z_thresh = SPACE_L / 4.0
    out = [i for i, z in candidates if z > z_thresh]
    if len(out) < 5:
        out = [i for i, _ in sorted(candidates, key=lambda t: -t[1])[:15]]
    return out


def decode_probs(field, output_nodes, vocab_size, temperature=1.0):
    """
    Decode next-token probability from output face phase state (h=1 only).
    Used in the non-RTR (W-projection) path.
    """
    if not output_nodes:
        return np.ones(vocab_size) / vocab_size

    thetas = np.array([field.nodes[i].theta for i in output_nodes], dtype=float)
    m_out  = np.mean(np.exp(1j * thetas))
    k_arr  = np.arange(vocab_size, dtype=float)
    e_k    = 2.0 * np.pi * k_arr / vocab_size
    psi    = np.real(np.exp(-1j * e_k) * m_out)
    psi_t  = psi / max(float(temperature), 1e-8)
    psi_t -= psi_t.max()
    probs  = np.exp(psi_t)
    probs /= probs.sum()
    return probs


def _valid_harmonics(harmonics, n_out_per_sink):
    """Subset of harmonics where 2h+1 ≤ n_out_per_sink."""
    valid = [h for h in harmonics if 2 * h + 1 <= n_out_per_sink]
    return valid if valid else [1]


def _sink_z(mode_nodes, region, out_set, h_valid):
    """
    Mass-weighted h-th order parameters for a sink region.
    Returns list of z_h (complex) for each h in h_valid, or zeros if empty.
    """
    r_out = [j for j in region if j in out_set]
    if not r_out:
        return [0j] * len(h_valid)
    th  = np.array([mode_nodes[j].theta for j in r_out], dtype=float)
    m   = np.array([mode_nodes[j].mass  for j in r_out], dtype=float)
    ms  = m.sum() + 1e-12
    return [np.sum(m * np.exp(1j * h * th)) / ms for h in h_valid]


def phase_dominant_sink(multi_field, out_nodes, sink_regions,
                        harmonics=(1, 3, 5, 7),
                        T0=0.10, T_floor=0.01,
                        return_h_contribs=False, return_psi=False):
    """
    Dominant sink by phase alignment + odd harmonics (Fix A+B+C).

    Uses mode 0 as the phase reference (coarsest scale, most stable).
    Returns: (dominant_idx, sink_weights, T)
      or, with return_h_contribs=True:
             (dominant_idx, sink_weights, T, h_contribs)
             where h_contribs = {h: [abs_contrib_per_sink, ...]}

    dominant_idx : int argmax of sink_weights
    sink_weights : softmax(psi_s / T), where psi_s is the harmonic resonance
                   of the field's current output-node order parameter with sink s
    T            : field-derived temperature from mode 0 coherence
    """
    n_sinks  = len(sink_regions)
    out_set  = set(out_nodes)
    n_per_sk = max(1, len(out_nodes) // n_sinks)
    h_valid  = _valid_harmonics(harmonics, n_per_sk)

    nodes0  = multi_field.modes[0].nodes
    th_out  = np.array([nodes0[j].theta for j in out_nodes], dtype=float)
    m1_out  = np.mean(np.exp(1j * th_out))
    ctx_ph  = float(np.angle(m1_out))
    T       = T0 * (1.0 - float(np.abs(m1_out))) + T_floor

    psi       = np.zeros(n_sinks)
    h_contrib = {h: [] for h in h_valid}   # h → [abs contribution per sink]
    for s, region in enumerate(sink_regions):
        zs = _sink_z(nodes0, region, out_set, h_valid)
        for hi, h in enumerate(h_valid):
            c = (1.0 / h) * float(np.real(np.exp(-1j * h * ctx_ph) * zs[hi]))
            psi[s] += c
            h_contrib[h].append(abs(c))

    shifted = psi - psi.max()
    weights = np.exp(shifted / max(T, 1e-6))
    weights /= weights.sum()
    if return_h_contribs and return_psi:
        return int(np.argmax(weights)), weights, T, h_contrib, psi
    if return_h_contribs:
        return int(np.argmax(weights)), weights, T, h_contrib
    if return_psi:
        return int(np.argmax(weights)), weights, T, psi
    return int(np.argmax(weights)), weights, T


def decode_triplet_resonance(multi_field, out_nodes, sink_regions, vocab_size,
                             harmonics=(1, 3, 5, 7),
                             T0=0.10, T_floor=0.01,
                             triplet=None):
    """
    RTR phase + odd-harmonic token readout (Fix A+B+C applied).

    Fix A: sink weights from phase resonance (not mass triplet)
    Fix B: odd harmonic sum with 1/h weights; capped at 2h+1 ≤ N_out_per_sink
    Fix C: soft sink assignment via softmax(psi_s / T)

    triplet : kept for API compatibility; not used in routing.
              (Mass entropy H is tracked separately via get_triplet().)
    """
    if not out_nodes:
        return np.ones(vocab_size) / vocab_size

    n_sinks  = len(sink_regions)
    n_modes  = multi_field.n_modes
    out_set  = set(out_nodes)
    n_per_sk = max(1, len(out_nodes) // n_sinks)
    h_valid  = _valid_harmonics(harmonics, n_per_sk)

    k_arr = np.arange(vocab_size, dtype=float)
    e_k   = 2.0 * np.pi * k_arr / vocab_size   # (V,)

    # ── Phase-based sink weights (Fix A, mode 0 reference) ──────────────────
    nodes0  = multi_field.modes[0].nodes
    th_out0 = np.array([nodes0[j].theta for j in out_nodes], dtype=float)
    m1_out0 = np.mean(np.exp(1j * th_out0))
    ctx_ph0 = float(np.angle(m1_out0))
    T       = T0 * (1.0 - float(np.abs(m1_out0))) + T_floor

    psi_ctx = np.zeros(n_sinks)
    for s, region in enumerate(sink_regions):
        zs = _sink_z(nodes0, region, out_set, h_valid)
        for hi, h in enumerate(h_valid):
            psi_ctx[s] += (1.0 / h) * float(
                np.real(np.exp(-1j * h * ctx_ph0) * zs[hi]))

    shifted_ctx = psi_ctx - psi_ctx.max()
    sink_w      = np.exp(shifted_ctx / max(T, 1e-6))
    sink_w     /= sink_w.sum()

    # ── Token score: Σ_mode coh_m · Σ_s sink_w_s · Σ_h (1/h) Re[e^{-ihe_k}·z_h_s]
    psi       = np.zeros(vocab_size)
    total_coh = 0.0

    for m_idx in range(n_modes):
        mode   = multi_field.modes[m_idx]
        nodes  = mode.nodes
        th_out = np.array([nodes[j].theta for j in out_nodes], dtype=float)
        coh_m  = float(np.abs(np.mean(np.exp(1j * th_out))))
        total_coh += coh_m

        psi_m = np.zeros(vocab_size)
        for s, region in enumerate(sink_regions):
            w_s = float(sink_w[s])
            zs  = _sink_z(nodes, region, out_set, h_valid)
            for hi, h in enumerate(h_valid):
                psi_m += w_s * (1.0 / h) * np.real(
                    np.exp(-1j * h * e_k) * zs[hi])

        psi += coh_m * psi_m

    if total_coh < 1e-12:
        return np.ones(vocab_size) / vocab_size

    psi   /= total_coh
    psi_t  = psi / max(T, 1e-6)
    psi_t -= psi_t.max()
    probs  = np.exp(psi_t)
    probs /= probs.sum()
    return probs


def sample_token(probs):
    """Sample one token ID from probability distribution."""
    return int(np.random.choice(len(probs), p=probs))


def top_k(probs, k=5):
    """Return top-k (token_id, probability) pairs, highest first."""
    idx = np.argsort(probs)[-k:][::-1]
    return [(int(i), float(probs[i])) for i in idx]
