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
Topological defect detection for the phase field.

A topological defect (vortex) exists where the phase field winds by ±2π
around a closed loop. In the graph-based field these are detected via
triangle plaquettes: for each triple of mutually-coupled nodes (i, j, k),
compute the winding number W = (Δθ_ij + Δθ_jk + Δθ_ki) / 2π.

|W| > 0.4  →  defect plaquette  →  all three nodes pinned permanently.

Defect nodes are topologically protected:
    - Excluded from mode.evolve() via the locked set
    - Excluded from ground state deformation
    - Cannot be erased by subsequent training

This gives a hard long-term memory layer (defects) alongside the
soft ground-state deformation memory.
"""

import numpy as np


# ── Triangle plaquette finding ─────────────────────────────────────────────────

def find_triangles(W, threshold=0.0):
    """
    Find all closed triangles in the coupling graph.

    Returns list of (i, j, k) with i < j < k and all pairwise
    couplings W[i,j], W[j,k], W[i,k] > threshold.
    """
    N         = W.shape[0]
    triangles = []
    for i in range(N):
        nbrs_i = np.where(W[i] > threshold)[0]
        for j in nbrs_i:
            if j <= i:
                continue
            if W[i, j] <= threshold:
                continue
            nbrs_j = np.where(W[j] > threshold)[0]
            for k in nbrs_j:
                if k <= j:
                    continue
                if W[i, k] > threshold and W[j, k] > threshold:
                    triangles.append((int(i), int(j), int(k)))
    return triangles


def _wrap(angle):
    """Wrap angle to [-π, π]."""
    return ((angle + np.pi) % (2.0 * np.pi)) - np.pi


# ── Defect detection ───────────────────────────────────────────────────────────

def detect_defects(mode, triangles=None, threshold=0.4, thetas=None):
    """
    Detect topological vortex defects via winding number on triangle plaquettes.

    Args:
        mode      : ModeField — field with .W and .nodes[i].theta
        triangles : precomputed list of (i, j, k); computed here if None
        threshold : |W| > threshold classifies as defect (default 0.4)
        thetas    : optional phase array (N,) to use instead of current node
                    thetas.  Pass mf.theta_0s[i] to detect defects on the
                    ground state rather than the current evolved state.

    Returns:
        set of node indices that are part of at least one defect plaquette
    """
    if triangles is None:
        triangles = find_triangles(mode.W)

    if thetas is None:
        thetas = np.array([n.theta for n in mode.nodes])

    defect_nodes = set()

    for (i, j, k) in triangles:
        dij     = _wrap(thetas[j] - thetas[i])
        djk     = _wrap(thetas[k] - thetas[j])
        dki     = _wrap(thetas[i] - thetas[k])
        winding = (dij + djk + dki) / (2.0 * np.pi)
        if abs(winding) > threshold:
            defect_nodes.update([i, j, k])

    return defect_nodes


# ── Defect registry update ─────────────────────────────────────────────────────

def update_defect_registry(mf, triangles_per_mode=None, threshold=0.4,
                           thetas_per_mode=None):
    """
    Scan all modes for new defects; permanently add them to mf.defect_nodes.

    Defects are monotonically accumulated — once pinned they cannot be removed.
    This is topological protection: a winding number can't be smoothed away
    without crossing an energy barrier.

    Args:
        mf                : MultiModeField with .defect_nodes set attribute
        triangles_per_mode: list of precomputed triangle lists per mode
        threshold         : winding number threshold
        thetas_per_mode   : optional list of phase arrays (one per mode).
                            When provided, defects are detected on these
                            phases (e.g. mf.theta_0s) rather than current
                            node thetas.  Use this to detect ground-state
                            topology, not transient evolved topology.

    Returns:
        int — number of newly pinned defect nodes
    """
    before = len(mf.defect_nodes)
    for i, mode in enumerate(mf.modes):
        tris   = triangles_per_mode[i] if triangles_per_mode else None
        thetas = thetas_per_mode[i]    if thetas_per_mode    else None
        found  = detect_defects(mode, tris, threshold, thetas=thetas)
        mf.defect_nodes |= found
    return len(mf.defect_nodes) - before


def precompute_triangles(mf):
    """
    Precompute triangle plaquettes for all modes (done once at init).
    Stores result in mf._triangles and returns it.
    """
    triangles = [find_triangles(mode.W) for mode in mf.modes]
    mf._triangles = triangles
    total = sum(len(t) for t in triangles)
    print(f"  Precomputed {total} triangles across {mf.n_modes} modes"
          f"  (defect detection ready)")
    return triangles
