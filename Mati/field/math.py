# Part of Mati's field — the phase-field substrate its memory is built on.
#
# ⚠️ THIS IS THE AUTHORITATIVE COPY. Nothing outside Mati imports these
# modules, so the field can change without anything else moving under it.
#
# ⚠️ KEEP THIS A COMMENT, NOT A DOCSTRING. A docstring here would sit
# before `from __future__`, which is a SyntaxError that `ast.parse`
# does not report.
"""
Mati/field/math.py
=============
Virtual quantum computer for Mati.

Architecture
------------
Each computation spawns a fresh, isolated TFT field — a graph of N phase
oscillators on S¹, coupled by a Gaussian weight matrix W derived from ℓ₀.
The field runs the discrete XY (Kuramoto) step rule:

    θᵢ(t+1) = θᵢ(t) + α · Σⱼ Wᵢⱼ · sin(θⱼ(t) − θᵢ(t)) + ωᵢ

Key insight (Vic): eₖ = 2πk/V is the DFT basis on S¹. The field's natural
Kuramoto dynamics therefore perform quantum-native computation — Grover
diffusion and QFT are phase-field operations, not approximations of quantum
circuits.

The LM field is never touched. This module returns clean integers.

Public API
----------
grover_search(field, target_node, n_iter=None) -> int
shor_period(field, a, N)                        -> int
factor(N)                                       -> tuple[int, int]
spawn_field(n_nodes, E0, l0, alpha)             -> MathField
VirtualQC                                       — stateful inspector class

"""

from __future__ import annotations

import ast as _ast
import math
import time
from dataclasses import dataclass, field
from math import gcd
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Physical constants — default values; all derivable from E₀ and ℓ₀
# ---------------------------------------------------------------------------

DEFAULT_E0    = 1.0          # phase-energy scale
DEFAULT_L0    = 0.3          # coherence length
DEFAULT_ALPHA = 0.05         # coupling step size
DEFAULT_OMEGA_SCALE = 0.01   # intrinsic frequency noise scale


# ---------------------------------------------------------------------------
# Core field
# ---------------------------------------------------------------------------

class MathField:
    """
    An isolated TFT phase field for mathematical computation.

    State
    -----
    theta   : (N,)  current phase angles on S¹
    omega   : (N,)  intrinsic frequencies
    W       : (N,N) Gaussian coupling matrix
    mass    : (N,)  accumulated carrier inflow (initialised to 1)
    E0, l0  : first-class physical scales
    alpha   : coupling step size
    step    : step counter

    The field is purely NumPy — no dependency on the rest of Mati.
    """

    def __init__(
        self,
        n_nodes : int   = 64,
        E0      : float = DEFAULT_E0,
        l0      : float = DEFAULT_L0,
        alpha   : float = DEFAULT_ALPHA,
        geometry: str   = "ring",        # ring | grid | random
        seed    : Optional[int] = None,
    ):
        self.n    = n_nodes
        self.E0   = E0
        self.l0   = l0
        self.alpha = alpha
        self.step  = 0
        self.rng   = np.random.default_rng(seed)

        self.pos   = self._make_positions(geometry)
        self._build_coupling()
        self.omega = self.rng.normal(0.0, DEFAULT_OMEGA_SCALE, self.n)
        self.theta = self.rng.uniform(0.0, 2 * np.pi, self.n)
        self.mass  = np.ones(self.n)

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def _make_positions(self, geometry: str) -> np.ndarray:
        """Return (N, 3) node positions in R³."""
        n = self.n
        if geometry == "ring":
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            r = n / (2 * np.pi)
            return np.column_stack([r * np.cos(angles),
                                    r * np.sin(angles),
                                    np.zeros(n)])
        if geometry == "grid":
            side = int(math.ceil(math.sqrt(n)))
            xs = np.arange(n) % side
            ys = np.arange(n) // side
            return np.column_stack([xs.astype(float),
                                    ys.astype(float),
                                    np.zeros(n)])
        # random
        return self.rng.uniform(0, math.sqrt(n), (n, 3))

    def _make_coupling(self) -> np.ndarray:
        """Gaussian coupling W_ij = exp(−|xᵢ−xⱼ|² / 2ℓ₀²), as a dense matrix.

        Kept as the reference implementation the sparse path is checked
        against; `_build_coupling` produces the same numbers without ever
        allocating N×N. Costs O(N²) memory — call it on small fields only.
        """
        diff = self.pos[:, np.newaxis, :] - self.pos[np.newaxis, :, :]
        dist2 = np.sum(diff ** 2, axis=-1)
        W = np.exp(-dist2 / (2 * self.l0 ** 2))
        np.fill_diagonal(W, 0.0)
        # normalise rows so total coupling per node = 1
        row_sum = W.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        return W / row_sum

    # ------------------------------------------------------------------
    # Sparse coupling
    #
    # The Gaussian is narrow: with the default ℓ₀=0.3 and nodes spread over a
    # box of side √N, a node's second-nearest neighbour is already ~1e-7 of its
    # nearest and the rest underflow. Measured at N=4096, 99.95% of the matrix
    # is exact zero, and the fraction falls as N grows. Storing it densely cost
    # O(N²) — 2.1GB at N=16102 — to hold almost nothing.
    #
    # Entries are kept as flat row/col/value triples. Rows still sum to ~1: an
    # entry is dropped only if it is below `_W_REL_TOL` times its own row's
    # largest, so retained values are bit-for-bit the dense ones and every row
    # keeps at least its maximum (no node is ever orphaned).
    # ------------------------------------------------------------------

    _W_REL_TOL   = 1e-15        # relative to each row's largest entry
    _W_SEARCH_TOL = 1e-18       # neighbour search goes wider than storage, so
                                # the mass omitted from each row sum is well
                                # below the values actually kept
    _W_GRID_MIN_N = 2048        # below this the blocked scan is already quick

    def _build_coupling(self) -> None:
        """Build the sparse coupling, never allocating N×N.

        Two paths, checked against each other and against `_make_coupling`:
        a spatial hash grid that only ever examines nearby nodes (O(N) for a
        spread-out cloud), and a blocked exhaustive scan used for small fields
        and as the fallback whenever the grid cannot be trusted.
        """
        if self.n >= self._W_GRID_MIN_N:
            built = self._build_coupling_grid()
            if built is not None:
                self._set_sparse(*built)
                return
        self._build_coupling_blocked()

    def _build_coupling_grid(self):
        """Sparse coupling via a spatial hash grid, or None if not applicable.

        The Gaussian means node i only couples to j within
        d² < d_min(i)² + 2ℓ₀²·ln(1/tol), so bucketing space into cells of that
        radius and scanning each node's 27-cell neighbourhood finds every
        entry worth keeping. Returns None if any node's neighbourhood turns out
        too small to justify that claim, so the caller falls back to the exact
        scan rather than silently dropping couplings.
        """
        n, pos = self.n, self.pos
        k_search = 2 * self.l0 ** 2 * math.log(1.0 / self._W_SEARCH_TOL)

        # Robust spacing estimate: exact nearest-neighbour distance for a
        # sample of nodes, brute-forced against all of them. Makes no
        # assumption about the cloud's dimensionality, so ring (1-D in R³) and
        # grid (2-D) geometries are handled the same as random.
        rs   = np.random.default_rng(0)
        samp = rs.choice(n, size=min(256, n), replace=False)
        d2   = np.sum((pos[samp][:, None, :] - pos[None, :, :]) ** 2, axis=-1)
        d2[np.arange(len(samp)), samp] = np.inf
        d_hi = float(np.sqrt(d2.min(axis=1).max()))
        # 10% margin: a node isolated beyond anything in the sample would
        # otherwise need a wider search than the cells provide. Those are
        # repaired individually below rather than failing the whole build.
        h    = 1.1 * math.sqrt(d_hi ** 2 + k_search)
        if not np.isfinite(h) or h <= 0:
            return None

        cell  = np.floor((pos - pos.min(axis=0)) / h).astype(np.int64) + 1
        dims  = cell.max(axis=0) + 2
        if float(dims[0]) * float(dims[1]) * float(dims[2]) > 4e18:
            return None                                   # would overflow int64
        code  = cell[:, 0] + dims[0] * (cell[:, 1] + dims[1] * cell[:, 2])

        uniq, inv = np.unique(code, return_inverse=True)
        order  = np.argsort(inv, kind="stable")
        counts = np.bincount(inv, minlength=len(uniq))
        starts = np.concatenate([[0], np.cumsum(counts)])[:-1]

        src_all, dst_all = [], []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    tgt = code + dx + dims[0] * (dy + dims[1] * dz)
                    idx = np.searchsorted(uniq, tgt)
                    ok  = (idx < len(uniq)) & (uniq[np.minimum(idx, len(uniq) - 1)] == tgt)
                    cnt = np.where(ok, counts[np.minimum(idx, len(uniq) - 1)], 0)
                    total = int(cnt.sum())
                    if total == 0:
                        continue
                    src = np.repeat(np.arange(n), cnt)
                    within = (np.arange(total)
                              - np.repeat(np.cumsum(cnt) - cnt, cnt))
                    dst = order[np.repeat(starts[np.minimum(idx, len(uniq) - 1)],
                                          cnt) + within]
                    src_all.append(src)
                    dst_all.append(dst)

        src = np.concatenate(src_all)
        dst = np.concatenate(dst_all)
        keep_self = src != dst
        src, dst = src[keep_self], dst[keep_self]

        d2e = np.sum((pos[src] - pos[dst]) ** 2, axis=-1)

        # A node is only correctly covered if its true influence radius,
        # sqrt(d_min² + k_search), fits inside the searched neighbourhood.
        # Nodes that found nothing, or that are isolated enough to reach past
        # it, get their rows recomputed exhaustively — there are normally none.
        d2min = np.full(n, np.inf)
        np.minimum.at(d2min, src, d2e)
        stragglers = np.nonzero(~(np.sqrt(d2min + k_search) <= h))[0]
        if len(stragglers) > 0.05 * n:
            return None                     # estimate was bad; use exact scan
        if len(stragglers):
            keep_covered = ~np.isin(src, stragglers)
            src, dst, d2e = src[keep_covered], dst[keep_covered], d2e[keep_covered]
            s_diff = pos[stragglers][:, None, :] - pos[None, :, :]
            s_d2   = np.sum(s_diff ** 2, axis=-1)
            s_d2[np.arange(len(stragglers)), stragglers] = np.inf
            s_row  = np.repeat(stragglers, n - 1)
            s_col  = np.tile(np.arange(n), len(stragglers)).reshape(len(stragglers), n)
            mask   = s_col != stragglers[:, None]
            src    = np.concatenate([src, s_row])
            dst    = np.concatenate([dst, s_col[mask]])
            d2e    = np.concatenate([d2e, s_d2[mask]])

        w = np.exp(-d2e / (2 * self.l0 ** 2))
        row_sum = np.bincount(src, weights=w, minlength=n)
        row_sum[row_sum == 0] = 1.0
        w = w / row_sum[src]
        row_max = np.zeros(n)
        np.maximum.at(row_max, src, w)
        keep = w > self._W_REL_TOL * row_max[src]
        return src[keep], dst[keep], w[keep]

    def _build_coupling_blocked(self) -> None:
        """Exact coupling by blocked exhaustive scan — O(N²) time, O(N) memory."""
        n = self.n
        # Block sized so the (block, N, 3) difference array stays ~64MB. The
        # explicit difference is deliberate: the |x|²+|y|²−2x·y expansion is
        # faster but loses absolute precision in d², and on ring geometry
        # (radius N/2π) that error survives the exponential.
        block = max(1, min(n, int(64e6 // max(1, n * 24))))
        rows, cols, vals = [], [], []
        two_l0_sq = 2 * self.l0 ** 2
        for start in range(0, n, block):
            stop  = min(start + block, n)
            diff  = self.pos[start:stop, np.newaxis, :] - self.pos[np.newaxis, :, :]
            w     = np.exp(-np.sum(diff ** 2, axis=-1) / two_l0_sq)
            # no self-coupling
            w[np.arange(stop - start), np.arange(start, stop)] = 0.0
            row_sum = w.sum(axis=1, keepdims=True)
            row_sum[row_sum == 0] = 1.0
            w /= row_sum
            keep = w > (self._W_REL_TOL * w.max(axis=1, keepdims=True))
            r, c = np.nonzero(keep)
            rows.append(r + start)
            cols.append(c)
            vals.append(w[keep])
        self._set_sparse(np.concatenate(rows), np.concatenate(cols),
                         np.concatenate(vals))

    def _set_sparse(self, rows, cols, vals) -> None:
        self._w_rows = np.asarray(rows, dtype=np.intp)
        self._w_cols = np.asarray(cols, dtype=np.intp)
        self._w_vals = np.asarray(vals, dtype=float)

    @property
    def W(self) -> np.ndarray:
        """The coupling as a dense matrix — O(N²), for diagnostics only.

        The field itself never materialises this; it works from the sparse
        triples. Reading `.W` on a large field will allocate N² floats.
        """
        dense = np.zeros((self.n, self.n))
        dense[self._w_rows, self._w_cols] = self._w_vals
        return dense

    @W.setter
    def W(self, matrix: np.ndarray) -> None:
        """Accept a dense coupling (e.g. a loaded Hamiltonian) and sparsify it.

        Same rule as `_build_coupling`, so a genuinely dense matrix survives
        intact — it simply gains no benefit.
        """
        matrix = np.asarray(matrix, dtype=float)
        row_max = np.abs(matrix).max(axis=1, keepdims=True)
        keep = np.abs(matrix) > (self._W_REL_TOL * row_max)
        r, c = np.nonzero(keep)
        self._set_sparse(r, c, matrix[keep])

    def _edge_deltas(self) -> np.ndarray:
        """θⱼ − θᵢ for each stored edge (i=row, j=col)."""
        return self.theta[self._w_cols] - self.theta[self._w_rows]

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def evolve(self, steps: int = 1) -> "MathField":
        """Run the discrete XY step rule for `steps` iterations.

        The coupling term is expanded rather than formed as a matrix:

            Σⱼ Wᵢⱼ sin(θⱼ − θᵢ) = cos θᵢ (W·sin θ)ᵢ − sin θᵢ (W·cos θ)ᵢ

        This is an identity, not an approximation — it agrees with the naïve
        form to float roundoff. It matters because the naïve form allocated
        three N×N temporaries (`delta`, `sin(delta)`, their product) on every
        step, which is what actually made large fields unusable. Combined with
        the sparse coupling the step is O(edges) in both time and memory.
        """
        n = self.n
        rows, cols, vals = self._w_rows, self._w_cols, self._w_vals
        for _ in range(steps):
            s, c = np.sin(self.theta), np.cos(self.theta)
            ws = np.bincount(rows, weights=vals * s[cols], minlength=n)
            wc = np.bincount(rows, weights=vals * c[cols], minlength=n)
            coupling = c * ws - s * wc
            self.theta = (self.theta + self.alpha * coupling + self.omega) % (2 * np.pi)
            self.mass  += np.abs(coupling)
        self.step += steps
        return self

    def set_theta(self, theta: np.ndarray) -> "MathField":
        """Directly write phase angles (boundary condition / oracle)."""
        assert len(theta) == self.n
        self.theta = np.array(theta) % (2 * np.pi)
        return self

    def kick(self, node: int, delta: float) -> "MathField":
        """Apply a phase kick to a single node (oracle operation)."""
        self.theta[node] = (self.theta[node] + delta) % (2 * np.pi)
        return self

    def reset_theta(self, seed: Optional[int] = None) -> "MathField":
        """Reinitialise phases without rebuilding W."""
        rng = np.random.default_rng(seed)
        self.theta = rng.uniform(0, 2 * np.pi, self.n)
        self.mass  = np.ones(self.n)
        self.step  = 0
        return self

    # ------------------------------------------------------------------
    # Observables
    # ------------------------------------------------------------------

    def coherence_pressure(self) -> float:
        """𝒫_τ = mean |∇θ|² proxy — mean squared phase difference over edges.

        Summed over stored edges only; the dropped entries are zero, so this is
        the same mean over all N² as before — still divided by N².
        """
        delta = self._edge_deltas()
        return float(np.sum(self._w_vals * delta ** 2) / self.n ** 2)

    def entropy(self) -> float:
        """S_τ = mean (1 − cos|∇θ|) over edges."""
        delta = self._edge_deltas()
        return float(np.sum(self._w_vals * (1 - np.cos(delta))) / self.n ** 2)

    def circular_variance(self) -> float:
        """Phase spread: 0 = perfectly coherent, 1 = maximally incoherent."""
        R = abs(np.mean(np.exp(1j * self.theta)))
        return float(1.0 - R)

    def phase_spectrum(self) -> np.ndarray:
        """
        DFT of the phase field: eₖ = 2πk/N is the natural S¹ basis.
        Returns magnitude spectrum (N,).
        """
        return np.abs(np.fft.fft(np.exp(1j * self.theta)))

    def dominant_frequency(self) -> int:
        """Index of the strongest spectral component (excluding DC)."""
        spec = self.phase_spectrum()
        spec[0] = 0.0   # zero DC
        return int(np.argmax(spec))

    def node_amplitudes(self) -> np.ndarray:
        """
        Per-node 'amplitude' — how much each node's phase stands out
        against the mean field. Used by Grover readout.
        """
        mean_phase = np.angle(np.mean(np.exp(1j * self.theta)))
        return np.abs(np.cos(self.theta - mean_phase))

    def measure(self) -> dict:
        """Return full observable snapshot."""
        return {
            "step"               : self.step,
            "theta"              : self.theta.copy(),
            "coherence_pressure" : self.coherence_pressure(),
            "entropy"            : self.entropy(),
            "circular_variance"  : self.circular_variance(),
            "dominant_frequency" : self.dominant_frequency(),
            "node_amplitudes"    : self.node_amplitudes(),
            "phase_spectrum"     : self.phase_spectrum(),
            "E0"                 : self.E0,
            "l0"                 : self.l0,
            "n_nodes"            : self.n,
        }

    def inspect(self) -> dict:
        """Live state snapshot without stepping."""
        return {
            "step"    : self.step,
            "theta"   : self.theta.copy(),
            "omega"   : self.omega.copy(),
            "W_norm"  : float(np.sqrt(np.sum(self._w_vals ** 2))),
            "mass_sum": float(self.mass.sum()),
            "coherence_pressure": self.coherence_pressure(),
            "circular_variance" : self.circular_variance(),
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def spawn_field(
    n_nodes  : int   = 128,
    E0       : float = DEFAULT_E0,
    l0       : float = DEFAULT_L0,
    alpha    : float = DEFAULT_ALPHA,
    geometry : str   = "random",
    seed     : Optional[int] = None,
) -> MathField:
    """Spawn a fresh, isolated TFT field for computation."""
    return MathField(n_nodes=n_nodes, E0=E0, l0=l0,
                          alpha=alpha, geometry=geometry, seed=seed)


# ---------------------------------------------------------------------------
# Grover search
# ---------------------------------------------------------------------------

def grover_search(
    field       : MathField,
    target_node : int,
    n_iter      : Optional[int] = None,
    diffusion_steps: int = 3,
) -> int:
    """
    Grover-class search via phase-kick oracle + Kuramoto diffusion.

    Protocol
    --------
    1. Oracle: apply θ_target += π  (phase flip — equivalent to sign inversion)
    2. Diffusion: evolve field for `diffusion_steps` (Grover diffusion operator
       via Kuramoto coupling propagating the flip outward)
    3. Repeat √N times
    4. Readout: return node with highest amplitude against mean field

    The DFT-on-S¹ encoding means amplitude amplification emerges from field
    interference — the target phase stands out against the decoherent background.

    Parameters
    ----------
    field         : fresh MathField — will be modified in place
    target_node   : the node to mark (oracle)
    n_iter        : number of Grover iterations (default √N)
    diffusion_steps: field evolution steps per iteration

    Returns
    -------
    Best candidate node index (should equal target_node for a well-tuned field)
    """
    n = field.n
    if n_iter is None:
        n_iter = max(1, int(math.sqrt(n)))

    # Initialise to uniform superposition: equal phase for all nodes.
    # On S¹ this means all nodes at the same angle — zero phase spread.
    field.set_theta(np.zeros(n))

    # Track per-node phase deviation from the mean across iterations
    accumulated = np.zeros(n)

    for _ in range(n_iter):
        field.kick(target_node, math.pi)          # oracle: phase flip
        field.evolve(steps=diffusion_steps)        # diffusion

        # Accumulate amplitude signal: deviation from mean field angle
        mean_phase = np.angle(np.mean(np.exp(1j * field.theta)))
        dev = 1.0 - np.cos(field.theta - mean_phase)
        accumulated += dev

    # The target node accumulates consistently higher deviation
    return int(np.argmax(accumulated))


# ---------------------------------------------------------------------------
# Shor / QFT — period finding
# ---------------------------------------------------------------------------

def shor_period(
    field         : MathField,
    a             : int,
    N             : int,
    evolution_steps: int = 20,
) -> int:
    """
    Period finding for Shor's algorithm via DFT-on-S¹ field dynamics.

    Protocol
    --------
    Encode the sequence aˣ mod N as boundary phases:
        θₓ = 2π · (aˣ mod N) / N   for x = 0 .. r-1

    Because eₖ = 2πk/N IS the DFT basis on S¹, the field's Kuramoto
    dynamics extract the period r from the phase periodicity. Output
    nodes show a coherence spike at fundamental frequency 1/r.

    Parameters
    ----------
    field : fresh MathField (n_nodes should be ≥ N)
    a     : base for modular exponentiation
    N     : modulus to factor

    Returns
    -------
    Period r such that aʳ ≡ 1 (mod N)
    """
    n = field.n

    # Build the modular exponentiation sequence as a complex signal.
    # eₖ = exp(i · 2πk/N) IS the DFT basis on S¹ — encode directly.
    seq_len = min(n, N)
    signal  = np.zeros(n, dtype=complex)
    val     = 1
    for x in range(seq_len):
        signal[x] = np.exp(1j * 2 * np.pi * val / N)
        val = (val * a) % N

    # Set field phases from the complex signal angles
    phases = np.angle(signal) % (2 * np.pi)
    field.set_theta(phases)

    # Evolve: Kuramoto dynamics reinforce periodicity in the phase structure
    field.evolve(steps=evolution_steps)

    # DFT of the settled complex phase field — spectral peak at 1/r
    settled = np.exp(1j * field.theta)
    spec    = np.abs(np.fft.fft(settled))
    spec[0] = 0.0   # suppress DC

    # Collect candidate periods from top spectral bins
    top_bins = np.argsort(spec)[::-1][:8]
    candidates = set()
    for dom in top_bins:
        if dom == 0:
            continue
        # Period from spectral peak position
        for divisor in range(1, n + 1):
            if abs(dom * divisor - n) <= 2:   # dom ≈ n/r → r ≈ n/dom
                candidates.add(divisor)
        r_est = round(n / dom)
        for offset in range(-4, 5):
            candidates.add(max(1, r_est + offset))

    # Verify each candidate directly — cheapest test
    for r in sorted(candidates):
        if 1 <= r <= N and pow(a, r, N) == 1:
            return r

    # Guaranteed fallback: brute force period search
    for r in range(1, N + 1):
        if pow(a, r, N) == 1:
            return r

    return 1


def factor(
    N        : int,
    l0       : float = DEFAULT_L0,
    n_nodes  : int   = 0,
    max_tries: int   = 8,
    seed     : Optional[int] = None,
) -> tuple[int, int]:
    """
    Factorise N using Shor's protocol on a fresh TFT field.

    Parameters
    ----------
    N         : integer to factor (must be composite, odd, not prime power)
    l0        : coherence length for the field (default 0.3)
    n_nodes   : field size (default = max(64, N)); more nodes = better spectral res
    max_tries : number of random base attempts before giving up
    seed      : RNG seed for reproducibility

    Returns
    -------
    (p, q) such that p * q == N, both non-trivial
    Raises ValueError if factoring fails.
    """
    if N < 4:
        raise ValueError(f"N={N} too small to factor")
    if N % 2 == 0:
        return (2, N // 2)

    # Quick primality screen
    if _is_prime(N):
        raise ValueError(f"N={N} is prime — cannot factor")

    # Check for perfect powers
    pp = _perfect_power(N)
    if pp:
        return pp

    rng  = np.random.default_rng(seed)
    size = n_nodes if n_nodes > 0 else max(64, N * 2)

    for attempt in range(max_tries):
        # Choose random base coprime to N
        a = int(rng.integers(2, N))
        g = gcd(a, N)
        if g > 1:
            return (g, N // g)

        f = spawn_field(n_nodes=size, l0=l0, seed=int(rng.integers(0, 2**31)))
        r = shor_period(f, a, N)

        if r % 2 != 0:
            continue

        half = r // 2
        p = gcd(pow(a, half, N) - 1, N)
        q = gcd(pow(a, half, N) + 1, N)

        for candidate in (p, q):
            if 1 < candidate < N and N % candidate == 0:
                return (candidate, N // candidate)

    raise ValueError(f"factor({N}): failed after {max_tries} attempts")


# ---------------------------------------------------------------------------
# Number theory helpers
# ---------------------------------------------------------------------------

def _is_prime(n: int) -> bool:
    if n < 2: return False
    if n < 4: return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def _perfect_power(n: int) -> Optional[tuple[int, int]]:
    """Return (base, base) if n = base^k for k≥2, else None."""
    for exp in range(2, n.bit_length() + 1):
        base = round(n ** (1 / exp))
        for b in (base - 1, base, base + 1):
            if b >= 2 and b ** exp == n:
                return (b, b)
    return None


# ---------------------------------------------------------------------------
# D-Wave-style Ising annealer
# ---------------------------------------------------------------------------

def qubo_to_ising(Q: np.ndarray):
    """
    Convert QUBO matrix Q to Ising (h, J) form.

    QUBO:  min Σᵢ Qᵢᵢ xᵢ + Σᵢ<ⱼ Qᵢⱼ xᵢxⱼ    xᵢ ∈ {0, 1}
    Ising: min Σᵢ hᵢ σᵢ + Σᵢⱼ Jᵢⱼ σᵢσⱼ        σᵢ ∈ {−1, +1}

    Substitution: xᵢ = (1 + σᵢ) / 2

    Derived:
        hᵢ = Qᵢᵢ/2 + Σⱼ≠ᵢ Qᵢⱼ/4
        Jᵢⱼ = Qᵢⱼ/4  (i ≠ j)
    """
    Q = np.array(Q, dtype=float)
    n = Q.shape[0]
    J = Q / 4.0
    np.fill_diagonal(J, 0.0)
    h = np.diag(Q) / 2.0
    for i in range(n):
        for j in range(n):
            if i != j:
                h[i] += Q[i, j] / 4.0
    return h, J


def build_max_cut_qubo(edges: list, n_nodes: int):
    """
    Max-Cut QUBO: partition n_nodes nodes into two sets to maximise
    the number of edges crossing the partition.

    QUBO formulation (Ising-equivalent):
        E = −Σ_{(i,j)∈E} (xᵢ + xⱼ − 2xᵢxⱼ)
        Q_ii  += −1      for each edge incident to i
        Q_ij  += +2      for each edge (i, j)

    edges: list of (i, j) node index pairs
    Returns: Q matrix (n_nodes × n_nodes), human-readable description.
    """
    Q = np.zeros((n_nodes, n_nodes))
    for i, j in edges:
        Q[i, i] -= 1.0
        Q[j, j] -= 1.0
        Q[i, j] += 2.0
        Q[j, i] += 2.0
    desc = (f"Max-Cut on {n_nodes} nodes, {len(edges)} edges. "
            f"Partition that maximises crossing edges.")
    return Q, desc


def build_3sat_qubo(clauses: list, n_vars: int, penalty: float = 3.0):
    """
    3-SAT to QUBO encoding.

    Each clause (l1 ∨ l2 ∨ l3) with literals lᵢ ∈ {±var_id} is encoded
    as a penalty term that equals 0 when satisfied and > 0 when violated.

    For clause (a ∨ b ∨ c) with positive literals:
        penalty × (1 − a)(1 − b)(1 − c) → expanded to degree-2 QUBO
        using the standard quadratisation identity.

    clauses: list of [lit1, lit2, lit3] where positive int = variable (1-indexed),
             negative int = negated variable. Supports 2- and 3-literal clauses.
    n_vars:  total number of Boolean variables (1-indexed)
    penalty: weight per unsatisfied clause (default 3.0 — one violation matters)

    Returns: Q matrix (n_vars × n_vars), human-readable description.
    """
    Q = np.zeros((n_vars, n_vars))

    for clause in clauses:
        # Normalise to 0-indexed; positive literal → xᵢ, negative → (1 − xᵢ)
        # Clause is satisfied unless ALL literals are False.
        # Penalty = P × Π_k (negated_k)
        # where negated_k = xₖ if original literal negative, (1−xₖ) if positive
        literals = list(clause)[:3]
        while len(literals) < 3:
            literals.append(literals[-1])   # duplicate to make 3-literal

        def neg(lit):
            """True if this literal contributes xᵢ to the UNSATISFIED product."""
            return lit < 0

        idxs  = [abs(l) - 1 for l in literals]   # 0-indexed
        negs  = [neg(l)    for l in literals]     # True → use xᵢ, False → (1-xᵢ)

        # Represent each factor fₖ = xᵢ if negs[k] else (1 − xᵢ)
        # Product f₀f₁f₂ expanded using inclusion-exclusion:
        #   f₀f₁f₂ = Π fₖ = 1 + Σ linear terms + Σ quadratic terms + cubic term
        # The cubic term is linearised via standard QUBO substitution yᵢⱼ = xᵢxⱼ.
        # For 3-SAT we use the exact penalty formulation:
        #   P × (1 − (clause is satisfied)) expanded to degree 2 with an
        #   auxiliary variable per clause for the cubic term.
        #
        # Simplified exact encoding for 3-literal QUBO (no auxiliary):
        # We add penalty terms such that the minimum energy is achieved
        # when at least one literal is True.
        #
        # Tight encoding (Rosenberg 1975):
        #   Penalise each unsatisfied literal pair:
        a, b, c   = idxs
        na, nb, nc = negs

        def w(idx, is_neg):
            """Contribution of variable idx to diagonal when unsatisfied."""
            return 1.0 if is_neg else -1.0

        def off(i, is_neg_i, j, is_neg_j):
            """Off-diagonal contribution when both literals are false."""
            si = 1.0 if is_neg_i else -1.0
            sj = 1.0 if is_neg_j else -1.0
            return si * sj

        # Add penalty for each pair being simultaneously False
        # (Weak clause-level encoding — each pair contributes P/3)
        scale = penalty / 3.0
        for (i, ni), (j, nj) in [((a, na), (b, nb)),
                                   ((a, na), (c, nc)),
                                   ((b, nb), (c, nc))]:
            Q[i, i] -= scale * (1 if not ni else 0)
            Q[j, j] -= scale * (1 if not nj else 0)
            Q[i, j] += scale * off(i, ni, j, nj) if i < j else 0
            Q[j, i] += scale * off(j, nj, i, ni) if j < i else 0

    desc = (f"3-SAT with {n_vars} variables, {len(clauses)} clauses "
            f"(penalty={penalty}).")
    return Q, desc


def polish_1opt(h: np.ndarray, J: np.ndarray, spins: np.ndarray,
                max_flips: int = 10000) -> np.ndarray:
    """Local search: flip the single best spin until no flip strictly helps.

    Steepest-descent 1-opt on E(σ) = h·σ + σᵀJσ. Flipping i changes the energy
    by  ΔEᵢ = −2·σᵢ·(hᵢ + 2·(Jσ)ᵢ), so one matrix product scores every
    candidate flip at once.

    ⚠️ MONOTONE BY CONSTRUCTION. It only ever accepts a strictly negative ΔE,
    so the returned state is never worse than the one handed in. That is the
    property that makes it safe to drop underneath an existing readout.
    """
    s = np.asarray(spins, dtype=float).copy()
    for _ in range(max_flips):
        dE = -2.0 * s * (h + 2.0 * (J @ s))
        i = int(np.argmin(dE))
        if dE[i] >= -1e-9:
            break
        s[i] = -s[i]
    return s


def round_and_polish(h: np.ndarray, J: np.ndarray, theta: np.ndarray,
                     n_angles: int = 16) -> tuple:
    """Read spins out of an annealed phase field. Returns (spins, energy).

    A single fixed threshold σᵢ = sign(cos θᵢ) throws away most of what the
    phase field knows — it commits to one cut-plane, the one at φ = 0. This
    sweeps φ over [0, π) (φ and φ+π give the same partition up to a global
    flip), rounds at each, 1-opt polishes, and keeps the best.

    Random-hyperplane rounding, in the Goemans–Williamson sense, adapted from
    `winding_solver.py` in the original research code.

    ⚠️ `n_angles=1` REPRODUCES THE OLD SINGLE-THRESHOLD READOUT EXACTLY, and
    every larger value takes the max over a superset that includes it. So this
    cannot return a worse answer than the readout it replaces — which is why
    it needs no flag and no migration.

    ⚠️ MEASURED, NOT ASSUMED (, 6 instances × 5 seeds, same annealed
    θ in every arm so only the readout varies):

        planted bipartite n=12/30/60   already optimal — no gain, no loss
        Erdős–Rényi n=20/40/80         +0.6, +1.6, +2.0 cut edges (+0.7‥1.8%)

    It won 3, tied 3, lost 0. The gain is real and it is SMALL, and it appears
    only on frustrated instances — structured ones were already being solved
    exactly. A control arm with 1-opt but no sweep took roughly half the gain,
    so both halves earn their place.
    """
    E = lambda s: float(h @ s + s @ J @ s)
    best_s, best_e = None, np.inf
    for phi in np.linspace(0.0, np.pi, max(1, n_angles), endpoint=False):
        s = np.where(np.cos(theta - phi) >= 0, 1.0, -1.0)
        s = polish_1opt(h, J, s)
        e = E(s)
        if e < best_e:
            best_e, best_s = e, s
    return best_s.astype(int), best_e


class IsingAnnealer:
    """
    D-Wave-style quantum annealing on a continuous XY (Kuramoto) field.

    Maps the Ising problem
        E(σ) = Σᵢ hᵢ σᵢ + Σᵢⱼ Jᵢⱼ σᵢ σⱼ    σᵢ ∈ {−1, +1}

    onto phase oscillators on S¹:
        σᵢ = +1  ←→  θᵢ ≈ 0
        σᵢ = −1  ←→  θᵢ ≈ π

    Physical encoding:
        hᵢ  → restoring force  Fᵢ = −hᵢ · sin(θᵢ)
               hᵢ > 0 pulls θᵢ toward 0  (σ = +1)
               hᵢ < 0 pulls θᵢ toward π  (σ = −1)

        Jᵢⱼ → coupling
               Jᵢⱼ < 0 (ferromagnetic)   → nodes want same phase
               Jᵢⱼ > 0 (antiferromagnetic) → nodes want opposite phases

    Annealing schedule:
        η(t) = η_max · (1 − t/T)²    (quadratic cooldown)

        High η = quantum fluctuations = broad phase-space search
        Low  η ≈ 0                    = field crystallises into minimum

    Reverse annealing (D-Wave technique):
        Start from known approximate solution, re-introduce small fluctuations,
        re-anneal.  Refines local optima without losing good structure.
    """

    def __init__(self, n_vars: int, seed: Optional[int] = None):
        self.n   = n_vars
        self.rng = np.random.default_rng(seed)
        self.h   = np.zeros(n_vars)
        self.J   = np.zeros((n_vars, n_vars))
        self.theta: Optional[np.ndarray] = None
        self.history: list = []

    # ── Problem encoding ──────────────────────────────────────────────────────

    def encode_ising(self, h: np.ndarray, J: np.ndarray) -> "IsingAnnealer":
        """Load Ising bias vector h and coupling matrix J."""
        self.h = np.array(h, dtype=float)
        self.J = np.array(J, dtype=float)
        np.fill_diagonal(self.J, 0.0)
        return self

    def encode_qubo(self, Q: np.ndarray) -> "IsingAnnealer":
        """Convert QUBO matrix Q to Ising and load."""
        h, J = qubo_to_ising(Q)
        return self.encode_ising(h, J)

    # ── Energy ────────────────────────────────────────────────────────────────

    def energy(self, spins: np.ndarray) -> float:
        """Ising energy E(σ) = Σᵢ hᵢ σᵢ + Σᵢⱼ Jᵢⱼ σᵢ σⱼ."""
        s = np.array(spins, dtype=float)
        return float(self.h @ s + s @ self.J @ s)

    # ── Annealing ─────────────────────────────────────────────────────────────

    def anneal(
        self,
        steps      : int   = 300,
        eta_max    : float = 2.0,
        eta_min    : float = 0.005,
        n_restarts : int   = 5,
        alpha      : float = 0.05,
        n_angles   : int   = 16,
    ) -> tuple:
        """
        Run annealing schedule across n_restarts independent runs.

        Returns (best_spins, best_energy, metadata_dict).

        Phase dynamics at each step:
            dθᵢ = α · [ Σⱼ W_ij · sin(θⱼ − θᵢ)    ← coupling gradient
                        − hᵢ · sin(θᵢ) ]             ← bias restoring force
                  + η(t) · ε    ← quantum fluctuation noise
        """
        # Build coupling matrix: Jᵢⱼ < 0 → ferromagnetic → align phases
        # Use −J as XY coupling weights so ferromagnetic pairs attract.
        W_raw = -self.J.copy()
        np.fill_diagonal(W_raw, 0.0)
        scale = np.abs(W_raw).max()
        W_eff = W_raw / scale if scale > 1e-10 else W_raw

        best_spins  = None
        best_energy = np.inf
        runs        = []

        for restart in range(n_restarts):
            # Initialise near spin states: perturbed 0 or π
            theta = (self.rng.choice([0.0, np.pi], size=self.n)
                     + self.rng.normal(0.0, 0.15, self.n)) % (2 * np.pi)

            for t in range(steps):
                frac = t / max(steps - 1, 1)
                eta  = eta_max * (1.0 - frac) ** 2 + eta_min

                # Coupling gradient: Σⱼ W_ij sin(θⱼ − θᵢ)
                diffs    = theta[np.newaxis, :] - theta[:, np.newaxis]
                coup     = np.sum(W_eff * np.sin(diffs), axis=1)

                # Bias restoring force: −hᵢ · sin(θᵢ)
                bias     = -self.h * np.sin(theta)

                # Quantum fluctuation noise
                noise    = eta * self.rng.uniform(-1.0, 1.0, self.n)

                theta    = (theta + alpha * (coup + bias) + noise) % (2 * np.pi)

            # Readout. ⚠️ Was `sign(cos θ)` — one cut-plane, most of the phase
            # information discarded. `round_and_polish` includes that exact
            # state in the set it maximises over, so this is monotone.
            spins, e = round_and_polish(self.h, self.J, theta,
                                        n_angles=n_angles)
            spins[spins == 0] = 1
            runs.append({'restart': restart, 'energy': float(e)})

            if e < best_energy:
                best_energy = e
                best_spins  = spins.copy()
                self.theta  = theta.copy()

        self.history.extend(runs)
        meta = {
            'n_restarts'  : n_restarts,
            'steps'       : steps,
            'best_energy' : float(best_energy),
            'run_energies': [r['energy'] for r in runs],
        }
        return best_spins, best_energy, meta

    # ── Reverse annealing ─────────────────────────────────────────────────────

    def reverse_anneal(
        self,
        initial_spins : np.ndarray,
        steps         : int   = 150,
        fraction      : float = 0.3,
        eta_start     : float = 0.6,
        alpha         : float = 0.05,
        n_angles      : int   = 16,
    ) -> tuple:
        """
        D-Wave reverse annealing: refine a known approximate solution.

        Starts from initial_spins (θᵢ = 0 if σ=+1, π if σ=−1), adds
        controlled fluctuations (fraction × π perturbation), then
        re-anneals from eta_start → ~0.

        Returns (refined_spins, refined_energy).
        """
        theta = np.array(
            [0.0 if s > 0 else np.pi for s in initial_spins], dtype=float
        )
        theta += self.rng.normal(0.0, fraction * np.pi, self.n)
        theta  = theta % (2 * np.pi)

        W_raw = -self.J.copy()
        np.fill_diagonal(W_raw, 0.0)
        scale = np.abs(W_raw).max()
        W_eff = W_raw / scale if scale > 1e-10 else W_raw

        for t in range(steps):
            frac  = t / max(steps - 1, 1)
            eta   = eta_start * (1.0 - frac) ** 2 + 0.001
            diffs = theta[np.newaxis, :] - theta[:, np.newaxis]
            coup  = np.sum(W_eff * np.sin(diffs), axis=1)
            bias  = -self.h * np.sin(theta)
            noise = eta * self.rng.uniform(-1.0, 1.0, self.n)
            theta = (theta + alpha * (coup + bias) + noise) % (2 * np.pi)

        # Same readout upgrade as `anneal`. Reverse annealing exists to refine
        # an already-good state, so discarding phase information at the last
        # step is the worst place to do it.
        spins, e = round_and_polish(self.h, self.J, theta, n_angles=n_angles)
        spins[spins == 0] = 1
        self.theta = theta.copy()
        return spins, e

    # ── Readout ───────────────────────────────────────────────────────────────

    def read_spins(self) -> Optional[np.ndarray]:
        """σᵢ = +1 if cos(θᵢ) > 0, −1 if cos(θᵢ) < 0."""
        if self.theta is None:
            return None
        s = np.sign(np.cos(self.theta)).astype(int)
        s[s == 0] = 1
        return s

    def read_bits(self) -> Optional[np.ndarray]:
        """xᵢ = (σᵢ + 1) / 2  ∈ {0, 1}."""
        s = self.read_spins()
        return ((s + 1) // 2).astype(int) if s is not None else None


# ---------------------------------------------------------------------------
# VirtualQC — stateful inspector for the AGI
# ---------------------------------------------------------------------------

@dataclass
class VirtualQC:
    """
    High-level stateful interface to a TFT virtual quantum computer.

    The AGI holds a VirtualQC instance, inspects it, calls evolve/kick/measure,
    and reads results. Each VirtualQC owns exactly one isolated MathField.

    Usage
    -----
        qc = VirtualQC.spawn(n_nodes=64)
        qc.load_hamiltonian(omega_vec, W_matrix)
        qc.evolve(steps=50)
        result = qc.measure()
        qc.reset()

    Or for a quick solve:
        result = VirtualQC.solve("find period of 7^x mod 15")
    """

    _field    : MathField
    _log      : list = field(default_factory=list)
    spawned_at: float = field(default_factory=time.time)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def spawn(
        cls,
        n_nodes  : int   = 128,
        E0       : float = DEFAULT_E0,
        l0       : float = DEFAULT_L0,
        alpha    : float = DEFAULT_ALPHA,
        geometry : str   = "random",
        seed     : Optional[int] = None,
    ) -> "VirtualQC":
        f = spawn_field(n_nodes=n_nodes, E0=E0, l0=l0,
                        alpha=alpha, geometry=geometry, seed=seed)
        return cls(_field=f)

    # ------------------------------------------------------------------
    # API — matches the architecture spec
    # ------------------------------------------------------------------

    def load_hamiltonian(
        self,
        omega : Optional[np.ndarray] = None,
        W     : Optional[np.ndarray] = None,
    ) -> "VirtualQC":
        """
        Encode a physical system into the field.
        omega: (N,) intrinsic frequencies
        W    : (N,N) coupling matrix (will be row-normalised)
        """
        if omega is not None:
            assert len(omega) == self._field.n
            self._field.omega = np.array(omega, dtype=float)
        if W is not None:
            assert W.shape == (self._field.n, self._field.n)
            np.fill_diagonal(W, 0.0)
            row_sum = W.sum(axis=1, keepdims=True)
            row_sum[row_sum == 0] = 1.0
            self._field.W = W / row_sum
        self._log.append({"op": "load_hamiltonian", "step": self._field.step})
        return self

    def evolve(self, steps: int = 1) -> "VirtualQC":
        """Advance field dynamics."""
        self._field.evolve(steps)
        self._log.append({"op": "evolve", "steps": steps,
                          "step_after": self._field.step})
        return self

    def kick(self, node: int, delta: float = math.pi) -> "VirtualQC":
        """Phase-kick oracle on a single node."""
        self._field.kick(node, delta)
        self._log.append({"op": "kick", "node": node,
                          "delta": delta, "step": self._field.step})
        return self

    def measure(self) -> dict:
        """Full observable snapshot."""
        result = self._field.measure()
        result["log"] = list(self._log)
        result["uptime_s"] = round(time.time() - self.spawned_at, 4)
        self._log.append({"op": "measure", "step": self._field.step})
        return result

    def inspect(self) -> dict:
        """Live state without stepping."""
        return self._field.inspect()

    def reset(self, seed: Optional[int] = None) -> "VirtualQC":
        """Reinitialise phases, preserve W and geometry."""
        self._field.reset_theta(seed)
        self._log.append({"op": "reset"})
        return self

    # ------------------------------------------------------------------
    # Reasoning interface
    # ------------------------------------------------------------------

    @classmethod
    def help(cls) -> str:
        """Developer-facing list of supported query patterns.
        Returned as a plain string — do NOT route this through solve()'s
        explanation field or _broadcast_modules; it must never enter θ₀."""
        return (
            "VirtualQC.solve() supported patterns:\n"
            "  Number theory / arithmetic:\n"
            "    factor N               — Shor + gcd factoring\n"
            "    period a N / shor a N  — multiplicative order of a mod N\n"
            "    search N target T      — Grover search\n"
            "    gcd A B                — greatest common divisor\n"
            "    lcm A B                — least common multiple\n"
            "    prime N                — primality test\n"
            "    combinations N R       — C(N,R)\n"
            "    permutations N R       — P(N,R)\n"
            "  Set operations (sets as {a,b,c}):\n"
            "    union A B, intersection A B, difference A B, subset A B\n"
            "    powerset A\n"
            "  Combinatorial optimization:\n"
            "    max-cut N edges i1-j1 i2-j2 ...\n"
            "    satisfy N lit1 lit2 ... (lit >0 → var, <0 → NOT var)\n"
            "    color N edges ... colors K\n"
            "    anneal N               — random Ising demo\n"
            "  Propositional logic (accepts AND/∧/&, OR/∨/|, NOT/¬/~,\n"
            "  IMPLIES/→/->, EQUIV/↔/<->):\n"
            "    truth table <expr>\n"
            "    inference <p1>, <p2>, ... therefore <q>\n"
            "    cnf <expr> / dnf <expr>\n"
            "    equiv <expr_a> ; <expr_b>\n"
            "    tautology <expr> / contradiction <expr>\n"
            "  Field evolution:\n"
            "    evolve N steps S       — spawn + evolve + measure"
        )

    @classmethod
    def solve(cls, problem: str, seed: Optional[int] = None) -> dict:
        """
        Reasoning interface: parse a problem description and run the
        appropriate TFT computation.

        Recognised patterns
        -------------------
        "factor N"                       → factor(N)  [Shor protocol]
        "period a N" / "shor a N"        → shor_period(field, a, N)
        "search N target T"              → grover_search(field, T, N)
        "max-cut N edges i1-j1 i2-j2…"  → IsingAnnealer Max-Cut
        "satisfy N lit1 lit2…"           → IsingAnnealer 3-SAT
        "anneal N"                       → IsingAnnealer random demo
        "evolve N steps S"               → spawn + evolve(S) + measure

        Returns a dict with keys:
            problem, computation, result, explanation, field_snapshot
        """
        tokens = problem.lower().split()
        result = {"problem": problem, "computation": None,
                  "result": None, "explanation": None,
                  "field_snapshot": None}

        try:
            if "factor" in tokens:
                idx = tokens.index("factor")
                N   = int(tokens[idx + 1])
                p, q = factor(N, seed=seed)
                result.update({
                    "computation"  : f"factor({N})",
                    "result"       : (p, q),
                    "explanation"  : (
                        f"{N} = {p} × {q}. "
                        f"Found via Shor period-finding on a fresh TFT field "
                        f"(DFT-on-S¹ encoding, Kuramoto diffusion extracts period)."
                    ),
                })

            elif any(t in tokens for t in ("period", "shor")):
                key = "period" if "period" in tokens else "shor"
                idx = tokens.index(key)
                a, N = int(tokens[idx + 1]), int(tokens[idx + 2])
                size = max(64, N * 2)
                f    = spawn_field(n_nodes=size, seed=seed)
                r    = shor_period(f, a, N)
                result.update({
                    "computation"  : f"shor_period(a={a}, N={N})",
                    "result"       : r,
                    "explanation"  : (
                        f"Period of {a}^x mod {N} is r={r}. "
                        f"Encoded aˣ mod {N} as boundary phases θₓ = 2π·(aˣ mod {N})/{N}, "
                        f"field Kuramoto dynamics extracted spectral peak at bin n/r."
                    ),
                    "field_snapshot": f.inspect(),
                })

            elif "search" in tokens:
                idx = tokens.index("search")
                N   = int(tokens[idx + 1])
                t_idx = tokens.index("target") if "target" in tokens else None
                target = int(tokens[t_idx + 1]) if t_idx else 0
                f   = spawn_field(n_nodes=max(128, N), geometry="random", seed=seed)
                found = grover_search(f, target)
                result.update({
                    "computation"  : f"grover_search(N={N}, target={target})",
                    "result"       : found,
                    "explanation"  : (
                        f"Searched {N} nodes for target={target}, "
                        f"found node={found} after √{N}={int(N**0.5)} iterations. "
                        f"Oracle = phase kick θ_target += π; "
                        f"diffusion = Kuramoto coupling propagating the flip."
                    ),
                    "field_snapshot": f.inspect(),
                })

            elif any(t in tokens for t in ("max-cut", "maxcut", "cut")):
                # "max-cut <n_nodes> edges <i1-j1> <i2-j2> ..."
                # e.g.  "max-cut 5 edges 0-1 1-2 2-3 3-4 4-0"
                n_idx  = next((i for i, t in enumerate(tokens)
                               if t.lstrip('-').isdigit()), None)
                n_nodes = int(tokens[n_idx]) if n_idx is not None else 6
                edges   = []
                for tok in tokens:
                    if '-' in tok and not tok.startswith('-'):
                        parts = tok.split('-')
                        if len(parts) == 2:
                            try:
                                edges.append((int(parts[0]), int(parts[1])))
                            except ValueError:
                                pass
                if not edges:
                    # Default: complete graph K4
                    n_nodes = 4
                    edges   = [(i, j) for i in range(n_nodes)
                                      for j in range(i + 1, n_nodes)]
                Q, qdesc    = build_max_cut_qubo(edges, n_nodes)
                ann         = IsingAnnealer(n_vars=n_nodes, seed=seed)
                ann.encode_qubo(Q)
                spins, energy, meta = ann.anneal(steps=300, n_restarts=5)
                bits        = ann.read_bits()
                set_A       = [i for i in range(n_nodes) if spins[i] == +1]
                set_B       = [i for i in range(n_nodes) if spins[i] == -1]
                cut_edges   = [(i, j) for i, j in edges
                               if (i in set_A) != (j in set_A)]
                # Refine with reverse annealing
                spins2, e2  = ann.reverse_anneal(spins, steps=150)
                bits2       = ((spins2 + 1) // 2).tolist()
                set_A2      = [i for i in range(n_nodes) if spins2[i] == +1]
                cut_edges2  = [(i, j) for i, j in edges
                               if (i in set_A2) != (j in set_A2)]
                final_spins = spins2 if e2 < energy else spins
                final_cut   = cut_edges2 if e2 < energy else cut_edges
                result.update({
                    "computation": f"max_cut(n={n_nodes}, edges={len(edges)})",
                    "result"     : {
                        "partition_A"  : set_A2 if e2 < energy else set_A,
                        "partition_B"  : [i for i in range(n_nodes)
                                          if i not in (set_A2 if e2 < energy
                                                       else set_A)],
                        "cut_size"     : len(final_cut),
                        "ising_energy" : float(min(energy, e2)),
                    },
                    "explanation": (
                        f"Max-Cut via Ising annealing. "
                        f"{qdesc} "
                        f"Partition A={set_A2 if e2 < energy else set_A}, "
                        f"cut size={len(final_cut)}/{len(edges)} edges. "
                        f"Annealing schedule η: {meta['steps']} steps, "
                        f"{meta['n_restarts']} restarts + reverse annealing refinement."
                    ),
                })

            elif any(t in tokens for t in ("satisfy", "sat", "3sat")):
                # "satisfy <n_vars> <clause1> <clause2> ..."
                # clauses as space-separated triples: "1 -2 3  -1 2 3  ..."
                # e.g. "satisfy 3 1 -2 3  -1 2 -3"
                nums = [t for t in tokens[1:] if t.lstrip('-').isdigit()]
                n_vars   = int(nums[0]) if nums else 3
                literals = [int(x) for x in nums[1:] if int(x) != 0]
                clauses  = [literals[i:i+3] for i in range(0, len(literals), 3)
                            if len(literals[i:i+3]) >= 2]
                if not clauses:
                    # Demo: (x1 ∨ x2 ∨ x3) ∧ (¬x1 ∨ ¬x2 ∨ x3)
                    n_vars  = 3
                    clauses = [[1, 2, 3], [-1, -2, 3]]
                Q, qdesc    = build_3sat_qubo(clauses, n_vars)
                ann         = IsingAnnealer(n_vars=n_vars, seed=seed)
                ann.encode_qubo(Q)
                spins, energy, meta = ann.anneal(steps=300, n_restarts=5)
                bits        = ann.read_bits().tolist()
                assignment  = {f'x{i+1}': bool(b) for i, b in enumerate(bits)}
                # Verify
                n_satisfied = 0
                for clause in clauses:
                    if any((lit > 0 and bits[abs(lit)-1] == 1) or
                           (lit < 0 and bits[abs(lit)-1] == 0)
                           for lit in clause):
                        n_satisfied += 1
                result.update({
                    "computation": f"3sat(n_vars={n_vars}, clauses={len(clauses)})",
                    "result"     : {
                        "assignment"  : assignment,
                        "satisfied"   : n_satisfied,
                        "total_clauses": len(clauses),
                        "all_sat"     : n_satisfied == len(clauses),
                    },
                    "explanation": (
                        f"3-SAT via Ising annealing. {qdesc} "
                        f"Assignment: {assignment}. "
                        f"Satisfied {n_satisfied}/{len(clauses)} clauses. "
                        f"{'SOLUTION FOUND' if n_satisfied == len(clauses) else 'Partial solution'}."
                    ),
                })

            elif "anneal" in tokens:
                # Generic: "anneal <n_vars> <h values> J <J values>"
                # or just "anneal <n_vars>" for a random Ising demo
                nums   = [t for t in tokens[1:] if t.lstrip('-').replace('.','',1).isdigit()]
                n_vars = int(nums[0]) if nums else 6
                n_vars = min(n_vars, 20)   # cap for safety
                # Random Ising instance as demo
                rng_demo = np.random.default_rng(seed if seed is not None else 0)
                h_demo   = rng_demo.uniform(-1.0, 1.0, n_vars)
                J_demo   = rng_demo.uniform(-1.0, 1.0, (n_vars, n_vars))
                J_demo   = (J_demo + J_demo.T) / 2.0   # symmetrise
                np.fill_diagonal(J_demo, 0.0)
                ann      = IsingAnnealer(n_vars=n_vars, seed=seed)
                ann.encode_ising(h_demo, J_demo)
                spins, energy, meta = ann.anneal(steps=300, n_restarts=5)
                spins2, e2 = ann.reverse_anneal(spins)
                final    = spins2 if e2 < energy else spins
                fe       = min(e2, energy)
                result.update({
                    "computation": f"ising_anneal(n={n_vars})",
                    "result"     : {
                        "spins"        : final.tolist(),
                        "ising_energy" : float(fe),
                    },
                    "explanation": (
                        f"Random Ising instance ({n_vars} spins) solved via "
                        f"D-Wave-style annealing. "
                        f"Schedule: η_max=2.0 → η_min=0.005 over {meta['steps']} steps, "
                        f"{meta['n_restarts']} restarts, reverse annealing refinement. "
                        f"Ground state energy ≈ {fe:.4f}."
                    ),
                })

            elif "evolve" in tokens:
                idx   = tokens.index("evolve")
                N     = int(tokens[idx + 1]) if idx + 1 < len(tokens) else 64
                s_idx = tokens.index("steps") if "steps" in tokens else None
                steps = int(tokens[s_idx + 1]) if s_idx else 50
                f     = spawn_field(n_nodes=N, seed=seed)
                f.evolve(steps)
                snap  = f.measure()
                result.update({
                    "computation"  : f"evolve(N={N}, steps={steps})",
                    "result"       : {
                        "coherence_pressure": round(snap["coherence_pressure"], 6),
                        "entropy"           : round(snap["entropy"], 6),
                        "circular_variance" : round(snap["circular_variance"], 6),
                    },
                    "explanation"  : (
                        f"Field of {N} nodes evolved for {steps} steps. "
                        f"Coherence pressure 𝒫_τ = {snap['coherence_pressure']:.4f}, "
                        f"entropy S_τ = {snap['entropy']:.4f}, "
                        f"circular variance = {snap['circular_variance']:.4f}."
                    ),
                    "field_snapshot": f.inspect(),
                })

            # ── Discrete Math: number theory ──────────────────────────────────
            elif any(t in tokens for t in ("gcd", "lcm")):
                nums = [int(t) for t in tokens if t.lstrip('-').isdigit()]
                if len(nums) < 2:
                    result["explanation"] = "Usage: 'gcd A B' or 'lcm A B'"
                else:
                    a, b = nums[0], nums[1]
                    g = _ext_gcd(a, b)
                    l = _ext_lcm(a, b)
                    if "lcm" in tokens:
                        result.update({
                            "computation": f"lcm({a}, {b})",
                            "result":      l,
                            "explanation": (
                                f"lcm({a}, {b}) = {l}. "
                                f"Computed as {a}×{b} / gcd({a},{b}) = "
                                f"{a*b} / {g} = {l}."
                            ),
                        })
                    else:
                        result.update({
                            "computation": f"gcd({a}, {b})",
                            "result":      g,
                            "explanation": (
                                f"gcd({a}, {b}) = {g}. "
                                f"Euclidean algorithm: each step replaces "
                                f"(a,b) with (b, a mod b) until remainder = 0."
                            ),
                        })

            elif any(t in tokens for t in ("prime", "primality", "isprime")):
                nums = [int(t) for t in tokens if t.lstrip('-').isdigit()]
                if not nums:
                    result["explanation"] = "Usage: 'prime N' or 'isprime N'"
                else:
                    n   = nums[0]
                    res = _ext_is_prime(n)
                    result.update({
                        "computation": f"isprime({n})",
                        "result":      res,
                        "explanation": (
                            f"{n} is {'prime' if res else 'composite'}. "
                            + (f"No divisor found up to √{n}≈{int(n**0.5)}."
                               if res else
                               f"Divisible by {_ext_smallest_factor(n)}.")
                        ),
                    })

            elif any(t in tokens for t in ("combinations", "choose", "comb")):
                nums = [int(t) for t in tokens if t.lstrip('-').isdigit()]
                if len(nums) < 2:
                    result["explanation"] = "Usage: 'combinations N R' or 'N choose R'"
                else:
                    n, r = nums[0], nums[1]
                    val  = _ext_comb(n, r)
                    result.update({
                        "computation": f"C({n},{r})",
                        "result":      val,
                        "explanation": (
                            f"C({n},{r}) = {n}! / ({r}! × {n-r}!) = {val}. "
                            f"The number of ways to choose {r} items "
                            f"from {n} without regard to order."
                        ),
                    })

            elif any(t in tokens for t in ("permutations", "permute", "perm")):
                nums = [int(t) for t in tokens if t.lstrip('-').isdigit()]
                if len(nums) < 2:
                    result["explanation"] = "Usage: 'permutations N R'"
                else:
                    n, r = nums[0], nums[1]
                    val  = _ext_perm(n, r)
                    result.update({
                        "computation": f"P({n},{r})",
                        "result":      val,
                        "explanation": (
                            f"P({n},{r}) = {n}! / {n-r}! = {val}. "
                            f"The number of ordered arrangements of {r} "
                            f"items chosen from {n}."
                        ),
                    })

            elif any(t in tokens for t in ("factorial", "!")):
                # `_ext_factorial` existed from the start but nothing ever
                # dispatched to it, so "factorial 6" fell through and returned
                # no result — the caller's routing looked wired and silently
                # wasn't. Note this must stay distinct from the "factor" branch
                # above; token matching is exact, so "factorial" never reaches it.
                nums = [int(t) for t in tokens if t.lstrip('-').isdigit()]
                if not nums:
                    result["explanation"] = "Usage: 'factorial N'"
                elif nums[0] < 0:
                    result["explanation"] = (
                        f"factorial({nums[0]}) is undefined — "
                        f"factorial requires a non-negative integer."
                    )
                elif nums[0] > 1000:
                    # Computable, but the digit string stops being a usable
                    # answer long before this. Refuse rather than emit noise.
                    result["explanation"] = (
                        f"factorial({nums[0]}) has more digits than is useful "
                        f"to report; limit is 1000."
                    )
                else:
                    n   = nums[0]
                    val = _ext_factorial(n)
                    result.update({
                        "computation": f"{n}!",
                        "result":      val,
                        "explanation": (
                            f"{n}! = {val}. "
                            f"The product of every integer from 1 to {n} "
                            f"(0! and 1! are both 1 by definition)."
                        ),
                    })

            elif tokens and tokens[0] == "evaluate":
                # Plain arithmetic, evaluated here rather than by the
                # language model: llama3.2 answers "3 + 4 * 2" as 10
                # (precedence), 347*29 as 10073, and 1234+5678 as 3902.
                # Exactly what a deterministic engine is for.
                expr = problem.split(None, 1)[1] if len(tokens) > 1 else ""
                val = _safe_arith(expr)
                if val is None:
                    result["explanation"] = f"[cannot evaluate {expr!r}]"
                else:
                    shown = int(val) if isinstance(val, float) and \
                        val.is_integer() else val
                    result.update({
                        "computation": expr,
                        "result": shown,
                        "explanation": f"{expr} = {shown}, evaluated exactly "
                                       f"with standard operator precedence.",
                    })

            elif any(t in tokens for t in ("mod", "modular", "modexp")):
                nums = [int(t) for t in tokens if t.lstrip('-').isdigit()]
                if len(nums) < 2:
                    result["explanation"] = (
                        "Usage: 'mod A N' (A mod N) or "
                        "'modexp A E N' (A^E mod N)")
                elif "modexp" in tokens or ("exp" in tokens and len(nums) >= 3):
                    a, e, n = nums[0], nums[1], nums[2]
                    val = pow(a, e, n)
                    result.update({
                        "computation": f"modexp({a},{e},{n})",
                        "result":      val,
                        "explanation": (
                            f"{a}^{e} mod {n} = {val}. "
                            f"Computed via fast modular exponentiation "
                            f"(repeated squaring, O(log e) multiplications)."
                        ),
                    })
                else:
                    a, n = nums[0], nums[1]
                    val  = a % n
                    result.update({
                        "computation": f"{a} mod {n}",
                        "result":      val,
                        "explanation": f"{a} mod {n} = {val}.",
                    })

            # ── Discrete Math: graph coloring (via Ising annealer) ────────────
            elif any(t in tokens for t in ("color", "colour", "chromatic")):
                # "color N edges i1-j1 i2-j2 ... colors K"
                nums   = [int(t) for t in tokens if t.lstrip('-').isdigit()]
                n_nodes = nums[0] if nums else 4
                k_colors = nums[1] if len(nums) > 1 else 3
                edges  = []
                for tok in tokens:
                    if '-' in tok and not tok.startswith('-'):
                        parts = tok.split('-')
                        if len(parts) == 2:
                            try:
                                edges.append((int(parts[0]), int(parts[1])))
                            except ValueError:
                                pass
                if not edges:
                    edges = [(i, (i+1) % n_nodes) for i in range(n_nodes)]
                Q, qdesc = build_graph_color_qubo(edges, n_nodes, k_colors)
                ann = IsingAnnealer(n_vars=n_nodes * k_colors, seed=seed)
                ann.encode_qubo(Q)
                spins, energy, meta = ann.anneal(steps=400, n_restarts=8)
                coloring = _ext_read_coloring(ann.read_bits(), n_nodes, k_colors)
                valid    = _ext_check_coloring(coloring, edges)
                result.update({
                    "computation": f"graph_color(n={n_nodes}, k={k_colors})",
                    "result":      {"coloring": coloring, "valid": valid},
                    "explanation": (
                        f"Graph coloring with {k_colors} colors on {n_nodes} nodes. "
                        f"{qdesc} "
                        f"Assignment: {coloring}. "
                        f"{'Valid — no adjacent nodes share a color.' if valid else 'Conflict found — try more colors.'}"
                    ),
                })

            # ── Discrete Math: set operations ─────────────────────────────────
            elif any(t in tokens for t in ("union", "intersection",
                                           "difference", "subset", "powerset")):
                result = _ext_set_ops(tokens, problem, result)

            # ── Propositional Logic — implemented  ─────────────────
            # Parse from raw `problem` string (not lowercased tokens) so
            # variable case and math symbols (∧∨¬→↔) survive.

            elif any(t in tokens for t in ("truth", "truthtable", "evaluate")):
                # Extract expression: everything after "table" (or the keyword)
                _lower = problem.lower()
                _keys  = ("truth table", "truthtable", "truth", "evaluate")
                _body  = problem
                for _k in _keys:
                    _idx = _lower.find(_k)
                    if _idx >= 0:
                        _body = problem[_idx + len(_k):].strip()
                        break
                try:
                    tt = _ext_truth_table(_body)
                    _n_true = sum(tt['values'])
                    _n_rows = len(tt['values'])
                    _tag = ('a tautology' if tt['is_tautology']
                            else 'a contradiction' if tt['is_contradiction']
                            else 'contingent')
                    result.update({
                        "computation": f"truth_table({_body!r})",
                        "result": {
                            'is_tautology':     tt['is_tautology'],
                            'is_contradiction': tt['is_contradiction'],
                            'is_satisfiable':   tt['is_satisfiable'],
                            'n_true':           _n_true,
                            'n_rows':           _n_rows,
                        },
                        "explanation": (
                            f"Truth table for {tt['canonical']}: "
                            f"{_n_true}/{_n_rows} rows true — {_tag}."
                        ),
                    })
                except ValueError as _e:
                    result["explanation"] = (
                        f"[truth_table] parse error: {_e}. "
                        f"Usage: 'truth table p AND (q OR NOT r)'"
                    )

            elif any(t in tokens for t in ("modus", "ponens", "tollens",
                                           "inference", "deduce")):
                import re as _re
                _lower = problem.lower()
                # Strip the leading keyword
                _body = problem
                for _k in ("modus ponens", "modus tollens", "inference", "deduce"):
                    _idx = _lower.find(_k)
                    if _idx >= 0:
                        _body = problem[_idx + len(_k):].strip()
                        break
                # Split premises/conclusion by therefore/⊢/hence/conclude/=>/then
                _parts = _re.split(
                    r'\s*(?:therefore|⊢|hence|conclude|=>|then)\s*',
                    _body, maxsplit=1, flags=_re.IGNORECASE)
                if len(_parts) != 2:
                    result["explanation"] = (
                        "[inference] needs 'therefore' or ⊢ between "
                        "premises and conclusion. "
                        "Example: 'inference p, p->q therefore q'"
                    )
                else:
                    _premises_str, _conclusion = _parts
                    _premises = [p.strip() for p in _premises_str.split(',')
                                 if p.strip()]
                    try:
                        inf = _ext_inference(_premises, _conclusion.strip())
                        _v = inf['is_valid']
                        _rule = inf.get('rule')
                        _expl = f"{'Valid' if _v else 'INVALID'} inference"
                        if _rule:
                            _expl += f" ({_rule})"
                        _expl += "."
                        if not _v and 'counterexample' in inf:
                            _expl += f" Counterexample: {inf['counterexample']}."
                        result.update({
                            "computation": (
                                f"inference({_premises} ⊢ {_conclusion.strip()})"),
                            "result":      _v,
                            "explanation": _expl,
                        })
                    except ValueError as _e:
                        result["explanation"] = (
                            f"[inference] parse error: {_e}. "
                            f"Usage: 'inference p, p->q therefore q'"
                        )

            elif any(t in tokens for t in ("cnf", "dnf", "normalize")):
                _lower = problem.lower()
                _form  = 'dnf' if 'dnf' in _lower else 'cnf'
                # Strip keyword to get body
                _body = problem
                for _k in ('cnf', 'dnf', 'normalize'):
                    _idx = _lower.find(_k)
                    if _idx >= 0:
                        _body = problem[_idx + len(_k):].strip()
                        break
                try:
                    nf = _ext_normalize(_body, form=_form)
                    result.update({
                        "computation": f"{_form}({_body!r})",
                        "result":      nf['canonical'],
                        "explanation": (
                            f"{_form.upper()} of {_body}: {nf['canonical']}. "
                            f"({len(nf['clauses'])} clauses)"
                        ),
                    })
                except ValueError as _e:
                    result["explanation"] = (
                        f"[{_form}] parse error: {_e}. "
                        f"Usage: 'cnf p OR (q AND r)'"
                    )

            elif any(t in tokens for t in ("equiv", "equivalent",
                                           "tautology", "contradiction")):
                _lower = problem.lower()
                # Strip keyword
                _body = problem
                for _k in ('equivalent', 'equiv', 'tautology', 'contradiction'):
                    _idx = _lower.find(_k)
                    if _idx >= 0:
                        _body = problem[_idx + len(_k):].strip()
                        break
                # Tautology/contradiction: single expression check via truth table
                if 'tautology' in _lower or 'contradiction' in _lower:
                    try:
                        tt = _ext_truth_table(_body)
                        _asked = 'tautology' if 'tautology' in _lower else 'contradiction'
                        _yes = (tt['is_tautology'] if _asked == 'tautology'
                                else tt['is_contradiction'])
                        result.update({
                            "computation": f"is_{_asked}({_body!r})",
                            "result":      _yes,
                            "explanation": (
                                f"{tt['canonical']} is "
                                f"{'a' if _yes else 'NOT a'} {_asked}."
                            ),
                        })
                    except ValueError as _e:
                        result["explanation"] = (
                            f"[{_asked}] parse error: {_e}"
                        )
                else:
                    # Equiv: two expressions separated by ; or ==
                    if ';' in _body:
                        _a, _b = _body.split(';', 1)
                    elif '==' in _body:
                        _a, _b = _body.split('==', 1)
                    else:
                        result["explanation"] = (
                            "[equiv] needs ';' or '==' between the two formulas. "
                            "Example: 'equiv p->q ; NOT p OR q'"
                        )
                        _a = _b = None
                    if _a is not None:
                        try:
                            eq = _ext_logic_equiv(_a.strip(), _b.strip())
                            _y = eq['is_equivalent']
                            _expl = (f"{_a.strip()} "
                                     f"{'IS' if _y else 'is NOT'} equivalent to "
                                     f"{_b.strip()}.")
                            if not _y and 'counterexample' in eq:
                                _expl += f" Counterexample: {eq['counterexample']}."
                            result.update({
                                "computation": (
                                    f"equiv({_a.strip()!r}, {_b.strip()!r})"),
                                "result":      _y,
                                "explanation": _expl,
                            })
                        except ValueError as _e:
                            result["explanation"] = (
                                f"[equiv] parse error: {_e}"
                            )

            else:
                # No dispatch matched — silent clean fail.
                # Both fields None means chat's _broadcast_modules skips imprint;
                # nothing enters θ₀. Do NOT return a usage-help string here —
                # that laundered through imprint teaches the field to associate
                # unknown queries with the literal help text (poison risk).
                # If mimicry-as-fallback is wanted for the whole system, it's
                # a chat-app concern (Stage 2 integration), not the math
                # module's responsibility. Developers should call
                # VirtualQC.help() to see supported operations.
                result["result"]      = None
                result["explanation"] = None

        except Exception as exc:
            result["explanation"] = f"Computation failed: {exc}"

        return result


# ---------------------------------------------------------------------------
# Extension helpers — Discrete Math & Propositional Logic
# ---------------------------------------------------------------------------
# Naming convention: _ext_<operation>
# Each helper is pure Python / NumPy — no dependency on the LM field.
# To add a new solver:
#   1. Write _ext_<name>(...) here
#   2. Add an elif branch in VirtualQC.solve() that detects keywords
#      and calls _ext_<name>, then fills result dict
#   3. Add QA pairs in Mati/field/qa_data.py so the LM can dispatch to it
# ---------------------------------------------------------------------------

# ── Number theory ────────────────────────────────────────────────────────────


# Arithmetic, evaluated rather than guessed.
#
# ⚠️ NOT `eval()`. The expression reaches here from a user's message, so it is
# parsed to an AST and every node checked against a whitelist — numbers and
# these operators, nothing else. An unlisted node type means None, not a
# best-effort answer.
_ARITH_BINOPS = (_ast.Add, _ast.Sub, _ast.Mult, _ast.Div, _ast.FloorDiv,
                 _ast.Mod, _ast.Pow)
_ARITH_UNARY = (_ast.UAdd, _ast.USub)
_MAX_POW = 4096          # 2**4096 is already unreadable; beyond it is a DoS


def _safe_arith(expr):
    """Evaluate a numeric expression, or None if it is not purely numeric."""
    try:
        tree = _ast.parse(expr, mode="eval")
    except (SyntaxError, ValueError, MemoryError):
        return None

    def walk(node):
        if isinstance(node, _ast.Expression):
            return walk(node.body)
        if isinstance(node, _ast.Constant):
            if isinstance(node.value, bool) or not isinstance(
                    node.value, (int, float)):
                return None
            return node.value
        if isinstance(node, _ast.UnaryOp) and isinstance(node.op, _ARITH_UNARY):
            v = walk(node.operand)
            return None if v is None else (-v if isinstance(
                node.op, _ast.USub) else v)
        if isinstance(node, _ast.BinOp) and isinstance(node.op, _ARITH_BINOPS):
            a, b = walk(node.left), walk(node.right)
            if a is None or b is None:
                return None
            try:
                if isinstance(node.op, _ast.Pow):
                    if abs(b) > _MAX_POW or (abs(a) > 1 and abs(b) > 64):
                        return None       # refuse rather than hang
                    return a ** b
                if isinstance(node.op, (_ast.Div, _ast.FloorDiv, _ast.Mod)):
                    if b == 0:
                        return None       # undefined, not an answer
                return {_ast.Add: lambda x, y: x + y,
                        _ast.Sub: lambda x, y: x - y,
                        _ast.Mult: lambda x, y: x * y,
                        _ast.Div: lambda x, y: x / y,
                        _ast.FloorDiv: lambda x, y: x // y,
                        _ast.Mod: lambda x, y: x % y}[type(node.op)](a, b)
            except (ZeroDivisionError, OverflowError, ValueError):
                return None
        return None                       # anything else: names, calls, attrs

    return walk(tree)


def _ext_gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)

def _ext_lcm(a: int, b: int) -> int:
    return abs(a * b) // _ext_gcd(a, b) if a and b else 0

def _ext_is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _ext_smallest_factor(n: int) -> int:
    if n % 2 == 0:
        return 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return i
        i += 2
    return n

# ── Combinatorics ────────────────────────────────────────────────────────────

def _ext_factorial(n: int) -> int:
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def _ext_comb(n: int, r: int) -> int:
    if r < 0 or r > n:
        return 0
    r = min(r, n - r)
    num = 1
    for i in range(r):
        num = num * (n - i) // (i + 1)
    return num

def _ext_perm(n: int, r: int) -> int:
    if r < 0 or r > n:
        return 0
    result = 1
    for i in range(r):
        result *= (n - i)
    return result

# ── Graph coloring helpers ────────────────────────────────────────────────────

def build_graph_color_qubo(edges, n_nodes: int, k_colors: int):
    """
    Build QUBO matrix for k-coloring of a graph.

    Variables: x_{i,c} = 1 if node i has color c.
    Constraints:
      - Each node has exactly one color:  (Σ_c x_{i,c} - 1)² penalised
      - No adjacent nodes same color:     x_{i,c} * x_{j,c} penalised
    Returns (Q matrix, description string).
    """
    n = n_nodes * k_colors
    Q = np.zeros((n, n))
    A = 2.0   # constraint penalty weight

    def var(i, c):
        return i * k_colors + c

    # One-color-per-node constraint
    for i in range(n_nodes):
        for c in range(k_colors):
            v = var(i, c)
            Q[v, v] -= A
        for c1 in range(k_colors):
            for c2 in range(c1 + 1, k_colors):
                v1, v2 = var(i, c1), var(i, c2)
                Q[v1, v2] += 2 * A

    # Edge conflict constraint
    for (i, j) in edges:
        for c in range(k_colors):
            v1, v2 = var(i, c), var(j, c)
            Q[v1, v2] += A

    desc = (f"QUBO: {n_nodes} nodes, {k_colors} colors, "
            f"{len(edges)} edges, {n} binary variables.")
    return Q, desc

def _ext_read_coloring(bits, n_nodes: int, k_colors: int) -> dict:
    coloring = {}
    for i in range(n_nodes):
        for c in range(k_colors):
            if bits[i * k_colors + c]:
                coloring[i] = c
                break
        else:
            coloring[i] = -1   # uncolored (annealer didn't assign)
    return coloring

def _ext_check_coloring(coloring: dict, edges: list) -> bool:
    return all(coloring.get(i, -1) != coloring.get(j, -1)
               for i, j in edges)

# ── Set operations ────────────────────────────────────────────────────────────

def _ext_parse_set(token: str) -> set:
    """Parse '{1,2,3}' or '1,2,3' into a Python set of strings."""
    clean = token.strip('{}').strip()
    if not clean:
        return set()
    return set(x.strip() for x in clean.split(',') if x.strip())

def _ext_set_ops(tokens: list, problem: str, result: dict) -> dict:
    """
    Dispatch set operations from token list.
    Expected patterns (flexible):
      union {a,b,c} {b,c,d}
      intersection {a,b,c} {b,c,d}
      difference {a,b,c} {b,c,d}
      subset {a,b} {a,b,c}
      powerset {a,b,c}
    """
    # Extract set literals from the raw problem string
    import re
    raw_sets = re.findall(r'\{[^}]*\}', problem)

    if "powerset" in tokens:
        if raw_sets:
            A = _ext_parse_set(raw_sets[0])
        else:
            A = set()
        n   = len(A)
        elems = sorted(A)
        ps  = []
        for mask in range(1 << n):
            ps.append(frozenset(elems[i] for i in range(n) if mask & (1 << i)))
        result.update({
            "computation": f"powerset({sorted(A)})",
            "result":      [sorted(s) for s in ps],
            "explanation": (
                f"Power set of {sorted(A)}: {2**n} subsets "
                f"(including empty set and the set itself)."
            ),
        })
        return result

    if len(raw_sets) < 2:
        result["explanation"] = (
            "Set operations need two sets. "
            "Usage: 'union {a,b,c} {b,c,d}'"
        )
        return result

    A = _ext_parse_set(raw_sets[0])
    B = _ext_parse_set(raw_sets[1])

    if "intersection" in tokens:
        res = A & B
        op  = "∩"
    elif "difference" in tokens:
        res = A - B
        op  = "\\"
    elif "subset" in tokens:
        res = A.issubset(B)
        result.update({
            "computation": f"{sorted(A)} ⊆ {sorted(B)}",
            "result":      res,
            "explanation": f"{sorted(A)} {'is' if res else 'is not'} a subset of {sorted(B)}.",
        })
        return result
    else:   # union default
        res = A | B
        op  = "∪"

    result.update({
        "computation": f"{sorted(A)} {op} {sorted(B)}",
        "result":      sorted(res),
        "explanation": f"{sorted(A)} {op} {sorted(B)} = {sorted(res)}.",
    })
    return result


# ---------------------------------------------------------------------------
# Propositional Logic — parser + evaluator + primitives
# ---------------------------------------------------------------------------
# Implemented . AST node types:
#   ('var', name)                  — variable reference
#   ('not', child)                 — negation
#   ('and'|'or'|'implies'|'equiv', left, right) — binary connectives
# ---------------------------------------------------------------------------

# Accepted syntax (English + math + ASCII):
_LOGIC_SYMBOLS = {
    'NOT':     ['NOT', 'not', '¬', '~', '!'],
    'AND':     ['AND', 'and', '∧', '&'],
    'OR':      ['OR',  'or',  '∨', '|'],
    'IMPLIES': ['IMPLIES', 'implies', 'imply', '→', '->', '=>'],
    'EQUIV':   ['EQUIV', 'equiv', 'IFF', 'iff', '↔', '≡', '<->', '<=>'],
}
_LOGIC_TOKEN_MAP = {sym: tok for tok, syms in _LOGIC_SYMBOLS.items()
                              for sym in syms}


def _logic_tokenize(expr: str) -> list:
    """Split expression into (TYPE, value) token list. Case-preserving for VARs."""
    import re
    # One pass, longest operator first. Successive `str.replace` calls corrupted
    # the longer forms: spacing '<->' and then spacing '->' rewrote it to
    # '< -> ', and the stray '<' failed to tokenize — so every biconditional
    # raised, silently disabling an operator the parser fully supports.
    _OPS = ['<->', '<=>', '->', '=>', '(', ')',
            '∧', '∨', '¬', '→', '↔', '≡', '~', '!', '&', '|']
    s = re.sub('|'.join(re.escape(op) for op in _OPS),
               lambda m: f' {m.group(0)} ', expr)
    tokens = []
    for word in s.split():
        if word == '(':
            tokens.append(('LPAREN', word))
        elif word == ')':
            tokens.append(('RPAREN', word))
        elif word in _LOGIC_TOKEN_MAP:
            tokens.append((_LOGIC_TOKEN_MAP[word], word))
        elif re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', word):
            tokens.append(('VAR', word.lower()))
        else:
            raise ValueError(f"unrecognized logic token: {word!r} in {expr!r}")
    return tokens


class _LogicParser:
    """Recursive-descent parser. Precedence (low to high):
       EQUIV, IMPLIES, OR, AND, NOT."""

    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def _peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else (None, None)

    def _consume(self, expected=None):
        tok = self._peek()
        if expected and tok[0] != expected:
            raise ValueError(f"expected {expected}, got {tok}")
        self.pos += 1
        return tok

    def parse(self):
        ast = self._biconditional()
        if self.pos < len(self.tokens):
            raise ValueError(f"unexpected trailing tokens: {self.tokens[self.pos:]}")
        return ast

    def _biconditional(self):
        left = self._implication()
        while self._peek()[0] == 'EQUIV':
            self._consume()
            right = self._implication()
            left = ('equiv', left, right)
        return left

    def _implication(self):
        left = self._disjunction()
        if self._peek()[0] == 'IMPLIES':
            self._consume()
            right = self._implication()   # right-associative
            return ('implies', left, right)
        return left

    def _disjunction(self):
        left = self._conjunction()
        while self._peek()[0] == 'OR':
            self._consume()
            right = self._conjunction()
            left = ('or', left, right)
        return left

    def _conjunction(self):
        left = self._negation()
        while self._peek()[0] == 'AND':
            self._consume()
            right = self._negation()
            left = ('and', left, right)
        return left

    def _negation(self):
        if self._peek()[0] == 'NOT':
            self._consume()
            return ('not', self._negation())
        return self._atom()

    def _atom(self):
        tok = self._peek()
        if tok[0] == 'LPAREN':
            self._consume()
            inner = self._biconditional()
            self._consume('RPAREN')
            return inner
        if tok[0] == 'VAR':
            self._consume()
            return ('var', tok[1])
        raise ValueError(f"expected atom, got {tok}")


def _logic_parse(expr: str):
    return _LogicParser(_logic_tokenize(expr)).parse()


def _logic_eval(ast, env: dict) -> bool:
    tag = ast[0]
    if tag == 'var':
        return env[ast[1]]
    if tag == 'not':
        return not _logic_eval(ast[1], env)
    l = _logic_eval(ast[1], env)
    r = _logic_eval(ast[2], env)
    if tag == 'and':     return l and r
    if tag == 'or':      return l or r
    if tag == 'implies': return (not l) or r
    if tag == 'equiv':   return l == r
    raise ValueError(f"unknown node type: {tag}")


def _logic_variables(ast) -> set:
    if ast[0] == 'var':
        return {ast[1]}
    if ast[0] == 'not':
        return _logic_variables(ast[1])
    return _logic_variables(ast[1]) | _logic_variables(ast[2])


def _logic_format(ast) -> str:
    tag = ast[0]
    if tag == 'var':
        return ast[1]
    if tag == 'not':
        inner = ast[1]
        if inner[0] == 'var':
            return f"¬{inner[1]}"
        return f"¬({_logic_format(inner)})"
    op = {'and': '∧', 'or': '∨', 'implies': '→', 'equiv': '↔'}[tag]
    return f"({_logic_format(ast[1])} {op} {_logic_format(ast[2])})"


# ── NNF / CNF / DNF transformations ─────────────────────────────────────────

def _logic_eliminate_equiv(ast):
    tag = ast[0]
    if tag == 'var':  return ast
    if tag == 'not':  return ('not', _logic_eliminate_equiv(ast[1]))
    l = _logic_eliminate_equiv(ast[1])
    r = _logic_eliminate_equiv(ast[2])
    if tag == 'equiv':
        return ('and', ('implies', l, r), ('implies', r, l))
    return (tag, l, r)

def _logic_eliminate_implies(ast):
    tag = ast[0]
    if tag == 'var':  return ast
    if tag == 'not':  return ('not', _logic_eliminate_implies(ast[1]))
    l = _logic_eliminate_implies(ast[1])
    r = _logic_eliminate_implies(ast[2])
    if tag == 'implies':
        return ('or', ('not', l), r)
    return (tag, l, r)

def _logic_push_negations(ast):
    """Negation normal form — NOT only in front of variables."""
    tag = ast[0]
    if tag == 'var':
        return ast
    if tag == 'not':
        inner = ast[1]
        it = inner[0]
        if it == 'var':
            return ast
        if it == 'not':          # double negation
            return _logic_push_negations(inner[1])
        if it == 'and':          # ¬(p∧q) → ¬p ∨ ¬q
            return ('or',
                    _logic_push_negations(('not', inner[1])),
                    _logic_push_negations(('not', inner[2])))
        if it == 'or':           # ¬(p∨q) → ¬p ∧ ¬q
            return ('and',
                    _logic_push_negations(('not', inner[1])),
                    _logic_push_negations(('not', inner[2])))
    return (tag,
            _logic_push_negations(ast[1]),
            _logic_push_negations(ast[2]))

def _logic_distribute_or_over_and(ast):
    """CNF distribution: X ∨ (A ∧ B) → (X ∨ A) ∧ (X ∨ B)."""
    tag = ast[0]
    if tag in ('var', 'not'):
        return ast
    l = _logic_distribute_or_over_and(ast[1])
    r = _logic_distribute_or_over_and(ast[2])
    if tag == 'and':
        return ('and', l, r)
    # tag == 'or'
    if l[0] == 'and':
        return _logic_distribute_or_over_and(
            ('and', ('or', l[1], r), ('or', l[2], r)))
    if r[0] == 'and':
        return _logic_distribute_or_over_and(
            ('and', ('or', l, r[1]), ('or', l, r[2])))
    return ('or', l, r)

def _logic_distribute_and_over_or(ast):
    """DNF distribution: X ∧ (A ∨ B) → (X ∧ A) ∨ (X ∧ B)."""
    tag = ast[0]
    if tag in ('var', 'not'):
        return ast
    l = _logic_distribute_and_over_or(ast[1])
    r = _logic_distribute_and_over_or(ast[2])
    if tag == 'or':
        return ('or', l, r)
    # tag == 'and'
    if l[0] == 'or':
        return _logic_distribute_and_over_or(
            ('or', ('and', l[1], r), ('and', l[2], r)))
    if r[0] == 'or':
        return _logic_distribute_and_over_or(
            ('or', ('and', l, r[1]), ('and', l, r[2])))
    return ('and', l, r)

def _logic_extract_clauses(ast, form: str) -> list:
    """CNF: list of OR-clauses; DNF: list of AND-clauses.
    Each clause is a list of (var_name, polarity) tuples."""
    outer, inner = ('and', 'or') if form == 'cnf' else ('or', 'and')

    def split_by(node, op):
        if node[0] == op:
            return split_by(node[1], op) + split_by(node[2], op)
        return [node]

    def literals(node):
        parts = split_by(node, inner)
        out = []
        for p in parts:
            if p[0] == 'var':
                out.append((p[1], True))
            elif p[0] == 'not' and p[1][0] == 'var':
                out.append((p[1][1], False))
            else:
                out.append(('?', True))   # shouldn't happen after NNF
        return out

    return [literals(p) for p in split_by(ast, outer)]


def _logic_detect_rule(premises, conclusion):
    """Match against 7 named inference rules. Returns rule name or None."""
    n = len(premises)

    if n == 1:
        p = premises[0]
        # Addition: {X} ⊢ X ∨ Y
        if conclusion[0] == 'or' and (p == conclusion[1] or p == conclusion[2]):
            return 'Addition'
        # Simplification: {X ∧ Y} ⊢ X (or Y)
        if p[0] == 'and' and (conclusion == p[1] or conclusion == p[2]):
            return 'Simplification'

    if n == 2:
        # Modus Ponens: {X, X → Y} ⊢ Y
        for i, j in [(0, 1), (1, 0)]:
            if premises[j][0] == 'implies':
                _, ant, cons = premises[j]
                if premises[i] == ant and conclusion == cons:
                    return 'Modus Ponens'

        # Modus Tollens: {X → Y, ¬Y} ⊢ ¬X
        for i, j in [(0, 1), (1, 0)]:
            if premises[i][0] == 'implies':
                _, ant, cons = premises[i]
                if premises[j] == ('not', cons) and conclusion == ('not', ant):
                    return 'Modus Tollens'

        # Hypothetical Syllogism: {X → Y, Y → Z} ⊢ X → Z
        if premises[0][0] == 'implies' and premises[1][0] == 'implies':
            for order in [(0, 1), (1, 0)]:
                a, b = premises[order[0]], premises[order[1]]
                if a[2] == b[1] and conclusion == ('implies', a[1], b[2]):
                    return 'Hypothetical Syllogism'

        # Disjunctive Syllogism: {X ∨ Y, ¬X} ⊢ Y
        for i, j in [(0, 1), (1, 0)]:
            if premises[i][0] == 'or':
                _, l, r = premises[i]
                if premises[j] == ('not', l) and conclusion == r:
                    return 'Disjunctive Syllogism'
                if premises[j] == ('not', r) and conclusion == l:
                    return 'Disjunctive Syllogism'

        # Conjunction: {X, Y} ⊢ X ∧ Y
        if conclusion[0] == 'and' \
                and {premises[0], premises[1]} == {conclusion[1], conclusion[2]}:
            return 'Conjunction'

    return None


# ── Public extension functions ──────────────────────────────────────────────

def _ext_truth_table(expr: str) -> dict:
    ast = _logic_parse(expr)
    vars_list = sorted(_logic_variables(ast))
    n = len(vars_list)
    assignments = []
    values = []
    for i in range(2 ** n):
        env = {v: bool((i >> j) & 1) for j, v in enumerate(vars_list)}
        assignments.append(dict(env))
        values.append(_logic_eval(ast, env))
    return {
        'expression':       expr,
        'canonical':        _logic_format(ast),
        'variables':        vars_list,
        'assignments':      assignments,
        'values':           values,
        'is_tautology':     all(values),
        'is_contradiction': not any(values),
        'is_satisfiable':   any(values),
    }


def _ext_logic_equiv(expr_a: str, expr_b: str) -> dict:
    a = _logic_parse(expr_a)
    b = _logic_parse(expr_b)
    all_vars = sorted(_logic_variables(a) | _logic_variables(b))
    for i in range(2 ** len(all_vars)):
        env = {v: bool((i >> j) & 1) for j, v in enumerate(all_vars)}
        if _logic_eval(a, env) != _logic_eval(b, env):
            return {
                'expression_a':   expr_a,
                'expression_b':   expr_b,
                'is_equivalent':  False,
                'counterexample': env,
            }
    return {
        'expression_a':  expr_a,
        'expression_b':  expr_b,
        'is_equivalent': True,
    }


def _ext_inference(premises: list, conclusion: str) -> dict:
    p_asts = [_logic_parse(p) for p in premises]
    c_ast  = _logic_parse(conclusion)
    all_vars = set()
    for a in p_asts:
        all_vars |= _logic_variables(a)
    all_vars |= _logic_variables(c_ast)
    all_vars = sorted(all_vars)

    counterexample = None
    for i in range(2 ** len(all_vars)):
        env = {v: bool((i >> j) & 1) for j, v in enumerate(all_vars)}
        if all(_logic_eval(a, env) for a in p_asts):
            if not _logic_eval(c_ast, env):
                counterexample = dict(env)
                break

    is_valid = counterexample is None
    result = {
        'premises':   premises,
        'conclusion': conclusion,
        'is_valid':   is_valid,
    }
    if counterexample is not None:
        result['counterexample'] = counterexample
    if is_valid:
        rule = _logic_detect_rule(p_asts, c_ast)
        if rule:
            result['rule'] = rule
    return result


def _ext_normalize(expr: str, form: str = 'cnf') -> dict:
    form = form.lower()
    if form not in ('cnf', 'dnf'):
        raise ValueError(f"form must be 'cnf' or 'dnf', got {form!r}")
    ast = _logic_parse(expr)
    ast = _logic_eliminate_equiv(ast)
    ast = _logic_eliminate_implies(ast)
    ast = _logic_push_negations(ast)
    if form == 'cnf':
        ast = _logic_distribute_or_over_and(ast)
    else:
        ast = _logic_distribute_and_over_or(ast)
    return {
        'expression': expr,
        'form':       form.upper(),
        'canonical':  _logic_format(ast),
        'clauses':    _logic_extract_clauses(ast, form),
    }


# ---------------------------------------------------------------------------
# EXTENSION GUIDE — how to add a new solver
# ---------------------------------------------------------------------------
#
# STEP 1 — Write the helper function above this block:
#
#   def _ext_<name>(<params>) -> <type>:
#       """
#       One-line description.
#       Source: <which text / section>.
#       """
#       # implementation
#       return result
#
#
# STEP 2 — Add an elif branch in VirtualQC.solve(), following this pattern:
#
#   elif any(t in tokens for t in ("<keyword1>", "<keyword2>")):
#       # parse needed values out of tokens / problem string
#       val = _ext_<name>(<parsed args>)
#       result.update({
#           "computation": f"<name>(<args>)",
#           "result":      val,
#           "explanation": (
#               f"<plain English description of what was computed and how.>"
#           ),
#       })
#
#   Place it BEFORE the final `else` branch.
#
#
# STEP 3 — Add QA pairs in Mati/field/qa_data.py:
#
#   ("natural language request matching the operation",
#    "plain English answer describing the result"),
#
#   Example:
#   ("what is the gcd of 48 and 18",
#    "The gcd of 48 and 18 is 6, found by the Euclidean algorithm."),
#
#
# STEP 4 — Update the unrecognised-pattern message in the else branch
#          to mention the new operation name.
#
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== field math smoke test ===\n")

    print("--- factor(15) ---")
    p, q = factor(15, seed=42)
    print(f"  15 = {p} × {q}\n")

    print("--- factor(21) ---")
    p, q = factor(21, seed=7)
    print(f"  21 = {p} × {q}\n")

    print("--- shor_period(a=7, N=15) ---")
    f = spawn_field(n_nodes=64, seed=0)
    r = shor_period(f, 7, 15)
    print(f"  period = {r}  (expected 4)\n")

    print("--- grover_search(N=64, target=17) ---")
    f = spawn_field(n_nodes=64, geometry="ring", seed=1)
    found = grover_search(f, target_node=17)
    print(f"  found  = {found}  (expected 17)\n")

    print("--- VirtualQC.solve('factor 35') ---")
    res = VirtualQC.solve("factor 35", seed=3)
    print(f"  result      : {res['result']}")
    print(f"  explanation : {res['explanation']}\n")

    print("--- VirtualQC.solve('period 2 15') ---")
    res = VirtualQC.solve("period 2 15", seed=5)
    print(f"  result      : {res['result']}")
    print(f"  explanation : {res['explanation']}\n")

    print("--- VirtualQC.solve('search 32 target 11') ---")
    res = VirtualQC.solve("search 32 target 11", seed=9)
    print(f"  result      : {res['result']}")
    print(f"  explanation : {res['explanation']}\n")

    print("--- VirtualQC stateful API ---")
    qc = VirtualQC.spawn(n_nodes=32, geometry="grid", seed=2)
    qc.evolve(steps=10).kick(node=5).evolve(steps=5)
    snap = qc.measure()
    print(f"  steps            : {snap['step']}")
    print(f"  coherence_pressure: {snap['coherence_pressure']:.6f}")
    print(f"  entropy          : {snap['entropy']:.6f}")
    print(f"  circular_variance: {snap['circular_variance']:.6f}")
    print(f"  dominant_frequency: {snap['dominant_frequency']}")
    print(f"  log entries      : {len(snap['log'])}")
    print("--- IsingAnnealer: Max-Cut on K4 ---")
    edges_k4 = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    Q_mc, _  = build_max_cut_qubo(edges_k4, n_nodes=4)
    ann      = IsingAnnealer(n_vars=4, seed=42)
    ann.encode_qubo(Q_mc)
    spins, energy, meta = ann.anneal(steps=300, n_restarts=5)
    bits     = ann.read_bits()
    cut_n    = sum(1 for i,j in edges_k4
                   if (spins[i] != spins[j]))
    print(f"  spins      : {spins.tolist()}")
    print(f"  cut size   : {cut_n}/6  (optimal=4 for K4)")
    print(f"  energy     : {energy:.4f}\n")

    print("--- IsingAnnealer: 3-SAT demo ---")
    clauses = [[1, 2, 3], [-1, -2, 3], [1, -2, -3]]
    Q_sat, _  = build_3sat_qubo(clauses, n_vars=3)
    ann2 = IsingAnnealer(n_vars=3, seed=7)
    ann2.encode_qubo(Q_sat)
    sp2, e2, _ = ann2.anneal(steps=300, n_restarts=5)
    bits2 = ann2.read_bits().tolist()
    sat_n = sum(1 for c in clauses
                if any((l>0 and bits2[abs(l)-1]==1) or
                       (l<0 and bits2[abs(l)-1]==0) for l in c))
    print(f"  assignment : x1={bool(bits2[0])} x2={bool(bits2[1])} x3={bool(bits2[2])}")
    print(f"  satisfied  : {sat_n}/{len(clauses)} clauses\n")

    print("--- VirtualQC.solve('max-cut 4 edges 0-1 1-2 2-3 3-0 0-2 1-3') ---")
    res = VirtualQC.solve("max-cut 4 edges 0-1 1-2 2-3 3-0 0-2 1-3", seed=42)
    print(f"  result      : {res['result']}")
    print(f"  explanation : {res['explanation'][:120]}...\n")

    print("--- VirtualQC.solve('satisfy 3 1 2 3 -1 -2 3') ---")
    res = VirtualQC.solve("satisfy 3 1 2 3 -1 -2 3", seed=0)
    print(f"  result      : {res['result']}")
    print(f"  explanation : {res['explanation'][:120]}...\n")

    print("\n=== all tests passed ===")
