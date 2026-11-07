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
FastPhaseField — vectorized drop-in replacement for PhaseField.

Overrides the three hot methods with fully numpy-vectorized equivalents:
    _gradient()       O(N) Python loop  →  (N,N) numpy broadcast
    _carrier_inflow() O(N) Python loop  →  (N,N) numpy broadcast
    step()            Python node loop  →  vectorized arrays + write-back

Also precomputes the distance matrix once in _build_coupling (positions
are fixed between migrate() calls) eliminating O(N²) recomputation per step.

Typical speedup: 20-50× on CPU relative to PhaseField.
v20.3 is untouched.
"""

import numpy as np
import sys, os
# (vendored: no sys.path hack — imports are package-relative)
from .base import PhaseField, l_0, g_eff, beta, DIST_EPS, STICKINESS

# Floor inside √m for the quantum-potential term — keeps ∇²√m/√m finite at
# empty nodes.  Small vs the observed θ₀ mass scale (mean|m| ≈ 0.02).
Q_EPS = 1e-6


class FastPhaseField(PhaseField):
    """
    Vectorized PhaseField — identical dynamics, faster execution.
    Safe to use anywhere PhaseField is expected.

    Phase 6 extensions (disabled by default):
        max_rho      : fermion exclusion — cap mass per node, redistribute excess
        eta_decohere : decoherence      — gradient-proportional phase noise
    """

    # Phase 6: set per-instance to enable (e.g. mode.max_rho = 0.05)
    max_rho      = None   # fermion exclusion cap; None = disabled
    eta_decohere = 0.0    # decoherence noise strength; 0 = disabled

    # TFTQ: quantum-potential (von-Weizsäcker / Fisher) term.  0 = disabled,
    # exactly the prior dynamics.  See _q_term() for the physics.
    q_coeff      = 0.0

    # Phase 7: Hebbian coupling modifier — set by init_j()
    J       = None        # (N, N) float32 modifier; None = disabled
    _J_mask = None        # sparsity mask: only update J where W_geom > 0

    # ── Coupling build (adds precomputed distance matrix + omega array) ───────
    def _build_coupling(self):
        super()._build_coupling()
        # Precompute (N,N) distance matrix — reused every step in _gradient
        pos            = np.array([n.position for n in self.nodes])
        delta          = pos[:, None, :] - pos[None, :, :]   # (N, N, 3)
        self._dist_mat = np.linalg.norm(delta, axis=2) + DIST_EPS  # (N, N)
        # Cache omega array — rebuilt whenever node omegas are assigned
        self._sync_omegas()

    def _sync_omegas(self):
        """Rebuild cached omega vector from node attributes."""
        self._omegas = np.array([n.omega for n in self.nodes], dtype=float)

    # ── Effective coupling (with Hebbian modifier) ────────────────────────────
    def _get_W(self):
        """
        Return effective coupling matrix.

        When J is set: W_eff = W_geom * (1 + J), row-normalised.
        Nodes whose coupling was strengthened by Hebbian experience route phase
        more strongly; weakened edges route less.  Geometry sets the prior;
        experience modulates it.

        When J is None: returns W_geom unchanged (zero overhead).
        """
        if self.J is None:
            return self.W
        W_eff     = self.W * (1.0 + self.J)
        row_sums  = W_eff.sum(axis=1, keepdims=True) + 1e-8
        return W_eff / row_sums

    # ── Vectorized gradient ───────────────────────────────────────────────────
    def _gradient(self):
        """∇θ at each node — fully vectorised sin-difference weighted sum."""
        thetas = self._thetas()                                   # (N,)
        diffs  = np.sin(thetas[None, :] - thetas[:, None])       # (N, N)
        return np.sum(self._get_W() * diffs / self._dist_mat, axis=1)   # (N,)

    # ── Quantum potential (TFTQ) ──────────────────────────────────────────────
    def _q_term(self):
        """
        ∇²√m / √m  — the Madelung quantum-potential contribution to ∂ₜθ.

        TFTQ result: the ONLY thing separating TFT-as-written from unitary QM is
        Q = −(ħ²/2m)·∇²√m/√m, which equals the Fisher information of the density.
        Without it the amplitude sector is a heat equation (diffusive), which is
        mechanically why a pure phase field cannot hold localized memory. Q is
        the anti-diffusion term: at a real-amplitude state Schrödinger gives
        ∂ₜθ = +(ħ/2m)∇²√ρ/√ρ, so it enters vtheta with a POSITIVE sign.

        Discrete Laplacian mirrors _gradient() exactly — same W, same 1/d
        weighting — but on √m (a real scalar) instead of sin(Δθ) (on S¹).
        Coefficient is supplied by the caller via q_coeff (TFT units: ħ²/2m ↔ E₀ℓ₀²).
        """
        m   = np.clip(self._masses(), 0.0, None)
        s   = np.sqrt(m + Q_EPS)                                  # (N,)
        lap = np.sum(self._get_W() * (s[None, :] - s[:, None])
                     / self._dist_mat, axis=1)                    # (N,)
        return lap / s

    # ── Vectorized carrier inflow ─────────────────────────────────────────────
    def _carrier_inflow(self):
        """Uphill phase flow — fully vectorised."""
        thetas = self._thetas()                                   # (N,)
        diffs  = np.sin(thetas[None, :] - thetas[:, None])       # (N, N)
        return g_eff * np.sum(self._get_W() * np.maximum(diffs, 0), axis=1)  # (N,)

    # ── Vectorized step ───────────────────────────────────────────────────────
    def step(self, dt=0.05, locked=None):
        """One evolution step — vectorised compute, minimal Python loop."""
        thetas  = self._thetas()
        grad    = self._gradient()
        lam     = self._compute_lambda()
        inflow  = self._carrier_inflow()
        masses  = self._masses()
        vthetas = np.array([n.vtheta for n in self.nodes], dtype=float)

        # Boolean mask of free (unlocked) nodes
        if locked:
            locked_set = locked if isinstance(locked, set) else set(locked)
            free = np.array([i not in locked_set for i in range(self.N)], dtype=bool)
        else:
            free = np.ones(self.N, dtype=bool)

        # Phase update — intrinsic frequency + coupling gradient
        vthetas[free] += self._omegas[free]                    # intrinsic drive
        vthetas[free] += (1.0 + lam) * (l_0 ** 2) * grad[free]
        # TFTQ quantum potential — opt-in anti-diffusion term (q_coeff=0 → no-op)
        if self.q_coeff != 0.0:
            vthetas[free] += self.q_coeff * self._q_term()[free]
        vthetas[free] *= (1.0 - STICKINESS)
        thetas_new = thetas.copy()
        thetas_new[free] = (thetas[free] + dt * vthetas[free]) % (2.0 * np.pi)

        # ── Phase 6: Decoherence — gradient-proportional phase noise ──────────
        # Unstable configurations (high |∇θ|) accumulate noise and wash out.
        # Stable attractors resist noise. Acts as field-native regularisation.
        if self.eta_decohere > 0.0:
            noise = (self.eta_decohere * np.abs(grad)
                     * np.random.uniform(-1.0, 1.0, self.N))
            thetas_new[free] = (thetas_new[free] + noise[free]) % (2.0 * np.pi)

        # Mass update (all nodes, no lock)
        new_mass = np.clip(masses + dt * (inflow - beta * masses), 0.0, None)

        # ── Phase 6: Fermion exclusion — cap ρ per node, redistribute excess ──
        # Prevents mass collapsing onto a few nodes (mode collapse).
        # Excess flows outward along coupling weights, forcing distributed use.
        if self.max_rho is not None:
            excess   = np.maximum(new_mass - self.max_rho, 0.0)
            new_mass = np.minimum(new_mass, self.max_rho)
            new_mass += self.W.T @ excess          # redistribute to neighbours
            new_mass  = np.minimum(new_mass, self.max_rho)   # re-cap

        # Write back (unavoidable Python loop, but O(N) not O(N²))
        for i, node in enumerate(self.nodes):
            node.vtheta = float(vthetas[i])
            node.theta  = float(thetas_new[i])
        self._set_masses(new_mass)

    # ── Alternating coupling (Gemma 4 5:1 local/global pattern) ─────────────
    def evolve_alternating(self, steps, locked=None, alt_W=None, ratio=5):
        """
        Evolve with alternating fine (local) and coarse (global) coupling.

        Gemma 4 uses a 5:1 ratio of local sliding-window layers to global
        full-context layers, with the final layer always global.  Mapped to
        Each evolution step is a 'layer'; fine coupling = local attention,
        coarse coupling = global attention.

        At step k:
            k % (ratio+1) == ratio  → coarse W (global synchronisation)
            k == steps-1             → coarse W (final step always global)
            otherwise                → fine W  + Hebbian J (local refinement)

        When alt_W is None falls back to standard evolve() with no overhead.

        Args:
            steps  : total evolution steps
            locked : set of locked node indices
            alt_W  : coarse (global) coupling matrix (N, N); None = standard
            ratio  : fine:coarse step ratio (default 5, matching Gemma 4)
        """
        if alt_W is None:
            self.evolve(steps=steps, locked=locked)
            return

        orig_W = self.W
        orig_J = self.J

        for k in range(steps):
            use_coarse = (k % (ratio + 1) == ratio) or (k == steps - 1)
            if use_coarse:
                # Global step: pure coarse geometry — no Hebbian modifier.
                # J was learned on the fine coupling; applying it to alt_W
                # would distort the long-range synchronisation signal.
                self.W = alt_W
                self.J = None
            else:
                # Local step: fine geometry + Hebbian experience.
                self.W = orig_W
                self.J = orig_J
            self.step(locked=locked)

        self.W = orig_W
        self.J = orig_J

    # ── Hebbian coupling learning ─────────────────────────────────────────────
    def init_j(self):
        """
        Initialise Hebbian coupling modifier J to zeros.

        J = 0 everywhere → W_eff = W_geom (no change until experience accumulates).
        _J_mask restricts updates to geometrically coupled node pairs only —
        experience modulates the field's natural connectivity but cannot rewire it.
        """
        self.J       = np.zeros((self.N, self.N), dtype=np.float32)
        self._J_mask = (self.W > 0).astype(np.float32)

    def update_j(self, alpha_j=0.001, decay=0.0001):
        """
        Hebbian update: nodes phase-correlated after evolution strengthen their edge.

        Rule:   ΔJ_ij = α_J · sin(θ_i) · sin(θ_j)
        Decay:  J ← (1 − decay) · J + ΔJ    (prevents runaway growth)
        Mask:   J_ij = 0 where W_geom_ij = 0 (geometry prior preserved)
        Bounds: J ∈ [−0.5, 1.0]              (W_eff stays positive, max 2× W)

        Called once per round (after full Q→A evolution), not per step.
        The current field state is whatever the last forward() left behind —
        the node phases after evolving the answer context.
        """
        if self.J is None:
            return
        thetas = self._thetas()                              # (N,)
        sin_t  = np.sin(thetas).astype(np.float32)
        delta  = alpha_j * np.outer(sin_t, sin_t)           # (N, N)
        self.J = (1.0 - decay) * self.J + delta
        self.J *= self._J_mask                               # enforce geometry sparsity
        np.clip(self.J, -0.5, 1.0, out=self.J)

    # ── Migration hook (rebuild distance matrix after nodes move) ─────────────
    def migrate(self, boundary_nodes, target_thetas, error_magnitude, rate=0.05):
        super().migrate(boundary_nodes, target_thetas, error_magnitude, rate)
        pos            = np.array([n.position for n in self.nodes])
        delta          = pos[:, None, :] - pos[None, :, :]
        self._dist_mat = np.linalg.norm(delta, axis=2) + DIST_EPS

