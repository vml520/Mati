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
Multi-scale field routing via mode decomposition.

From cheat sheet §8:
    θ(x) = Σₙ θₙ(x)
    gₙ = Eₙ / ℓ₀           (coupling strength per mode)
    ξₙ = ξ₀ · rⁿ            (coupling cutoff, octave-spaced)

Each mode n is an independent FastPhaseField evolving the SAME context
boundary conditions but with a different spatial coupling scale.

Short-range modes (small ξ) capture local character patterns.
Long-range modes (large ξ) capture phrase/word-level patterns.

The output features from all modes are concatenated before the
projection layer — equivalent to multi-head attention but derived
entirely from phase field coherence at different scales.

Gauge projection (P² = P): implicit in the row-normalisation of W —
only phase-coherent subspaces (eigenvectors with eigenvalue 1) survive.

REGIME (tft-corrected-derivations §2):
    All modes run the DISSIPATIVE FIRST-ORDER dynamics (Kuramoto/gradient
    flow). The coupling cutoff ξₙ per mode approximates the kink width
    w = ℓ₀/√Λ from the continuous sine-Gordon action at each scale — ξₙ
    is not Λ, but both set the spatial scale at which phase gradients
    resolve into distinct tokens.

λ_eff — Λ connection (tft-corrected-derivations §2):
    The discrete λ_eff (dynamic, input-driven) and the continuous Λ
    (fixed dimensionless coupling in V(θ) = (Λ/ℓ₀²)(1−cos θ)) both
    modulate the Laplacian term. They are not equal numerically. The
    kink mass M_K = 8√Λ·E₀/c² and kink width w = ℓ₀/√Λ are exact
    results of the conservative regime (1D, verified); they inform the
    choice of ξₙ scale spacing but do not appear directly in the
    dissipative update rule.
"""

import numpy as np
import sys, os
# (vendored: no sys.path hack — imports are package-relative)
from .base import (
    Node, SPACE_L, N_NODES, N_BOUNDARY, E_0, DIST_EPS, STICKINESS,
    _boundary_positions,
)
from .fast_field import FastPhaseField

# Physical omega bound: ω = E₀·m/ℏ with ℏ = dt (field time unit) and m_max = 0.05 (max_rho).
# Derived: OMEGA_MAX = E₀ · max_rho / dt = 1.0 · 0.05 / 0.05 = 1.0.
# At this cap: vθ_ss = OMEGA_MAX / STICKINESS = 20, giving Δθ = dt·vθ = 1 rad/step.
# The runaway hit 1.6×10⁸ — six orders of magnitude above this physical bound.
_DT       = 0.05
_MAX_RHO  = 0.05   # fermion exclusion cap (max_rho in trainer)
OMEGA_MAX = E_0 * _MAX_RHO / _DT   # = 1.0


# ── Mode field: custom coupling scale ─────────────────────────────────────────
class ModeField(FastPhaseField):
    """
    FastPhaseField with a custom coupling cutoff (xi) and width (sigma).
    Identical dynamics to FastPhaseField — only the connectivity differs.
    """

    def __init__(self, nodes, xi, sigma):
        self.xi    = float(xi)
        self.sigma = float(sigma)
        # Call PhaseField.__init__ directly to use our _build_coupling
        self.nodes      = nodes
        self.N          = len(nodes)
        self.W          = None
        self.lambda_eff = 0.0
        self._build_coupling()

    def _build_coupling(self):
        """Build W and distance matrix using this mode's xi and sigma."""
        N   = self.N
        pos = np.array([n.position for n in self.nodes])

        # Full pairwise distances  (N, N)
        delta          = pos[:, None, :] - pos[None, :, :]
        raw_dist       = np.linalg.norm(delta, axis=2)
        self._dist_mat = raw_dist + DIST_EPS

        # Coupling: Gaussian within xi, zero outside
        W    = np.zeros((N, N))
        mask = (raw_dist < self.xi) & (raw_dist > DIST_EPS)
        W[mask] = np.exp(-raw_dist[mask] ** 2 / (2.0 * self.sigma ** 2))

        # Row-normalise (gauge projection P² = P)
        row_sums = W.sum(axis=1, keepdims=True) + 1e-8
        self.W   = W / row_sums

        # Cache omega array (nodes default to omega=0 until set externally)
        self._sync_omegas()


# ── Shared node positions ──────────────────────────────────────────────────────
def _make_shared_positions(seed=42):
    """
    Generate a single set of node positions to share across all modes.
    All modes use the same geometry — only coupling scale differs.
    """
    rng  = np.random.default_rng(seed)
    pos  = list(_boundary_positions())
    for _ in range(N_NODES - N_BOUNDARY):
        pos.append(rng.uniform(-SPACE_L / 2, SPACE_L / 2, 3).tolist())
    return pos


def _nodes_from_positions(positions):
    return [Node(p) for p in positions]


# ── Multi-mode field ───────────────────────────────────────────────────────────
class MultiModeField:
    """
    N_modes independent ModeFields sharing the same node positions.

    Each mode has coupling scale ξₙ = xi_base · scale^n.
    Modes are evolved in parallel (sequentially on CPU) with the same
    context boundary conditions.

    Features for the projection layer (projection readout path):
        [cos θ, sin θ, mass]  from each mode's output nodes, concatenated.
        Dimension: N_modes × 3 × N_out

    RTR readout path (no projection matrix):
        Phase order parameters and sink-region mass are read directly.
        ξₙ spacing determines which scale sees which token boundary —
        the physical analogue is that different mode scales resolve
        phase gradients at different kink-width-equivalent separations.

    All modes run dissipative first-order dynamics. See module docstring.
    """

    def __init__(self, n_modes=3, xi_base=1.0, sigma_base=1.0, scale=2.0,
                 seed=42, omega_scale=0.02):
        self.n_modes    = n_modes
        self.seed       = seed   # stored for checkpoint cloning
        self.positions  = _make_shared_positions(seed)
        self.modes      = []

        print(f"  MultiModeField: {n_modes} modes  (field-native multi-scale routing)")
        for i in range(n_modes):
            xi    = xi_base * (scale ** i)
            sigma = sigma_base * (scale ** i)
            nodes = _nodes_from_positions(self.positions)
            mf    = ModeField(nodes, xi=xi, sigma=sigma)
            self.modes.append(mf)

        # ── Intrinsic node frequencies ────────────────────────────────────────
        # Each interior node has a small unique natural frequency on S¹.
        # The node knows itself through its omega; coupling drives harmony
        # between nodes that also know themselves.  Boundary nodes (input
        # face, first N_BOUNDARY) receive context and do not self-oscillate.
        #
        # Same omega per node across all modes: the node is the same node,
        # connected at different scales.  Only connectivity differs, not identity.
        if omega_scale > 0.0:
            rng_ω = np.random.default_rng(seed + 7919)   # prime offset
            omegas = np.zeros(len(self.positions))
            omegas[N_BOUNDARY:] = rng_ω.normal(0.0, omega_scale,
                                                len(self.positions) - N_BOUNDARY)
            for mi, mode in enumerate(self.modes):
                for j, node in enumerate(mode.nodes):
                    node.omega = float(omegas[j])
                mode._sync_omegas()
                nonzero = (mode.W > 0).sum()
                xi_i    = xi_base * (scale ** mi)
                sigma_i = sigma_base * (scale ** mi)
                print(f"    mode {mi}: ξ={xi_i:.1f}  σ={sigma_i:.1f}"
                      f"  nonzero_couplings={nonzero}")

        # Ground states — one per mode
        self.theta_0s = [None] * n_modes
        self.mass_0s  = [None] * n_modes

        # ── Phase 6 extensions ────────────────────────────────────────────────
        # Differential mode evolution (lepton-generation analogy):
        #   mode 0 (fine,   ξ small): evolve every forward pass
        #   mode 1 (medium, ξ mid)  : evolve every 3rd pass
        #   mode 2 (coarse, ξ large): evolve every 10th pass
        self.differential_evolution = False
        self._fwd_count    = 0
        self._cached_states = [None] * n_modes   # last-evolved state per mode
        self._update_freqs  = [1, 3, 10]          # fine / medium / coarse

        # Topological defect registry (monotonically growing, never cleared)
        self.defect_nodes = set()
        self._triangles   = None   # precomputed by defects.precompute_triangles()

        # Constitution nodes (identified by embed_constitution, then frozen)
        self.constitution_nodes = set()

        # Learned boundary encoder (set by MultiModeFieldLM if enc_lr > 0)
        self.encoder = None

        # Hebbian coupling (Phase 7) — enabled by enable_hebbian()
        self.hebbian = False

        # Alternating coupling (Phase 8, Gemma 4) — enabled by enable_alternating()
        self._alt_W     = None   # coarse coupling matrix; None = standard evolve
        self._alt_ratio = 5      # fine:coarse step ratio

    @property
    def N(self):
        return self.modes[0].N

    def compute_ground_states(self, steps=200, std_threshold=0.5, max_retries=5):
        """
        Relax each mode independently to its natural ground state.

        If a mode's phase std exceeds std_threshold after relaxation it likely
        landed in a vortex-dominated state rather than a smooth ground.
        Re-initialise that mode's phases to a uniform value and retry.
        """
        print(f"\n  Computing {self.n_modes} mode ground states ({steps} steps each)...")
        for i, mode in enumerate(self.modes):
            for attempt in range(max_retries):
                mode.evolve(steps=steps)
                th, m = mode.get_state()
                if th.std() <= std_threshold:
                    break
                # Vortex state detected — reset phases uniformly and retry
                print(f"    mode {i}: vortex state (std={th.std():.4f})"
                      f"  retry {attempt + 1}/{max_retries}...")
                uniform = float(th.mean())
                for node in mode.nodes:
                    node.theta  = uniform
                    node.vtheta = 0.0
            self.theta_0s[i] = th
            self.mass_0s[i]  = m
            print(f"    mode {i}: θ₀ mean={th.mean():.4f}  std={th.std():.4f}"
                  f"  λ_eff={mode.lambda_eff:.6f}")

    def restore_all(self):
        """Reset all modes to their ground states with vtheta=0."""
        for i, mode in enumerate(self.modes):
            mode.set_state(self.theta_0s[i].copy(), self.mass_0s[i].copy())
            for node in mode.nodes:
                node.vtheta = 0.0

    def encode_and_evolve(self, context_ids, vocab_size, context_len, steps,
                          phases=None):
        """
        Apply context boundary to all modes, evolve each.

        Phase 6: if differential_evolution is True, each mode evolves only
        on its own schedule (fine every step, medium every 3, coarse every 10).
        Between updates, slow modes reuse their cached state from last evolution.

        Defect nodes and constitution nodes are always included in locked,
        protecting them from phase updates regardless of context.
        """
        from .encoder import encode_context

        # Nodes that are permanently protected (topology + constitution)
        always_locked = self.defect_nodes | self.constitution_nodes
        locked        = None

        for i, mode in enumerate(self.modes):
            freq = (self._update_freqs[i]
                    if self.differential_evolution and i < len(self._update_freqs)
                    else 1)

            if not self.differential_evolution or \
               self._fwd_count % freq == 0 or \
               self._cached_states[i] is None:
                # This mode's turn: restore ground state, encode context, evolve
                mode.set_state(self.theta_0s[i].copy(), self.mass_0s[i].copy())
                for node in mode.nodes:
                    node.vtheta = 0.0
                if self.encoder is not None:
                    locked = self.encoder.encode_context(
                        mode, context_ids, context_len, phases)
                else:
                    locked = encode_context(
                        mode, context_ids, vocab_size, context_len, phases)
                eff_locked = locked | always_locked
                mode.evolve_alternating(
                    steps=steps, locked=eff_locked,
                    alt_W=self._alt_W, ratio=self._alt_ratio,
                )
                self._cached_states[i] = mode.get_state()
            else:
                # Not this mode's turn: restore cached state (no re-evolution)
                th_c, m_c = self._cached_states[i]
                mode.set_state(th_c.copy(), m_c.copy())

        if self.differential_evolution:
            self._fwd_count += 1

        # locked guaranteed non-None: mode 0 (freq=1) always runs
        if locked is None:
            from .base import BOUNDARY_IDS
            locked = set(BOUNDARY_IDS)

        return locked

    def encode_and_evolve_traj(self, context_ids, vocab_size, context_len, steps,
                               out_nodes, sample_steps, phases=None):
        """
        Encode context, evolve step-by-step, collect vtheta_signal at out_nodes
        (mode 0) at sample_steps. Returns (locked, traj_feat).

        vtheta_signal = vtheta - omega/STICKINESS removes omega-baseline drift,
        isolating the coupling/input-driven transient. Matches the gate diagnostic.

        Always steps all modes every step (no differential_evolution shortcutting)
        to give mode 0 the same dynamics as the gate test.
        """
        from .encoder import encode_context as _encode_context

        always_locked = self.defect_nodes | self.constitution_nodes

        locked = None
        for i, mode in enumerate(self.modes):
            mode.set_state(self.theta_0s[i].copy(), self.mass_0s[i].copy())
            for node in mode.nodes:
                node.vtheta = 0.0
            if self.encoder is not None:
                locked = self.encoder.encode_context(mode, context_ids, context_len, phases)
            else:
                locked = _encode_context(mode, context_ids, vocab_size, context_len, phases)

        if locked is None:
            from .base import BOUNDARY_IDS as _BID
            locked = set(_BID)
        eff_locked = locked | always_locked

        step_set   = set(sample_steps)
        traj_parts = []
        for step in range(1, steps + 1):
            for mode in self.modes:
                mode.step(locked=eff_locked)
            if step in step_set:
                vth_sig = np.array([
                    self.modes[0].nodes[j].vtheta - self.modes[0].nodes[j].omega / STICKINESS
                    for j in out_nodes
                ])
                traj_parts.append(vth_sig)

        traj_feat = np.concatenate(traj_parts) if traj_parts else np.zeros(len(out_nodes))
        return locked, traj_feat

    def get_features(self, out_nodes):
        """
        Concatenate [cos θ, sin θ, mass] from each mode's output nodes.
        Returns array of shape (N_modes × 3 × N_out,).
        """
        parts = []
        for mode in self.modes:
            thetas = np.array([mode.nodes[i].theta for i in out_nodes])
            masses = np.array([mode.nodes[i].mass  for i in out_nodes])
            parts.extend([np.cos(thetas), np.sin(thetas), masses])
        return np.concatenate(parts)

    def get_gs_features(self, out_nodes):
        """
        Extract [cos θ₀, sin θ₀] from ground state at output nodes.

        These features represent the imprinted question state directly —
        not mixed with boundary evolution.  Concatenating with get_features()
        gives W two separate signals:
            evolved  → 'what is the current answer context?'
            gs       → 'what question shaped this field?'

        Returns array of shape (N_modes × 2 × N_out,).
        """
        out = np.array(out_nodes)
        parts = []
        for th0 in self.theta_0s:
            th0_out = th0[out]
            parts.extend([np.cos(th0_out), np.sin(th0_out)])
        return np.concatenate(parts)

    def get_all_states(self):
        """Return list of (theta, mass) for each mode."""
        return [mode.get_state() for mode in self.modes]

    def get_omegas(self):
        """Return omega array (N,) from mode 0 — identical across all modes."""
        return np.array([nd.omega for nd in self.modes[0].nodes])

    def set_omegas(self, omegas):
        """Restore saved omega array to all modes."""
        for mode in self.modes:
            for i, nd in enumerate(mode.nodes):
                nd.omega = float(omegas[i])
            mode._sync_omegas()

    def update_omegas(self, coherence, alpha_omega=0.0001):
        """
        Deform node frequencies toward their current activity level.

        Nodes activated by frequently-used tokens develop higher omega.
        Zipf-distributed word frequencies emerge naturally from usage
        patterns without being programmed.

        Update rule:  ω_i ← clip(ω_i + α_ω · coherence · |vθ_i|, 0, OMEGA_MAX)

        Capped at OMEGA_MAX = E₀·max_rho/dt = 1.0 (physical bound: ω = E₀·m/ℏ,
        m ≤ max_rho). Prevents the vθ_ss = ω/STICKINESS feedback runaway that
        previously drove ω to 1.6×10⁸.

        Boundary nodes (input receivers) are excluded — they should
        remain silent so context encoding is not contaminated by
        self-oscillation.
        """
        alpha = float(alpha_omega) * float(coherence)
        if alpha == 0.0:
            return

        n = len(self.positions)

        # Average |vtheta| across all modes — robust signal
        avg_act = np.zeros(n)
        for mode in self.modes:
            avg_act += np.abs(np.array([nd.vtheta for nd in mode.nodes],
                                       dtype=float))
        avg_act /= max(len(self.modes), 1)

        # Interior nodes only — capped at OMEGA_MAX
        for mode in self.modes:
            for i in range(N_BOUNDARY, n):
                mode.nodes[i].omega = float(np.clip(
                    mode.nodes[i].omega + alpha * avg_act[i],
                    0.0, OMEGA_MAX
                ))
            mode._sync_omegas()

    def continuous_step(self, context_ids, vocab_size, context_len, n_steps=1,
                        phases=None):
        """
        Evolve the field for n_steps WITHOUT resetting the interior.

        Boundary phases are updated in-place each step. The interior retains
        its live dynamics — previous context has already propagated inward
        and continues to influence the field.  This is the continuous field
        model: the field runs perpetually and new input is a perturbation,
        not a restart.

        Differential evolution is respected: coarse modes skip steps per
        their cadence.  Mode 0 (fine) always runs — it anchors locked set.

        Returns locked set (boundary + protected nodes).
        """
        from .encoder import encode_context

        always_locked = self.defect_nodes | self.constitution_nodes
        locked        = None

        for _ in range(n_steps):
            for i, mode in enumerate(self.modes):
                freq = (self._update_freqs[i]
                        if self.differential_evolution and i < len(self._update_freqs)
                        else 1)

                if not self.differential_evolution or self._fwd_count % freq == 0:
                    # Update boundary phases in-place — interior untouched
                    if self.encoder is not None:
                        locked = self.encoder.encode_context(
                            mode, context_ids, context_len, phases)
                    else:
                        locked = encode_context(
                            mode, context_ids, vocab_size, context_len, phases)
                    eff_locked = locked | always_locked
                    mode.step(locked=eff_locked)

            if self.differential_evolution:
                self._fwd_count += 1

        if locked is None:
            from .base import BOUNDARY_IDS
            locked = set(BOUNDARY_IDS)

        return locked

    def update_ground_states(self, states, locked, lm_alpha_gs, weight=1.0,
                             occupancy=None, sink_regions=None, sink_weights=None,
                             beta_resist=0.0):
        """
        Deform each mode's ground state toward its residual, weighted by
        prediction confidence. Identical rule to Phase 1, applied per mode.

        Phase 6: defect_nodes and constitution_nodes are excluded from
        deformation — their ground state is topologically / constitutionally
        protected and cannot drift through training.

        Phase-aligned consolidation (sink_weights provided — primary):
            Deposit θ₀ deformation in proportion to the soft phase-resonance
            weights _sw from phase_dominant_sink.  Aligns the slow channel
            (consolidation) with the fast channel (readout) so both follow
            phase, not mass.
            Formula: node_ags[j] = ags * n_sk * sw[s] * brake_w[s]
            n_sk factor: preserves total update magnitude when phase is uniform.

        Consolidation resistance (occupancy provided — secondary brake):
            Under-visited sinks receive stronger deformation; over-visited
            sinks receive weaker deformation.  Self-extinguishes as occupancy
            balances toward 1/n_sinks each.
            brake_w[s] = (1/n_sk) / max(occ_s, 1/(n_sk·C))  C=4 cap
        """
        ags       = lm_alpha_gs * float(weight)
        protected = locked | self.defect_nodes | self.constitution_nodes

        n_nodes  = len(self.theta_0s[0])
        node_ags = np.full(n_nodes, ags)

        if sink_regions is not None:
            n_sk = len(sink_regions)

            # Inverse-occupancy brake (secondary; ones = no brake when no occupancy)
            brake_w = np.ones(n_sk)
            if occupancy is not None:
                occ = np.asarray(occupancy, dtype=float)
                occ_sum = occ.sum()
                if occ_sum > 0:
                    occ = occ / occ_sum
                else:
                    occ = np.ones(n_sk) / max(n_sk, 1)
                C = 4.0
                for s in range(n_sk):
                    brake_w[s] = (1.0 / n_sk) / max(float(occ[s]), 1.0 / (n_sk * C))

            if sink_weights is not None:
                # Phase-aligned: deposit proportional to soft phase-resonance weights.
                # n_sk normalizes so uniform sw → factor = brake_w (magnitude preserved).
                sw = np.asarray(sink_weights, dtype=float)
                for s, region in enumerate(sink_regions):
                    factor = n_sk * float(sw[s]) * float(brake_w[s])
                    for j in region:
                        if j < n_nodes:
                            node_ags[j] = ags * factor
            else:
                # Legacy: inverse-occupancy brake only, no phase alignment
                for s, region in enumerate(sink_regions):
                    for j in region:
                        if j < n_nodes:
                            node_ags[j] = ags * float(brake_w[s])

        for i, (th_res, m_res) in enumerate(states):
            # §4.4 mass-proportional plasticity (opt-in, beta_resist>0):
            # committed (high-mass) nodes resist deformation.  Per-mode, using
            # THIS mode's own accumulated ground-state mass.  A continuous
            # plasticity spectrum that generalizes the binary constitution freeze
            # (which still applies below via `protected`).  beta_resist=0 → no-op.
            if beta_resist > 0.0:
                eff_ags = node_ags / (1.0 + beta_resist * np.abs(self.mass_0s[i]))
            else:
                eff_ags = node_ags
            d_theta = np.sin(th_res - self.theta_0s[i])
            th_new  = (self.theta_0s[i] + eff_ags * d_theta) % (2 * np.pi)
            m_new   =  self.mass_0s[i]  + eff_ags * (m_res - self.mass_0s[i])
            for j in protected:
                th_new[j] = self.theta_0s[i][j]
                m_new[j]  = self.mass_0s[i][j]
            self.theta_0s[i] = th_new
            self.mass_0s[i]  = m_new

    def teacher_force_step(self, target_nodes, e_tgt, K_t, n_steps, locked,
                           K_mass=0.0, sink_regions=None, sink_weights=None):
        """
        Evolve all modes with teacher oscillator coupling on target_nodes.

        Phase channel: adds K_t · sin(e_tgt − θⱼ) to vtheta before each step().
        Kuramoto coupling to a frozen teacher oscillator — forcing through the EOM.

        Mass channel (K_mass > 0): after each step(), adds K_mass · dt to the
        carrier inflow for target nodes.

        Soft forcing (sink_regions + sink_weights provided):
            Instead of concentrating K_t on one argmax sink, distribute coupling
            proportional to soft phase-resonance weights _sw across ALL sinks in
            the SAME step loop. Each sink s receives K_t * sw[s] coupling strength.
            This is the phase-aligned forcing: the same move applied to consolidation
            (update_ground_states), now applied one layer up at the forcing step.
            Prevents concentrated m_res from creating mass-depth bias in θ₀.

        Constitution and defect nodes are always protected. Boundary (locked)
        nodes remain at their encoded values.

        Args:
            target_nodes  : iterable of node indices — used when sink_regions=None
            e_tgt         : teacher phase ∈ [0, 2π) — token embedding angle
            K_t           : phase coupling strength per step
            n_steps       : dynamics steps to run with coupling active
            locked        : locked node indices from current context encoding
            K_mass        : carrier inflow boost (0 = disabled)
            sink_regions  : list of node-index lists per sink (soft forcing)
            sink_weights  : soft phase-resonance weights _sw per sink (soft forcing)
        """
        always_locked = self.defect_nodes | self.constitution_nodes
        eff_locked    = set(locked) | always_locked
        _dt           = 0.05   # matches fast_field.py default dt

        if sink_regions is not None and sink_weights is not None:
            # Soft forcing: distribute K_t by phase weight across all sinks
            sw = np.asarray(sink_weights, dtype=float)
            force_pairs = [(list(reg), float(sw[s]))
                           for s, reg in enumerate(sink_regions)
                           if float(sw[s]) > 1e-6]
        else:
            force_pairs = [(list(target_nodes), 1.0)]

        for _ in range(n_steps):
            for mode in self.modes:
                for nodes, w_s in force_pairs:
                    K_eff = K_t * w_s
                    for j in nodes:
                        mode.nodes[j].vtheta += K_eff * float(
                            np.sin(e_tgt - mode.nodes[j].theta))
                mode.step(locked=eff_locked)
                if K_mass > 0.0:
                    for nodes, w_s in force_pairs:
                        for j in nodes:
                            mode.nodes[j].mass = max(
                                0.0, mode.nodes[j].mass + _dt * K_mass * w_s)

    # ── Phase 7: Hebbian coupling ─────────────────────────────────────────────

    def enable_alternating(self, ratio=5):
        """
        Enable Gemma 4-style alternating local/global coupling.

        Uses the coarsest mode's W (mode[-1], largest ξ) as the global
        coupling matrix.  Every (ratio+1)-th evolution step, and the final
        step, the field switches from fine (local) to coarse (global) coupling —
        allowing long-range coherence pulses to propagate through the field at
        regular intervals rather than only at the scale of mode 2's slow cadence.

        Call after compute_ground_states() so coupling matrices are built.
        """
        self._alt_W     = self.modes[-1].W.copy()
        self._alt_ratio = ratio
        xi_coarse = self.modes[-1].xi
        steps_per_global = ratio + 1
        print(f"  Alternating coupling enabled  {ratio}:1 local/global  "
              f"coarse ξ={xi_coarse:.1f}  "
              f"(global pulse every {steps_per_global} steps, always on final step)")

    def enable_hebbian(self):
        """
        Initialise Hebbian coupling modifier J on all modes.

        J starts at zero — effective coupling equals geometric coupling until
        experience accumulates.  Safe to call even if already enabled (idempotent:
        existing J is preserved; only uninitialized modes get init_j()).
        """
        for mode in self.modes:
            if mode.J is None:
                mode.init_j()
        self.hebbian = True
        print(f"  Hebbian coupling (J_ij) enabled across {self.n_modes} modes")

    def update_all_j(self, alpha_j=0.001, decay=0.0001):
        """
        Hebbian update on all modes using their current evolved phase state.

        Call once per round, after the generation loop and before restore_ground_state.
        The field is in the state left by the last forward() — phase correlations
        from processing the Q→A pair are fresh and informative.
        """
        if not self.hebbian:
            return
        for mode in self.modes:
            mode.update_j(alpha_j=alpha_j, decay=decay)

    def get_sink_regions(self, out_nodes, n_sinks=3):
        """
        Partition output nodes into n_sinks groups by nearest sink.

        Sinks are discovered from mass_0s[0] (ground state mass after
        relaxation) — carrier has already concentrated there naturally.
        Uses k-means with centroids initialised from the n_sinks
        highest-mass output nodes.

        Returns list of n_sinks lists of node indices.
        Cached on self._sink_regions after first call.
        """
        if getattr(self, '_sink_regions', None) is not None:
            return self._sink_regions

        out_arr  = list(out_nodes)
        n        = len(out_arr)
        pos_all  = np.array(self.positions)          # (N_total, 3)
        out_pos  = pos_all[out_arr]                  # (N_out, 3)

        # Initial centroids: n_sinks highest-mass output nodes
        if self.mass_0s[0] is not None:
            out_masses  = self.mass_0s[0][np.array(out_arr)]
        else:
            out_masses  = np.ones(n)                 # flat fallback

        top_idx    = np.argsort(out_masses)[-n_sinks:]
        centroids  = out_pos[top_idx].copy()         # (n_sinks, 3)

        # k-means (6 iterations — fast convergence on small N_out)
        labels = np.zeros(n, dtype=int)
        for _ in range(6):
            dists  = np.linalg.norm(
                out_pos[:, None, :] - centroids[None, :, :], axis=2)  # (N_out, n_sinks)
            labels = np.argmin(dists, axis=1)
            for k in range(n_sinks):
                mask = labels == k
                if mask.any():
                    centroids[k] = out_pos[mask].mean(axis=0)

        regions = [[] for _ in range(n_sinks)]
        for idx, label in enumerate(labels):
            regions[label].append(out_arr[idx])

        self._sink_regions = regions
        return regions

    def get_triplet(self, out_nodes, sink_regions):
        """
        Normalized mass vector [w0, w1, w2] at the 3 sink regions,
        read from mode 0's CURRENT evolved mass (not ground state).

        Returns numpy array of shape (n_sinks,) summing to 1.
        """
        out_set = set(out_nodes)
        ws = []
        for region in sink_regions:
            region_out = [i for i in region if i in out_set]
            m_s = (sum(self.modes[0].nodes[i].mass for i in region_out)
                   if region_out else 1e-8)
            ws.append(m_s)
        ws = np.array(ws, dtype=float)
        return ws / (ws.sum() + 1e-8)

    def get_j_arrays(self):
        """Return list of J arrays (one per mode) for checkpoint saving."""
        if not self.hebbian:
            return []
        return [mode.J.copy() for mode in self.modes if mode.J is not None]

    def set_j_arrays(self, j_arrays):
        """Restore J arrays from checkpoint. Enables hebbian if not already on."""
        if not j_arrays:
            return
        if not self.hebbian:
            self.enable_hebbian()
        for mode, J in zip(self.modes, j_arrays):
            mode.J[:] = np.asarray(J, dtype=np.float32)
        print(f"  Hebbian J loaded ({len(j_arrays)} modes)")
