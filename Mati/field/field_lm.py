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
Field language model — core.

Wraps PhaseField with the encode → evolve → decode cycle for
next-token prediction and autoregressive generation.

Learning: ground state deformation weighted by prediction confidence
(field-native Hebbian rule — no backpropagation through field dynamics).
"""

import numpy as np
import sys, os
# (vendored: no sys.path hack — imports are package-relative)
from .base import (
    restore_ground_state, BOUNDARY_IDS, alpha_gs as _DEFAULT_ALPHA_GS
)
from .encoder import encode_context, CONTEXT_LEN, LearnedBoundaryEncoder
from .decoder import decode_triplet_resonance, sample_token, top_k


class FieldProjection:
    """
    Thin learned readout: field observables → token logits.

    Features (per output node):
        [cos(θᵢ), sin(θᵢ), mᵢ]   →  3 × N_out dimensional vector

    Using cos/sin instead of raw θ keeps the feature space continuous
    and consistent with the S¹ topology of the phase field.

    This is a field observable readout, not a neural network layer —
    it computes inner products between field state and learned reference
    vectors, equivalent to projecting onto learned basis states in S¹.

    Parameters: W  (vocab_size × n_features)
    Update rule: ΔW = -lr · (p - 1[k*]) ⊗ features   (cross-entropy gradient)
    """

    def __init__(self, n_out_nodes, vocab_size, lr=0.1):
        n_feat    = 3 * n_out_nodes          # cos, sin, mass per node
        self.W    = np.zeros((vocab_size, n_feat))   # zero-init → uniform start
        self.lr   = lr
        self.n_feat = n_feat

    def features(self, field, out_nodes):
        """Extract [cos θ, sin θ, mass] feature vector from output nodes."""
        thetas = np.array([field.nodes[i].theta for i in out_nodes])
        masses = np.array([field.nodes[i].mass  for i in out_nodes])
        return np.concatenate([np.cos(thetas), np.sin(thetas), masses])

    @staticmethod
    def _normalize(feat):
        """L2-normalise feature vector — makes lr scale-invariant."""
        norm = np.linalg.norm(feat)
        return feat / (norm + 1e-8)

    def logits(self, feat):
        """feat: (n_feat,) → logits: (vocab_size,)"""
        return self.W @ self._normalize(feat)

    def probs(self, feat, temperature=1.0):
        """logits → softmax probabilities."""
        lg  = self.logits(feat) / max(temperature, 1e-8)
        lg -= lg.max()
        p   = np.exp(lg)
        return p / p.sum()

    def update(self, feat, probs, target_id):
        """One gradient step on cross-entropy loss."""
        nfeat           = self._normalize(feat)
        grad            = probs.copy()
        grad[target_id] -= 1.0
        self.W         -= self.lr * np.outer(grad, nfeat)
        self.W[0]       = 0.0   # UNK has no representation — zero after every update


def find_output_nodes(field):
    """Nodes in upper z-quarter, excluding boundary (which sit at z=0)."""
    from .base import SPACE_L
    boundary_set = set(BOUNDARY_IDS)
    z_thresh = SPACE_L / 4.0
    out = [i for i in range(field.N)
           if field.nodes[i].position[2] > z_thresh
           and i not in boundary_set]
    if len(out) < 5:
        interior = [(i, field.nodes[i].position[2])
                    for i in range(field.N) if i not in boundary_set]
        out = [i for i, _ in sorted(interior, key=lambda t: -t[1])[:15]]
    return out


class FieldLM:
    """
    Field-native character-level language model.

    Architecture (pure phase-field theory):
        encode_context  →  phase boundary on 8×8 input grid
        field.evolve    →  coherence pressure minimises δP_τ = 0
        decode_probs    →  mean-field order parameter → token probs
        update_gs       →  ground state deformation = persistent memory
    """

    def __init__(self, field, theta_0, mass_0, vocab_size,
                 context_len=CONTEXT_LEN, steps=80,
                 temperature=1.0, lm_alpha_gs=None, proj_lr=0.1):
        self.field        = field
        self.theta_0      = theta_0.copy()
        self.mass_0       = mass_0.copy()
        self.vocab_size   = vocab_size
        self.context_len  = context_len
        self.steps        = steps
        self.temperature  = temperature
        self.lm_alpha_gs  = lm_alpha_gs if lm_alpha_gs is not None \
                            else _DEFAULT_ALPHA_GS * 10
        self.output_nodes = find_output_nodes(field)
        self.proj         = FieldProjection(len(self.output_nodes),
                                            vocab_size, lr=proj_lr)

        print("  FieldLM initialised:")
        print(f"    vocab={vocab_size}  context_len={context_len}")
        print(f"    output_nodes={len(self.output_nodes)}"
              f"  steps={steps}  proj_lr={proj_lr}"
              f"  lm_alpha_gs={self.lm_alpha_gs:.5f}")

    # ── Forward pass ──────────────────────────────────────────────────────────
    def forward(self, context_ids, steps=None, temperature=None):
        """
        One forward pass:
            restore ground state → encode context → evolve → decode.

        Returns:
            probs    : (vocab_size,) probability distribution
            locked   : set of locked node indices
            th_res   : phase array after evolution
            m_res    : mass array after evolution
        """
        steps       = steps       if steps       is not None else self.steps
        temperature = temperature if temperature is not None else self.temperature

        restore_ground_state(self.field, self.theta_0, self.mass_0)
        for node in self.field.nodes:   # vtheta NOT reset by restore_ground_state
            node.vtheta = 0.0
        locked = encode_context(self.field, context_ids,
                                self.vocab_size, self.context_len)
        self.field.evolve(steps=steps, locked=locked)
        th_res, m_res = self.field.get_state()
        feat  = self.proj.features(self.field, self.output_nodes)
        probs = self.proj.probs(feat, temperature)
        # Return 4-tuple: (probs, locked, residual, feat)
        # residual = (th_res, m_res) for single-field ground state update
        return probs, locked, (th_res, m_res), feat

    # ── Prediction ────────────────────────────────────────────────────────────
    def predict_next(self, context_ids, steps=None, temperature=None):
        """Return (top token ID, full probability array)."""
        probs, _, _, _ = self.forward(context_ids, steps, temperature)
        return int(np.argmax(probs)), probs

    def predict_top_k(self, context_ids, k=5, steps=None):
        """Return top-k (token_id, probability) pairs."""
        probs, _, _, _ = self.forward(context_ids, steps)
        return top_k(probs, k)

    # ── Generation ────────────────────────────────────────────────────────────
    def generate(self, prompt_ids, n_tokens=100,
                 steps=None, temperature=1.0, sample=True):
        context   = list(prompt_ids[-(self.context_len):])
        generated = []
        for _ in range(n_tokens):
            probs, _, _, _ = self.forward(context, steps=steps,
                                          temperature=temperature)
            tok = sample_token(probs) if sample else int(np.argmax(probs))
            generated.append(tok)
            context = (context + [tok])[-(self.context_len):]
        return generated

    # ── Ground state update (field-native learning) ───────────────────────────
    def output_coherence(self, residual):
        """
        Phase coherence |⟨e^{iθ}⟩| at output nodes after evolution.

        This is the field's self-assessment: how cleanly did it settle
        into an attractor?  High coherence = confident, stable state.
        Low coherence = competing attractors, uncertain.

        Field-native replacement for p_correct from W's softmax.
        """
        th_res, _m_res = residual
        thetas_out = th_res[self.output_nodes]
        return float(np.abs(np.mean(np.exp(1j * thetas_out))))

    def update_ground_state(self, residual, locked, weight=1.0):
        """residual = (th_res, m_res) from forward()."""
        th_res, m_res = residual
        """
        Deform ground state toward field residual, weighted by coherence.

        Math (spec §7.3):
            θ₀ ← θ₀ + α_gs · weight · sin(θ_res − θ₀)
            m₀ ← m₀ + α_gs · weight · (m_res − m₀)

        Boundary / context nodes are protected (not deformed).

        weight ∈ [0, 1]: output_coherence — how cleanly the field settled.
            High weight → deep attractor (field was confident).
            Low weight  → shallow update (field was uncertain).
        """
        ags = self.lm_alpha_gs * float(weight)

        d_theta     = np.sin(th_res - self.theta_0)
        th_new      = (self.theta_0 + ags * d_theta) % (2 * np.pi)
        m_new       = self.mass_0   + ags * (m_res - self.mass_0)

        # Protect locked (context boundary) nodes
        for i in locked:
            th_new[i] = self.theta_0[i]
            m_new[i]  = self.mass_0[i]

        self.theta_0 = th_new
        self.mass_0  = m_new


# ── Phase 2: Multi-mode LM ────────────────────────────────────────────────────
class MultiModeFieldLM:
    """
    Phase 2 language model: N independent fields at different coupling scales.

    Mode decomposition (cheat sheet §8):
        θ(x) = Σₙ θₙ(x)   with ξₙ = ξ₀ · 2ⁿ

    Features from all modes are concatenated before the projection layer.
    This captures short-range (character) and long-range (word/phrase)
    coherence simultaneously — field-native replacement for multi-head attention.

    Interface identical to FieldLM so trainer.py works unchanged.
    """

    def __init__(self, multi_field, vocab_size,
                 context_len=CONTEXT_LEN, steps=80,
                 temperature=1.0, lm_alpha_gs=None, proj_lr=0.1, enc_lr=0.0,
                 use_gs_features=True,
                 use_traj_features=False,
                 traj_sample_steps=(5, 10, 20, 30, 40)):
        from .decoder import find_output_nodes
        from .base import BOUNDARY_IDS, alpha_gs as _DEFAULT_ALPHA_GS

        self.mf              = multi_field
        self.vocab_size      = vocab_size
        self.context_len     = context_len
        self.steps           = steps
        self.temperature     = temperature
        self.lm_alpha_gs     = lm_alpha_gs if lm_alpha_gs is not None \
                               else _DEFAULT_ALPHA_GS * 10
        self.use_gs_features   = use_gs_features
        self.use_traj_features = use_traj_features
        self.traj_sample_steps = list(traj_sample_steps)

        # Output nodes identified from mode 0 (same positions for all modes)
        ref_field         = multi_field.modes[0]
        self.output_nodes = find_output_nodes(ref_field, set(BOUNDARY_IDS))

        n_out    = len(self.output_nodes)
        n_modes  = multi_field.n_modes

        if use_traj_features:
            # vtheta_signal at output nodes (mode 0) at each sample step
            n_feat      = len(traj_sample_steps) * n_out
            n_feat_evol = n_feat
            n_feat_gs   = 0
        else:
            # Evolved features: [cos θ, sin θ, mass] × n_modes × n_out
            # GS features:      [cos θ₀, sin θ₀]    × n_modes × n_out
            n_feat_evol = n_modes * 3 * n_out
            n_feat_gs   = n_modes * 2 * n_out
            n_feat      = n_feat_evol + n_feat_gs if use_gs_features else n_feat_evol

        self.proj = FieldProjection(n_out * n_modes, vocab_size, lr=proj_lr)
        # Override W/n_feat since FieldProjection uses 3×n_out internally
        self.proj.W      = np.zeros((vocab_size, n_feat))
        self.proj.n_feat = n_feat

        # Learned boundary encoder (Phase 7): enc_lr=0 disables (fixed encoding)
        if enc_lr > 0:
            self.encoder        = LearnedBoundaryEncoder(vocab_size, lr=enc_lr)
            multi_field.encoder = self.encoder
        else:
            self.encoder = None

        # RTR readout mode — "projection" keeps W path; "rtr" uses native readout
        self.readout_mode   = "projection"
        self._sink_regions  = None   # cached after first call to _ensure_sink_regions
        self._traj_baseline = None   # cached baseline trajectory; None = needs compute

        print(f"  MultiModeFieldLM: {multi_field.n_modes} modes")
        print(f"    vocab={vocab_size}  context_len={context_len}"
              f"  output_nodes={n_out}")
        if use_traj_features:
            print(f"    features={n_feat}  (traj: {len(traj_sample_steps)} steps"
                  f" × {n_out} nodes, mode 0)  proj_params={vocab_size * n_feat}"
                  f"  proj_lr={proj_lr}"
                  f"  enc_lr={enc_lr if enc_lr > 0 else 'fixed'}")
        else:
            print(f"    features={n_feat}  (evol={n_feat_evol}"
                  f"  gs={n_feat_gs if use_gs_features else 'off'})"
                  f"  proj_params={vocab_size * n_feat}"
                  f"  proj_lr={proj_lr}"
                  f"  enc_lr={enc_lr if enc_lr > 0 else 'fixed'}")

    def _get_traj_baseline(self, steps):
        """
        Return the cached baseline trajectory (empty-context forward pass).

        The baseline captures the common damping envelope all tokens share.
        Subtracting it before normalizing isolates the token-specific direction.

        Cache is invalidated by restore_ground_state (θ₀ changed). Computed
        lazily on first traj-features forward call after any θ₀ change.
        """
        if self._traj_baseline is None:
            steps_b = max(steps, max(self.traj_sample_steps))
            _, bl = self.mf.encode_and_evolve_traj(
                [], self.vocab_size, self.context_len, steps_b,
                self.output_nodes, self.traj_sample_steps)
            self._traj_baseline = bl
        return self._traj_baseline

    def _ensure_sink_regions(self):
        """Compute and cache output-node sink partition (k-means on ground state mass)."""
        if self._sink_regions is None:
            self._sink_regions = self.mf.get_sink_regions(self.output_nodes)
        return self._sink_regions

    def forward(self, context_ids, steps=None, temperature=None, phases=None):
        steps       = steps       if steps       is not None else self.steps
        temperature = temperature if temperature is not None else self.temperature

        self.mf.restore_all()

        if self.use_traj_features:
            steps = max(steps, max(self.traj_sample_steps))
            baseline = self._get_traj_baseline(steps)
            locked, raw_feat = self.mf.encode_and_evolve_traj(
                context_ids, self.vocab_size, self.context_len, steps,
                self.output_nodes, self.traj_sample_steps, phases)
            feat   = raw_feat - baseline   # token-specific deviation from envelope
            states = self.mf.get_all_states()
            probs  = self.proj.probs(feat, temperature)
            return probs, locked, states, feat

        locked = self.mf.encode_and_evolve(
            context_ids, self.vocab_size, self.context_len, steps, phases)
        states = self.mf.get_all_states()

        if self.readout_mode == "rtr":
            sink_regions = self._ensure_sink_regions()
            triplet      = self.mf.get_triplet(self.output_nodes, sink_regions)
            probs = decode_triplet_resonance(
                self.mf, self.output_nodes, sink_regions,
                self.vocab_size, T0=temperature, triplet=triplet)
            feat  = None   # no W-based feature vector in RTR path
        else:
            feat  = self.mf.get_features(self.output_nodes)
            if self.use_gs_features:
                feat = np.concatenate([feat, self.mf.get_gs_features(self.output_nodes)])
            probs = self.proj.probs(feat, temperature)

        # Pack residual states for ground state update
        return probs, locked, states, feat

    def predict_next(self, context_ids, steps=None, temperature=None):
        probs, _, _, _ = self.forward(context_ids, steps, temperature)
        return int(np.argmax(probs)), probs

    def predict_top_k(self, context_ids, k=5, steps=None):
        probs, _, _, _ = self.forward(context_ids, steps)
        return top_k(probs, k)

    def generate(self, prompt_ids, n_tokens=100,
                 steps=None, temperature=1.0, sample=True):
        context   = list(prompt_ids[-(self.context_len):])
        generated = []
        for _ in range(n_tokens):
            probs, _, _, _ = self.forward(context, steps=steps,
                                          temperature=temperature)
            tok = sample_token(probs) if sample else int(np.argmax(probs))
            generated.append(tok)
            context = (context + [tok])[-(self.context_len):]
        return generated

    def output_coherence(self, states):
        """
        Phase coherence |⟨e^{iθ}⟩| at output nodes, averaged over modes.

        The field's self-assessment of how cleanly it settled into an
        attractor after evolving.  Replaces p_correct from W's softmax
        as the ground state deformation weight.
        """
        scores = []
        for th, _m in states:
            thetas_out = th[self.output_nodes]
            scores.append(float(np.abs(np.mean(np.exp(1j * thetas_out)))))
        return float(np.mean(scores))

    def update_ground_state(self, states, locked, weight=1.0,
                            occupancy=None, sink_regions=None, sink_weights=None):
        # §4.4 mass-resistance is opt-in: set lm.beta_resist>0 to enable.
        # Absent attribute → 0.0 → identical to prior behavior.
        self.mf.update_ground_states(states, locked, self.lm_alpha_gs, weight,
                                     occupancy=occupancy, sink_regions=sink_regions,
                                     sink_weights=sink_weights,
                                     beta_resist=getattr(self, 'beta_resist', 0.0))

    # ── Phase 7: Question imprinting ──────────────────────────────────────────

    def snapshot_ground_state(self):
        """Return a copy of the current ground state for later restoration."""
        return (
            [th.copy() for th in self.mf.theta_0s],
            [m.copy()  for m  in self.mf.mass_0s],
        )

    def restore_ground_state(self, snapshot):
        """Restore ground state to a previously saved snapshot."""
        theta_0s, mass_0s = snapshot
        self.mf.theta_0s    = [th.copy() for th in theta_0s]
        self.mf.mass_0s     = [m.copy()  for m  in mass_0s]
        self._traj_baseline = None   # θ₀ changed — recompute baseline next forward

    def imprint_question(self, question_ids, n_passes=5, imprint_strength=10.0,
                         steps=None, phases=None, lift_protection=True,
                         should_stop=None):
        """
        Deform the ground state toward the question's attractor landscape.

        The question does not occupy a context window — it shapes θ₀ itself.
        Subsequent generation reads from this deformed landscape for the full
        length of the answer, with no token expiry.

        This is field-native Q&A: the question becomes part of what the field
        IS, not a buffer it holds. Analogous to how lived experience shapes
        future perception in Nahuatl process philosophy.

        Args:
            question_ids    : token IDs of the question
            n_passes        : how many times to run the question through
            imprint_strength: α_gs multiplier during imprinting (default 10×)
            steps           : field evolution steps (default self.steps)

        Returns:
            snapshot of ground state BEFORE imprinting, for restoration.
        """
        from .trainer import make_examples

        steps    = steps if steps is not None else self.steps
        snapshot = self.snapshot_ground_state()
        examples = make_examples(question_ids, self.context_len)
        if not examples:
            return snapshot

        # Build aligned phases windows matching each context window in examples.
        # make_examples window at position i = question_ids[max(0,i-ctx_len+1):i+1]
        phases_list = None
        if phases is not None:
            phases_list = []
            for i in range(len(question_ids)):
                start = max(0, i - self.context_len + 1)
                phases_list.append(phases[start:i + 1])

        saved_alpha      = self.lm_alpha_gs
        self.lm_alpha_gs = saved_alpha * imprint_strength

        # Whether a question may reshape the most stable attractors.
        #
        # The original stance was that it always may — "a genuine question
        # re-opens what seemed settled" — and it is a coherent one. But it made
        # the constitution unenforceable: this method is the PRIMARY training
        # signal (every user turn goes through it), so protection lifted here
        # is protection that never applies anywhere. Measured: "frozen" nodes
        # drifted 1.7476 over six ordinary turns against 1.5343 for unfrozen
        # ones — they moved MORE. Measured in the constitution study.
        #
        # Default stays True so training runs and the diagnostics behind the
        # results ladder are unchanged. Mati passes False: there, a question
        # must not be able to reshape the constitution (user).
        saved_defects       = self.mf.defect_nodes.copy()
        saved_constitution  = self.mf.constitution_nodes.copy()
        if lift_protection:
            self.mf.defect_nodes       = set()
            self.mf.constitution_nodes = set()

        # `should_stop` is checked ONLY at pass boundaries, and never before the
        # first pass has finished. That is the one interruption point that
        # leaves a coherent field: every token has been imprinted, just fewer
        # times. Stopping mid-pass would imprint the early tokens more than the
        # late ones and skew the memory rather than shallow it.
        #
        # This exists so a queued episode write (~55s at TRIPLET_CAP=64) cannot
        # make an arriving turn wait for the whole thing. The trade is real and
        # deliberate: under load a memory forms less deeply. The caller is told
        # how many passes actually ran so it can say so rather than hide it.
        self.passes_run = 0
        for _p in range(n_passes):
            for idx, (ctx, _) in enumerate(examples):
                ctx_phases = phases_list[idx] if phases_list else None
                _, locked, states, _ = self.forward(ctx, steps=steps,
                                                    phases=ctx_phases)
                self.update_ground_state(states, locked, weight=1.0)
            self.passes_run += 1
            if should_stop is not None and should_stop():
                break                      # a turn is waiting; this is enough

        # Restore protection
        self.mf.defect_nodes       = saved_defects
        self.mf.constitution_nodes = saved_constitution
        self.lm_alpha_gs           = saved_alpha
        self._traj_baseline        = None   # imprint deformed θ₀; answer forwards must recompute from imprinted state
        return snapshot

    def generate_answer(self, seed_ids, max_tokens=80, steps=None,
                        temperature=0.85, stop_token=None):
        """
        Generate answer tokens from the current (possibly imprinted) ground state.

        Args:
            seed_ids   : starting token IDs (e.g. tok.encode("A: "))
            max_tokens : maximum tokens to generate
            steps      : field evolution steps
            temperature: sampling temperature
            stop_token : token ID that ends generation (e.g. EOS)

        Returns:
            list of generated token IDs (not including seed)
        """
        from .decoder import sample_token

        steps   = steps if steps is not None else self.steps
        context = list(seed_ids[-(self.context_len):])
        generated = []

        for _ in range(max_tokens):
            probs, _, _, _ = self.forward(context, steps=steps,
                                          temperature=temperature)
            tid = sample_token(probs)
            if stop_token is not None and tid == stop_token:
                break
            generated.append(tid)
            context = (context + [tid])[-(self.context_len):]

        return generated

    def generate_answer_continuous(self, seed_ids, max_tokens=80,
                                   steps_per_token=10, temperature=0.85,
                                   stop_token=None, gen_alpha_gs=0.01,
                                   warm_start=True):
        """
        Generate answer tokens using continuous field dynamics.

        The field is NOT reset between tokens.  Each new token updates the
        boundary in-place; the interior dynamics carry forward — previous
        context remains encoded in the live field, not just the boundary window.

        This is the biological model: the field runs perpetually, input is a
        perturbation, output is always readable.  Effective context is no longer
        limited to context_len tokens — it is whatever the interior dynamics
        still retain from earlier in the sequence.

        Args:
            seed_ids        : starting token IDs (e.g. tok.encode("A: "))
            max_tokens      : maximum tokens to generate
            steps_per_token : field evolution steps between successive reads
            temperature     : sampling temperature
            stop_token      : token ID that ends generation
            gen_alpha_gs    : mild GS deformation per token so the field
                              remembers what it said (0 = disabled)
            warm_start      : if True, restore ground state and run a full
                              warmup pass before generation begins.  Set False
                              when the field is already live (e.g. called from
                              inside a continuous dreaming loop) — the field
                              picks up from wherever it currently is.

        Returns:
            list of generated token IDs (not including seed)
        """
        from .decoder import sample_token

        context   = list(seed_ids[-(self.context_len):])
        generated = []

        # Traj-feature LMs require a fresh trajectory per token — use forward()
        if self.use_traj_features:
            for _ in range(max_tokens):
                probs, locked, states, _ = self.forward(
                    context, steps=self.steps, temperature=temperature)
                tid = sample_token(probs)
                if stop_token is not None and tid == stop_token:
                    break
                generated.append(tid)
                context = (context + [tid])[-(self.context_len):]
                if gen_alpha_gs > 0.0:
                    coherence        = self.output_coherence(states)
                    saved_alpha      = self.lm_alpha_gs
                    self.lm_alpha_gs = gen_alpha_gs
                    self.update_ground_state(states, locked, weight=coherence)
                    self.lm_alpha_gs = saved_alpha
            return generated

        # ── Initialize: restore ground state ONCE, establish live dynamics ────
        # Skip when field is already running continuously (warm_start=False).
        if warm_start:
            self.mf.restore_all()
            locked = self.mf.encode_and_evolve(
                context, self.vocab_size, self.context_len, self.steps)

        # ── Continuous generation loop ─────────────────────────────────────────
        for _ in range(max_tokens):

            # Boundary perturbation + n steps — interior runs free
            locked = self.mf.continuous_step(
                context, self.vocab_size, self.context_len,
                n_steps=steps_per_token)

            states = self.mf.get_all_states()

            if self.readout_mode == "rtr":
                sink_regions = self._ensure_sink_regions()
                triplet      = self.mf.get_triplet(self.output_nodes, sink_regions)
                probs = decode_triplet_resonance(
                    self.mf, self.output_nodes, sink_regions,
                    self.vocab_size, T0=temperature, triplet=triplet)
            else:
                feat  = self.mf.get_features(self.output_nodes)
                if self.use_gs_features:
                    feat = np.concatenate([feat, self.mf.get_gs_features(self.output_nodes)])
                probs = self.proj.probs(feat, temperature)

            tid = sample_token(probs)
            if stop_token is not None and tid == stop_token:
                break
            generated.append(tid)
            context = (context + [tid])[-(self.context_len):]

            # Mild GS deformation: the field remembers what it just said.
            # gen_alpha_gs << lm_alpha_gs so the question imprint is not erased.
            if gen_alpha_gs > 0.0:
                coherence        = self.output_coherence(states)
                saved_alpha      = self.lm_alpha_gs
                self.lm_alpha_gs = gen_alpha_gs
                self.update_ground_state(states, locked, weight=coherence)
                self.lm_alpha_gs = saved_alpha

        return generated

    # ── Phase 6: Ground state inversion ───────────────────────────────────────

    def ground_state_inversion(self, top_k=10):
        """
        Read θ₀ to infer what most shaped this field.

        Builds a feature vector matching W's training layout exactly:
            evolved proxy : [cos θ₀, sin θ₀, mass₀] × modes × out_nodes (270)
            GS features   : [cos θ₀, sin θ₀]        × modes × out_nodes (180)

        Using θ₀ and mass₀ as proxies for the evolved state reads W from the
        perspective of the field at rest — what attractors are present in the
        ground state, regardless of what input triggered them.

        This is the field's primary self-knowledge operation:
            not 'I believe X' but 'I have been shaped by X'.

        Returns:
            list of (token_id, affinity) pairs, highest affinity first.
        """
        if self.use_traj_features:
            # vtheta_signal is transient — θ₀ carries no meaningful traj proxy.
            return []

        feat_parts = []

        # Evolved feature proxy: θ₀ and mass₀ as the resting state
        for i in range(len(self.mf.modes)):
            th0    = self.mf.theta_0s[i]
            m0     = self.mf.mass_0s[i]
            out_th = th0[self.output_nodes]
            out_m  = m0[self.output_nodes]
            feat_parts.extend([np.cos(out_th), np.sin(out_th), out_m])

        # GS features (appended only when use_gs_features=True)
        if self.use_gs_features:
            for i in range(len(self.mf.modes)):
                th0    = self.mf.theta_0s[i]
                out_th = th0[self.output_nodes]
                feat_parts.extend([np.cos(out_th), np.sin(out_th)])

        feat  = np.concatenate(feat_parts)
        feat /= (np.linalg.norm(feat) + 1e-8)

        if self.readout_mode == "rtr":
            # W is not used in RTR mode.
            # Field self-knowledge: read order parameters from resting θ₀ directly.
            sink_regions = self._ensure_sink_regions()
            triplet      = self.mf.get_triplet(self.output_nodes, sink_regions)
            probs = decode_triplet_resonance(
                self.mf, self.output_nodes, sink_regions,
                self.vocab_size, T0=0.10, triplet=triplet)
            top_idx = np.argsort(probs)[::-1][:top_k]
            return [(int(idx), float(probs[idx])) for idx in top_idx]

        affinities = self.proj.W @ feat                          # (vocab_size,)
        affinities[0] = -np.inf                                  # UNK has no meaning
        top_idx    = np.argsort(affinities)[::-1][:top_k]
        return [(int(idx), float(affinities[idx])) for idx in top_idx]
