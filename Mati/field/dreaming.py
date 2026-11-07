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
Field dreaming / consolidation loop.

The field dreams by evolving freely from its own ground state θ₀,
measuring how coherently it settles at each step, and deepening the
attractors it naturally produces — weighted by that coherence.

No external dataset.  No W update.  No teacher.
The field grades itself by its own phase coherence and writes only
what it is already confident about deeper into θ₀.

Biological analogue: during sleep the brain replays the activation
patterns strongest in its current synaptic state — not a fixed tape
of memories, but a resonance of whatever the field IS right now.
Patterns that cause coherent settling are reinforced; incoherent
patterns fade.  The self-instructor is the field's own topology.

Architecture:
    Each dream cycle alternates between two modes:

    FREE (even cycles):
        1. Start from a minimal seed (single space token).
        2. Autoregressively generate `dream_length` tokens.
        3. At each step: measure output_coherence of the evolved state.
        4. Deform θ₀ toward that state, weighted by coherence.
        Only θ₀ is modified.  W is untouched.

    REPLAY (odd cycles, when replay_buffer is populated):
        1. Restore post-turn ground state from the most recent turn record.
        2. Re-broadcast module experiences at dream_alpha strength.
        3. Re-imprint question at dream_alpha strength.
        4. Generate from the same echo seed used during conversation.
        5. Update θ₀ weighted by replay coherence.
        The field re-lives what it just learned from the vantage of having
        already learned it.  Stable basins deepen; unstable ones are left alone.

Three timescales, one mechanism:
    Fast   — conversation imprint (α_gs × IMPRINT_STRENGTH)
    Medium — dream replay        (α_gs × dream_alpha, same sequence)
    Slow   — free dream          (α_gs × dream_alpha, self-generated)

Thread safety:
    DreamingLoop uses a threading.Lock shared with the chat loop.
    Acquire dreamer.lock before any field-modifying operation in chat.

Usage (background thread in chat):
    from .dreaming import DreamingLoop
    dreamer = DreamingLoop(lm, tok, replay_buffer=replay_buffer,
                           broadcast_fn=_broadcast_modules,
                           notes_conn=notes_conn)
    dreamer.start()
    ...
    with dreamer.lock:
        lm.imprint_question(...)
        answer = lm.generate_answer(...)
    dreamer.stop()
"""

import threading
import numpy as np
from .tokenizer import encode as _encode_char
from .decoder import sample_token


class DreamingLoop:
    """
    Background field self-consolidation thread.

    Alternates between free generation and replay consolidation.
    Free cycles: field runs on its own structure, deepens natural attractors.
    Replay cycles: field re-lives the most recent conversation turn at
                   dream_alpha strength — medium-timescale consolidation.

    Args:
        lm             : MultiModeFieldLM instance
        tokenizer      : BPESubsetTokenizer; uses char tokenizer if None
        cycle_secs     : seconds between dream cycles (default 60)
        dream_length   : tokens generated per free cycle (default 40)
        steps          : field evolution steps per token (default 20)
        dream_alpha    : α_gs used during dreaming — kept very small so dreams
                         consolidate gently without drifting θ₀ away from the
                         distribution W was trained on (default 0.001, 50× below
                         the training α_gs of 0.05)
        verbose        : print cycle metrics
        replay_buffer  : list of turn records for replay consolidation
        broadcast_fn   : _broadcast_modules function from chat_qa.py
        notes_conn     : sqlite3 connection to notes DB
    """

    def __init__(self, lm, tokenizer=None,
                 cycle_secs=60, dream_length=40, steps=20,
                 dream_alpha=0.001, verbose=True,
                 replay_buffer=None,
                 broadcast_fn=None,
                 notes_conn=None):
        self.lm           = lm
        self.tokenizer    = tokenizer
        self.encode       = tokenizer.encode if tokenizer is not None else _encode_char
        self.cycle_secs   = cycle_secs
        self.dream_length = dream_length
        self.steps        = steps
        self.dream_alpha  = dream_alpha
        self.verbose      = verbose

        self.replay_buffer  = replay_buffer if replay_buffer is not None else []
        self.broadcast_fn   = broadcast_fn
        self.notes_conn     = notes_conn

        self.lock        = threading.Lock()
        self._stop_event = threading.Event()
        self._thread     = None

        self.cycle_count         = 0
        self._cycle_count        = 0   # internal counter for alternation
        self._coherence_history  = []   # (cycle_index, mean_coherence)

    # ── Free dream cycle ───────────────────────────────────────────────────────

    def _dream_cycle(self):
        """
        One self-consolidation cycle — continuous field dynamics.

        The field evolves via continuous_step(): boundary phases update
        in-place, the interior never resets.  Between conversations the
        field runs the same way it runs during answer generation — one
        unbroken dynamical process.  The user sees nothing; the field
        consolidates quietly.

        θ₀ is gently deformed toward high-coherence states.  W untouched.
        """
        lm      = self.lm
        seed    = self.encode(" ")[:1] or [1]
        context = list(seed)

        coherences = []

        saved_alpha    = lm.lm_alpha_gs
        lm.lm_alpha_gs = self.dream_alpha

        with self.lock:
            for _ in range(self.dream_length):
                if getattr(lm, 'use_traj_features', False):
                    probs, locked, states, _ = lm.forward(context)
                else:
                    locked = lm.mf.continuous_step(
                        context, lm.vocab_size, lm.context_len,
                        n_steps=self.steps)
                    states = lm.mf.get_all_states()
                    feat   = lm.mf.get_features(lm.output_nodes)
                    if lm.use_gs_features:
                        feat = np.concatenate(
                            [feat, lm.mf.get_gs_features(lm.output_nodes)])
                    probs = lm.proj.probs(feat)

                coherence = lm.output_coherence(states)
                coherences.append(coherence)

                tid = sample_token(probs)

                # Only deepen attractors for real tokens — UNK is noise.
                if tid != 0:
                    lm.update_ground_state(states, locked, weight=coherence)

                context = (context + [tid])[-lm.context_len:]

        lm.lm_alpha_gs = saved_alpha

        mean_coherence = float(np.mean(coherences)) if coherences else 0.0
        self._coherence_history.append((self.cycle_count, mean_coherence))
        self.cycle_count += 1

        if self.verbose:
            prev = self._coherence_history[-2][1] if len(self._coherence_history) >= 2 else mean_coherence
            arrow = '↑' if mean_coherence > prev + 0.001 else ('↓' if mean_coherence < prev - 0.001 else '→')
            print(f"\n  [dream free {self.cycle_count}]"
                  f"  coherence={mean_coherence:.4f} {arrow}",
                  flush=True)

    # ── Replay cycle ───────────────────────────────────────────────────────────

    def _replay_cycle(self):
        """
        Re-live the most recent conversation turn at dream_alpha strength.

        The field processes the same question, the same module experiences,
        and generates from the same echo seed — but from the ground state
        as it stood after that turn, and at a fraction of the original
        imprint strength.

        High coherence on replay: the basin is stable — deepen it gently.
        Low coherence on replay: the basin is unstable — apply smaller
        deformation, field self-corrects toward equilibrium.

        θ₀ is updated in-place at dream_alpha weight × replay coherence.
        W is never touched.

        This is the medium timescale between fast conversation imprint
        and slow free dream generation. Three timescales, one mechanism.
        """
        if not self.replay_buffer:
            self._dream_cycle()
            return

        lm  = self.lm
        tok = self.tokenizer

        # Pick most recent turn
        record = self.replay_buffer[-1]

        saved_alpha    = lm.lm_alpha_gs
        lm.lm_alpha_gs = self.dream_alpha

        with self.lock:
            # ── Restore post-turn ground state ────────────────────────────
            # The replay starts from where the conversation left off —
            # not from the pre-turn state, not from training baseline.
            for i in range(lm.mf.n_modes):
                lm.mf.theta_0s[i] = record['theta_0_post'][i].copy()
                lm.mf.mass_0s[i]  = record['mass_0_post'][i].copy()

            # ── Re-broadcast module experiences at dream strength ─────────
            # The same notes recall and math result that shaped the original
            # turn are re-imprinted at reduced strength. The field re-lives
            # the pre-thought context before re-processing the question.
            if self.broadcast_fn and self.notes_conn and tok is not None:
                q_text = tok.decode(record['q_ids'])
                try:
                    experiences, _ = self.broadcast_fn(
                        q_text, lm, tok, self.notes_conn)
                    for label, text, strength in experiences:
                        exp_ids = tok.encode(text)
                        if exp_ids:
                            lm.imprint_question(
                                exp_ids,
                                n_passes         = 1,
                                imprint_strength = strength * self.dream_alpha,
                                steps            = 10,
                            )
                except Exception:
                    pass   # broadcast silent-fails in dream — never blocks replay

            # ── Re-imprint question at dream strength ─────────────────────
            lm.imprint_question(
                record['q_ids'],
                n_passes         = 3,
                imprint_strength = 5.0 * self.dream_alpha,
                steps            = 20,
            )

            # ── Generate from echo seed ───────────────────────────────────
            # Same seed as original turn. Measures whether the field settles
            # into the same basin it found during the conversation.
            if getattr(lm, 'use_traj_features', False):
                _, locked, states, _ = lm.forward(record['echo_seed'])
            else:
                locked = lm.mf.continuous_step(
                    record['echo_seed'],
                    lm.vocab_size,
                    lm.context_len,
                    n_steps=self.steps,
                )
                states = lm.mf.get_all_states()
                feat   = lm.mf.get_features(lm.output_nodes)
                if lm.use_gs_features:
                    feat = np.concatenate(
                        [feat, lm.mf.get_gs_features(lm.output_nodes)])

            # ── Coherence = basin stability signal ────────────────────────
            # High coherence: field settled into same basin as original turn.
            # Low coherence: basin is unstable, smaller deformation applied.
            coherence = lm.output_coherence(states)

            # ── Update ground state weighted by coherence ─────────────────
            # dream_alpha is already active via lm_alpha_gs.
            # Stable replays deepen the basin; unstable replays barely touch it.
            if coherence > 0.0:
                lm.update_ground_state(states, locked, weight=coherence)

        lm.lm_alpha_gs = saved_alpha

        self._coherence_history.append((self.cycle_count, coherence))
        self.cycle_count += 1

        if self.verbose:
            print(f"\n  [dream replay {self.cycle_count}]"
                  f"  coherence={coherence:.4f}"
                  f"  q_len={len(record['q_ids'])}"
                  f"  alpha={self.dream_alpha:.4f}",
                  flush=True)

    # ── Alternating cycle dispatch ─────────────────────────────────────────────

    def _run_one_cycle(self):
        """
        Alternate between replay and free generation.

        Odd cycles:  dream replay — consolidate most recent experience
        Even cycles: free generation — field runs on its own structure

        If replay buffer is empty, always run free generation.
        """
        if self._cycle_count % 2 == 1 and self.replay_buffer:
            self._replay_cycle()
        else:
            self._dream_cycle()
        self._cycle_count += 1

    # ── Thread management ──────────────────────────────────────────────────────

    def _run(self):
        while not self._stop_event.is_set():
            try:
                self._run_one_cycle()
            except Exception as e:
                if self.verbose:
                    print(f"\n  [dream error: {e}]", flush=True)
            self._stop_event.wait(timeout=self.cycle_secs)

    def start(self):
        """Start background dreaming thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name='mati-dream')
        self._thread.start()
        if self.verbose:
            print(f"\n  [dreaming started — cycle every {self.cycle_secs}s"
                  f"  ·  {self.dream_length} tokens/free cycle"
                  f"  ·  alternating free / replay consolidation]",
                  flush=True)

    def stop(self):
        """Stop dreaming thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.cycle_secs + 5)
        if self.verbose:
            print(f"\n  [dreaming stopped — {self.cycle_count} cycles"
                  f"  ·  θ₀ self-consolidated]",
                  flush=True)

    def dream_now(self, n_cycles=1):
        """Run N dream cycles synchronously (for testing or manual trigger)."""
        for _ in range(n_cycles):
            self._run_one_cycle()

    def measure_coherence(self) -> float:
        """
        Measure current output coherence WITHOUT deforming θ₀.

        Runs a short generation pass read-only: field evolves from ground
        state but update_ground_state() is never called.  Safe to use during
        settle_before_save — θ₀ is untouched.

        Returns mean output coherence over dream_length steps.
        """
        lm      = self.lm
        seed    = self.encode(" ")[:1] or [1]
        context = list(seed)
        coherences = []

        with self.lock:
            snap = lm.snapshot_ground_state()
            try:
                for _ in range(self.dream_length):
                    if getattr(lm, 'use_traj_features', False):
                        probs, _, states, _ = lm.forward(context)
                    else:
                        lm.mf.continuous_step(
                            context, lm.vocab_size, lm.context_len,
                            n_steps=self.steps)
                        states = lm.mf.get_all_states()
                        feat   = lm.mf.get_features(lm.output_nodes)
                        if lm.use_gs_features:
                            feat = np.concatenate(
                                [feat, lm.mf.get_gs_features(lm.output_nodes)])
                        probs = lm.proj.probs(feat)
                    coherence = lm.output_coherence(states)
                    coherences.append(coherence)
                    tid = sample_token(probs)
                    context = (context + [tid])[-lm.context_len:]
            finally:
                lm.restore_ground_state(snap)

        return float(np.mean(coherences)) if coherences else 0.0

    # ── Status ────────────────────────────────────────────────────────────────

    def stats(self):
        recent = [c for _, c in self._coherence_history[-10:]]
        return {
            'cycles':          self.cycle_count,
            'coherence_now':   float(np.mean(recent)) if recent else None,
            'coherence_all':   float(np.mean([c for _, c in self._coherence_history]))
                               if self._coherence_history else None,
            'running':         self._thread is not None and self._thread.is_alive(),
            'cycle_secs':      self.cycle_secs,
            'dream_length':    self.dream_length,
            'replay_buffered': len(self.replay_buffer),
        }
