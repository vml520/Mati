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
Field-native training loop for FieldLM.

Learning mechanism (Phase 1):
    Forward pass → cross-entropy loss → ground state deformation.
    No backprop through field dynamics.
    Weight of each deformation = p(correct token) — high confidence
    correct predictions deepen attractors more.
"""

import numpy as np
import sys, os
from collections import deque
# (vendored: no sys.path hack — imports are package-relative)
from .tokenizer import encode as _encode_char, decode as _decode_char
from .scorer    import reference_overlap
from .decoder   import phase_dominant_sink


def _log_encoder_phase_spread(lm, sink_regions, out_set, label=""):
    """
    Log encoder.phi distribution and per-sink home assignment from theta_0.

    A. Histogram of phi across vocabulary (24 bins over [0, 2π)).
    B. home_hist: for each token k, which sink does encoder.phi[k] resonate
       with given current ground-state sink mean phases?
    Diagnostic only — no dynamics change.
    """
    enc = getattr(lm, 'encoder', None)
    if enc is None:
        print(f"  [encoder diag{label}] No learned encoder — phi fixed at 2π·k/V")
        return

    V   = lm.vocab_size
    phi = enc.phi   # (V,) ∈ [0, 2π)

    # A. phi histogram
    n_bins = 24
    counts, edges = np.histogram(phi % (2 * np.pi), bins=n_bins,
                                 range=(0.0, 2 * np.pi))
    bin_w = 2 * np.pi / n_bins
    max_c = max(int(counts.max()), 1)
    print(f"\n  [encoder diag{label}] phi histogram (V={V}, {n_bins} bins over [0,2π)):")
    for i, ct in enumerate(counts):
        bar = '█' * int(round(ct * 20 / max_c))
        print(f"    [{edges[i]:.2f},{edges[i]+bin_w:.2f})  {ct:4d}  {bar}")

    # B. home_hist using ground-state theta_0s[0] for sink mean phases
    th0 = lm.mf.theta_0s[0]   # (N,) ground-state phases
    sink_phs = []
    for reg in sink_regions:
        r_out = [j for j in reg if j in out_set]
        if r_out:
            th = th0[np.array(r_out)]
            sink_phs.append(float(np.angle(np.mean(np.exp(1j * th)))))
        else:
            sink_phs.append(0.0)

    home_counts = np.zeros(len(sink_regions), dtype=int)
    for k in range(V):
        s = int(np.argmax([np.cos(phi[k] - sp) for sp in sink_phs]))
        home_counts[s] += 1

    home_frac = home_counts / max(int(home_counts.sum()), 1)
    sp_str  = '  '.join(f"s{s}={sink_phs[s]:.3f}rad" for s in range(len(sink_regions)))
    hom_str = '  '.join(
        f"s{s}={home_frac[s]:.2f}({home_counts[s]})" for s in range(len(sink_regions)))
    print(f"  [encoder diag{label}] sink phases (θ₀): {sp_str}")
    print(f"  [encoder diag{label}] home_hist:         {hom_str}\n")


def cross_entropy(probs, target_id):
    """Per-token cross-entropy loss."""
    return -np.log(max(float(probs[target_id]), 1e-12))


def perplexity(losses):
    """Perplexity from a list of per-token cross-entropy values."""
    return float(np.exp(np.mean(losses)))


def make_examples(token_ids, context_len=8):
    """
    Sliding-window (context, target) pairs from a token ID sequence.

    Each position i is a prediction target; context is the preceding
    context_len tokens (left-padded with BOS if needed).
    """
    examples = []
    for i in range(len(token_ids) - 1):
        ctx = token_ids[max(0, i - context_len + 1): i + 1]
        tgt = token_ids[i + 1]
        examples.append((list(ctx), int(tgt)))
    return examples


def train(lm, text, n_epochs=3, steps_per_fwd=80,
          temperature=1.0, log_every=200, shuffle=True, tokenizer=None,
          mask_unk=True, alpha_omega=0.0001):
    """
    Train FieldLM on text via ground state deformation.

    Args:
        lm           : FieldLM instance
        text         : training text string
        n_epochs     : passes over the data
        steps_per_fwd: field evolution steps per forward pass
                       (80 for speed during training; use 160 at inference)
        temperature  : softmax temperature during training
        log_every    : print progress every N examples
        shuffle      : shuffle example order each epoch
        tokenizer    : optional BPESubsetTokenizer; uses char tokenizer if None
        mask_unk     : if True, skip W/GS updates when target is UNK (id 0).
                       Field still evolves through UNK for context propagation
                       but UNK cannot leave topological traces in θ₀ or W.
        alpha_omega  : learnable omega step size (0 = fixed omegas).

    Returns:
        list of {'epoch', 'loss', 'ppl'} dicts
    """
    encode = tokenizer.encode if tokenizer is not None else _encode_char
    decode = tokenizer.decode if tokenizer is not None else _decode_char
    token_ids = encode(text)
    examples  = make_examples(token_ids, lm.context_len)

    unk_count = sum(1 for _, t in examples if t == 0)
    unk_pct   = 100 * unk_count / max(len(examples), 1)
    print("\n=== Training FieldLM ===")
    print(f"  text:        {len(text)} chars")
    print(f"  examples:    {len(examples)}")
    print(f"  vocab size:  {lm.vocab_size}")
    print(f"  UNK targets: {unk_count} ({unk_pct:.1f}%)"
          f"  {'masked — no W/GS update' if mask_unk else 'unmasked'}")
    print(f"  epochs:      {n_epochs}  steps/fwd: {steps_per_fwd}\n")

    epoch_log = []

    for epoch in range(n_epochs):
        order = list(range(len(examples)))
        if shuffle:
            np.random.shuffle(order)

        losses = []
        correct = 0

        for n, idx in enumerate(order):
            ctx, tgt = examples[idx]

            result = lm.forward(ctx, steps=steps_per_fwd,
                                temperature=temperature)
            probs, locked, residual, feat = result

            loss = cross_entropy(probs, tgt)
            losses.append(loss)

            pred = int(np.argmax(probs))
            if pred == tgt:
                correct += 1

            # UNK masking: field evolves through UNK for context propagation
            # but UNK target tokens must not imprint topology into θ₀ or W.
            if mask_unk and tgt == 0:
                continue

            # 1. Projection update — skipped in RTR mode (feat is None, W not used)
            if feat is not None:
                lm.proj.update(feat, probs, tgt)

            # 2. Learned boundary encoder update (Phase 7, no-op if enc_lr=0)
            if feat is not None and getattr(lm, 'encoder', None) is not None:
                lm.encoder.update(
                    ctx, tgt, feat, lm.proj.W, probs,
                    len(lm.output_nodes), lm.mf.n_modes,
                )

            # 3. Ground state deformation — field grades itself by coherence,
            #    not by W's softmax probability.  The field deepens whatever
            #    attractor it actually settled into, proportional to how
            #    clearly it settled.  No external teacher required.
            coherence = lm.output_coherence(residual)
            lm.update_ground_state(residual, locked, weight=coherence)

            # 4. Omega update — learnable node frequencies (Zipf emergence).
            #    Nodes activated by frequent tokens develop higher omega,
            #    letting word-frequency structure appear without programming.
            if alpha_omega > 0.0 and hasattr(lm, 'mf'):
                lm.mf.update_omegas(coherence, alpha_omega)

            if (n + 1) % log_every == 0:
                window    = losses[-log_every:]
                recent_ppl = perplexity(window)
                acc        = correct / (n + 1)
                print(f"  epoch {epoch+1}  step {n+1:>5}/{len(examples)}"
                      f"  loss={loss:.4f}  ppl={recent_ppl:.1f}"
                      f"  acc={acc:.3f}"
                      f"  pred={repr(decode([pred]))}"
                      f"  true={repr(decode([tgt]))}")

        mean_loss  = float(np.mean(losses))
        epoch_ppl  = perplexity(losses)
        epoch_acc  = correct / len(examples)
        epoch_log.append({
            'epoch': epoch + 1,
            'loss':  mean_loss,
            'ppl':   epoch_ppl,
            'acc':   epoch_acc,
        })
        print(f"\n  Epoch {epoch+1} complete —"
              f" loss={mean_loss:.4f}"
              f"  ppl={epoch_ppl:.1f}"
              f"  acc={epoch_acc:.3f}\n")

    return epoch_log


def train_on_qa_imprint(lm, qa_pairs, tokenizer=None,
                        n_imprint_passes=3, imprint_strength=5.0,
                        imprint_steps=20, steps_per_fwd=20,
                        n_epochs=3, log_every=5, alpha_omega=0.0001,
                        mask_unk=True,
                        reinforce_threshold=0.1, reinforce_strength=1.0,
                        n_gen_tokens=20,
                        reinforce_mode='scorer',
                        collect_per_pair=False,
                        victim_ids=None, upweight=1.0):
    """
    Imprint-aligned Q&A training with external scorer reinforcement.

    Aligns training with inference: for each pair, the question is first
    imprinted into θ₀ (identical to what the chat does), then W is trained
    on answer tokens from that imprinted state.

    External check (scorer reinforcement):
        After W update, the model generates a short answer from the imprinted
        state and scores it via token-F1 against the reference answer.
        If overlap >= reinforce_threshold, the base ground state θ₀ is
        deformed toward the imprinted state weighted by overlap score.
        This lets the field remember which question-shapes produce good answers.
        If the answer is poor, θ₀ is simply restored unchanged.

    Args:
        lm                   : MultiModeFieldLM
        qa_pairs             : list of (question_str, answer_str) tuples
        tokenizer            : BPESubsetTokenizer; uses char tokenizer if None
        n_imprint_passes     : imprint passes per question (match chat setting)
        imprint_strength     : α_gs multiplier during imprint (match chat setting)
        imprint_steps        : field steps during imprint (fast, default 20)
        steps_per_fwd        : field steps during answer token training
        n_epochs             : passes over all pairs
        log_every            : print every N pairs
        reinforce_threshold  : minimum token-F1 to trigger GS reinforcement
        reinforce_strength   : multiplier on lm_alpha_gs for reinforcement
        n_gen_tokens         : tokens to generate for scoring (default 20)

    Returns:
        list of {'epoch', 'loss', 'ppl', 'ppl_nu', 'reinforce_rate'} dicts.
        If collect_per_pair=True, each dict also contains
        'per_pair_ppl_nu': {pair_idx (0-based): ppl_nu_float} — the per-pair
        non-UNK perplexity accumulated during that epoch's training pass.
        This is the only instrument that resolves floor-vs-descent without a
        separate no-update eval; log it every epoch.
    """
    encode = tokenizer.encode if tokenizer is not None else _encode_char

    answer_prefix = encode("A: ")

    # Measure UNK rate across all answer tokens upfront
    all_ans_ids = [tid for _, a in qa_pairs for tid in encode(a)]
    unk_count   = all_ans_ids.count(0)
    unk_pct     = 100 * unk_count / max(len(all_ans_ids), 1)

    print(f"\n=== Q&A Imprint Training: {len(qa_pairs)} pairs × {n_epochs} epochs ===")
    print(f"    imprint: {n_imprint_passes} passes  strength={imprint_strength}×  "
          f"steps={imprint_steps}")
    print(f"    answer:  steps={steps_per_fwd}")
    print(f"    answer UNK rate: {unk_pct:.1f}%"
          f"  ({'masked' if mask_unk else 'unmasked'}) — "
          f"{unk_count}/{len(all_ans_ids)} tokens\n")

    epoch_log = []

    for epoch in range(n_epochs):
        losses          = []
        losses_nu       = []   # non-UNK targets only — what W actually trains on
        reinforced      = 0
        per_pair_ppl_nu = {} if collect_per_pair else None

        for qi, (q, a) in enumerate(qa_pairs):
            pair_losses_nu = [] if collect_per_pair else None

            # Loss upweighting for confirmed interference victims
            _is_victim   = (victim_ids is not None and qi in victim_ids
                            and upweight != 1.0)
            _orig_proj_lr = lm.proj.lr
            _orig_enc_lr  = getattr(getattr(lm, 'encoder', None), 'lr', 0.0)
            if _is_victim:
                lm.proj.lr = _orig_proj_lr * upweight
                if hasattr(lm, 'encoder') and lm.encoder is not None:
                    lm.encoder.lr = _orig_enc_lr * upweight

            # ── 1. Imprint question into θ₀ ──────────────────────────────────
            q_ids    = encode(q)
            snapshot = lm.imprint_question(
                q_ids,
                n_passes         = n_imprint_passes,
                imprint_strength = imprint_strength,
                steps            = imprint_steps,
            )

            # ── 2. Teacher-force answer from imprinted state ──────────────────
            # Seed with "A: " so the model knows it's in answer mode.
            # Collect field residuals as they emerge — these are the states
            # the field physically evolved into under its own dynamics.
            # W is updated here; GS update is deferred to step 3.
            a_ids     = answer_prefix + encode(a)
            examples  = make_examples(a_ids, lm.context_len)
            residuals = []   # (states, locked) — physics states only, no UNK

            for ctx, tgt in examples:
                probs, locked, states, feat = lm.forward(ctx, steps=steps_per_fwd)
                loss = cross_entropy(probs, tgt)
                losses.append(loss)
                if tgt != 0:
                    losses_nu.append(loss)
                    if pair_losses_nu is not None:
                        pair_losses_nu.append(loss)

                if mask_unk and tgt == 0:
                    continue

                # Update W only — GS deformation happens in step 3 if scored
                # Skipped in RTR mode (feat is None, W not used in readout path)
                if feat is not None:
                    lm.proj.update(feat, probs, tgt)

                if feat is not None and getattr(lm, 'encoder', None) is not None:
                    lm.encoder.update(
                        ctx, tgt, feat, lm.proj.W, probs,
                        len(lm.output_nodes), lm.mf.n_modes,
                    )

                if alpha_omega > 0.0 and hasattr(lm, 'mf'):
                    coherence = lm.output_coherence(states)
                    lm.mf.update_omegas(coherence, alpha_omega)

                # Deep-copy residual — forward() overwrites arrays each call
                if reinforce_threshold > 0.0:
                    residuals.append((
                        [(th.copy(), m.copy()) for th, m in states],
                        set(locked),
                    ))

            # ── 3. External check: score → let field reinforce from residuals ─
            # The scorer is a gate only — it decides whether reinforcement
            # happens and how strongly.  It never constructs or targets a state.
            # When the field got the answer right, deform the BASE θ₀ toward
            # the residuals the field physically settled into during step 2.
            # Properties emerge from dynamics; nothing is forced.
            if reinforce_threshold > 0.0 and residuals:
                try:
                    if reinforce_mode == 'coherence':
                        # §10.4 field-native gate: the field grades itself by its
                        # own output coherence (|⟨e^{iθ}⟩| at output nodes) over
                        # the residual states — NO external scorer, NO generation
                        # pass.  Same self-assessment the dreaming loop already
                        # uses (dreaming.py), relocated onto the foundation path.
                        cohs = [lm.output_coherence(st) for st, _lk in residuals]
                        gate_val = float(np.mean(cohs)) if cohs else 0.0
                    else:
                        # Default 'scorer': continuous generation (warm_start=True)
                        # matches the chat's inference mode.  gen_alpha_gs=0 keeps
                        # θ₀ clean so the scoring pass doesn't pollute the decision.
                        # In RTR mode temperature is T0; use 0.10 (coherence-derived)
                        _gen_temp = 0.10 if getattr(lm, 'readout_mode', '') == 'rtr' \
                                    else 0.85
                        gen_ids = lm.generate_answer_continuous(
                            seed_ids        = answer_prefix,
                            max_tokens      = n_gen_tokens,
                            steps_per_token = steps_per_fwd,
                            temperature     = _gen_temp,
                            gen_alpha_gs    = 0.0,
                            warm_start      = True,
                        )
                        gate_val = reference_overlap(gen_ids, encode(a))

                    # Restore base θ₀ regardless — then deepen if gate passes
                    lm.restore_ground_state(snapshot)

                    if gate_val >= reinforce_threshold:
                        # Deform base toward field-evolved residuals, weighted
                        # by the gate value.  update_ground_state uses
                        # sin(θ_res − θ₀) — correct S¹ geometry, no forcing.
                        w = gate_val * reinforce_strength
                        for states_copy, locked_copy in residuals:
                            lm.update_ground_state(states_copy, locked_copy,
                                                   weight=w)
                        reinforced += 1
                except Exception:
                    lm.restore_ground_state(snapshot)
            else:
                lm.restore_ground_state(snapshot)

            # Restore lr after victim pair
            if _is_victim:
                lm.proj.lr = _orig_proj_lr
                if hasattr(lm, 'encoder') and lm.encoder is not None:
                    lm.encoder.lr = _orig_enc_lr

            if per_pair_ppl_nu is not None:
                per_pair_ppl_nu[qi] = perplexity(pair_losses_nu) if pair_losses_nu else float('nan')

            if (qi + 1) % log_every == 0:
                recent    = losses[-min(100, len(losses)):]
                recent_nu = losses_nu[-min(100, len(losses_nu)):] if losses_nu else []
                ppl_nu_str = f"{perplexity(recent_nu):.1f}" if recent_nu else "n/a"
                print(f"  epoch {epoch+1}  pair {qi+1:>4}/{len(qa_pairs)}"
                      f"  loss={np.mean(recent):.4f}"
                      f"  ppl={perplexity(recent):.1f}"
                      f"  ppl_nu={ppl_nu_str}"
                      f"  reinforced={reinforced}/{qi+1}")

        mean_loss    = float(np.mean(losses))
        ep_ppl       = perplexity(losses)
        ep_ppl_nu    = perplexity(losses_nu) if losses_nu else float('nan')
        reinforce_rt = reinforced / max(len(qa_pairs), 1)
        record = {
            'epoch': epoch + 1, 'loss': mean_loss,
            'ppl': ep_ppl, 'ppl_nu': ep_ppl_nu, 'reinforce_rate': reinforce_rt,
        }
        if per_pair_ppl_nu is not None:
            record['per_pair_ppl_nu'] = per_pair_ppl_nu
        epoch_log.append(record)
        print(f"\n  Imprint epoch {epoch+1}: loss={mean_loss:.4f}"
              f"  ppl={ep_ppl:.1f}"
              f"  ppl_nu={ep_ppl_nu:.1f}"
              f"  reinforced={reinforced}/{len(qa_pairs)}"
              f"  ({100*reinforce_rt:.1f}%)\n")

    return epoch_log


def train_supervised_rtr(lm, qa_pairs, tokenizer=None,
                         n_imprint_passes=3, imprint_strength=5.0,
                         imprint_steps=80, answer_steps=40,
                         rtr_alpha=0.10, pin_steps=20, K_mass=0.0,
                         n_epochs=5, log_every=5,
                         mask_unk=True, alpha_omega=0.0001,
                         sup_imprint_strength=1.0,
                         ckpt_fn=None, ckpt_every=20,
                         start_epoch=0, start_pair=0):
    """
    Supervised RTR imprinting — physics-compliant via teacher oscillator.

    For each answer token, adds K_t·sin(e_tgt − θⱼ) to the equation of
    motion (via vtheta injection before each step()) for nodes in the
    *dominant sink region only*. The other two sink regions evolve freely,
    preserving the basin differentiation that Layers 2+2b read.

    Signal balance (Run 2):
        Question imprint uses sup_imprint_strength (default 1.0) instead of
        imprint_strength (default 5.0) so the answer-pin signal dominates.
        The a/q ratio = (K_t · pin_steps / answer_steps) / (sup_imprint_strength
        · lm_alpha_gs) is logged per pair so you can see which channel dominates.

    Mass channel (K_mass > 0):
        Adds carrier inflow forcing to dominant-sink nodes after each step,
        equivalent to boosting the inflow term in the mass EOM. Drives mass
        toward the target sink so triplet_H moves off the frozen-uniform value.
        Success metric: triplet_H moves off 1.052 by end of epoch 1.

    Args:
        rtr_alpha            : teacher coupling strength K_t (default 0.10)
        pin_steps            : dynamics steps with teacher coupling (default 20)
        K_mass               : carrier inflow boost for dominant sink (0 = off)
        answer_steps         : field evolution steps per answer token (default 40)
        sup_imprint_strength : imprint_strength used during question imprint in
                               this supervised pass (default 1.0, down-weighted
                               from the normal 5.0 so answer pin dominates)
        Other args same as train_on_qa_imprint.
    """
    encode = tokenizer.encode if tokenizer is not None else _encode_char
    decode = tokenizer.decode if tokenizer is not None else _decode_char

    answer_prefix = encode("A: ")
    V = lm.vocab_size

    all_ans_ids = [tid for _, a in qa_pairs for tid in encode(a)]
    unk_count   = all_ans_ids.count(0)
    unk_pct     = 100 * unk_count / max(len(all_ans_ids), 1)

    # Signal balance diagnostic
    q_push = sup_imprint_strength * lm.lm_alpha_gs
    a_push = rtr_alpha * pin_steps / max(answer_steps, 1)
    aq_ratio_init = a_push / max(q_push, 1e-12)

    print(f"\n=== Supervised RTR Imprint (physics-compliant): "
          f"{len(qa_pairs)} pairs × {n_epochs} epochs ===")
    print(f"    imprint: {n_imprint_passes} passes  "
          f"strength={sup_imprint_strength}×  steps={imprint_steps}"
          f"  (supervised; normal={imprint_strength}×)")
    print(f"    answer:  rtr_alpha={rtr_alpha}  answer_steps={answer_steps}  "
          f"pin_steps={pin_steps}  K_mass={K_mass}")
    print(f"    a/q ratio ≈ {aq_ratio_init:.3f}  "
          f"(>1 = answer dominates; <1 = question dominates)")
    print("    pinning: dominant-sink only (Layers 2+2b basin structure preserved)")
    print(f"    UNK rate: {unk_pct:.1f}%  (masked) — "
          f"{unk_count}/{len(all_ans_ids)} tokens\n")

    # Sink regions are geometry-based (cached) — computed once
    sink_regions = lm._ensure_sink_regions()
    n_sinks      = len(sink_regions)
    epoch_log    = []

    # Fix 2: fermion exclusion — cap mass per node to prevent runaway accumulation
    for _m in lm.mf.modes:
        _m.max_rho = 0.05

    # Fix 1 pre-computation: encoder and output-node set, reused every context step
    _enc     = getattr(lm, 'encoder', None)
    _out_set = set(lm.output_nodes)

    # Part 3: startup confirmation — if this line doesn't print, stale .pyc was loaded
    print("  harmonic logging active  h_requested=(1,3,5,7)"
          "  — epoch summary will show h1/h3/h5/h7 with cap status")
    print("  Forcing: argmax encoder-phase (force_idx) — proven, no change")
    print("  Consolidation: force_idx one-hot (encoder phase) — feedback loop cut")
    print(f"  Psi diagnostic: skips answer prefix ({len(answer_prefix)} tok) — fires on content words")

    # Sliding-window resistance: W_LEN derived from α_gs consolidation timescale.
    # The field can move ~1/α_gs steps to deform by O(1) rad — that is the
    # timescale over which a winning sink deepens its attractor.  Resistance
    # must respond within that window to brake before the attractor sets.
    _alpha_gs_eff = lm.lm_alpha_gs * 1.0   # weight=1.0 per consolidation call
    _W_LEN = int(np.clip(round(1.0 / max(_alpha_gs_eff, 1e-6)), 15, 50))
    print(f"  sliding-window resistance: W_LEN={_W_LEN}"
          f"  (from α_gs={lm.lm_alpha_gs:.5f}  →  1/α_gs={1.0/max(_alpha_gs_eff,1e-6):.0f}"
          f"  clipped to [15,50])")
    if _W_LEN in (15, 50):
        print(f"  NOTE: W_LEN clipped to bound — raw 1/α_gs="
              f"{1.0/max(_alpha_gs_eff,1e-6):.0f} was outside [15,50]."
              f" Check α_gs if unexpected.")

    # occ_window carries across epochs — resistance responds to current conditions,
    # not epoch-cumulative history.  Reset only between distinct training runs.
    occ_window = deque(maxlen=_W_LEN)   # recent sink assignments for resistance

    # Part 2: encoder phase-spread diagnostic — before any consolidation runs
    _log_encoder_phase_spread(lm, sink_regions, _out_set, label=" pre-training")

    for epoch in range(start_epoch, n_epochs):
        _skip_pairs = start_pair if epoch == start_epoch else 0
        pair_win_fracs = []   # per-pair window-occ snapshots for oscillation check
        losses             = []   # all token losses (including UNK)
        non_unk_losses     = []   # losses for representable (non-UNK) tokens only
        triplet_ents       = []   # per pair: H of sink mass vector (after pair)

        # ── Structure-formation diagnostics ──────────────────────────────────
        # These are the PRIMARY signals for the native-grow run.
        # Run 2's fatal signature: flip_count_consec=0, contested_count=0,
        # sink_occupancy collapsed to one sink (~100% on s0 or s1).
        #
        # flip_count_consec : argmax(triplet) changed between CONSECUTIVE
        #                     forward passes within the same pair's answer.
        #                     flip > 0 = field uses >1 sink across a single answer.
        # contested_count   : forward passes where top-2 sinks within ε=0.05.
        #                     contested > 0 = basin boundary is active.
        # sink_occupancy    : absolute count of tokens assigned to each sink.
        #                     collapsed = one sink > ~95%.
        # token_sink_history: tid → [sink_idx, ...] for cross-pair consistency.
        flip_count_consec = 0
        contested_count   = 0
        sink_occupancy    = np.zeros(n_sinks, dtype=int)
        total_contexts    = 0
        token_sink_history = {}   # tid → list[int]
        epoch_h_contribs   = {}   # h → [abs_contrib, ...]  — harmonic diagnostic

        for qi, (q, a) in enumerate(qa_pairs):
            if qi < _skip_pairs:
                continue
            # Imprint question — permanently deforms θ₀; no restore (accumulation).
            # Uses sup_imprint_strength (down-weighted) so the answer pin dominates.
            q_ids = encode(q)
            lm.imprint_question(
                q_ids,
                n_passes         = n_imprint_passes,
                imprint_strength = sup_imprint_strength,
                steps            = imprint_steps,
            )

            a_ids    = answer_prefix + encode(a)
            examples = make_examples(a_ids, lm.context_len)

            # Reset per pair — consecutive flips measured within each answer
            prev_sink = None
            _psi_logged_this_pair = False   # psi diagnostic: fire once per sampled pair
            _ans_pos = 0                    # position within answer (0 = first example)

            for ctx, tgt in examples:
                _ans_pos += 1
                probs, locked, states, _ = lm.forward(ctx, steps=answer_steps)
                loss = cross_entropy(probs, tgt)
                losses.append(loss)
                if tgt != 0:
                    non_unk_losses.append(loss)

                if mask_unk and tgt == 0:
                    continue

                # Mass triplet: for H entropy metric only (not for routing)
                triplet      = lm.mf.get_triplet(lm.output_nodes, sink_regions)

                # Fix A+B+C: phase + harmonic dominant sink (replaces mass argmax)
                dominant_idx, _sw, _T, _hc, _psi = phase_dominant_sink(
                    lm.mf, lm.output_nodes, sink_regions,
                    return_h_contribs=True, return_psi=True)
                for _h, _vals in _hc.items():
                    epoch_h_contribs.setdefault(_h, []).extend(_vals)

                # Psi diagnostic: log raw pre-softmax resonances for sampled pairs.
                # Skips answer-prefix tokens so we see content words, not ':'.
                # Distinguishes: peaked-different-sinks (healthy) vs peaked-same-sink
                # (encoder over-concentration).
                if qi % 57 == 0 and not _psi_logged_this_pair \
                        and _ans_pos > len(answer_prefix):
                    _psi_str  = '  '.join(f"s{s}={_psi[s]:+.3f}" for s in range(n_sinks))
                    _sw_str   = '  '.join(f"s{s}={_sw[s]:.3f}"  for s in range(n_sinks))
                    _tok_str  = repr(decode([tgt]))
                    print(f"  [psi qi={qi:3d}] tok={_tok_str:12s}"
                          f"  psi=[{_psi_str}]  sw=[{_sw_str}]"
                          f"  T={_T:.4f}  dominant=s{dominant_idx}")
                    _psi_logged_this_pair = True

                # Fix 1: phase-resonance forcing target — breaks Matthew loop.
                # Forcing goes to whichever sink's current phase resonates with
                # the token's encoder phase, NOT whichever sink has the most mass.
                # Nothing is hardcoded; the assignment follows field phase geometry.
                _tok_ph   = (float(_enc.phi[tgt]) if _enc is not None
                             else 2.0 * np.pi * float(tgt) / V)
                _sink_phs = []
                for _reg in sink_regions:
                    _r_out = [j for j in _reg if j in _out_set]
                    if _r_out:
                        _th = np.array([lm.mf.modes[0].nodes[j].theta
                                        for j in _r_out])
                        _sink_phs.append(
                            float(np.angle(np.mean(np.exp(1j * _th)))))
                    else:
                        _sink_phs.append(0.0)
                force_idx    = int(np.argmax([np.cos(_tok_ph - sp)
                                              for sp in _sink_phs]))
                target_nodes = sink_regions[force_idx]

                # ── Structure signals ─────────────────────────────────────────
                sink_occupancy[dominant_idx] += 1   # cumulative (display/summary)
                occ_window.append(dominant_idx)      # sliding window (resistance)
                total_contexts += 1

                if prev_sink is not None and dominant_idx != prev_sink:
                    flip_count_consec += 1
                prev_sink = dominant_idx

                sorted_t = np.sort(triplet)[::-1]
                if sorted_t[0] - sorted_t[1] < 0.05:
                    contested_count += 1

                # Cross-pair token→sink history (basin consistency check)
                if tgt not in token_sink_history:
                    token_sink_history[tgt] = []
                token_sink_history[tgt].append(dominant_idx)

                e_tgt = 2.0 * np.pi * float(tgt) / V
                lm.mf.teacher_force_step(
                    target_nodes, e_tgt,
                    K_t=rtr_alpha, n_steps=pin_steps, locked=locked,
                    K_mass=K_mass,
                )

                pinned_states = lm.mf.get_all_states()
                # Build window-based occupancy counts for resistance.
                # Responds to current conditions (last W_LEN tokens), not epoch average.
                _occ_win = np.zeros(n_sinks)
                for _s in occ_window:
                    _occ_win[_s] += 1
                # Consolidate by force_idx (encoder phase) — NOT readout-sw.
                # Cuts the feedback loop: consolidation no longer reads back from
                # the attractor depths it is shaping.
                _force_onehot = np.zeros(n_sinks)
                _force_onehot[force_idx] = 1.0
                lm.update_ground_state(pinned_states, locked, weight=1.0,
                                       occupancy=_occ_win,
                                       sink_regions=sink_regions,
                                       sink_weights=_force_onehot)

                if alpha_omega > 0.0:
                    coherence = lm.output_coherence(pinned_states)
                    lm.mf.update_omegas(coherence, alpha_omega)

            # After each pair: measure H from current θ₀ (field already restored
            # by restore_all inside the next forward(), but we call it explicitly
            # so the H reading is from a clean ground-state start, not live state)
            lm.mf.restore_all()
            triplet_now = lm.mf.get_triplet(lm.output_nodes, sink_regions)
            H = float(-np.sum(triplet_now * np.log(triplet_now + 1e-12)))
            triplet_ents.append(H)

            # Snapshot current window occ frac for oscillation check
            _snap = np.zeros(n_sinks)
            for _s in occ_window:
                _snap[_s] += 1
            _snap_sum = max(float(_snap.sum()), 1)
            pair_win_fracs.append(_snap / _snap_sum)

            if ckpt_fn is not None and (qi + 1) % ckpt_every == 0:
                ckpt_fn(epoch, qi + 1)   # epoch=0-indexed; qi+1=next pair to run

            if (qi + 1) % log_every == 0:
                recent    = losses[-min(100, len(losses)):]
                recent_nu = non_unk_losses[-min(100, len(non_unk_losses)):]

                # Basin coherences from restored θ₀
                coh_parts = []
                for s, region in enumerate(sink_regions):
                    region_out = [j for j in region if j in set(lm.output_nodes)]
                    if region_out:
                        th_arr = np.array(
                            [lm.mf.modes[0].nodes[j].theta for j in region_out])
                        coh_parts.append(
                            f"r{s}={float(np.abs(np.mean(np.exp(1j * th_arr)))):.3f}")

                # H range over last log_every pairs
                H_win   = triplet_ents[-min(log_every, len(triplet_ents)):]
                H_lo    = float(np.min(H_win))
                H_hi    = float(np.max(H_win))
                H_mu    = float(np.mean(H_win))

                # Cumulative structure rates
                n_possible = max(total_contexts - (qi + 1), 1)
                flip_rate  = flip_count_consec / n_possible
                cntstd_rt  = contested_count / max(total_contexts, 1)

                # Per-sink occupancy fractions (cumulative epoch total)
                occ_frac = sink_occupancy / max(int(sink_occupancy.sum()), 1)
                occ_str  = ' '.join(f"s{i}={occ_frac[i]:.2f}" for i in range(n_sinks))

                # Sliding-window occ (current conditions, last W_LEN tokens)
                _win_now = pair_win_fracs[-1] if pair_win_fracs \
                           else np.ones(n_sinks) / n_sinks
                win_str  = ' '.join(f"s{i}={_win_now[i]:.2f}" for i in range(n_sinks))

                # Oscillation: variance of window-occ across last log_every pairs
                _osc_hist = pair_win_fracs[-min(log_every, len(pair_win_fracs)):]
                _osc_var  = float(np.mean(
                    [np.var([h[s] for h in _osc_hist]) for s in range(n_sinks)]
                )) if len(_osc_hist) > 1 else 0.0

                ppl_nu_str = (f"  ppl_nu={perplexity(recent_nu):.1f}"
                              if recent_nu else "")
                q_push_eff = sup_imprint_strength * lm.lm_alpha_gs
                a_push_eff = rtr_alpha * pin_steps / max(answer_steps, 1)
                aq_ratio   = a_push_eff / max(q_push_eff, 1e-12)

                print(f"  epoch {epoch+1}  pair {qi+1:>4}/{len(qa_pairs)}"
                      f"  ppl={perplexity(recent):.1f}{ppl_nu_str}"
                      f"  H=[{H_lo:.3f},{H_hi:.3f}]μ{H_mu:.3f}"
                      f"  flip={flip_count_consec}({flip_rate:.3f})"
                      f"  cntstd={contested_count}({cntstd_rt:.3f})"
                      f"  cum=[{occ_str}]  win=[{win_str}]  osc={_osc_var:.4f}"
                      f"  a/q={aq_ratio:.2f}"
                      f"  {'  '.join(coh_parts)}")

        # ── Epoch summary ─────────────────────────────────────────────────────
        mean_loss = float(np.mean(losses))
        ep_ppl    = perplexity(losses)
        ep_ppl_nu = perplexity(non_unk_losses) if non_unk_losses else float('nan')
        H_lo_ep   = float(np.min(triplet_ents))
        H_hi_ep   = float(np.max(triplet_ents))
        H_mu_ep   = float(np.mean(triplet_ents))

        # H frozen: range < 0.01 across all pairs → not differentiating
        H_frozen  = (H_hi_ep - H_lo_ep) < 0.01

        # Occupancy
        occ_total_ep = max(int(sink_occupancy.sum()), 1)
        occ_frac_ep  = sink_occupancy / occ_total_ep
        occ_ep_str   = '/'.join(f"{occ_frac_ep[i]:.2f}" for i in range(n_sinks))
        collapsed    = float(np.max(occ_frac_ep)) > 0.90   # one sink > 90%

        n_poss_ep    = max(total_contexts - len(qa_pairs), 1)
        ep_flip_rate = flip_count_consec / n_poss_ep
        ep_cntstd_rt = contested_count / max(total_contexts, 1)

        # Cross-pair flip rate (token consistency across pairs)
        xpair_obs = []
        for sinks in token_sink_history.values():
            if len(sinks) > 1:
                modal = max(range(n_sinks), key=lambda s: sinks.count(s))
                xpair_obs.extend(1 if s != modal else 0 for s in sinks)
        xpair_flip = float(np.mean(xpair_obs)) if xpair_obs else 0.0

        epoch_log.append({
            'epoch': epoch + 1, 'loss': mean_loss,
            'ppl': ep_ppl, 'ppl_nu': ep_ppl_nu,
            'H_lo': H_lo_ep, 'H_hi': H_hi_ep, 'H_mu': H_mu_ep,
            'flip_consec': flip_count_consec, 'flip_rate': ep_flip_rate,
            'contested': contested_count, 'contested_rate': ep_cntstd_rt,
            'occupancy': occ_frac_ep.tolist(),
        })

        flags = []
        if H_frozen:
            flags.append('H-FROZEN(range<0.01)')
        if collapsed:
            flags.append(f'BASINS-COLLAPSED(max={float(np.max(occ_frac_ep)):.2f})')
        if flip_count_consec == 0:
            flags.append('flip=0(Run2-signature)')
        flag_str = '  *** ' + ' | '.join(flags) + ' ***' if flags else ''

        # Per-harmonic contribution summary — all requested h shown, capped ones noted
        _n_per_sk_diag = max(1, len(lm.output_nodes) // n_sinks)
        h_summary = []
        h1_mean = float(np.mean(epoch_h_contribs.get(1, [0.0]))) + 1e-12
        for _h in (1, 3, 5, 7):
            if 2 * _h + 1 <= _n_per_sk_diag:
                _mu = float(np.mean(epoch_h_contribs.get(_h, [0.0])))
                _ratio = _mu / h1_mean if _h > 1 else 1.0
                h_summary.append(
                    f"h{_h}={_mu:.4f}({'x1' if _h==1 else f'x{_ratio:.2f}'})")
            else:
                h_summary.append(
                    f"h{_h}=cap(need {2*_h+1}>{_n_per_sk_diag}n/sk)")
        h_str = '  '.join(h_summary)

        print(f"\n  RTR epoch {epoch+1}:"
              f"  ppl={ep_ppl:.1f}  ppl_nu={ep_ppl_nu:.1f}"
              f"  H=[{H_lo_ep:.3f},{H_hi_ep:.3f}]μ{H_mu_ep:.3f}"
              f"  flip={flip_count_consec}({ep_flip_rate:.3f})"
              f"  cntstd={contested_count}({ep_cntstd_rt:.3f})"
              f"  occ=[{occ_ep_str}]"
              f"  xpair_flip={xpair_flip:.3f}"
              f"  (log3={np.log(3):.3f})"
              f"{flag_str}"
              f"\n  harmonics: {h_str}"
              f"  [n_out={len(lm.output_nodes)} n_per_sk={_n_per_sk_diag}"
              f"  h5_needs={2*5+1} h7_needs={2*7+1}]\n")
        _log_encoder_phase_spread(lm, sink_regions, _out_set,
                                  label=f" post-epoch{epoch+1}")

        if ckpt_fn is not None:
            ckpt_fn(epoch + 1, 0)   # start of next epoch; epoch+1 = completed count

    return epoch_log


def embed_constitution(lm, principles, tokenizer=None,
                       weight=3.0, steps_per_fwd=20, n_passes=3,
                       coherence_threshold=0.4):
    """
    Embed constitutional principles as deep attractors in the field.

    Step 1 — Weighted training: trains on principle text with 'weight'× loss,
              deepening attractors in the ground state. In W mode also updates
              the projection matrix; in RTR mode W is skipped (feat is None).
    Step 2 — Node identification:
              W mode  : measures bulk phase coherence |<e^{iθ}>| at output nodes;
                        high-coherence nodes (> coherence_threshold) are returned.
              RTR mode: measures per-node phase displacement |sin(θ − θ₀)| from
                        ground state after running constitution context. Nodes most
                        displaced are the ones whose attractors were engaged by the
                        constitution text — these are the constitution nodes.

    The CORRECT embedding order is: train first (semantics before topology).
    The caller should then set mf.constitution_nodes = returned set, which
    freezes those nodes during all future evolution and GS updates.

    Args:
        lm         : MultiModeFieldLM
        principles : list of principle strings
        tokenizer  : BPESubsetTokenizer; uses char tokenizer if None
        weight     : training weight multiplier (default 3.0×)
        n_passes   : training passes over the combined constitution text

    Returns:
        set of output node indices that encode constitutional attractors
    """
    encode = tokenizer.encode if tokenizer is not None else _encode_char

    constitution_text = '\n\n'.join(principles)
    token_ids = encode(constitution_text)
    examples  = make_examples(token_ids, lm.context_len)

    rtr_mode = (getattr(lm, 'readout_mode', '') == 'rtr')

    print(f"\n=== Constitution Embedding: {len(principles)} principles"
          f"  weight={weight}×  passes={n_passes}"
          f"  mode={'RTR' if rtr_mode else 'W'} ===")
    for p in principles:
        print(f"  · {p}")

    # Step 1: Weighted training on constitutional text
    for pass_i in range(n_passes):
        losses = []
        for ctx, tgt in examples:
            probs, locked, residual, feat = lm.forward(ctx, steps=steps_per_fwd)
            loss = cross_entropy(probs, tgt)
            losses.append(loss)
            if feat is not None:   # W path only; skipped in RTR mode
                lm.proj.update(feat, probs, tgt)
            coherence = lm.output_coherence(residual)
            lm.update_ground_state(residual, locked, weight=min(coherence * weight, 1.0))
        print(f"  pass {pass_i + 1}/{n_passes}:"
              f"  loss={np.mean(losses):.4f}  ppl={perplexity(losses):.1f}")

    # Step 2: Identify constitution nodes per principle
    constitution_nodes = set()
    out_arr = np.array(lm.output_nodes)

    for principle in principles:
        ids = encode(principle)
        ctx = ids[-lm.context_len:]
        _, _, states, _ = lm.forward(ctx, steps=80)

        for i, (th, _m) in enumerate(states):
            thetas_out = th[lm.output_nodes]

            if rtr_mode:
                # RTR path: nodes most displaced from ground state by this context
                # are the ones whose attractors were engaged — those are the
                # constitutional anchor nodes.
                th0_out    = lm.mf.theta_0s[i][out_arr]
                activation = np.abs(np.sin(thetas_out - th0_out))   # ∈ [0, 1]
                top5       = np.argsort(activation)[-5:]
                for idx in top5:
                    constitution_nodes.add(lm.output_nodes[idx])
            else:
                # W path: bulk coherence threshold
                coherence = float(np.abs(np.mean(np.exp(1j * thetas_out))))
                if coherence > coherence_threshold:
                    c_each = np.abs(np.exp(1j * thetas_out))
                    top5   = np.argsort(c_each)[-5:]
                    for idx in top5:
                        constitution_nodes.add(lm.output_nodes[idx])

    id_method = ('RTR displacement top-5/principle/mode'
                 if rtr_mode else f'coherence > {coherence_threshold}')
    print(f"  Constitution nodes identified: {len(constitution_nodes)}"
          f"  ({id_method})")
    return constitution_nodes


def evaluate(lm, text, steps=None, tokenizer=None):
    """
    Compute loss and perplexity on held-out text (no ground state update).
    """
    encode = tokenizer.encode if tokenizer is not None else _encode_char
    token_ids = encode(text)
    examples  = make_examples(token_ids, lm.context_len)
    losses    = []
    for ctx, tgt in examples:
        probs, _, _, _ = lm.forward(ctx, steps=steps)
        losses.append(cross_entropy(probs, tgt))
    return {'loss': float(np.mean(losses)), 'ppl': perplexity(losses)}
