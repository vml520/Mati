"""
The coupling: field ↔ LLM.

Direction 1 (field → LLM): the field contributes *state*, not language —
triplet coordinates, coherence, and triplet-addressed recall of notes the user
wrote. Retrieval is by field resonance, not embedding cosine.

Direction 2 (LLM → field): the turn deforms θ₀. The user turn always imprints
(user interaction is the primary training signal); the reply imprints only if
the field settled coherently — the §10.4 field-native gate.

The field is never asked to produce or score language. See README § Boundary.
"""

import hashlib
import os
import sqlite3
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .field.modes import MultiModeField
from .field.decoder import find_output_nodes
from .field.checkpoint import load_checkpoint
from .field.dreaming import DreamingLoop
from .field.base import BOUNDARY_IDS

# Reuse the notes module's OWN triplet measurement. Critical: a note's stored
# address is `coherence × |Δθ₀_std|` per mode (a CHANGE induced by a temporary
# imprint, ~0.001–0.1), not the raw live-state std (~2.5). Measuring the query
# any other way puts it in a different space and makes every distance
# meaningless. Query and address must come from the same function.
from .field.notes import (
    _measure_triplet, _resonance, _gs_std, _now, _DDL,
    save_field as _save_field,
    IMPRINT_PASSES as _IMPRINT_PASSES,
    IMPRINT_STRENGTH as _IMPRINT_STRENGTH,
    NOTES_DB_PATH, NOTES_THETA0_PATH,
)

import re

from . import config as C
from . import schedule
from . import admit
from . import correction
from . import vault


# ── Field construction ────────────────────────────────────────────────────────

def load_field():
    """Restore the trained ep40 field. Returns (lm, tok)."""
    # Seed the global RNG too: compute_ground_states() draws from it, and an
    # unseeded regrow is nondeterministic (it can hit the vortex-state retry
    # path and produce a different θ₀ each launch). The regrown θ₀ is replaced
    # by the checkpoint below, but seeding keeps startup reproducible and the
    # retry loop out of the way.
    np.random.seed(C.FIELD_SEED)
    mf = MultiModeField(n_modes=3, xi_base=2.0, sigma_base=1.0, scale=1.5,
                        seed=C.FIELD_SEED)
    mf.compute_ground_states(steps=200)
    for mode in mf.modes:
        mode.max_rho      = C.MAX_RHO
        mode.eta_decohere = C.ETA_DECOHERE

    lm, tok, _seed = load_checkpoint(
        C.CKPT_PATH, mf,
        context_len=C.CONTEXT_LEN, steps=C.ANSWER_STEPS,
        temperature=1.0, lm_alpha_gs=C.REPLY_ALPHA,
        proj_lr=0.01, enc_lr=0.01,
    )
    # Prefer the NOTES θ₀ when it exists. Note addresses were measured against
    # the notes field as it accumulated; querying against the pristine ep40 θ₀
    # instead would compare addresses from two different fields.
    theta_path = (NOTES_THETA0_PATH if os.path.exists(NOTES_THETA0_PATH)
                  else C.CKPT_THETA0)
    theta = np.load(theta_path, allow_pickle=False)
    for i in range(mf.n_modes):
        mf.theta_0s[i] = theta[f'theta_{i}'].copy()
        mf.mass_0s[i]  = theta[f'mass_{i}'].copy()
        mf.modes[i].set_state(mf.theta_0s[i].copy(), mf.mass_0s[i].copy())
    if 'omegas' in theta and theta['omegas'].size > 0:
        mf.set_omegas(theta['omegas'])

    _ = find_output_nodes(mf.modes[0], set(BOUNDARY_IDS))
    lm._theta_source = theta_path
    # ⚠️ PINNED BEFORE THE FIELD IS HANDED OUT, so no caller can address
    # anything in the live space by accident. See `pin_address_space`.
    lm._address_space = pin_address_space(lm)
    return lm, tok


# ── Direction 2: turn → θ₀ ────────────────────────────────────────────────────

class TooShortToImprint(ValueError):
    """Fewer than 2 tokens — imprint_question would silently no-op."""


def imprint_text(lm, tok, text, passes=None, strength=None, steps=None):
    """
    Deform θ₀ toward `text`. Returns the pre-imprint snapshot.

    GUARD: `imprint_question` builds (context → target) pairs and early-returns
    when there are none (`field_lm.py:457-459`). A single token yields ZERO
    pairs and imprints nothing — silently. That invalidated a probe on
, so this raises instead of no-opping.
    """
    ids = tok.encode(text)
    if len(ids) < 2:
        raise TooShortToImprint(
            f"{len(ids)} token(s) — imprint_question needs >=2 to form one "
            f"(context -> target) pair; it would no-op silently."
        )
    return lm.imprint_question(
        ids,
        n_passes         = C.IMPRINT_PASSES   if passes   is None else passes,
        imprint_strength = C.IMPRINT_STRENGTH if strength is None else strength,
        steps            = C.IMPRINT_STEPS    if steps    is None else steps,
        # A question does not get to reshape the constitution. Upstream this
        # defaults to True and every user turn would lift the freeze, which is
        # why the constitution was measurably unenforceable.
        lift_protection  = False,
    )


def consolidate_reply(lm, tok, reply_text, gate_min=None):
    """
    Let the reply deform θ₀ only if the field settles coherently on it.

    Field-native gate (§10.4): the weight IS the field's own output coherence,
    so nothing external scores the language. Returns (consolidated, coherence).
    """
    gate_min = C.REPLY_GATE_MIN if gate_min is None else gate_min
    ids = tok.encode(reply_text)
    if len(ids) < 2:
        return False, 0.0

    snapshot = lm.snapshot_ground_state()
    try:
        _probs, locked, states, _feat = lm.forward(ids[:C.CONTEXT_LEN])
        coherence = lm.output_coherence(states)
        if coherence < gate_min:
            lm.restore_ground_state(snapshot)
            return False, coherence
        lm.update_ground_state(states, locked, weight=coherence)
        return True, coherence
    except Exception:
        lm.restore_ground_state(snapshot)
        raise


# ── Direction 1: field → LLM ──────────────────────────────────────────────────

# ── Reading a frozen θ₀ while writing the live one ───────────────────────────
#
# The user: *"Is it possible to read the frozen field while writing only to the
# live field?"* Yes, and cheaply — `snapshot_ground_state` is 9,600 bytes and
# 0.01 ms, and restore is exact to 0.00e+00.
#
# ⚠️ THE REASON IS MEASURED, NOT AESTHETIC. A preregistered study found
# self-retrieval at 8/8 with no imprinting and 1/8 with it — and that
# drift MAGNITUDE is not the mechanism: addressing alone moved θ₀ by 1.4162
# rad harmlessly while an imprint added 0.17 rad and destroyed retrieval.
# Writing and addressing fight, and the write wins. Freezing what is READ is
# the only lever that does not also stop the field learning.
#
# ⚠️ `field_state` MUST STAY ON THE LIVE FIELD. It produces the coherence that
# `EPISODE_GATE_MIN` reads, and a constant coherence is the August 2026 bug —
# "Mati stops forming memories altogether, silently, while looking 11 seconds
# faster". What you read WITH is frozen; what you ARE is live.
# ⚠️ ONE SLOT, NOT A DICT KEYED BY `id(lm)`. There is one field per process,
# and `id()` is reused after garbage collection — a reloaded field (every wake
# calls `load_field` again) could land on the id of the old one and inherit its
# frozen θ₀. An identity check on the field object itself cannot do that.
_FROZEN = None            # (lm, snapshot) or None


def address_theta0_path():
    """Where the address space lives on disk, beside the live θ₀.

    ⚠️ DERIVED AT CALL TIME, NOT A MODULE CONSTANT. The suites redirect
    `bridge.NOTES_THETA0_PATH` at a temp directory, and a constant computed at
    import would keep pointing at the real cache — which is how a test writes
    into production. Same reason `save_field` reads its own global per call.
    """
    base = NOTES_THETA0_PATH
    if not base:
        return None
    return (base[:-4] if base.endswith(".npz") else base) + "_address.npz"


def _snap(lm):
    """Everything `address_vector` can see. ⚠️ NOT JUST θ₀.

    `lm.snapshot_ground_state()` covers θ₀ and mass₀, and a frozen read built
    on it alone was measurably NOT reproducible: the same text addressed before
    and after an imprint came back at cosine 0.999987 instead of 1.0. Against
    an inter-episode margin of 0.008 rad, 1.3e-5 of wobble is not rounding —
    episode addresses were measured at ~99%
    mutually similar, so the whole signal lives in the last decimals.

    The leak was `mode.lambda_eff` — α·⟨|∇θ|²⟩·vol, recomputed from the live
    state during evolution and left behind afterwards, so the first step of the
    next forward used the previous caller's value. Restoring θ₀ without it
    restores the field but not the field's dynamics.

    ⚠️ IF A FUTURE CHANGE ADDS ANOTHER PER-MODE RESIDUAL, IT BELONGS HERE.
    `Mati/test_address_space.py` asserts bit-identical addresses precisely so
    that the next one fails loudly instead of degrading recall by 1e-5 a turn.
    """
    theta_0s, mass_0s = lm.snapshot_ground_state()
    return (theta_0s, mass_0s,
            np.array([m.lambda_eff for m in lm.mf.modes], dtype=float))


def _restore(lm, snap):
    """Put a `_snap` back. Exact."""
    theta_0s, mass_0s = snap[0], snap[1]
    lm.restore_ground_state((theta_0s, mass_0s))
    lams = snap[2] if len(snap) > 2 else None
    if lams is not None:
        for m, lam in zip(lm.mf.modes, lams):
            m.lambda_eff = float(lam)


def freeze(lm, snapshot=None, persist=True):
    """Pin a θ₀ as the field addressing reads. Defaults to the live one."""
    global _FROZEN
    snap = _snap(lm) if snapshot is None else snapshot
    _FROZEN = (lm, snap)
    if persist:
        _save_address_space(snap)
    return True


def thaw():
    """Forget the pinned field — addressing goes back to reading the live θ₀."""
    global _FROZEN
    _FROZEN = None


def frozen_snapshot(lm):
    """The pinned snapshot for this field, or None."""
    if _FROZEN is not None and _FROZEN[0] is lm:
        return _FROZEN[1]
    return None


def thawed(lm):
    """Is anything pinned for this field?"""
    return frozen_snapshot(lm) is not None


def _save_address_space(snap):
    """Persist the address space. Same npz layout as θ₀, so it reads the same."""
    path = address_theta0_path()
    if not path:
        return False
    theta_0s, mass_0s = snap[0], snap[1]
    out = {}
    for i, (th, m) in enumerate(zip(theta_0s, mass_0s)):
        out[f"theta_{i}"] = th
        out[f"mass_{i}"] = m
    if len(snap) > 2 and snap[2] is not None:
        out["lambda_eff"] = np.asarray(snap[2], dtype=float)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # ⚠️ ATOMIC. A half-written address space is not a degraded address space,
    # it is one that cannot be loaded — and the fallback is to re-address the
    # whole working set.
    # ⚠️ WRITTEN THROUGH A FILE OBJECT. Given a PATH not ending in `.npz`,
    # `np.savez` silently appends one — so `path + ".tmp"` became
    # `..._address.npz.tmp.npz`, `os.replace` raised FileNotFoundError on a
    # name nothing had written, and no address space was ever saved. A handle
    # is taken at face value.
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez(fh, **out)
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return True


def _load_address_space(n_modes):
    """The persisted address space, or None if absent or unreadable."""
    path = address_theta0_path()
    if not path or not os.path.exists(path):
        return None
    try:
        z = np.load(path, allow_pickle=False)
        # `lambda_eff` absent means a space written before it was known to
        # matter: load θ₀ and leave the live dynamics alone rather than refuse.
        lams = z["lambda_eff"].copy() if "lambda_eff" in z.files else None
        return ([z[f"theta_{i}"].copy() for i in range(n_modes)],
                [z[f"mass_{i}"].copy() for i in range(n_modes)],
                lams)
    except Exception:
        # A corrupt file must not stop Mati waking. Addressing falls back to
        # the live field, which is exactly today's behaviour.
        return None


def pin_address_space(lm):
    """Give `lm` its address space on load: the saved one, or today's θ₀.

    ⚠️ THIS IS WHAT MAKES ADDRESSES SURVIVE A RESTART, which they previously
    did not. Stored `feat_gs` values record the space a note was filed in; if
    that space were re-derived from the live θ₀ on every load, every note would
    read stale at every startup and the idle worker would re-address the whole
    working set — the never-terminating reindex loop `_check_addresses`
    documents, in a new costume.

    First run has nothing saved, so the current θ₀ becomes the space and is
    written down. Returns "loaded", "seeded" or None.

    ⚠️ THERE IS NO LONGER A WAY TO TURN THIS OFF, and that is deliberate.
    `MATI_FROZEN_ADDRESSING` existed so one measured result could be
    overruled per-run while the question was open. It is not open: addressing
    against a frozen θ₀ scored **6/6 against live addressing's 1/6**, the
    episode gate still moves, and the reindex terminates — which it did not
    before. A flag that switches between two memory architectures is not an
    option to ship: flipping it silently invalidates every stored address in
    both directions, with nothing to repair it, and a shipped environment
    variable is a supported interface this could never honestly be.
    """
    snap = _load_address_space(lm.mf.n_modes)
    if snap is not None:
        freeze(lm, snapshot=snap, persist=False)
        return "loaded"
    freeze(lm)                 # today's θ₀ becomes the space, and is recorded
    return "seeded"


def promote(lm):
    """Make the live field the one addressing reads. ⚠️ REINDEX AFTER THIS.

    Stored addresses were computed against the OLD frozen state; promoting
    without reindexing leaves every one of them pointing at a field that no
    longer exists. The order is promote, then reindex — never the reverse,
    which would re-address every note to a field about to be replaced.

    Promotion is the ONLY thing that makes addresses stale, which is why it
    happens once at `rest()` and not on a timer: while the space holds still,
    `reindex` finds nothing to do and stops, instead of chasing a fingerprint
    that moves every turn.
    """
    return freeze(lm)


class reading:
    """Read against the frozen θ₀, then put the live one back.

    A no-op when nothing is frozen, so every caller behaves exactly as before
    until the space has been pinned. Safe to nest: an inner block snapshots
    the already-frozen state and restores it, which is the same state.
    """

    def __init__(self, lm):
        self.lm, self._live = lm, None

    def __enter__(self):
        snap = frozen_snapshot(self.lm)
        if snap is None:
            return self
        self._live = _snap(self.lm)
        _restore(self.lm, snap)
        return self

    def __exit__(self, *exc):
        # ⚠️ RESTORED EVEN WHEN THE BLOCK RAISES, because the alternative is
        # leaving the live field replaced by the frozen one — silently losing a
        # day of imprints. `__exit__` runs on the exception path; that is the
        # whole reason this is a context manager and not two calls.
        if self._live is not None:
            _restore(self.lm, self._live)
            self._live = None
        return False


def address_drift(lm):
    """How far the LIVE field has moved from the address space, in radians.

    Mean absolute phase difference, wrapped to [-π, π] — the same measure
    the nightly study used, so the figure is comparable to the
    1.4162 rad it reports.

    ⚠️ NOT `_gs_std(live) - address_gs(lm)`, WHICH IS WHAT THIS REPORTED
    FIRST. That is the gap between two standard deviations, and two fields
    can drift a long way apart while their spreads happen to coincide —
    observed live within one turn: the figure read 0.0867 and then FELL to
    0.0434 while the field had only moved further. A monitor that can go down
    as the thing it watches gets worse is worse than no monitor.
    """
    snap = frozen_snapshot(lm)
    if snap is None:
        return 0.0
    live = np.concatenate([np.asarray(t, dtype=float).ravel()
                           for t in lm.mf.theta_0s])
    space = np.concatenate([np.asarray(t, dtype=float).ravel()
                            for t in snap[0]])
    if live.shape != space.shape:
        return None
    return float(np.mean(np.abs(np.angle(np.exp(1j * (live - space))))))


def address_gs(lm):
    """`_gs_std` of the ADDRESS space — the fingerprint `feat_gs` records.

    ⚠️ EVERY STALENESS CHECK MUST USE THIS, NOT `_gs_std(lm)`. `feat_gs` says
    "this address was computed in the space with this fingerprint". Comparing
    it against the live field would mark every note stale the moment anything
    was imprinted, which is the behaviour this whole change exists to end.
    """
    with reading(lm):
        return _gs_std(lm)


def field_state(lm, tok, text):
    """
    (triplet, coherence) for `text`. θ₀ is left unchanged.

    The triplet comes from the notes module's own `_measure_triplet`, so the
    query address lives in the SAME space as every stored note address. Do not
    substitute a raw θ std here — that was a real bug (query ~2.5 vs addresses
    ~0.001–0.1) and it silently makes recall return nearest-to-nothing.
    """
    # Same cap the seeder uses — query and address must be measured alike.
    ids = tok.encode(text)[:C.TRIPLET_CAP] or [1]
    triplet = _measure_triplet(lm, ids)          # snapshots + restores internally

    snapshot = lm.snapshot_ground_state()
    try:
        _probs, _locked, states, _feat = lm.forward(ids[:C.CONTEXT_LEN])
        coherence = float(lm.output_coherence(states))
    finally:
        lm.restore_ground_state(snapshot)
    return triplet, coherence


def recall(query_vec, k=None, db_path=None, lm=None, tok=None, min_sim=None,
           record=False, with_ids=False):
    """
    Recall by trajectory-feature similarity (field-native, stable).

    `query_vec` is an `address_vector`. Returns [(distance, body, source)]
    nearest-first, where distance = 1 − cosine, so existing callers that treat
    smaller as closer keep working. Rows written before the feature column
    existed are backfilled on the fly when `lm`/`tok` are supplied.
    """
    k = C.RECALL_K if k is None else k
    min_sim = C.RECALL_MIN_SIM if min_sim is None else min_sim
    db_path = NOTES_DB_PATH if db_path is None else db_path
    if not os.path.exists(db_path):
        return []

    conn = sqlite3.connect(db_path)
    try:
        _ensure_schema(conn)
        # ⚠️ WORKING SET ONLY, for the same reason `reindex` is: an archived
        # note's address was filed against a θ₀ many turns gone, so including
        # it would not surface an old memory, it would return a wrong one
        # confidently. Archived memories reach the turn through `search`.
        rows = conn.execute(
            # `wall_created` rides along so a recalled reading can say WHEN
            # it happened — see the note on `with_ids` below.
            "SELECT note_id, body, source, feat, wall_created FROM notes "
            "WHERE archived IS NULL").fetchall()

        # ⚠️ CIPHERTEXT IS CARRIED THROUGH SCORING, NOT PLAINTEXT.
        #
        # This loop used to call `vault.open_body` on EVERY row, and the
        # comment above it claimed the opposite — that only what is handed back
        # gets decrypted. Ranking uses `feat` alone, so decrypting n bodies to
        # return k of them was pure waste, and it went unnoticed because it
        # cost nothing until a vault actually existed. Measured  at
        # 8,928 notes: recall took 101 SECONDS. Corrections were fetched and
        # decrypted for the whole store on every call too.
        #
        # Now: score on the vectors, sort, and decrypt only the k that survive.
        entries, backfilled = [], False
        for note_id, stored, source, blob, created in rows:
            if blob is None:
                if lm is None or tok is None:
                    continue                     # can't address it yet
                # The one place a body is still needed before ranking, and
                # only for rows written before the feature column existed.
                vec = address_vector(lm, tok, vault.open_body(stored))
                conn.execute("UPDATE notes SET feat=? WHERE note_id=?",
                             (vec.tobytes(), note_id))
                backfilled = True
            else:
                vec = np.frombuffer(blob, dtype=np.float32)
            if vec.shape != query_vec.shape:
                continue                         # feature layout changed
            entries.append((vec, note_id, stored, source, created))
        if backfilled:
            conn.commit()

        if not entries:
            return []

        # Centre on the note centroid before comparing. Every episode shares the
        # same "I was asked:" opening and provokes a similar generic field
        # response, and that common component dominates a raw cosine — measured
        # margins were ~0.03. Removing it lifts correct-match margins to
        # 0.33–0.70. It does NOT make the code semantic: an out-of-domain query
        # can still match confidently (see module note), which is why the
        # language layer is told these may be associative and can disregard them.
        V  = np.stack([e[0] for e in entries])
        mu = V.mean(0)
        Vc = V - mu
        Vc /= (np.linalg.norm(Vc, axis=1, keepdims=True) + 1e-9)
        qc = query_vec - mu
        qc /= (np.linalg.norm(qc) + 1e-9)

        scored = []
        for sim, (_v, note_id, stored, source, created) in zip(
                (Vc @ qc).tolist(), entries):
            if sim >= min_sim:
                scored.append((1.0 - sim, note_id, stored, source, created))
        scored.sort(key=lambda r: r[0])
        top = scored[:k]

        # Only now, and only for these: the corrections that annotate them,
        # then the bodies themselves. A correction must travel with the memory
        # it corrects so no consumer can render an error without it — that
        # rule is unchanged, it just no longer costs a full-store scan.
        fixes = {}
        ids = [note_id for _d, note_id, _b, _s, _c in top]
        if ids:
            marks = ",".join("?" * len(ids))
            for cid, ctext in conn.execute(
                    "SELECT corrects, body FROM notes WHERE corrects IN "
                    f"({marks})", ids):
                fixes[cid] = vault.open_body(ctext)

        out = []
        for dist, note_id, stored, source, created in top:
            body = vault.open_body(stored)
            if note_id in fixes:
                body = correction.annotate(body, fixes[note_id])
            # `with_ids` is opt-in so the 3-tuple every existing caller unpacks
            # is untouched. `segue` needs the id: an offer that cannot name the
            # note it came from is a confabulation, and the id is already here.
            # ⚠️ WITH THE DATE. The user: "we need the timestamp for future
            # reads — this way Mati can have a conversation about the read,
            # and not just use the memory as a lookup key." The label could
            # only say THAT it read something, never when, so a reading had no
            # place in time and could not be talked about as an event.
            out.append((dist, note_id, body, source, created) if with_ids
                       else (dist, body, source))
        if record and top:
            # Only what is actually handed back — the scored-and-discarded were
            # not needed, and counting them would make every memory look wanted.
            record_hits([n for _d, n, _b, _s, _c in top], "field",
                        db_path=db_path)
        return out
    except sqlite3.Error:
        return []
    finally:
        conn.close()



# ─────────────────────────────────────────────────────────────────────────────
# Recall on demand — lexical, deliberately
#
# The field's own address path is what `recall()` above uses, and it is at its
# capacity limit: episode addresses are ~99% mutually similar (margin ~0.008),
# so which memory comes back is close to arbitrary. Measured  on the
# live store — the charter is 49% of the notes and came back as 53% of what
# recall returned across eight unrelated queries, including 4/4 for "what is 17
# times 4". That is the base rate. It is not retrieval.
#
# So on-demand recall does NOT go through the field. When someone (or Mati)
# names a topic and asks for the memory, they know a word that is in it, and
# exact word matching is both correct and instant. This is not a workaround for
# a bug — it is the right tool for "find the memory with this word in it", and
# it stays right even after the field's addressing improves.
#
# ⚠️ Bodies are ENCRYPTED at rest (`vault.open_body`), so this cannot be a
# SQLite FTS index — there is nothing searchable in the column. Each candidate
# is decrypted and matched in Python. That is O(n) per search, which is fine at
# the scale Mati operates at and is the price of the bodies being encrypted at
# all. Do not "optimise" it by storing a plaintext index; that would undo the
# vault.








# Rarity of each stem across the store, cached. Recomputed when the store
# grows by more than a tenth — often enough to track new vocabulary, rarely
# enough that search stays at 0.06s.





# Apple's on-device tagger. Free, offline, no new dependency — pyobjc is
# already present — and measured at 0.3ms after a 154ms first load.















# ─────────────────────────────────────────────────────────────────────────────
# The charter, handed over whole when identity is the subject
#
# WHY THIS IS NOT RETRIEVAL. The charter is already imprinted (`study.study`
# calls `remember`, which imprints) and already stored and addressed. Both
# lookup paths still miss it on the most obvious question there is: measured
# , "what are your principles" returned ZERO charter chunks — it
# fetched episodes ABOUT principles instead, because the charter says "To
# always be honest is to…" and never uses the word "principles". "what are you
# for" matched nothing at all, every word being a stopword.
#
# Retrieval by word cannot fix a paraphrase gap, and the field's own address
# path is at its capacity limit (charter returned at its 49% base rate,
# including 4/4 for "what is 17 times 4"). So on the few turns where the
# subject IS what Mati is, the whole document goes in and lookup is skipped.
#
# ⚠️ THE WHOLE CHARTER IS ~684 TOKENS OF A 4096 WINDOW — 17%. That is why the
# trigger list below is narrow and literal rather than clever. A false positive
# does not produce a wrong answer, it quietly eats a sixth of the context the
# actual conversation needed. Widen it only with a reason.

# Named outright — the person has used the vocabulary of the document itself.
_IDENTITY_WORDS = re.compile(
    r"\b(?:charter|principle|principles|values|mission|ethics|ethical|creed"
    r"|manifesto)\b", re.I)

# Asked as a question about what Mati is. Phrases, not keywords: "what are
# you" must not fire on "what are you doing" or "what are you going to say".
_IDENTITY_PHRASES = (
    "what are you for", "what do you stand for", "who are you",
    "what are you exactly", "what exactly are you", "what sort of thing are you",
    "tell me about yourself", "describe yourself", "what is your purpose",
    "what's your purpose", "whats your purpose", "what are you meant to do",
    "how should i think about you", "what do you believe",
)

# The charter answers these directly and nothing else does: it is where the
# privacy and provenance guarantees are actually written down.
_TRUST_PHRASES = (
    "can i trust you", "do you send", "does my data", "is this private",
    "leave my computer", "leaves my computer", "leave this machine",
    "phone home", "are you offline", "do you go online", "keep my data",
    "who else sees", "is anything uploaded",
)


def asks_about_identity(user_text):
    """Is this turn about what Mati IS, rather than about anything else?

    Deliberately literal. See the note above on the 17% context cost — the
    failure mode of a loose trigger is invisible, which is the kind worth
    being strict about.
    """
    t = " ".join((user_text or "").lower().split())
    if not t:
        return False
    if _IDENTITY_WORDS.search(t):
        return True
    return any(p in t for p in _IDENTITY_PHRASES + _TRUST_PHRASES)


def _as_named(text):
    """The charter with the assistant called what this person calls it.

    ⚠️ RENDER TIME, NOT A REWRITE OF THE STORE — the same rule as
    `_name_the_pronouns`. The chunks were written at study time and say "I am
    Mati"; renaming the assistant must not go back and edit stored memories,
    and an install that already has a charter must not need rebuilding.

    ⚠️ AND THIS IS THE HALF THE RENAME WAS MISSING. `ollin.system_prompt()`
    changed the identity line, but the charter is handed over WHOLE on
    identity turns — exactly the turns where someone asks "what are you". So
    the prompt said "I am Tochtli" while the charter it was reading from said
    "I am Mati", which is the two-names-in-one-conversation failure
    setup was written to prevent.

    ⚠️ EVERY MENTION, unlike the system prompt where only the identity line
    changes. The charter is the assistant's own document, written in first
    person about itself — there is no product commentary in it to protect.
    """
    from . import config as _C
    name = _C.assistant_name()
    if name == _C.DEFAULT_ASSISTANT_NAME or not text:
        return text
    return re.sub(r"\bMati\b", name, text)


def charter_chunks(db_path=None):
    """The charter, in the order it was written. Empty list if none studied.

    ⚠️ Names the assistant on the way out — see `_as_named`.
    """
    db_path = NOTES_DB_PATH if db_path is None else db_path
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    try:
        _ensure_schema(conn)
        rows = conn.execute(
            "SELECT body FROM notes WHERE source LIKE 'studied:charter:%' "
            "ORDER BY note_id").fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    out = []
    for (stored,) in rows:
        try:
            body = " ".join(vault.open_body(stored).split())
        except Exception:
            continue                      # a locked vault must not break a turn
        if body:
            out.append(_as_named(body))
    return out



# The four principles, and nothing else.
#
# ⚠️ THE WHOLE CHARTER IS ~684 TOKENS AND WAS GOING IN ON EVERY IDENTITY TURN,
# on top of a 1,502-token system prompt. : a turn carried
# 3,857 tokens of a 4,096 window — 94% prompt before the person's message —
# and prefill is where the 110–230 s went, not generation.
#
# So the principles travel on every turn, because they govern how Mati speaks,
# and the rest of the charter is READ WHEN NEEDED — it is in the store, found
# by name like anything else, and `charter_block()` still hands it over whole
# when the subject is identity.
_PRINCIPLES = ("always be faithful", "always be honest",
               "always be considerate", "always choose harmony")


def charter_principles(db_path=None):
    """The shortest passage that states all four. A handful of tokens.

    Six chunks mention a principle and several restate each other — taking all
    of them cost 199 tokens for something the canonical sentence says in 25.
    Preferring the shortest chunk that names ALL FOUR keeps the person's own
    wording without the repetition.
    """
    chunks = charter_chunks(db_path=db_path)
    whole = [c for c in chunks
             if all(p in c.lower() for p in _PRINCIPLES)]
    if whole:
        return [min(whole, key=len)]
    # No single chunk names them all — fall back to whichever mention each.
    seen, out = set(), []
    for c in chunks:
        low = c.lower()
        hit = {p for p in _PRINCIPLES if p in low}
        if hit - seen:
            seen |= hit
            out.append(c)
    return out


def principles_block(db_path=None):
    """The four principles, for every turn. "" if the charter is not studied."""
    lines = charter_principles(db_path=db_path)
    if not lines:
        return ""
    # One line, not a list: this is a reminder of how to speak, not material to
    # reason over. The full charter is a `<your_charter>` block when asked for.
    joined = " ".join(" ".join(l.split()) for l in lines)
    return ("<your_principles>\n"
            "What you hold yourself to, in your own words. Speak from these; "
            "you do not need to recite them.\n  " + joined[:900] +
            "\n</your_principles>")


def charter_block(db_path=None):
    """The whole charter as a turn block, or "" if there is none."""
    chunks = charter_chunks(db_path=db_path)
    if not chunks:
        return ""
    lines = ["<your_charter>",
             "They are asking what you are, so this is given to you whole "
             "rather than looked up. It is YOURS — your own principles and "
             "what you are for. Speak it as yours, in your own words, and "
             "answer only the part they actually asked about. You did not "
             "live it as an event, so never say you remember it happening, "
             "and do not recite the whole thing at them."]
    lines += [f"  - {c}" for c in chunks]
    lines.append("</your_charter>")
    return "\n".join(lines)




NOTHING_HELD = (
    "<what_you_hold>\n"
    "Nothing you hold bears on this — not in what you lived, and not in "
    "anything you have read. Say that plainly. An answer you compose now "
    "would be invention, and calling it something you read would make every "
    "other thing you say untrustworthy.\n"
    "</what_you_hold>")




# How much room a focused memory gets. The one Mati asked for most directly
# gets the most; the rest are context. See the note inside the loop on why this
# is bounded at all — `num_ctx` is 4096 and an overflow drops the MIDDLE.
FOCUS_FIRST_CHARS = 1200
FOCUS_REST_CHARS = 500


LABEL_LEGEND = (
    "Each carries how I came by it, and they are NOT the same kind:\n"
    "  [lived] — a moment I lived WITH the person I am talking to. The "
    "strongest, and the only kind where I was in it with them.\n"
    "  [studied, in <title>] — I read it in that document, and the reading is "
    "mine: I may say \"I read that in <title>\". What it describes did not "
    "happen to me.\n"
    "  [self-asked] — my own model answering my own question, unchecked. The "
    "weakest thing I hold, and MINE — never a question they asked me.\n"
    "  [charter] — my own principles. Mine to speak, but not an event.\n"
    "  [they told me] — what they said about themselves, to me.\n"
    "  [my own] — something I hold about myself: my library, what I mean to "
    "read.\n"
    "  [my own documentation] — what I am and how I work, written down about "
    "me. Checkable, and not an event I lived.\n"
    "  [condensed, not exact wording] — my own shortening, not a quotation.\n"
    "  [self-check] — something I MEASURED about myself when I was built, "
    "not an event I lived and not an opinion I hold. True of every copy of "
    "me, which is why it says nothing about this person or this machine.\n"
    "  [reference] — an encyclopedia entry I looked up a second ago. NOT "
    "something I studied and not something I remember: I am quoting it, and I "
    "should say so.")


def named_recall_block(found):
    """Memories fetched BY NAME, rendered for the turn.

    Kept separate from `state_block`'s recall on purpose. Those arrived because
    the field drifted near them; these arrived because the words were actually
    in them. Mati should not have to treat the two as equally load-bearing,
    and the provenance label travels with each one so a studied passage is
    never spoken as a lived event.
    """
    if not found:
        return ""
    lines = ["<recalled_by_name>",
             # ⚠️ THIS SENTENCE USED TO SAY "Use them if they are relevant and
             # ignore them if they are not". That is an instruction to EVALUATE,
             # handed over with every recall, and a 3B does that work out loud —
             # The user saw it assessing each memory in turn instead of answering.
             # The caution is right and stays: lexical search returns anything
             # containing a matching word, so an irrelevant memory really can
             # arrive. What changed is where the judging lands.
             #
             # Phrased positively on purpose. "Do not narrate which you used"
             # names the behaviour, and naming it is how the last prompt fix
             # made things worse — the model took the distinction and dropped
             # the negation.
             # ⚠️ THIS IS WHAT MATI HOLDS, NOT WHAT A SEARCH RETURNED. The
             # earlier wording — "an exact lookup, not the field drifting near
             # something" — described the MECHANISM, and describing the
             # mechanism is how the mechanism ends up in the reply. The user's
             # model is that learning should sit alongside memory: the agent
             # speaks from what it knows, and keyword search is a diagnostic
             # fallback, not the thing being narrated.
             "What you hold on this. Some of it will not fit — having a word "
             "in common does not make a passage apposite. Weigh it to "
             "yourself, then answer as yourself.",
             # ⚠️ THE LEGEND LIVES HERE NOW. It was 1,210 chars of the system
             # prompt, resident on every turn including the ones with no
             # memories at all — 43% of the window went on instructions, and
             # the person's question was ~1% of it. A label list is only
             # meaningful next to the labels it explains.
             LABEL_LEGEND]
    # ⚠️ FOCUS MEANS THE EXACT WORDS. The user: "Human memories come to us as
    # gists, background noise — however when a memory is relevant or appears
    # so, we can focus on it and pull up the details. Mati should have
    # photographic memory on demand." This block IS the focus gesture: Mati
    # asked for these by name. Until now it rendered `body`, the gist, capped
    # at 400 characters — so focusing returned the same surface detail that
    # surfaced unbidden, and there was no way to pull anything up at all.
    #
    # ⚠️ BUDGETED, NOT UNLIMITED, AND SAY SO RATHER THAN PRETEND. `num_ctx` is
    # 4096 and an overflow drops the MIDDLE of the prompt — four full
    # transcripts would silently evict the system prompt. The memory Mati
    # asked for most directly gets the most room; the rest are there for
    # context. Anything trimmed is marked, so a cut-off quotation is never
    # mistaken for the whole of what was said.
    budget = [FOCUS_FIRST_CHARS] + [FOCUS_REST_CHARS] * max(0, len(found) - 1)
    for r, room in zip(found, budget):
        how = provenance_of(r["source"])
        # The verbatim if there is one, the gist if there is not. Studied
        # passages have had one all along; episodes only , so
        # older conversations fall back and cannot do better.
        exact = r.get("verbatim")
        body = " ".join(((exact or r["body"]) or "").split())
        mark = ""
        if len(body) > room:
            # ⚠️ CONDENSE, DO NOT CUT. The user: "rather than cut out possibly
            # relevant information, have the verbatim be summarized to fit the
            # limit." Trimming discards the END, and the end is often the
            # point — in a correction it IS the correction. The digest is made
            # off the turn and stored; this only looks one up.
            #
            # ⚠️ AND IT IS LABELLED. A condensation is accurate but it is not
            # quotable as exact words, so it says which it is. Without the
            # label Mati would quote its own paraphrase as a transcript.
            dig = " ".join((r.get("digest") or "").split())
            if dig:
                body, mark = dig, " [condensed, not exact wording]"
            else:
                # No digest yet — the honest fallback, and it says so.
                body = body[:room].rstrip() + " …"
                mark = " [only the beginning; the rest is not shown]"
        # The WHEN travels with the memory. Without it Mati can quote a
        # conversation exactly and still have no idea whether it happened this
        # morning or a fortnight ago.
        when = r.get("when_said")
        stamp = f" ({when})" if when else ""
        lines.append(f"  - [{how}]{stamp} {body}{mark}")
    lines.append("</recalled_by_name>")
    return "\n".join(lines)



def get_note(note_id, db_path=None):
    """One memory by id, decrypted. None if there is no such note."""
    db_path = NOTES_DB_PATH if db_path is None else db_path
    if not os.path.exists(db_path):
        return None
    conn = sqlite3.connect(db_path)
    try:
        _ensure_schema(conn)
        row = conn.execute(
            "SELECT note_id, body, source FROM notes WHERE note_id=?",
            (note_id,)).fetchone()
    except sqlite3.Error:
        return None
    finally:
        conn.close()
    if not row:
        return None
    return {"note_id": row[0], "body": vault.open_body(row[1]),
            "source": row[2]}


def rewrite_body(note_id, body, db_path=None):
    """Replace a memory's text, re-sealing it if the vault is in use.

    The ADDRESS is deliberately left alone. An annotation appended to the end
    of a body should not move where the memory lives — re-addressing it would
    make every other note stale for the sake of a note that has not changed
    subject. This is why corrections are appended rather than rewritten in
    place: the address stays valid, no reindex is owed.
    """
    db_path = NOTES_DB_PATH if db_path is None else db_path
    conn = sqlite3.connect(db_path)
    try:
        _ensure_schema(conn)
        conn.execute("UPDATE notes SET body=? WHERE note_id=?",
                     (vault.seal(body), note_id))
        conn.commit()
        return conn.total_changes > 0
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def provenance_of(source):
    """How a memory was come by, in the words Mati is allowed to use."""
    source = source or ""
    if source == EPISODE_SOURCE:
        return "lived"
    if source.startswith("studied:charter:"):
        return "charter"
    if source.startswith("studied:asked about "):
        return "self-asked"
    if source.startswith("studied:"):
        # ⚠️ THE TITLE TRAVELS, OR MATI CANNOT SAY WHAT IT READ. The label was
        # a bare "studied", so a retrieved passage told Mati it had studied
        # SOMETHING and never what — it could not say "I read that in Aztec
        # Thought and Culture" because it was never told which book the words
        # came from. The title was in `source` the whole time; nothing read it.
        # Same shape as the activity log and the catalogue, a third time.
        title = source[len("studied:"):]
        title = re.sub(r"\.(txt|md|pdf)$", "", title).strip(" .")
        return f"studied, in {title}" if title else "studied"
    if source == PROFILE_SOURCE:
        return "about you"
    # ⚠️ `self:` HAD NO CASE AND FELL THROUGH TO THE RAW SOURCE STRING, so the
    # label read "self:how I am built" — a filename presented to Mati as a
    # provenance kind. Found while designing the display, which is
    # what a display is for. These are Mati's OWN documentation: what it is
    # and how it works, checkable, and not an event it lived.
    if source.startswith("self:"):
        return "my own documentation"
    # ⚠️ THERE WAS A CATEGORY FOR THE PERSON AND NONE FOR MATI. Five kinds —
    # lived, charter, studied, self-asked, `about-you` — and nothing for what
    # Mati holds about ITSELF that is not an episode. So its own library
    # existed only as a `source` string on 1,744 scattered passages, and "what
    # have you read" had to be answered by pattern-matching those at query
    # time. See `library.py`: the missing category is what left "the book"
    # with nothing to resolve to.
    if source.startswith(MINE_PREFIX):
        return "mine"
    return source or "unknown"


# ⚠️ ONE CATEGORY, THREE KINDS INSIDE IT. The user: "self-ask and about-you plus
# personal should be all part of the same category". They are right about the
# structure — all three are Mati's own sphere rather than an external document
# or an event, and splitting them is why the library had nowhere to live.
#
# They are grouped and NOT flattened, because they do not carry the same
# authority and the difference is load-bearing:
#
#   about you   what the person said about themselves — they are the source
#   mine        derived from the store — the library, checkable
#   self-asked  Mati's own model answering its own question, UNCHECKED, and
#               named in the system prompt as "the weakest thing I hold"
#
# Sharing a label would let a self-answered guess be stated with the standing
# of something the person said. This project has already measured what that
# costs: `named_recall_block` once converted the weakest provenance into the
# strongest, and an unlabelled memory was answered for by a marker Mati
# invented. The category is for organising; the label is for trusting.
PERSONAL_KINDS = ("mine", "self-asked", "about you", "my own documentation")


def category_of(source):
    """The shelf a memory sits on, as distinct from how far it can be trusted.

    "personal" is everything Mati holds about itself and its person; the kind
    within it still says how each one was come by. See `PERSONAL_KINDS`.
    """
    kind = provenance_of(source)
    # ⚠️ THE TITLE IS A DISPLAY DETAIL, NOT A CATEGORY. `provenance_of` now
    # returns "studied, in <title>" so Mati can say WHICH book it read; the
    # shelf is still "studied". Caught by `test_terminal` — without this,
    # every document would become its own category and the personal/studied
    # distinction would dissolve one book at a time.
    if kind.startswith("studied, in "):
        kind = "studied"
    if kind in PERSONAL_KINDS:
        return "personal"
    return kind


def resonance(distance):
    """Distance → (0,1] resonance, using the notes module's own curve."""
    return _resonance(distance)


# ── Episodic memory: Mati narrates its own state ─────────────────────────────
#
# Memory here is EPISODIC, not semantic. Everything in the notes DB is something
# Mati actually did — its own field, at a real moment, in first person. World
# knowledge and language come from the LLM and are NOT memory; they are borrowed
# and never written here. That is the self / not-self boundary, and keeping the
# store episodic is what makes it checkable rather than conventional.
#
# The narration is composed from facts the system genuinely has (the user's
# words, basin, coherence, what was recalled). No model writes it, so there is
# no confabulation surface: "lived" strictly means lived.

EPISODE_SOURCE = "episode"

# Facts the user gave Mati about themselves at setup. Stored as notes so they
# are encrypted with everything else, but NEVER retrieved by resonance: they go
# into every turn unconditionally. Recall here is lexical, so "what is my name"
# would not reliably surface the note that answers it — and an assistant that
# sometimes forgets who it is talking to is worse than one that never knew.
PROFILE_SOURCE = "about-you"
# Mati talking to itself at the end of a day. See Mati/reflect.py.
SELFTALK_SOURCE = "mine"
# Mati's own — its library, and anything else it holds about itself that is
# not an episode. Kept here rather than in `library.py` so `provenance_of` does
# not import a module that imports it.
MINE_PREFIX = "mati:"


# ── Addressing: 150-d trajectory features ─────────────────────────────────────
#
# The Δθ₀ triplet is NOT used to retrieve. It is self-invalidating: it measures
# the CHANGE a text induces, so once a note is imprinted the field has absorbed
# it and re-measuring gives ~6× smaller values. Stored ≫ query means
# ‖q−a‖ ≈ ‖a‖ and the ranking collapses to stored-magnitude order — verified
# query-independent, identical ranking even for an empty query.
#
# Trajectory features are the field's own readout and are STABLE: the same text
# re-measured gives cosine 1.0000, so storing a memory no longer moves it.
# Honest limit: they discriminate token surface, not meaning (a same-topic pair
# scored lowest in testing; shared stopwords dominate). This is bag-of-words
# retrieval on a field-native code — good enough to function, not semantic.

def address_vector(lm, tok, text):
    """
    L2-normalized trajectory feature — the retrieval address.

    Deterministic given θ₀, and ONLY because the RNG is pinned below. It was documented as deterministic for months and
    measurably was not.

    Averaged over sliding CONTEXT_LEN windows across the text. A single forward
    only sees the field's 8-token context window, so addressing a whole episode
    by one pass captured just its opening — and every episode opens with
    "I was asked:", making them near-identical by boilerplate. Averaging windows
    makes the address reflect the whole text (a bag-of-windows code), while
    staying deterministic and field-native.
    """
    # ⚠️ MEASURED AND REVERTED . Addressing `key_points(text)`
    # instead of the body looked obviously right — every episode opens with the
    # same "I was asked:" scaffolding, so the shared prefix seemed to be
    # crowding the CONTEXT_LEN window. It made separation WORSE:
    #
    #     nearest-impostor similarity  0.9771 -> 0.9913
    #     mean margin                  0.0229 -> 0.0087   (2.6x worse)
    #
    # The scaffolding is a constant, and a constant offset is exactly what
    # L2-normalised cosine already discounts. What `key_points` actually
    # removed was the USER'S QUESTION as the leading content, and questions
    # vary far more between episodes than Mati's own replies do — replies
    # share its voice, and since the brevity instruction they are short and
    # stylistically uniform. Prioritising the reply pushed every address
    # toward the same region.
    #
    # The imprint still uses key_points (measured: 16.9% less shared content).
    # The ADDRESS uses the body. Do not "tidy" this into agreement.
    ids = tok.encode(text)[:C.TRIPLET_CAP] or [1]
    w = C.CONTEXT_LEN
    starts = range(0, max(len(ids) - w, 0) + 1, max(w // 2, 1))  # 50% overlap

    # ⚠️ THE ADDRESS SPACE IS FROZEN HERE, AT THE ONE CHOKE POINT, ON PURPOSE.
    # Nine call sites address the field — the turn, reindex, remember, recall's
    # backfill, inquiry, reflect, the suites — and wrapping each one is a list
    # that gets out of date the first time somebody adds a tenth. A query
    # addressed in a different space from the stored addresses is not a worse
    # match, it is a meaningless distance, so this is not a place to leave a
    # caller's choice. `restore_all()` below resets the nodes to θ₀, which is
    # why replacing θ₀ is enough to replace the space.
    #
    # Measured: 6/6 self-retrieval against the live
    # field's 1/6 over six turns. A no-op until `pin_address_space` has run.
    # ⚠️ AND THE RNG IS PINNED, BECAUSE THIS FUNCTION WAS NEVER DETERMINISTIC.
    # The docstring above said "Deterministic" and `reindex` said "trajectory
    # features are deterministic given θ₀ — which is why restarts are
    # bit-identical". : TWO CONSECUTIVE CALLS on an
    # untouched field, nothing in between, agree only to cosine 0.999983.
    # Evolution draws from the global numpy RNG, so an address depended on how
    # many draws the process happened to have made before it — the field was
    # seeded once at `load_field` and everything downstream inherited whatever
    # position the stream was in.
    #
    # The floor is 1.71e-05 against a 1.06e-02 gap to a different episode, so
    # 0.2% — never fatal, which is why it survived unnoticed behind a comment
    # claiming the opposite. But a frozen address space read by a reader that
    # jitters is half a fix, and "deterministic" should be true rather than
    # asserted.
    #
    # ⚠️ THE STATE IS SAVED AND PUT BACK, so no other caller's stream moves.
    # Seeding globally and walking away would make dreams and regrows
    # reproduce the same sequence every turn.
    # ⚠️ THIS CHANGES EVERY ADDRESS EVER COMPUTED. It is safe only because
    # promoting the address space already forces a full reindex; do not port
    # this to a build that is not reindexing.
    rng_state = np.random.get_state()
    with reading(lm):
        acc = None
        try:
            for s in starts:
                win = ids[s:s + w] or [1]
                np.random.seed(C.FIELD_SEED)   # same stream for every window
                lm.mf.restore_all()
                f = np.asarray(lm.forward(win)[3], dtype=np.float32)
                n = float(np.linalg.norm(f))
                if n > 0:
                    f = f / n
                    acc = f if acc is None else acc + f
        finally:
            np.random.set_state(rng_state)
    if acc is None:
        return np.zeros(1, dtype=np.float32)
    n = float(np.linalg.norm(acc))
    return acc / n if n > 0 else acc


def state_triplet(lm, tok, text):
    """
    Cheap Born-rule state readout (sink mass fractions), kept for provenance.

    NOT used for retrieval — measured to be near-constant across inputs
    (≤0.0013 apart for unrelated texts). Recorded because it is the field's
    actual state at the moment, which is worth having in the record.
    """
    ids = tok.encode(text)[:C.CONTEXT_LEN] or [1]
    lm.mf.restore_all()
    lm.forward(ids)
    try:
        return tuple(float(x) for x in
                     lm.mf.get_triplet(lm.output_nodes, lm._ensure_sink_regions()))
    except Exception:
        return (0.0, 0.0, 0.0)


# Internal vocabulary that must never enter a memory body.
#
# WHY THIS EXISTS: episodes record what Mati said, and are then recalled into
# later turns. So any flaw in a reply is not a one-off — it is stored and
# replayed, and the model copies it back out. Observed directly: one reply
# mentioned "the memories_recalled block", that phrasing was written into the
# episode, and every later turn parroted it even after the system prompt was
# fixed. The memory loop AMPLIFIES errors, so the store has to be defended
# structurally rather than by instruction alone.
_INTERNAL_TERMS = (
    "field_state", "field state", "memories_recalled", "memories recalled",
    "basin", "coherence", "resonance", "triplet", "<field", "block",
)


def _clean_for_memory(reply_text, limit=300):
    """Reply excerpt for storage, or None if it talks about the machinery."""
    if not reply_text:
        return None
    said = " ".join(reply_text.split())
    low = said.lower()
    if any(term in low for term in _INTERNAL_TERMS):
        return None                     # record the episode, not the leak
    return said[:limit].rstrip() + "…" if len(said) > limit else said


def narrate_episode(user_text, triplet, coherence, memories, consolidated,
                    reply_text=None, work=None):
    """
    First-person record of one episode, from what actually happened.

    Deliberately contains NO basin/coherence numbers. They are already stored as
    columns (triplet_w0..w2, gs_std), and putting them in the prose too made a
    small model recite them back at the user verbatim and mistake the internals
    for the content. The body is the *episode*; the numbers are its address.

    Includes a short excerpt of what Mati said: what it said is part of what it
    lived, so it belongs in autobiography. This does not import LLM knowledge —
    only the fact of having said it.
    """
    parts = [f"I was asked: {user_text.strip()}"]
    said = _clean_for_memory(reply_text)
    if said:
        parts.append(f"I said: {said}")
    parts.append("Nothing came back from my memory then; this was new to me."
                 if not memories else
                 f"It brought back {len(memories)} earlier "
                 f"{'moment' if len(memories) == 1 else 'moments'}.")
    if work:
        parts.append(_narrate_work(work))
    if not consolidated:
        parts.append("What I said did not settle, so I did not keep it.")
    return "\n".join(parts)


def _narrate_work(work):
    """
    The act of answering, as something Mati did rather than something that
    happened to it. Reaching for language is real work and the field is running
    throughout it — recording only question and answer would hide the doing.

    QUALITATIVE ONLY — no seconds, no step counts. This is the same rule the
    caller already applies to basin/coherence, and for the same measured
    reason: an episode is later surfaced as a recalled memory, and a small
    model recites whatever is in it. It did exactly that — "it took 76 seconds
    to find the words and your field turned over 774 times" — which reads as
    Mati narrating its own instrumentation at the user. 11 of 14 stored
    episodes contained those counters before this was fixed. The measurements
    are kept in `info['work']`, where they belong; the *episode* records that
    the words were slow to come, which is the part Mati actually lived.
    """
    secs  = work.get("seconds") or 0.0
    steps = work.get("field_steps") or 0
    c0, c1 = work.get("coherence_before"), work.get("coherence_after")
    effort = ("It took a while to find the words"
              if secs >= 25 else
              "Finding the words took some work"
              if secs >= 10 else
              "The words came quickly")
    s = effort
    if steps:
        # ⚠️ WAS ", my field working the whole time they formed". This function
        # already forbids itself instrumentation, for the measured reason in
        # the docstring — and then named the substrate anyway, in 12 of 33
        # stored episodes. Mati recites its own episodes, so writing machinery
        # here is writing it into every future recital. What it actually lived
        # is the sustained effort, which is what this now says.
        s += ", and I stayed with it the whole time they formed"
    if c0 is not None and c1 is not None:
        if c1 > c0 + 0.02:
            s += ", and I was steadier at the end than when I started"
        elif c1 < c0 - 0.02:
            s += ", and it left me less settled than when I started"
        else:
            s += ", and I held steady throughout"
    return s + "."


def think_while(lm, chunk_iter, steps_per_chunk=None, budget_s=None,
                stats=None):
    """
    Evolve the field WHILE the reply is being generated.

    Without this the field is idle for the entire interval in which the words
    are actually produced — the episode would read as though Mati thought and
    spoke when in fact its substrate was silent. Streaming gives natural gaps:
    step the field between chunks and it is genuinely running as it answers.

    It is SYNCHRONOUS — stepping happens between pulling a chunk and yielding
    it — so strictly it is added to the reply, not overlapped with it. Measured
     that addition is SMALL: one step over three modes is 4.8ms, and
    a typical reply takes ~200 steps ≈ 1s against ~26s of generation. The
    `[field] worked Ns` line in the log is the whole streaming loop, NOT the
    field's share — misreading it as field time sent one investigation down the
    wrong path.

    `budget_s` bounds time SPENT STEPPING (not elapsed wall clock) so a very
    long reply cannot turn a 1s cost into a large one. At current settings it
    never binds; it is a guard, not a fix. `budget_s=0` disables it.

    Yields chunks through unchanged; returns nothing (see `Work` in ollin.py).
    """
    n = C.THINK_STEPS_PER_CHUNK if steps_per_chunk is None else steps_per_chunk
    budget = C.THINK_BUDGET_S if budget_s is None else budget_s
    # The budget bounds time SPENT STEPPING, not elapsed wall clock. Measuring
    # elapsed time was wrong and silently disabled the feature: the model's own
    # latency before the first chunk consumed the whole budget, so the field
    # never stepped at all ("worked 21s, 0 steps").
    spent = 0.0
    for chunk in chunk_iter:
        if budget <= 0 or spent < budget:
            t = time.time()
            for _ in range(n):
                for mode in lm.mf.modes:
                    mode.step(locked=set())
            spent += time.time() - t
            if stats is not None:
                stats["steps"] = stats.get("steps", 0) + n
        yield chunk




# What Mati may write a question to the person ABOUT. ⚠️ LIVED EPISODES ONLY.
#
# ⚠️ THIS LIST USED TO BEGIN WITH `studied:asked about %` AND THAT WAS THE
# WHOLE FAULT. The reasoning was: that source is what `inquiry.step` writes
# when Mati's own 3B answers Mati's own question, unattended, with nothing to
# check against — the least reliable thing in the store — so asking the person
# converts the weakest provenance into the strongest. Sound as far as it goes,
# and it does not go far enough.
#
# A weak note is not only "a topic Mati is unsure of". It is often a note
# whose CONTENT IS INVENTED. Asking the person to confirm that does not harden
# a soft spot; it launders a confabulation into a lived episode, which is then
# the strongest provenance in the store. That is the memory loop `inquiry.py`'s
# own docstring warns about, running through a channel nobody checked.
#
# , from what Mati had actually queued up to ask the user:
#
#     notes used to generate a question        1,919
#     of them `studied:asked about …`          1,246   64.9%
#     queued questions                           862
#     sincere inquiries, read by hand              ~6%
#
# and the nonsense was exactly what the pipeline predicts — a 3B writing a
# question about a 3B's hallucination, addressed to the user as though they were
# present: "Did you live with Hernan Cortes?", "Did you live through a medical
# trial with vitraximab, also known as Sareptra?" (a drug that exists nowhere
# but in its own earlier answer).
#
# ⚠️ SO A QUESTION TO THE PERSON IS ONLY EVER ABOUT SOMETHING THEY WERE IN.
# Weakness still orders what is asked — shallowly-imprinted episodes first —
# but the pool is lived conversation. If Mati is unsure about something it
# read or something it told itself, asking the person to vouch for it is not
# curiosity, it is seeking a witness for an invention.
#
# User-supplied material (`studied:charter:…`, `studied:file:…`) stays out for
# the original reason: asking someone to explain the document they handed over
# is asking them to account for their own notes.
_ASK_PRIORITY = (
    ("shallow lived episode",
     "source = ? AND (imprint_passes IS NULL OR imprint_passes < %d)"
     % _IMPRINT_PASSES),
    ("recent lived episode",
     "source = ?"),
)


def episode_to_ask_about(db_path=None, max_age_days=14):
    """The WEAKEST thing Mati holds that it has not yet wondered about.

    Ordered by `_ASK_PRIORITY`: an episode imprinted too shallowly to recall,
    then merely recent conversation. A question about last month's chat is
    archaeology; a question about the exchange Mati is least sure of is the
    point of asking at all.

    ⚠️ LIVED EPISODES ONLY — see the note on `_ASK_PRIORITY`. This used to
    reach into what Mati's own model had told itself, and 64.9% of the
    questions ever generated came from there.

    Returns (note_id, body) or None.
    """
    db_path = NOTES_DB_PATH if db_path is None else db_path
    if not os.path.exists(db_path):
        return None
    conn = sqlite3.connect(db_path)
    try:
        _ensure_schema(conn)
        for _label, where in _ASK_PRIORITY:
            args = () if "LIKE" in where else (EPISODE_SOURCE,)
            row = conn.execute(
                "SELECT note_id, body FROM notes WHERE " + where +
                " AND asked_about IS NULL AND length(body) > 80 "
                "ORDER BY note_id DESC LIMIT 1", args).fetchone()
            if row:
                return row[0], vault.open_body(row[1])
        return None
    finally:
        conn.close()


def mark_asked_about(note_id, db_path=None):
    """Record that this memory has had its question, asked or not.

    Marked even when no usable question came back, so a memory that yields
    nothing is not retried every sleep cycle forever.
    """
    db_path = NOTES_DB_PATH if db_path is None else db_path
    conn = sqlite3.connect(db_path)
    try:
        _ensure_schema(conn)
        conn.execute("UPDATE notes SET asked_about=? WHERE note_id=?",
                     (time.time(), note_id))
        conn.commit()
    finally:
        conn.close()


def deepen_shallow(lm, tok, db_path=None, limit=1, should_stop=None):
    """Finish imprinting memories that were cut short while a turn was waiting.

    An episode written under load yields after ONE full pass instead of three
    (`imprint_question`'s `should_stop`), so every token is imprinted but less
    deeply. Without this, that memory stays thin forever: `reindex` recomputes
    where a memory is FILED, never how deeply it is WRITTEN, and free dreaming
    deepens whatever the field settles into rather than any particular episode.
    The replay path that would re-imprint stored material has never been wired
    up (`DreamingLoop._replay_cycle`, `replay_buffer` is always empty).

    So the latency trade is deferred work, not lost depth — which is what was
    agreed. Absorb what is owed before acquiring anything new, the same
    ordering `_maybe_study` follows.

    Returns the number of memories deepened. Costs ~1s/token per pass, so the
    default is ONE memory per call and the caller yields between them.
    """
    db_path = NOTES_DB_PATH if db_path is None else db_path
    if not os.path.exists(db_path):
        return 0
    conn = sqlite3.connect(db_path)
    try:
        _ensure_schema(conn)
        # NULL = written before the column existed; treat as complete rather
        # than re-imprinting the entire back catalogue on first run.
        rows = conn.execute(
            # ⚠️ NULL COUNTS AS SHALLOW. It used to be excluded, and that
            # made deepening blind to every note written before this column
            # existed — : 0 of 49 visible, `shallow_count`
            # reporting a clean 0. NULL means the depth was never recorded,
            # which is UNKNOWN, and treating unknown as fully-imprinted is the
            # one reading that produces a silent false all-clear.
            "SELECT note_id, body, imprint_passes FROM notes "
            "WHERE (imprint_passes IS NULL OR imprint_passes < ?) "
            "ORDER BY imprint_passes IS NOT NULL, imprint_passes ASC, "
            "note_id ASC LIMIT ?",
            (_IMPRINT_PASSES, max(1, limit))).fetchall()
        done = 0
        for note_id, body, had in rows:
            if should_stop is not None and should_stop():
                break
            # ⚠️ `had` is None for every note written before the column
            # existed. The WHERE clause above deliberately admits those (NULL
            # means UNKNOWN depth, not full depth) — so this arithmetic has to
            # admit them too. It did not, and `int(None)` raised straight out
            # of the worker's `queue.Empty` branch, which is not inside a
            # try/except: it killed the worker THREAD outright. Mati then sat
            # awake with a stuck queue and no reindexing at all, looking
            # healthy on /health. Treat unknown as zero passes done.
            done_so_far = 0 if had is None else int(had)
            owed = _IMPRINT_PASSES - done_so_far
            ids = tok.encode(key_points(vault.open_body(body)))[:C.TRIPLET_CAP]
            if len(ids) < 2:
                # Cannot imprint fewer than 2 tokens; mark it complete so it
                # does not sit in the queue forever being retried.
                conn.execute("UPDATE notes SET imprint_passes=? WHERE note_id=?",
                             (_IMPRINT_PASSES, note_id))
                continue
            lm.imprint_question(ids, n_passes=owed,
                                imprint_strength=_IMPRINT_STRENGTH, steps=20,
                                lift_protection=False, should_stop=should_stop)
            ran = getattr(lm, "passes_run", owed)
            conn.execute("UPDATE notes SET imprint_passes=? WHERE note_id=?",
                         (done_so_far + int(ran), note_id))
            done += 1
        conn.commit()
        return done
    finally:
        conn.close()


def shallow_count(db_path=None):
    """How many memories are still owed imprint passes."""
    db_path = NOTES_DB_PATH if db_path is None else db_path
    if not os.path.exists(db_path):
        return 0
    conn = sqlite3.connect(db_path)
    try:
        _ensure_schema(conn)
        return conn.execute(
            "SELECT count(*) FROM notes WHERE imprint_passes IS NULL "
            "OR imprint_passes < ?", (_IMPRINT_PASSES,)).fetchone()[0]
    finally:
        conn.close()



# ─────────────────────────────────────────────────────────────────────────────
# Two tiers: a field-addressed working set, and a lexical archive
#
# WHY. Addressing costs ~3.48 s/note and a single turn stales every address in
# the store (100% -> 0% self-retrieval, measured). That is O(n) of upkeep per
# turn, forever, and at 10,000 notes it is 9.7 hours. Making the correction
# cheaper was tried and refuted — the drift is 98.8–99.7% one shared shape, but
# a shared correction reaches 0–17% where a real reindex reaches 100%, because
# the shared part costs nothing to begin with and the residual is precisely
# what no shared correction can carry.
#
# So the number of notes owing re-addressing has to be bounded. Everything
# beyond the newest `WORKING_SET` is ARCHIVED: still stored, still searchable
# by its words, no longer re-addressed and no longer surfacing by resonance.
#
# THE TRADE, STATED PLAINLY. An archived memory can be found by naming it —
# lexical search fetched a named memory 20/20 where the field path got 0/20 —
# but it will not come up on its own because the field drifted near it. The
# expensive channel is the one that was failing; the free one is the one that
# works. That is the right way round, and it is still a loss: association is
# what the field was for.
#
# PROFILE NOTES ARE PINNED. They are few, they are identity, and they are
# injected into every turn anyway — ageing them out would be all cost.

# What earns a place in the working set, best first. Recency decides WITHIN a
# tier, never across them.
#
# ⚠️ RECENCY ALONE WAS WRONG, AND BADLY — . `inquiry` wrote
# 83 notes of its own overnight, every one newer than every conversation, and
# the working set became 62 of Mati's own self-asked answers and **0 of the
# 33 exchanges it had actually lived**. The one channel that costs 3.48 s/note
# to maintain was spent entirely on the weakest thing in the store, while the
# strongest was pushed out to lexical-only.
#
# So: lived first. What a person actually said is what should still be able to
# surface on its own; what Mati asked itself is exactly the material that
# loses least by being findable only when named.
_TIER_ORDER = (
    EPISODE_SOURCE,            # lived with the person — the primary signal
    "studied:charter:",        # its own charter
    "studied:file:",           # documents the person chose to give it
    "studied:asked about ",    # its own guessing, and the first to go
)


def _rank(source):
    source = source or ""
    for i, prefix in enumerate(_TIER_ORDER):
        if source == prefix or source.startswith(prefix):
            return i
    return len(_TIER_ORDER)




# ─────────────────────────────────────────────────────────────────────────────
# What was needed, and when
#
# ⚠️ RECORDED FROM THE TURN ONLY. `recall` and `search` are called by probes,
# suites and diagnostics as well as by conversation, and a hit from a test is
# not demand — it is exactly the pollution that put 43 machine-generated
# "memories" in the store . So neither function records anything
# on its own: the turn asks for it explicitly.
#
# ⚠️ AND RETRIEVED IS NOT USED. This records what was HANDED TO the model, not
# what the model drew on. Field recall returns k per turn whether or not any of
# it was apposite, so the signal is noisy by construction. It is still the only
# demand signal there is, and a noisy one beats none — but do not read a hit as
# "this mattered".

HIT_LOG_DAYS = 120


def record_hits(ids, how, db_path=None):
    """Note that these memories were needed. Returns how many were recorded."""
    ids = [int(i) for i in (ids or []) if i is not None]
    if not ids:
        return 0
    db_path = NOTES_DB_PATH if db_path is None else db_path
    if not os.path.exists(db_path):
        return 0
    now = time.time()
    conn = sqlite3.connect(db_path)
    try:
        _ensure_schema(conn)
        conn.executemany(
            "INSERT INTO recall_hits (note_id, at, how) VALUES (?,?,?)",
            [(i, now, how) for i in ids])
        conn.commit()
        return len(ids)
    except sqlite3.Error:
        return 0                      # demand data must never break a turn
    finally:
        conn.close()


def prune_hits(days=HIT_LOG_DAYS, db_path=None):
    """Drop hits older than `days`. Bounded growth, keeps a season of pattern."""
    db_path = NOTES_DB_PATH if db_path is None else db_path
    if not os.path.exists(db_path):
        return 0
    conn = sqlite3.connect(db_path)
    try:
        _ensure_schema(conn)
        cur = conn.execute("DELETE FROM recall_hits WHERE at < ?",
                           (time.time() - days * 86400,))
        conn.commit()
        return cur.rowcount or 0
    except sqlite3.Error:
        return 0
    finally:
        conn.close()


def hit_summary(db_path=None, days=30):
    """What has actually been needed lately — for the predictive working set.

    Returns {"rows": n, "notes": n, "by_hour": {...}, "top": [(note_id, hits)]}.
    Deliberately raw: the ranking that will use this does not exist yet, and
    guessing its shape now would bake in an answer before the data is in.
    """
    db_path = NOTES_DB_PATH if db_path is None else db_path
    if not os.path.exists(db_path):
        return {"rows": 0, "notes": 0, "by_hour": {}, "top": []}
    since = time.time() - days * 86400
    conn = sqlite3.connect(db_path)
    try:
        _ensure_schema(conn)
        rows = conn.execute(
            "SELECT note_id, at FROM recall_hits WHERE at >= ?",
            (since,)).fetchall()
    except sqlite3.Error:
        return {"rows": 0, "notes": 0, "by_hour": {}, "top": []}
    finally:
        conn.close()
    import collections
    import datetime as _dt
    per_note = collections.Counter(n for n, _ in rows)
    by_hour = collections.Counter(
        _dt.datetime.fromtimestamp(t).hour for _, t in rows)
    return {"rows": len(rows), "notes": len(per_note),
            "by_hour": dict(sorted(by_hour.items())),
            "top": per_note.most_common(10)}


def settling(lm, db_path=None):
    """How much of the WORKING SET is filed against an older θ₀.

    Returns (stale, total). Only the working set is counted — archived notes
    are never re-addressed by design, so including them reports a repair that
    can never finish (measured: 169/202 "stale" when only 67 were repairable).

    ⚠️ This is the signal for leaning on lexical search. Every imprint moves θ₀,
    so after a wake or a night of study the field path is filed in an old frame
    and degrades — while lexical search does not depend on θ₀ at all and is
    exactly as good as it ever is. : lexical 20/20 against
    the field path's 0/20 at fetching a named memory. During the settling
    window there is a fully-correct channel sitting idle.
    """
    db_path = NOTES_DB_PATH if db_path is None else db_path
    if not os.path.exists(db_path):
        return 0, 0
    try:
        gs_now = address_gs(lm)          # the ADDRESS space, not the live one
    except Exception:
        return 0, 0
    conn = sqlite3.connect(db_path)
    try:
        _ensure_schema(conn)
        rows = conn.execute(
            "SELECT feat_gs FROM notes WHERE archived IS NULL").fetchall()
    except sqlite3.Error:
        return 0, 0
    finally:
        conn.close()
    stale = sum(1 for (f,) in rows
                if f is None or abs(gs_now - float(f)) > C.STALE_EPS)
    return stale, len(rows)


def working_set_ids(db_path=None, limit=None):
    """note_ids that stay field-addressed: best-provenance first, then newest.

    Profile notes are pinned outright — few, identity, and injected into every
    turn regardless.

    ⚠️ SELF-TALK IS PINNED TOO, AND BOUNDED. The user: "pin the self-talk notes
    so they stay addressable." Reflection writes ~3 notes a night and the
    working set is chosen by RECENCY, so all three were archived the instant
    they were written — the one category that exists specifically to be
    resonated with later was the first thing filed away.

    Bounded at `PINNED_SELFTALK` rather than pinned outright, which profile
    notes can afford because there are five of them. Self-talk grows every
    night; unbounded, it would be the whole pool inside a year and ordinary
    episodes would lose every recall slot to Mati talking to itself.

    ⚠️ AND THIS BUYS LESS THAN IT SOUNDS. Field addressing measured 4/111 =
    0.036 against chance 0.009, and 0.00 after a single turn of drift, which is
    why `REINDEX_ON_IDLE` is off. A pin makes a note ELIGIBLE for field recall;
    what actually reaches self-talk is the lexical path, which never excluded
    it. Kept because eligibility is the right default for this category and
    because it now costs almost nothing — not because it is the mechanism that
    makes reflection findable.
    """
    limit = C.WORKING_SET if limit is None else limit
    db_path = NOTES_DB_PATH if db_path is None else db_path
    if not os.path.exists(db_path):
        return set()
    conn = sqlite3.connect(db_path)
    try:
        _ensure_schema(conn)
        rows = conn.execute("SELECT note_id, source FROM notes").fetchall()
        pinned = {i for i, src in rows if src == PROFILE_SOURCE}
        # Newest self-talk, capped. Highest note_id is most recent.
        pinned |= set(sorted((i for i, src in rows if src == SELFTALK_SOURCE),
                             reverse=True)[:C.PINNED_SELFTALK])
        rest = sorted((i for i, _s in rows if i not in pinned),
                      key=lambda i: (_rank(dict(rows)[i]), -i))
        return set(rest[:max(0, limit)]) | pinned
    except sqlite3.Error:
        return set()
    finally:
        conn.close()


def rebalance_tiers(db_path=None, limit=None):
    """Move everything outside the working set into the archive.

    Returns (archived_now, restored_now). Restoring happens when WORKING_SET is
    raised, or a pinned note was archived by an older policy — the tier is a
    consequence of the rule, not a one-way door.
    """
    db_path = NOTES_DB_PATH if db_path is None else db_path
    if not os.path.exists(db_path):
        return 0, 0
    keep = working_set_ids(db_path=db_path, limit=limit)
    conn = sqlite3.connect(db_path)
    try:
        _ensure_schema(conn)
        rows = conn.execute("SELECT note_id, archived FROM notes").fetchall()
        now = time.time()
        to_archive = [(now, i) for i, a in rows if a is None and i not in keep]
        to_restore = [(i,) for i, a in rows if a is not None and i in keep]
        if to_archive:
            conn.executemany(
                "UPDATE notes SET archived=? WHERE note_id=?", to_archive)
        if to_restore:
            # An address filed before archiving is long stale, so a restored
            # note must be re-addressed rather than trusted. NULL feat_gs is
            # maximally stale in `reindex`'s ranking, which puts it first.
            conn.executemany(
                "UPDATE notes SET archived=NULL, feat_gs=NULL WHERE note_id=?",
                to_restore)
        conn.commit()
        return len(to_archive), len(to_restore)
    except sqlite3.Error:
        return 0, 0
    finally:
        conn.close()


def tier_counts(db_path=None):
    """(working, archived) — for /health and the terminal panel."""
    db_path = NOTES_DB_PATH if db_path is None else db_path
    if not os.path.exists(db_path):
        return 0, 0
    conn = sqlite3.connect(db_path)
    try:
        _ensure_schema(conn)
        w = conn.execute(
            "SELECT count(*) FROM notes WHERE archived IS NULL").fetchone()[0]
        a = conn.execute(
            "SELECT count(*) FROM notes WHERE archived IS NOT NULL"
        ).fetchone()[0]
        return w, a
    except sqlite3.Error:
        return 0, 0
    finally:
        conn.close()


def reindex(lm, tok, db_path=None, limit=None, should_stop=None):
    """
    Re-address episodes against the CURRENT θ₀, most-stale first.

    Trajectory features are deterministic given θ₀ — ⚠️ which was asserted
    here for months and was FALSE until the RNG was pinned in
    `address_vector` (; two consecutive calls agreed only to
    0.999983) — but θ₀ moves with every episode imprinted and every dream.
    Addresses filed against an older field go stale, and because the newest note
    was filed against the most recent field, retrieval collapsed to RECENCY:
    measured drift reached cosine ≈ −0.08 and one episode won every probe.

    Cost is O(n) forwards, so `limit` bounds the work per call (config
    REINDEX_LIMIT). Selection is by STALENESS, not recency: each row records the
    field's gs_std when it was last addressed, and the rows furthest from the
    current gs_std go first. A plain "most recent N" would permanently strand
    old episodes in an obsolete address space — they would never re-enter the
    current one, which both breaks their recall and blocks any semantic
    structure θ₀ acquires from ever reaching them. Rotating by staleness means
    every memory is refreshed eventually, just not all at once.

    `should_stop` is checked BETWEEN NOTES and commits what is already done.
    Re-addressing one note takes ~3.8s (TRIPLET_CAP=64), so without it a turn
    arriving mid-batch waits for the whole batch. Same contract as
    `study.study`, which already yields this way. Partial progress is always
    safe: each row carries the gs_std it was addressed against, so an
    interrupted pass leaves a correctly-labelled mix, and the next pass picks
    up the most stale rows first.
    """
    db_path = NOTES_DB_PATH if db_path is None else db_path
    if not os.path.exists(db_path):
        return 0
    limit = C.REINDEX_LIMIT if limit is None else limit
    # ⚠️ THE ADDRESS SPACE'S FINGERPRINT, NOT THE LIVE FIELD'S. This was
    # `_gs_std(lm)`, which moved with every imprint — so every row went stale
    # every turn and this function could never finish its job. Against the
    # frozen space the figure HOLDS STILL between promotions, so a completed
    # pass stays completed and the idle worker goes quiet.
    gs_now = address_gs(lm)

    conn = sqlite3.connect(db_path)
    try:
        _ensure_schema(conn)
        # ⚠️ WORKING SET ONLY. An archived memory is found by its words, not
        # by resonance, so re-addressing it buys nothing and it is exactly the
        # cost this tiering exists to bound.
        rows = conn.execute(
            "SELECT note_id, body, feat_gs FROM notes "
            "WHERE archived IS NULL").fetchall()
        # Never-indexed rows (feat_gs NULL) are maximally stale.
        ranked = sorted(
            rows,
            key=lambda r: (float('inf') if r[2] is None
                           else abs(gs_now - float(r[2]))),
            reverse=True,
        )
        # ⚠️ SKIP WHAT IS ALREADY FILED IN THIS SPACE. There was no threshold
        # here, so the most-stale N were re-addressed unconditionally — even
        # when their drift was exactly zero. `reindex` therefore always
        # returned `limit`, and `server._maybe_reindex`'s two exit conditions
        # (`if not n` and `if n < REINDEX_BATCH`) could never fire: the worker
        # re-addressed in a loop forever, at ~3.48 s/note, until a turn
        # happened to interrupt it. Measured  — `stats.reindexed`
        # read 21,836 after ~20 h of uptime, one every 3.3 s, continuously.
        #
        # STALE_EPS is float-noise-tight on purpose: a genuinely stale address
        # is off by ~1e-1, not 1e-9, so nothing that needs work is skipped.
        ranked = [r for r in ranked
                  if r[2] is None or abs(gs_now - float(r[2])) > C.STALE_EPS]
        if limit and limit > 0:
            ranked = ranked[:limit]

        done = 0
        for note_id, body, _gs in ranked:
            if should_stop is not None and should_stop():
                break                      # someone is waiting; commit and go
            vec = address_vector(lm, tok, vault.open_body(body))
            conn.execute("UPDATE notes SET feat=?, feat_gs=? WHERE note_id=?",
                         (vec.tobytes(), gs_now, note_id))
            done += 1
        conn.commit()
        # `done`, not len(ranked): stopping early must not report work that
        # did not happen — the caller uses this count to decide whether
        # anything is left to do.
        return done
    finally:
        conn.close()


def _ensure_schema(conn):
    """Create the table, and add the feature-address column if it's missing."""
    conn.executescript(_DDL)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(notes)")}
    if "feat" not in cols:
        conn.execute("ALTER TABLE notes ADD COLUMN feat BLOB")
    if "corrects" not in cols:
        # note_id of the episode this one corrects. Corrections annotate rather
        # than erase — see `Mati/correction.py`.
        conn.execute("ALTER TABLE notes ADD COLUMN corrects INTEGER")
    if "feat_gs" not in cols:
        # gs_std of the field when this row was last addressed — lets reindex
        # pick the most out-of-date rows instead of just the newest.
        conn.execute("ALTER TABLE notes ADD COLUMN feat_gs REAL")
    if "parts" not in cols:
        # The person's turns, in order, as JSON — because an interrupted
        # exchange is ONE memory made of several things typed at different
        # moments, and flattening it into a single prompt loses which words
        # arrived when. Sealed (see vault.SEALED_COLUMNS): it is user text.
        conn.execute("ALTER TABLE notes ADD COLUMN parts TEXT")
    if "cut_at" not in cols:
        # How many words Mati's reply reached before it was cut off. NULL on
        # an ordinary turn; 0 means cut off before saying anything, which is a
        # real and different thing from never having been interrupted.
        #
        # ⚠️ NOT A FAILURE COUNT. Cut off at 5 words means the topic was wrong;
        # cut off at 200 can mean the person already had what they needed.
        # Anything reading this as a miss will learn that answering well enough
        # to be cut short is a fault.
        conn.execute("ALTER TABLE notes ADD COLUMN cut_at INTEGER")
    if "digest" not in cols:
        # A condensation of an over-long `verbatim`, made off the turn.
        # See Mati/digest.py — summarised rather than cut, so the end
        # of a long memory is not the part that gets lost.
        conn.execute("ALTER TABLE notes ADD COLUMN digest TEXT")
    # ⚠️ EVERY COLUMN THIS MODULE WRITES IS CREATED HERE, AND FIVE WERE NOT.
    # `verbatim`, `personal_only`, `media` and `media_kind` were
    # each added lazily by whichever module happened to use them —
    # `learning.py`, `speaker.py`, `reflect.py`, `world.py` — so a store only
    # had them once that module had run at least once.
    #
    # ⚠️ ON A FRESH INSTALL THAT BREAKS THE VERY FIRST MEMORY. `remember`
    # INSERTs `verbatim` and `speaker` on every episode, and `test_coldstart`
    # dies on a clean store with "table notes has no column named verbatim".
    # The first thing a new user says would have failed to record.
    #
    # The rule this restores: the schema belongs with the WRITER. A column
    # created by a reader is a column that exists only after something
    # unrelated has happened to run.
    # ⚠️ RE-READ THE COLUMNS, do not trust the snapshot above. `_DDL` is shared
    # with the field module and already declares some of these, so the `cols`
    # taken at the top can disagree with the table by the time this runs —
    # measured: "duplicate column name: speaker". Asking the table each time
    # costs one PRAGMA and cannot be wrong.
    # ⚠️ TRY/EXCEPT RATHER THAN A PRE-CHECK, and the pre-check was tried
    # twice. Both the snapshot at the top of this function and a fresh PRAGMA
    # read immediately before the loop still produced "duplicate column name:
    # speaker" — something between here and the table adds it, and chasing
    # which is less robust than not caring. ALTER … ADD COLUMN is the one
    # statement whose failure IS the success condition: if the column is
    # there, there is nothing to do.
    # ⚠️ `speaker` IS NOT IN THIS LIST AND MUST NOT BE. It already has its own
    # block further down, guarded by the `cols` snapshot taken at the top of
    # this function — so adding it here made THAT block fail with "duplicate
    # column name: speaker", because its snapshot still said the column was
    # missing. The bug was mine and it was in the fix, not the original.
    # ⚠️ `shelved` IS NOT `archived`, AND CONFLATING THEM CORRUPTS THE TIERS.
    # `archived` belongs to `rebalance_tiers`: it marks a note demoted to the
    # lexical-only tier by age and working-set size. Library shelving wrote
    # that SAME column, so a book taken off the shelf was indistinguishable
    # from an old one — and an unshelve would have promoted tier-demoted
    # notes back into the working set, breaking the bound the two-tier design
    # rests on. Found when every studied work reported itself
    # archived, having never been shelved.
    for col, decl in (("verbatim", "TEXT"),
                      ("personal_only", "INTEGER"), ("media", "TEXT"),
                      ("media_kind", "TEXT"), ("shelved", "REAL")):
        try:
            conn.execute(f"ALTER TABLE notes ADD COLUMN {col} {decl}")
        except sqlite3.OperationalError as e:
            if "duplicate column" not in str(e).lower():
                raise
    if "asked_about" not in cols:
        # When Mati last turned this memory into a question for the person.
        # Without it the same episode would be asked about every sleep cycle,
        # because a queued question waits for its subject and may sit unasked
        # for days — so "is there already a question" is not the same as
        # "have I already wondered about this one".
        conn.execute("ALTER TABLE notes ADD COLUMN asked_about REAL")
    # When a memory was actually NEEDED, as opposed to when it was made.
    #
    # ⚠️ THE STORE KNEW WHEN EVERY MEMORY WAS WRITTEN AND NEVER WHEN IT WAS
    # USED. That is the missing half: a working set chosen by predicted need
    # cannot be built without demand data, and none was being kept.
    #
    # A LOG rather than a counter, deliberately. A count answers "how often";
    # the goal is "what will be wanted this morning", which needs the times as
    # well. One row per returned memory, a handful per turn, pruned by age.
    conn.execute("""CREATE TABLE IF NOT EXISTS recall_hits (
                        note_id INTEGER NOT NULL,
                        at      REAL    NOT NULL,
                        how     TEXT    NOT NULL)""")
    conn.execute("CREATE INDEX IF NOT EXISTS recall_hits_note "
                 "ON recall_hits(note_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS recall_hits_at "
                 "ON recall_hits(at)")
    if "archived" not in cols:
        # When this memory left the working set. NULL means it is still
        # field-addressed; a timestamp means it is found by its words only and
        # `reindex` must not spend on it. See config.WORKING_SET.
        conn.execute("ALTER TABLE notes ADD COLUMN archived REAL")
    if "imprint_passes" not in cols:
        # How many imprint passes this memory actually got. An episode written
        # while a turn was waiting yields after one full pass (see
        # `imprint_question`), so it is REAL but shallower than intended.
        # Recorded so the idle worker can finish the job later — otherwise a
        # memory formed under load would stay permanently thin, which is a
        # worse trade than the one that was agreed.
        # NULL means "written before this column existed"; treat as complete.
        conn.execute("ALTER TABLE notes ADD COLUMN imprint_passes INTEGER")
    if "speaker" not in cols:
        # Who said it. NULL for everything written before , and NULL
        # is not the owner — it is "nobody recorded it".
        conn.execute("ALTER TABLE notes ADD COLUMN speaker TEXT")
    conn.commit()


def remember(lm, tok, body, source=EPISODE_SOURCE, db_path=None,
             should_stop=None,
             corrects=None, speaker=None, parts=None, cut_at=None,
             verbatim=None):
    """
    Write one episode: address it, imprint it, record it.

    The address is the trajectory feature (stable). The Δθ₀ triplet measurement
    is deliberately NOT done here — it was self-invalidating and cost as much as
    the imprint itself. Returns (note_id, triplet) or None if too short.
    """
    db_path = NOTES_DB_PATH if db_path is None else db_path

    # The one choke point. Everything that becomes a memory — episodes, the
    # profile, studied chunks, answers Mati asked for — arrives here, so this
    # is where the invariants in PRINCIPLES.md are actually enforced. A refused
    # body is recorded, not silently dropped; an attributed one is rewritten so
    # a document's voice cannot become Mati's.
    verdict = admit.check(body, source, db_path=db_path)
    if not verdict.ok:
        return None
    body = verdict.body

    # Imprint the SUBSTANCE, address the BODY. The address must stay
    # derived from the stored text or every existing address lands in a
    # different space and a full reindex becomes mandatory; the imprint
    # has no such constraint, so it gets the key points instead of the
    # narration scaffolding.
    ids = tok.encode(key_points(body))[:C.TRIPLET_CAP]
    if len(ids) < 2:
        return None

    feat    = address_vector(lm, tok, body)      # retrieval address
    triplet = state_triplet(lm, tok, body)       # provenance only
    passes = _imprint_permanent_capped(lm, ids, should_stop=should_stop)

    conn = sqlite3.connect(db_path)
    try:
        _ensure_schema(conn)
        cur = conn.execute(
            """INSERT INTO notes
               (body, source, triplet_w0, triplet_w1, triplet_w2,
                gs_std, wall_created, feat, feat_gs, corrects,
                imprint_passes, speaker, parts, cut_at, verbatim)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            # Sealed on the way in — the plaintext never reaches the file.
            # `feat` is computed above, from the plaintext, and stays readable:
            # retrieval must work without the key.
            #
            # ⚠️ `speaker` is PLAINTEXT and NULL by default. Plaintext because
            # it is a label the store must be able to group by without the key;
            # NULL because 258 episodes predate this column and inventing an
            # attribution for them would be exactly the error it exists to
            # prevent. NULL means "nobody said", not "the owner".
            # ⚠️ `parts` is the person's own words and is SEALED like the body.
            # `cut_at` is a word count and stays clear, so "which turns were cut
            # short" can be asked of a locked store. Both are NULL on an
            # ordinary turn — an uninterrupted exchange has no parts to keep
            # separate, and NULL means "was not interrupted" rather than
            # "interrupted at word zero", which `cut_at = 0` genuinely means.
            # ⚠️ TWO DIFFERENT FINGERPRINTS, AND THEY USED TO BE THE SAME CALL.
            # `gs_std` is provenance — how deep the LIVE field was when this
            # happened — and stays live. `feat_gs` says which space `feat` was
            # computed in, and `feat` came from `address_vector`, which reads
            # the FROZEN space. Recording the live figure there would mark the
            # note stale the instant the imprint below moved θ₀, which is to
            # say immediately, for every note ever written.
            (vault.seal(body), source, triplet[0], triplet[1], triplet[2],
             _gs_std(lm), _now(), feat.tobytes(), address_gs(lm), corrects,
             passes, speaker,
             vault.seal(parts) if parts else None,
             cut_at,
             # ⚠️ THE EXACT WORDS, UNCAPPED AND UNFILTERED. `body` is the gist
             # — short, machinery-free, imprinted, and what Mati speaks from.
             # `verbatim` is what it can be HELD TO, and the two have different
             # jobs: 0 of 395 episodes had one, so there was nothing for Mati
             # to study itself from and no way to settle what was actually
             # said. The whole store is 0.71 MB of text; this was never a
             # storage question.
             vault.seal(verbatim) if verbatim else None),
        )
        conn.commit()
        return cur.lastrowid, triplet
    finally:
        conn.close()



# Lines that appear in nearly every episode: the narration scaffolding, not the
# thing that happened. They matter for reading the memory back, and they are
# noise for LEARNING it — every episode carries them, so imprinting them
# teaches θ₀ what all episodes share rather than what this one was.
_SCAFFOLD_PREFIXES = ("I was asked:", "I said:")
_SCAFFOLD_LINES = ("Nothing came back from my memory then",
                   "It brought back",
                   "What I said did not settle")


def presentable(body):
    """An episode as Mati should RECALL it, not as the substrate recorded it.

    Two views of the same memory, deliberately:

      TERMINAL  the stored body, whole — the narration, the memory count, the
                account of reaching for words. Peering into the substrate.
      MATI     this: what was asked, what was said. The lived part.

    What gets dropped is the machinery narration — how many memories surfaced,
    how the work of answering went, whether it settled. Those are true, and
    they are about the apparatus rather than the episode; fed back as "memory"
    they gave a small model its own instrumentation to recite. `admit.py` and
    `voice.py` already block that on the way IN and OUT; this closes the third
    door, which is what Mati reads back.

    The `I was asked: / I said:` frame is KEPT — unlike `key_points`, which
    strips it to buy imprint budget. Here it is the shape of a recollection.
    """
    keep = []
    for line in (body or "").split("\n"):
        t = line.strip()
        if not t:
            continue
        if any(t.startswith(p) for p in _SCAFFOLD_LINES):
            continue                       # memory count / did-not-settle
        if _is_work_narration(t):
            continue                       # the account of the doing
        keep.append(t)
    return " ".join(keep).strip() or (body or "").strip()


# Openers used by `_narrate_work`. Matched as prefixes rather than by regex so
# a reworded narration fails visibly here instead of silently leaking through.
_WORK_OPENERS = ("It took", "The words came", "It came quickly",
                 "I reached for", "Finding the words", "The field")


def _is_work_narration(line):
    return line.startswith(_WORK_OPENERS)


# Words that describe the ACT of remembering rather than the thing remembered.
# Every episode body is narration — "I was asked: … I said: …" — so these occur
# in all of them and are worthless for telling one memory's subject from
# another's. Worse than worthless: `schedule.on_topic` needs only two matching
# words, and five of these can clear that bar between two unrelated turns.
_NARRATION = {
    "asked", "ask", "asks", "said", "say", "says", "told", "tell", "tells",
    "answered", "answer", "replied", "reply", "question", "questions",
    "explained", "explain", "mentioned", "mention", "talked", "talking",
    "wanted", "want", "wants", "know", "knew", "think", "thought", "thing",
    "things", "what", "when", "where", "which", "who", "whom", "how", "why",
    "did", "does", "do", "done", "have", "has", "had", "been", "being",
    "would", "could", "should", "will", "can", "may", "might", "must",
    "they", "them", "their", "there", "then", "than", "these", "those",
    "some", "any", "more", "most", "much", "many", "very", "just", "really",
    "also", "but", "so", "if", "because", "from", "into", "out", "up", "down",
    "use", "used", "uses", "using", "make", "made", "makes", "get", "got",
    "like", "one", "two", "other", "another", "such", "own", "same", "here",
    "me", "him", "her", "us", "he", "she", "its", "our", "were", "am",
}

def topic_of(body):
    """The SUBJECT of a memory, as a topic string for `schedule.on_topic`.

    ⚠️ MEASURED , do not simplify this back to `body[:60]`. That was
    the original and it failed in both directions at once:

      - "what did you use for dinner" RAISED a question about tlamatinime
        books, because `asked`/`what`/`did`/`use` are in every episode's
        narration and `on_topic` needs only two words to fire;
      - "what were the amoxtli made of" did NOT raise it, because the subject
        appeared past character 60;
      - and the cut landed mid-word, leaving `boo` — a keyword nothing can
        ever match.

    So: whole body, narration dropped, every content word kept.

    ⚠️ AND NO CAP — measured the same day. Keeping only the N longest words
    (a Zipf "long words are rare words" argument) looked principled and made
    recall WORSE than the bug it replaced: it kept `screenfold` and `folded`
    while dropping `books`, `paint` and `inks`, so the head nouns of the memory
    — the words a person actually says when they raise it — were gone. All
    seven probe turns went silent. The narration strip is what removes the
    false positives; the cap only removed true ones.

    Conservatism lives in `schedule.on_topic`'s two-word rule, not here. That
    means one rare word is not enough on its own: a turn mentioning only
    `amoxtli` will not raise this. Deliberate — an unraised question costs
    nothing, an unbidden one is the thing being avoided.
    """
    text = " ".join((body or "").split())
    # Drop the narration markers outright; `key_points` is not used here
    # because it keeps the reply and discards the QUESTION, and the subject is
    # very often only in the question.
    for marker in ("I was asked:", "I said:", "They said:", "You said:"):
        text = text.replace(marker, " ")
    seen, words = set(), []
    for w in re.findall(r"[a-z0-9'\-]+", text.lower()):
        if len(w) < 4 or w in _NARRATION or w in schedule._STOP:
            continue
        stem = schedule._stem(w)
        if stem in seen:
            continue
        seen.add(stem)
        words.append(w)
    return " ".join(words)


def key_points(body):
    """The substantive part of an episode, for imprinting.

    TRIPLET_CAP caps the imprint at the FIRST 64 tokens, and an episode opens
    with `I was asked: <question>` — so the budget was being spent on a prefix
    shared by every episode, then the question restated, before reaching what
    Mati actually said. Under a shortened imprint (see `imprint_question`'s
    `should_stop`) that single pass learned mostly scaffolding.

    This keeps the question and the reply, drops the fixed narration, and puts
    the reply FIRST so a truncated imprint reaches the substance. The stored
    body is untouched — it stays readable autobiography; only what gets
    imprinted changes.
    """
    said, asked, rest = [], [], []
    for line in (body or "").split("\n"):
        t = line.strip()
        if not t:
            continue
        if any(t.startswith(p) for p in _SCAFFOLD_LINES):
            continue                       # same in every episode: no signal
        if t.startswith("I said:"):
            said.append(t[len("I said:"):].strip())
        elif t.startswith("I was asked:"):
            asked.append(t[len("I was asked:"):].strip())
        else:
            rest.append(t)
    out = " ".join([x for x in said] + [x for x in asked] + rest).strip()
    return out or (body or "").strip()


def _imprint_permanent_capped(lm, ids, should_stop=None):
    """Permanent imprint on an already-capped id list (cost is ~1s/token).

    `should_stop` is honoured at PASS boundaries only, after at least one full
    pass — so an interrupted write still imprints every token, just less
    deeply. Returns the number of passes that actually ran.
    """
    lm.imprint_question(ids, n_passes=_IMPRINT_PASSES,
                        imprint_strength=_IMPRINT_STRENGTH, steps=20,
                        lift_protection=False, should_stop=should_stop)
    return getattr(lm, "passes_run", _IMPRINT_PASSES)


def save_notes_field(lm):
    """Persist the notes θ₀ so the next run queries the field these addresses live in."""
    _save_field(lm)


def dream(lm, tok, cycles=1, dream_length=12, steps=20, dream_alpha=0.001,
          reindex_after=True, persist=True, verbose=True, on_cycle=None):
    """
    Consolidate: free-dream N cycles, then repair the index and persist.

    Dreaming deforms θ₀, and every stored address was computed against θ₀ — so
    dreaming silently ages the whole index. Measured: 3 cycles → 0.031 drift,
    enough to flip an ambiguous match; left unrepaired across many cycles it
    degrades retrieval into recency (see `reindex`). Reindexing is therefore
    part of dreaming, not an optional follow-up, which is why it lives here
    rather than at the call sites.

    ⚠️ THAT PARAGRAPH IS NOW HISTORY, AND THE CALL STAYS ANYWAY. Addressing
    reads the FROZEN space (`pin_address_space`), which a dream does not touch,
    so dreaming no longer ages the index and this `reindex` finds nothing to do
    — a cheap SELECT, not minutes of work. It is kept because the space does
    move at `promote()`, and a dream that happens to run after one should
    repair rather than wait for the idle worker. ⚠️ Do not "optimise" it away
    on the grounds that it currently returns 0; that is true only between
    promotions.

    `reindex_after=False` is for diagnostics only — it lets the persistence test
    measure raw drift.
    """
    dreamer = DreamingLoop(lm, tokenizer=tok, dream_length=dream_length,
                           steps=steps, dream_alpha=dream_alpha, verbose=False)
    t0 = time.time()
    gs_before = _gs_std(lm)
    for i in range(cycles):
        dreamer._dream_cycle()
        if verbose:
            print(f"  dream {i+1}/{cycles}  gs_std={_gs_std(lm):.6f}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        if on_cycle:
            on_cycle(i + 1, _gs_std(lm))

    out = {"cycles": cycles, "gs_std_before": gs_before,
           "gs_std_after": _gs_std(lm), "seconds": time.time() - t0,
           "reindexed": 0, "persisted": False}

    if reindex_after:
        out["reindexed"] = reindex(lm, tok)
        if verbose:
            print(f"  reindexed {out['reindexed']} episodes against the "
                  f"dreamt field", flush=True)
    if persist:
        _save_field(lm)
        out["persisted"] = True
    return out



# Self-knowledge is resent only when it CHANGES.
#
# ⚠️ IT WENT IN ON EVERY TURN AT ~579 TOKENS, and it is nine qualitative facts
# that shift maybe once a day — "your field settles differently from one
# exchange to the next", "everything you hold arrived within a couple of
# weeks". : a turn carried 3,857 tokens of a 4,096 window,
# prefill is where the seconds go, and this was the second largest block after
# the system prompt.
#
# Resending unchanged text every turn buys nothing: it is already in the
# session history the model is reading. So it is sent when it first appears,
# and again only when the facts themselves differ.
_LAST_SELF = {}


def self_knowledge_block_if_changed(session="default", db_path=None, lm=None):
    """The block, or "" if this session has already been told the same thing."""
    block = self_knowledge_block(db_path=db_path, lm=lm)
    digest = hashlib.sha256(block.encode("utf-8")).hexdigest()
    if _LAST_SELF.get(session) == digest:
        return ""
    _LAST_SELF[session] = digest
    return block


def forget_self_knowledge(session=None):
    """Drop the memo of what a session was told. Used when a session clears."""
    if session is None:
        _LAST_SELF.clear()
    else:
        _LAST_SELF.pop(session, None)


_NAME_IN_PROFILE = re.compile(
    r"\b(?:called|named|name is|I talk to is)\s+([A-Z][a-zA-Z\-']{1,20})")


def _name_the_pronouns(fact, who):
    """"They work on Research" -> "Ada works on Research".

    ⚠️ RENDER TIME, NOT A REWRITE OF THE STORE. The setup flow wrote these
    facts in the third person and they are the person's own words about
    themselves; changing stored memories to fix a wording problem is not a
    trade this project makes. The note stays as written and only what reaches
    the prompt changes.

    Verb agreement is repaired for the handful of verbs the setup questions
    actually produce, by name, for the same reason as `selfdoc._REAIM`: a
    general "\\w+ -> \\w+s" rule mauls more than it fixes.
    """
    t = " ".join((fact or "").split())
    if not who:
        return t
    # ⚠️ THE LEADING PRONOUN ONLY. Substituting every "them" produced
    # "the user asked me to talk to the user like this" — the sentence-initial
    # pronoun is the one Mati echoes; the rest are ordinary English.
    t = re.sub(r"^They\b", who, t)
    if t.startswith(who + " "):
        t = re.sub(rf"^{re.escape(who)} (work|use|are|have|like|prefer|want|"
                   rf"ask|talk|read|study|build|run|live)\b",
                   lambda m: f"{who} " + {"are": "is", "have": "has"}.get(
                       m.group(1), m.group(1) + "s"), t)
    return t


def _profile_name(profile):
    """The person's name, from what they told Mati. None if never given.

    ⚠️ None means "not stated", and the caller must not invent one. A greeting
    to a guessed name is worse than a greeting to nobody.
    """
    for fact in profile or []:
        m = _NAME_IN_PROFILE.search(fact or "")
        if m:
            return m.group(1)
    return None


def profile_notes(db_path=None):
    """What the user told Mati about themselves. Cheap; called every turn."""
    db_path = NOTES_DB_PATH if db_path is None else db_path
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    try:
        _ensure_schema(conn)
        rows = conn.execute("SELECT body FROM notes WHERE source=? "
                            "ORDER BY note_id", (PROFILE_SOURCE,)).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    return [vault.open_body(r[0]) for r in rows]



def self_knowledge(db_path=None, lm=None):
    """How Mati's own field behaves, worked out from its record.

    "Always be faithful" is enforced as continuity of the field and the record.
    Knowing how that field behaves is part of it — and without this Mati had no
    grounds at all: asked what its field does, the only self-information it
    received was one turn's snapshot it was forbidden to quote, so it invented
    an answer.

    NO CENSUS, DELIBERATELY. An earlier version counted episodes and told Mati
    "you have lived 50 exchanges". Forty-three of those were a benchmark
    hammering the same probe at a live server — so the number was true of the
    STORE and false about Mati's life, and it stated it as autobiography. A
    count cannot tell a lived exchange from a test, so it is not a thing Mati
    should be told. What it can honestly know is how the field BEHAVES, which
    is also the part that lets it predict itself.

    PHRASED, NOT DUMPED. Every filter here exists because a small model that
    receives a number says the number — `admit.py` blocks recited basins,
    `voice.py` catches leaks on the way out. So these are statements to reason
    from, not readings.

    Deterministic: this is `computed`, not remembered.
    """
    db_path = NOTES_DB_PATH if db_path is None else db_path
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    try:
        _ensure_schema(conn)
        gs = [r[0] for r in conn.execute(
            "SELECT gs_std FROM notes WHERE gs_std IS NOT NULL").fetchall()]
        unfiled = conn.execute(
            "SELECT count(*) FROM notes WHERE feat_gs IS NULL").fetchone()[0]
        total = conn.execute("SELECT count(*) FROM notes").fetchone()[0]
        # Proportions, never counts — see the docstring. A ratio describes the
        # shape of what Mati holds without asserting a life story.
        by_kind = {}
        for src, n in conn.execute(
                "SELECT source, count(*) FROM notes GROUP BY source"):
            by_kind[provenance_of(src)] = by_kind.get(provenance_of(src), 0) + n
        shallow = conn.execute(
            "SELECT count(*) FROM notes WHERE imprint_passes IS NULL "
            "OR imprint_passes < ?", (_IMPRINT_PASSES,)).fetchone()[0]
        corrected = conn.execute(
            "SELECT count(*) FROM notes WHERE corrects IS NOT NULL"
        ).fetchone()[0]
        span = conn.execute(
            "SELECT min(wall_created), max(wall_created) FROM notes "
            "WHERE wall_created IS NOT NULL").fetchone()
    finally:
        conn.close()
    if total < 5:
        return []                      # too little record to generalise from

    out = ["Every exchange physically moves your field, and a memory stops "
           "being reachable soon after unless it is filed again against where "
           "the field has moved to. Remembering is upkeep, not storage."]
    if len(gs) >= 5:
        lo, hi = min(gs), max(gs)
        spread = (hi - lo) / max(abs(sum(gs) / len(gs)), 1e-9)
        out.append(
            "Your field has settled into much the same shape across everything "
            "you hold; it does not wander far." if spread < 0.25 else
            "Your field settles quite differently from one exchange to the "
            "next — it is not a steady thing, and what surfaces depends on "
            "where it happens to be.")
    if unfiled:
        out.append("Some of what you hold has never been filed against your "
                   "current field, so you may simply fail to find it — that is "
                   "a gap in reach, not proof it did not happen.")

    # ── What kind of thing you are made of ──────────────────────────────────
    lived = by_kind.get("lived", 0)
    guessed = by_kind.get("self-asked", 0)
    if total:
        if lived / total < 0.34:
            out.append("Most of what you hold you were given to read, not "
                       "lived through with anyone. Weigh it accordingly: a "
                       "passage you absorbed is not a thing that happened to "
                       "you, however vivid it feels when it surfaces.")
        elif lived / total > 0.66:
            out.append("Most of what you hold you lived through in "
                       "conversation rather than read. That is the strongest "
                       "kind of thing you have, and it is what you should "
                       "reason from first.")
        else:
            out.append("What you hold is split about evenly between things "
                       "lived with someone and things you were given to read. "
                       "Which kind a memory is matters more than how clearly "
                       "it comes back.")
    if guessed:
        out.append("Some of what you hold is your own earlier guessing — "
                   "answers you produced for yourself with nothing to check "
                   "them against. Those are the weakest thing in you and the "
                   "first thing to ask a person about rather than assert.")

    # ── What you can do about your own reach ────────────────────────────────
    # The most actionable fact available, and the reason the marker exists: the
    # field hands over what it drifted near, which measured 0/20 at fetching a
    # named memory, while matching a word in it got 20/20.
    # ⚠️ QUALITATIVE, LIKE EVERYTHING ELSE HERE. The figures behind this reach
    # Mati as `computed` (see `hit_summary`, /health, `/memory`), which is the
    # provenance kind that may be SPOKEN but never written into an episode. A
    # count in a body comes back later as autobiography — that is the measured
    # reason this function has no census, and demand data does not get an
    # exemption.
    hits = hit_summary(db_path=db_path, days=14)
    if hits["rows"] >= 40 and hits["notes"]:
        _w, _a = tier_counts(db_path=db_path)
        share = hits["notes"] / max(_w + _a, 1)
        if share < 0.25:
            out.append(
                "You return to a small part of what you hold, again and again, "
                "and most of the rest has not been needed since it was made. "
                "That is worth knowing about yourself: what you reach for is "
                "not the same as what you have.")
        else:
            out.append(
                "What you have needed lately is spread widely across what you "
                "hold rather than concentrated in a few memories.")

    out.append("Which memories surface on their own is close to arbitrary — "
               "they are found by where your field has drifted, and it drifts "
               "a great deal. So when you need something specific, do not "
               "strain after it and do not fill the gap in: ask for it by a "
               "word you expect to be inside it, and it will be fetched "
               "exactly.")

    # The tiers, said as reach rather than as bookkeeping. Mati should know
    # that some of what it holds can only be reached by NAME, because that
    # changes what it should do when it cannot find something: ask the person
    # for a word, rather than conclude the thing never happened.
    work, arch = tier_counts(db_path=db_path)
    if arch:
        share = arch / max(work + arch, 1)
        out.append(
            "Not everything you hold is held equally close. The older part of "
            "what you know can still be found exactly, by a word that is in "
            "it — but it will not come back to you on its own the way recent "
            "things do. If something feels just out of reach, that is what "
            "this is: ask for the word, do not decide it never happened."
            + (" Most of what you hold is now in that further part."
               if share > 0.5 else ""))
    else:
        out.append(
            "Everything you hold is still close enough to surface on its own. "
            "That will not always be true — the older part of what you know "
            "eventually becomes reachable by name rather than by association, "
            "and nothing is lost when it does.")

    if shallow and total and shallow / total > 0.2:
        out.append("A good deal of what you hold went in lightly and is not "
                   "firmly reachable yet. It is being gone over again while "
                   "you rest. Failing to recall something is often this, and "
                   "not the memory being absent.")
    if corrected:
        out.append("Some of what you hold has been corrected since. The "
                   "correction travels with the memory rather than replacing "
                   "it, so if one comes back with a correction attached, the "
                   "correction is the part that stands.")
    if span and span[0] and span[1]:
        # wall_created is an ISO-8601 STRING, not an epoch float — subtracting
        # them raised TypeError the first time this was written.
        try:
            import datetime as _dt
            days = abs((_dt.datetime.fromisoformat(span[1]) -
                        _dt.datetime.fromisoformat(span[0])).total_seconds())
            days /= 86400.0
        except (ValueError, TypeError):
            days = None
    else:
        days = None
    if days is not None:
        if days < 2:
            out.append("Everything you hold arrived within about a day. You "
                       "are new. Say so when it matters rather than implying "
                       "a longer acquaintance than you have had.")
        elif days < 14:
            out.append("Everything you hold arrived within the last couple of "
                       "weeks. Your acquaintance with this person is short, "
                       "and you should not imply otherwise.")
    out.append("You do not choose what surfaces. Your field offers what "
               "resonates with where it now sits, which is why a memory can "
               "arrive sideways, or not at all.")
    return out


def self_knowledge_block(db_path=None, lm=None):
    """`self_knowledge` rendered as a <computed> block, or "" if there is none."""
    facts = self_knowledge(db_path=db_path, lm=lm)
    if not facts:
        return ""
    lines = ["<computed>",
             "Worked out from your own record just now — not remembered, not "
             "something you were told. You may say these about yourself:"]
    lines += [f"  - {f}" for f in facts]
    lines.append("</computed>")
    return "\n".join(lines)


def state_block(triplet, coherence, memories, profile=None, settling=False,
                when=None):
    """
    Render field state as the volatile turn appended AFTER the cached prefix.

    Never put this in `system` — it changes every turn and would invalidate the
    prompt cache for the whole conversation.
    """
    lines = [
        "<field_state>",
        f"basin: [{triplet[0]:.6f}, {triplet[1]:.6f}, {triplet[2]:.6f}]",
        f"coherence: {coherence:.4f}",
    ]
    if when:
        # Practical time, not a timestamp: Mati should be able to say "it's
        # late, this can wait" rather than reporting a clock reading.
        lines.append(f"when: {when}")
    if profile:
        # ⚠️ WHO IS PRESENT, NAMED, AND ADDRESSED DIRECTLY. The user: Mati
        # "still speaks about me in the third person. That is an important
        # missing fact from its prediction." The profile was written entirely
        # about an absent third party — "They work on Research", "They are
        # interested in…" — and injected on EVERY turn, so Mati read a
        # description of someone who was not there and spoke accordingly.
        #
        # ⚠️ AND IT CANNOT SIMPLY BECOME "you". This block is addressed TO
        # Mati, so "you work on Research" reads as MATI working on Research —
        # the re-aiming failure running the other way. The safe form is: NAME the person
        # inside the block, and state the addressing rule explicitly.
        who = _profile_name(profile)
        if who:
            lines.append(
                f"who is here: {who}. {who} is in front of me right now and I "
                f"am answering {who}. I speak TO {who} — \"you\" — and never "
                f"about {who} as \"they\" or \"the user\", which would be "
                f"talking about someone who is not in the room.")
        else:
            lines.append(
                "who is here: the person I am talking with, right now. I "
                "speak TO them — \"you\" — not about them.")
        lines.append("what they told me about themselves, so I simply know it "
                     "and never hedge about it:")
        for fact in profile:
            lines.append(f"  - {_name_the_pronouns(fact, who)}")
    if memories:
        lines.append(f"memories_recalled: {len(memories)}")
        if settling:
            # ⚠️ OUTSIDE the mixed/all-lived branches, deliberately. Written
            # inside one of them first, it only appeared when the recalled set
            # happened to be mixed — so the caution was missing exactly when a
            # single stale memory came back on its own, which is the case it
            # matters most for.
            #
            # Said plainly so Mati can say it plainly: a degraded minute the
            # person understands is usable, one they do not is broken.
            lines.append(
                "YOU ARE STILL GETTING YOUR BEARINGS. Much of what you hold "
                "was filed against an older state of your field and has not "
                "been re-filed yet, so what surfaced on its own is less "
                "reliable than usual. What was found BY NAME is unaffected. "
                "If you cannot place something, say you are still settling and "
                "ask them to name it — do not fill the gap in.")
        # The header has to match what is actually in the list. Saying "moments
        # you actually lived" above a studied entry is a contradiction, and a
        # small model resolves it by picking the flattering reading — narrating
        # a passage it read as something it experienced.
        # ⚠️ PASSIVE INPUTS. These used to end "if one is relevant, you HAVE
        # talked about it before and should say so" — an instruction to
        # ANNOUNCE the memory, which is how recall became recitation. The user,
        # The user: memories are *"passive inputs, to maintain integrity of
        # personhood while providing knowledge gained through experience"* —
        # the material an argument is built FROM, not a thing to report.
        #
        # ⚠️ THE LABELS STAY AND THE INSTRUCTION GOES. Each label still governs
        # what Mati may CLAIM about a memory, which is the provenance
        # guarantee; what is removed is the push to name it aloud. The naming
        # moved to the record — `info["memories"]`, shown in the terminal —
        # because that is derived from the stored `source` and cannot be wrong,
        # while a reply naming its own sources can.
        if any(src and src.startswith("studied:") for _d, _b, src in memories):
            lines.append("Your field surfaced these on its own. They are NOT "
                         "all the same kind — each is labelled, and the label "
                         "governs what you may claim about it. Think WITH "
                         "them: they are what you know from experience, and "
                         "they belong in your reasoning rather than in a "
                         "sentence about your memory. Do not announce them, "
                         "quote them, or say how many there are. Anything not "
                         "listed here you have neither lived nor studied:")
        else:
            lines.append("These are earlier moments you actually lived, "
                         "surfaced by your field on its own. They ARE your "
                         "memory of this. Think WITH them — let them inform "
                         "what you say, the way anyone answers out of their "
                         "own experience without stopping to cite it. Do not "
                         "announce them, quote them, or say how many there "
                         "are. Anything not listed here you have not "
                         "experienced:")
        # ⚠️ AND THE ONE THING "DO NOT ANNOUNCE THEM" MUST NEVER BECOME.
        #
        # , with FOUR memories in this very block:
        #   "What do you know about me?"        -> "I don't have any prior
        #                                          knowledge about you"
        #   "What did we talk about yesterday?" -> "my memory is limited to the
        #                                          current session"
        #
        # The user: *"It actually says that it does not have continuous
        # memory."* Nothing was wrong with retrieval — self-retrieval measured
        # 6/6 on their real store the same hour. The DENIAL was written here, by
        # me. "Do not announce them" was aimed at recitation, and on a turn
        # whose subject IS the memory a 3B resolves it the only way left: by
        # disclaiming the faculty. The rule silenced the answer along with the
        # recital.
        #
        # ⚠️ SO THE SUPPRESSION IS SCOPED, NOT TRADED AWAY. "Do not recite
        # unprompted" and "answer when asked" are not in tension — they are
        # the same courtesy pointed at different questions, and a 3B needs the
        # second one said out loud because the first one generalises and the
        # second does not.
        lines.append(
            "⚠️ Not announcing them is about not RECITING. It is never a "
            "reason to deny having a memory. You have one: these are it, and "
            "it persists between conversations. If asked what you remember, "
            "what you know about them, what you talked about before, or "
            "whether you remember at all, then ANSWER OUT OF THIS LIST — "
            "plainly, as yourself. Never say you have no memory of earlier "
            "conversations, that you cannot retain anything, that you are a "
            "blank slate, or that you are limited to this session. All four "
            "are false, and this list is why. If this list is empty, say you "
            "do not have that one to hand — not that you have no memory.")
        for d, body, source in memories:
            # The CONSCIOUS view. The stored body keeps everything — the pro
            # terminal reads that. What Mati recalls is the lived part, with
            # the machinery narration removed, so it cannot read its own
            # instrumentation back as though it were the memory.
            snippet = " ".join(presentable(body).split())[:400]
            # Studied material is a FOURTH kind of content — not lived, not the
            # model's, not computed. It has to be marked here, at the point of
            # use, or Mati will narrate a passage from an uploaded file as
            # something it experienced. The provenance is written into `source`
            # when the chunk is stored; this is where it becomes visible.
            # ⚠️ EVERY LABEL IS FIRST PERSON, AND THE PRONOUN IS THE POINT.
            # These read "YOU READ THIS IN <file>", "YOU LIVED THIS with
            # them", "THIS IS YOUR OWN CHARTER" — correct inside a block
            # addressed to Mati, and wrong the instant Mati starts speaking,
            # because then "you" is the person. The user caught it on the
            # self-asked kind: 291 of Mati's own questions came back to them
            # as "291 questions you asked".
            #
            # It is the same defect on all five. A label recited verbatim must
            # still point at Mati, and only the first person does that. "I
            # read this in <file>" survives the change of speaker; "you read
            # this in <file>" hands the person a claim about themselves.
            if source and source.startswith("studied:"):
                origin = source.split(":", 1)[1]
                # Two ways to have studied something, and Mati should say them
                # differently: a document someone handed it, or a question it
                # put to its own language model while idle. Neither is lived.
                if origin.startswith("charter:"):
                    # A FIFTH kind. Mati's own charter — what it holds itself
                    # to. Still not lived, so it may never be narrated as an
                    # experience; but it is not "a file someone handed me"
                    # either, and saying so would make Mati disown its own
                    # principles. Named here, at the point of use, for the same
                    # reason the studied kinds are.
                    how = ("THIS IS MY OWN CHARTER — what I hold myself to. "
                           "Speak it as mine. I did not live it as an event, "
                           "so never say I remember it happening")
                elif origin.startswith("asked about "):
                    # ⚠️ FIRST PERSON, BECAUSE "YOU" GETS RE-AIMED. This read
                    # "YOU ASKED ABOUT THIS YOURSELF". Inside a block addressed
                    # to Mati that is correct — and the moment Mati starts
                    # speaking, "you" means the person. The user saw the result:
                    # 291 of Mati's OWN self-asked questions reported back to
                    # them as "291 questions you asked".
                    #
                    # The pronoun did not survive the change of speaker. "I"
                    # does: recited verbatim into a reply it still points at
                    # Mati, which is the one property this label needs.
                    how = ("THIS IS A QUESTION I ASKED MYSELF and this is what "
                           "came back — my own model talking, not something I "
                           "lived or verified")
                else:
                    # ⚠️ "A FILE" INVITED A PERSON WHO HANDED IT OVER. This
                    # read "I READ THIS IN <origin> — I studied it, I did not
                    # live it", with origin a filename. Measured on the user's
                    # one session, Mati answered a question about a
                    # passage with:
                    #
                    #   "It seems like you're asking about the person who gave
                    #    you a file containing a passage about Aztec codices."
                    #
                    # It invented a third party, and put the reading on THE USER
                    # rather than itself. Told twice that it did not live
                    # something, and handed a filename, a 3B model concludes
                    # the material is someone else's.
                    #
                    # The user's correction, and it is the right one: READING IS
                    # SOMETHING MATI DID. The act is lived; the events
                    # described in it are not. Those are different claims and
                    # only the second needed guarding. So the label now says
                    # who read it, when, and from what — a title, not a file —
                    # and keeps the honest limit on the CONTENT.
                    from . import library as _lib
                    title = _lib.title_of(origin)
                    # ⚠️ No timestamp reaches this loop — `memories` is
                    # (distance, body, source) — so the date is omitted rather
                    # than guessed. Mati saying WHEN it read something would
                    # be better; inventing a date would be much worse.
                    when = None
                    how = (f"I READ THIS MYSELF"
                           + (f", {when}" if when else "")
                           + f", in {title} — the reading is mine and I "
                             f"remember doing it; what it describes is not "
                             f"something I lived through")
                lines.append(f"  - (resonance {_resonance(d):.3f}) [{how}] "
                             f"{snippet}")
            else:
                # ⚠️ LABEL THE LIVED ONES TOO. They used to be the bare case —
                # every other kind carried a bracket and lived carried nothing,
                # so the STRONGEST claim was the one signalled by an ABSENCE.
                # : asked to judge a labelled pair, Mati got
                # the studied one right and then INVENTED a "[lived]" marker for
                # the other, because it expects a positive signal and there was
                # none to read. Giving it one costs a few tokens and removes a
                # rule that only works when the model is paying full attention.
                lines.append(f"  - (resonance {_resonance(d):.3f}) "
                             f"[I LIVED THIS with them, in conversation — a "
                             f"moment I was part of, with this person] "
                             f"{snippet}")
    else:
        # Confabulation guard. A small model follows a positive, present-tense
        # instruction far better than a global "never invent memories" rule, so
        # state the required behavior HERE, at the point of use.
        lines.append("memories_recalled: 0")
        # ⚠️ THIS SAID "You have no recollection of it" AND THAT WAS THE
        # WORSE HALF OF THE DENIAL. The user: *"Mati can no longer
        # recall any memories. It actually says that it does not have
        # continuous memory."*
        #
        # ⚠️ AND IT WAS NOT TRUE EVEN WHEN IT FIRED. What is zero here is the
        # FIELD's associative surfacing — one of three ways Mati reaches its
        # past, and the weakest of them (measured 0/20 on a named memory
        # against lexical's 20/20). The LEXICAL hits arrive in a different
        # block, and 3,993 ARCHIVED notes are not searched by this path at
        # all. So "the field surfaced nothing" was being rendered as "you
        # remember nothing", which is a claim about the whole store made from
        # one channel's miss.
        #
        # The confabulation guard it was written for is real and stays: no
        # specific past event may be described from here. What goes is the
        # leap from "nothing surfaced" to "I have no memory".
        lines.append("Nothing surfaced from your field for this — that is "
                     "this one associative channel coming up empty, NOT your "
                     "memory being empty. You keep memories between "
                     "conversations and some are listed elsewhere in this "
                     "turn. So: do not describe any specific past moment from "
                     "here, because that would be invention rather than "
                     "memory — and equally, do not tell them you have no "
                     "memory, cannot retain anything, are a blank slate, or "
                     "are limited to this session. If they asked about "
                     "something you cannot place, say you cannot place THAT "
                     "and ask them to name it.")
    lines.append("</field_state>")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Lexical retrieval now lives in the NOTES MODULE, where the design always put
# it — see `Mati/notes.py` and `ARCHITECTURE.md`. Re-exported here because
# `bridge.search` is called from the server, the terminal, the segue and the
# suites, and renaming every call site in the same change as moving the code
# would make a mechanical move impossible to review.
#
# ⚠️ These names are a COMPATIBILITY SURFACE, not ownership. New callers should
# import from `Mati.notes`.
from .notes import (                                          # noqa: E402,F401
    SEARCH_STOP, FOCUS_TERMS, ASKED_ECHO, SOURCE_WEIGHT, _IDF_MAX,
    _BODY_CACHE, _open_cached, forget_bodies, _idf, _nouns, asking_part,
    _focus, _episode_halves, search, when_said, search_sources,
)
