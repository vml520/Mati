# Part of Mati's field — the phase-field substrate its memory is built on.
#
# ⚠️ THIS IS THE AUTHORITATIVE COPY. Nothing outside Mati imports these
# modules, so the field can change without anything else moving under it.
#
# ⚠️ KEEP THIS A COMMENT, NOT A DOCSTRING. A docstring here would sit
# before `from __future__`, which is a SyntaxError that `ast.parse`
# does not report.
"""
Mati's phase-field engine, v20.3
───────────────────────────────────────────────────────────────────────────────
M = R³ × S¹

θ(x,t) : R³ → S¹  phase field on mobile nodes in R³.

REGIME (required distinction — tft-corrected-derivations §2):
──────────────────────────────────────────────────────────────
This module implements the DISSIPATIVE FIRST-ORDER regime: gradient flow of the
TFT energy functional (Kuramoto-type settling). The conservative SECOND-ORDER
regime — the sine-Gordon equation (1/c²)∂²θ/∂t² − ∇²θ + (Λ/ℓ₀²)sin θ = 0 —
is the physics-paper dynamics. Same functional, two regimes. Do not conflate.

New in v20.3 — Dynamic field size from energy density:
────────────────────────────────────────────────────────
Nodes have positions x ∈ R³ and can migrate. Coupling is by spatial distance,
cut off at coherence length ℓ₀. Field size is not fixed — it emerges from the
energy density of the input via λ_eff:

    λ_eff = α · ⟨|∇θ|²⟩ · vol_element       (from cheat sheet §10)

λ_eff amplifies the gradient term in each step:
    dθ/dt = (1 + λ_eff) · ℓ₀² · ∇²θ

λ_eff is the discrete-simulation counterpart of the dimensionless coupling Λ in
the continuous action V(θ) = (Λ/ℓ₀²)(1 − cos θ). The kink width w = ℓ₀/√Λ
sets the spatial scale for phase gradient resolution; ξₙ per mode approximates
this at each scale. λ_eff ≠ Λ numerically — λ_eff is dynamic (input-driven),
Λ is a fixed coupling constant — but both modulate the same gradient term.

High-energy input → large ⟨|∇θ|²⟩ → large λ_eff → carriers propagate deeper.
The active field volume expands with input energy density.

Node migration (from v13):
    Nodes with high phase disagreement move toward input centroid.
    Coupling matrix recomputed after each migration.
    Field finds its own geometry in R³.

All v20.2 functionality preserved:
    Ground state θ₀, deform_ground_state, identify_sinks,
    encode_triplet, decode_triplet, measure_settlement,
    learn_patterns, test_recognition, print_triplet_space,
    print_settlement_report, save_state.
───────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import json, os

# ── First-class scales ────────────────────────────────────────────────────────
E_0   = 1.0
l_0   = 1.0
g_eff = E_0 / l_0
beta  = E_0 / l_0**2
SPACE_L = 6.0                          # physical extent of R³ domain
TAU_0   = 1.0                          # phase-time unit
E0_phys = 2 * np.pi / SPACE_L         # physical phase-energy scale
alpha   = 1.0 / (E0_phys * TAU_0)     # expansion coupling (from §10)
alpha_gs = g_eff / 200                 # ground state deformation rate

STICKINESS  = 0.05                     # velocity damping
DIST_EPS    = 1e-6
XI          = l_0 * 3.0               # spatial coupling cutoff = 3 ℓ₀
SIGMA       = l_0                      # Gaussian coupling width = ℓ₀

# ── Node ─────────────────────────────────────────────────────────────────────
class Node:
    """
    A point in R³ carrying a phase value on S¹.
    position: x ∈ R³
    theta:    θ ∈ S¹
    mass:     accumulated carrier inflow
    vtheta:   phase velocity (S¹ tangent)
    omega:    intrinsic frequency.
              Each node knows itself through its own natural rate of
              phase rotation on S¹.  Coupling drives harmony between
              nodes; omega gives each node an identity that persists
              even when decoupled.  Default 0 = no intrinsic drive.
    """
    def __init__(self, position, theta=None, omega=0.0):
        self.position = np.array(position, dtype=float)
        self.theta    = float(theta) if theta is not None else \
                        np.random.uniform(0, 2*np.pi)
        self.mass     = 0.0
        self.vtheta   = 0.0
        self.omega    = float(omega)   # intrinsic frequency

    def to_dict(self):
        return {
            'position': self.position.tolist(),
            'theta':    self.theta,
            'mass':     self.mass,
        }

    @staticmethod
    def from_dict(d):
        n = Node(d['position'], d['theta'])
        n.mass = d.get('mass', 0.0)
        return n

# ── Field ─────────────────────────────────────────────────────────────────────
class PhaseField:
    """
    Phase field θ(x): R³ → S¹ on mobile nodes.
    Coupling by spatial distance — no fixed lattice.
    λ_eff drives dynamic field expansion from input energy.

    Dynamics: dissipative first-order (gradient flow / Kuramoto).
    Not the conservative sine-Gordon EOM — see module docstring.
    """

    def __init__(self, nodes):
        self.nodes      = nodes
        self.N          = len(nodes)
        self.W          = None          # coupling matrix, computed on demand
        self.lambda_eff = 0.0
        self._build_coupling()

    def _build_coupling(self):
        """
        Coupling weight between nodes i and j:
            W[i,j] = exp(-dist²/2σ²)  if dist < ξ, else 0
        Row-normalized so total influence is uniform.
        """
        N   = self.N
        pos = np.array([n.position for n in self.nodes])
        W   = np.zeros((N, N))
        for i in range(N):
            dists = np.linalg.norm(pos - pos[i], axis=1)
            mask  = (dists < XI) & (dists > DIST_EPS)
            W[i, mask] = np.exp(-dists[mask]**2 / (2 * SIGMA**2))
            row_sum = W[i].sum() + 1e-8
            W[i] /= row_sum
        self.W = W

    def _thetas(self):
        return np.array([n.theta for n in self.nodes])

    def _positions(self):
        return np.array([n.position for n in self.nodes])

    def _masses(self):
        return np.array([n.mass for n in self.nodes])

    def _set_masses(self, m):
        for i, node in enumerate(self.nodes):
            node.mass = float(m[i])

    def _set_thetas(self, th):
        for i, node in enumerate(self.nodes):
            node.theta = float(th[i]) % (2*np.pi)

    # ── Gradient (phase difference weighted by coupling) ──────────────────────
    def _gradient(self):
        """
        ∇θ at each node: weighted sum of sin(θⱼ - θᵢ) over neighbors.
        sin() wraps correctly on S¹.
        """
        thetas = self._thetas()
        pos    = self._positions()
        grad   = np.zeros(self.N)
        for i in range(self.N):
            diffs   = np.sin(thetas - thetas[i])
            dists   = np.linalg.norm(pos - pos[i], axis=1) + DIST_EPS
            grad[i] = np.sum(self.W[i] * diffs / dists)
        return grad

    def _gradient_magnitude(self):
        """⟨|∇θ|²⟩ — mean squared gradient over all nodes."""
        return float(np.mean(self._gradient()**2))

    # ── λ_eff — dynamic expansion scalar ─────────────────────────────────────
    def _compute_lambda(self):
        """
        λ_eff = α · ⟨|∇θ|²⟩ · vol_element
        From cheat sheet §10: global phase gradient drives expansion.
        High-energy input → large λ_eff → field expands deeper.

        Relationship to the continuous action (tft-corrected-derivations §2):
            Continuous: V(θ) = (Λ/ℓ₀²)(1 − cos θ), Λ fixed dimensionless coupling
            Discrete:   λ_eff is dynamic — computed each step from ⟨|∇θ|²⟩
        Both modulate the same Laplacian term. λ_eff is not Λ numerically;
        it is the input-driven analogue that makes field volume input-responsive.
        Kink width in continuous theory: w = ℓ₀/√Λ — sets gradient resolution
        scale that each mode's ξₙ approximates.
        """
        gm         = self._gradient_magnitude()
        vol_element = SPACE_L**3 / self.N
        self.lambda_eff = alpha * gm * vol_element
        return self.lambda_eff

    # ── Carrier inflow (uphill phase flow, S¹-correct) ────────────────────────
    def _carrier_inflow(self):
        """
        Inflow at node i = g_eff · Σⱼ W[i,j] · max(sin(θⱼ - θᵢ), 0)
        Carriers flow uphill along ∇θ. Mass accumulates where they concentrate.
        """
        thetas  = self._thetas()
        inflow  = np.zeros(self.N)
        for i in range(self.N):
            diffs      = np.sin(thetas - thetas[i])
            inflow[i]  = g_eff * np.sum(self.W[i] * np.maximum(diffs, 0))
        return inflow

    # ── Single evolution step ──────────────────────────────────────────────────
    def step(self, dt=0.05, locked=None):
        """
        One step of dissipative first-order field dynamics (Kuramoto/gradient flow):
            dθ/dt = ω + (1 + λ_eff) · ℓ₀² · ∇θ
            dm/dt = carrier_inflow − β · m

        This is gradient flow of the TFT energy functional — NOT the conservative
        sine-Gordon EOM (1/c²)∂²θ/∂t² − ∇²θ + (Λ/ℓ₀²)sin θ = 0.
        Settling into an attractor IS the computation. See module docstring.

        ω (omega): each node's intrinsic frequency.
        Coupling harmonises nodes; omega preserves each node's identity.
        locked nodes are held fixed (context boundary).
        """
        thetas     = self._thetas()
        grad       = self._gradient()
        lam        = self._compute_lambda()
        inflow     = self._carrier_inflow()
        masses     = self._masses()

        for i, node in enumerate(self.nodes):
            if locked and i in locked:
                continue
            # Phase update — intrinsic frequency + coupling gradient
            node.vtheta += node.omega                          # intrinsic drive
            node.vtheta += (1.0 + lam) * l_0**2 * grad[i]
            node.vtheta *= (1.0 - STICKINESS)
            node.theta   = (node.theta + dt * node.vtheta) % (2*np.pi)

        # Mass update — all nodes including boundary
        new_mass = np.clip(masses + dt * (inflow - beta * masses), 0, None)
        self._set_masses(new_mass)

    # ── Node migration ─────────────────────────────────────────────────────────
    def migrate(self, boundary_nodes, target_thetas, error_magnitude, rate=0.05):
        """
        Nodes near the boundary with high phase disagreement migrate
        toward the input centroid. Field finds its own geometry in R³.
        After migration, coupling is recomputed.
        """
        if not boundary_nodes:
            return
        centroid = np.mean([self.nodes[i].position for i in boundary_nodes], axis=0)
        for i in boundary_nodes:
            node        = self.nodes[i]
            target      = target_thetas.get(i, node.theta)
            disagreement = abs(np.sin(target - node.theta))   # S¹-correct
            mob          = error_magnitude * disagreement * rate
            direction    = centroid - node.position
            norm         = np.linalg.norm(direction) + DIST_EPS
            node.position += mob * direction / norm
            node.position  = np.clip(node.position, -SPACE_L/2, SPACE_L/2)
        self._build_coupling()

    def evolve(self, steps, dt=0.05, locked=None, migrate_every=None,
               boundary_nodes=None, target_thetas=None, error_fn=None):
        """
        Evolve field for `steps` steps.
        If migrate_every set, call migrate() at that interval.
        """
        for s in range(steps):
            self.step(dt=dt, locked=locked)
            if (migrate_every and boundary_nodes and
                    target_thetas and s % migrate_every == 0):
                err = error_fn() if error_fn else 0.1
                self.migrate(boundary_nodes, target_thetas, err)

    # ── State access ──────────────────────────────────────────────────────────
    def get_state(self):
        return self._thetas().copy(), self._masses().copy()

    def set_state(self, thetas, masses):
        self._set_thetas(thetas)
        self._set_masses(masses)

    def field_energy(self):
        thetas = self._thetas()
        E = 0.0
        for i in range(self.N):
            E += np.sum(self.W[i] * (1 - np.cos(thetas - thetas[i])))
        return E / self.N

# ── Field construction ────────────────────────────────────────────────────────
rng = np.random.default_rng(42)

N_NODES       = 200        # total nodes in R³
N_BOUNDARY    = 64         # boundary nodes seeded on z=0 face (8×8)
BOUNDARY_IDS  = list(range(N_BOUNDARY))

def _boundary_positions():
    """8×8 grid of boundary nodes on z=0 plane, scaled to SPACE_L."""
    scale = SPACE_L / 8.0
    return [[x * scale - SPACE_L/2 + scale/2,
             y * scale - SPACE_L/2 + scale/2,
             0.0]
            for x in range(8) for y in range(8)]

BOUNDARY_POS_R3 = np.array(_boundary_positions())   # (64, 3)
BOUNDARY_XY     = BOUNDARY_POS_R3[:, :2]            # (64, 2) for pattern lookup

def make_field():
    """
    Construct fresh field: 64 boundary nodes on z=0, remainder random in R³.
    """
    nodes = []
    # Boundary nodes
    for pos in _boundary_positions():
        nodes.append(Node(pos))
    # Interior nodes — random in R³ volume
    for _ in range(N_NODES - N_BOUNDARY):
        pos = rng.uniform(-SPACE_L/2, SPACE_L/2, 3).tolist()
        nodes.append(Node(pos))
    return PhaseField(nodes)

# ── Patterns — spatial functions over R³ boundary ────────────────────────────
def pat_A(p): return (np.pi/2) * (((p[:,0] + SPACE_L/2) / (SPACE_L/8)).astype(int) % 2)
def pat_B(p): return (np.pi/2) * ((((p[:,0] + SPACE_L/2) / (SPACE_L/8)).astype(int) // 2) % 2)
def pat_C(p): return (np.pi/2) * ((((p[:,0] + SPACE_L/2) / (SPACE_L/8)).astype(int) // 4) % 2)
def pat_D(p): return (np.pi/2) * (((p[:,1] + SPACE_L/2) / (SPACE_L/8)).astype(int) % 2)
def pat_E(p): return (np.pi/2) * ((((p[:,0] + SPACE_L/2) / (SPACE_L/8)).astype(int) +
                                    ((p[:,1] + SPACE_L/2) / (SPACE_L/8)).astype(int)) % 2)

PATTERNS = {'A': pat_A, 'B': pat_B, 'C': pat_C, 'D': pat_D, 'E': pat_E}

def apply_boundary(field, pat_fn):
    """Set boundary node phases to spatial pattern. Returns locked set."""
    phases = pat_fn(BOUNDARY_POS_R3)
    for i, bid in enumerate(BOUNDARY_IDS):
        field.nodes[bid].theta = float(phases[i])
    return set(BOUNDARY_IDS)

# ── Ground state ──────────────────────────────────────────────────────────────
def compute_ground_state(steps=200, dt=0.05):
    """
    Relax fresh field with no boundary condition.
    Returns (thetas, masses) — the resting state θ₀.
    """
    print("=== Computing ground state θ₀ ===")
    field = make_field()
    field.evolve(steps=steps, dt=dt)
    th, m = field.get_state()
    print(f"  θ₀: mean={th.mean():.4f}  std={th.std():.4f}  λ_eff={field.lambda_eff:.6f}")
    print(f"  m₀: mean={m.mean():.4f}   std={m.std():.4f}")
    return th, m

def restore_ground_state(field, theta_0, mass_0):
    """Reset field to ground state θ₀."""
    field.set_state(theta_0.copy(), mass_0.copy())

# ── Ground state deformation — learning by field shaping ─────────────────────
def deform_ground_state(theta_0, mass_0, theta_res, mass_res):
    """
    θ₀ ← θ₀ + α_gs · sin(θ_res − θ₀)     S¹-correct
    m₀ ← m₀ + α_gs · (m_res − m₀)
    Boundary nodes not deformed — they belong to the input.
    """
    d_theta  = np.sin(theta_res - theta_0)
    th_new   = (theta_0 + alpha_gs * d_theta) % (2*np.pi)
    m_new    = mass_0 + alpha_gs * (mass_res - mass_0)
    # Protect boundary
    th_new[BOUNDARY_IDS] = theta_0[BOUNDARY_IDS]
    m_new[BOUNDARY_IDS]  = mass_0[BOUNDARY_IDS]
    return th_new, m_new

# ── Sink identification ───────────────────────────────────────────────────────
def identify_sinks(field, theta_0, mass_0, n_probes=30, steps=320, dt=0.05):
    """
    Drive field from θ₀ with random boundary conditions.
    Sinks = nodes of persistent high mass. 3 emerge from S¹ topology on R³.
    """
    print("\n=== Calibration: identifying natural sinks ===")
    acc = np.zeros(field.N)
    for _ in range(n_probes):
        restore_ground_state(field, theta_0, mass_0)
        for bid in BOUNDARY_IDS:
            field.nodes[bid].theta = float(rng.uniform(0, 2*np.pi))
        field.evolve(steps=steps, dt=dt)
        _, m = field.get_state()
        acc += m
    acc /= n_probes

    sink_idx = np.sort(np.argsort(acc)[-3:])
    contrast  = acc[sink_idx].std() / (acc[sink_idx].mean() + 1e-12)
    sink_pos  = [field.nodes[i].position.round(3).tolist() for i in sink_idx]
    print(f"  Sink positions in R³: {sink_pos}")
    print(f"  Sink mass values:     {acc[sink_idx].round(5)}")
    print(f"  Sink contrast:        {contrast:.6f}")
    print(f"  Global mass range:    [{acc.min():.5f}, {acc.max():.5f}]")
    print(f"  λ_eff at calibration: {field.lambda_eff:.6f}")
    return sink_idx, acc

# ── Triplet encoding / decoding ───────────────────────────────────────────────
def encode_triplet(mass, sink_idx):
    raw   = mass[sink_idx]
    total = raw.sum()
    return raw / total if total > 1e-10 else np.ones(3) / 3.0

def decode_triplet(triplet, library):
    distances = {lbl: np.linalg.norm(triplet - ref) for lbl, ref in library.items()}
    return min(distances, key=distances.get), distances

# ── Settlement measurement ────────────────────────────────────────────────────
def measure_settlement(field, theta_0, mass_0, pat_fn, sink_idx,
                        max_steps=600, dt=0.05, tol=1e-4, window=20):
    restore_ground_state(field, theta_0, mass_0)
    locked = apply_boundary(field, pat_fn)
    history = []
    for step in range(max_steps):
        field.step(dt=dt, locked=locked)
        _, m = field.get_state()
        history.append(encode_triplet(m, sink_idx))
        if step >= window:
            recent_var = np.std(history[-window:], axis=0).mean()
            if recent_var < tol:
                return step - window // 2, history[-1]
    return max_steps, history[-1]

# ── Learning ──────────────────────────────────────────────────────────────────
def learn_patterns(field, theta_0, mass_0, sink_idx,
                   steps=320, dt=0.05, n_presentations=3):
    print("\n=== Learning phase (persistent ground state) ===")
    print(f"  α_gs={alpha_gs:.6f}  presentations={n_presentations}\n")

    library            = {}
    settlement_history = {}
    th_0 = theta_0.copy()
    m_0  = mass_0.copy()

    for lbl, pat_fn in PATTERNS.items():
        settlements = []

        for p in range(n_presentations):
            # Measure settlement
            s_step, triplet = measure_settlement(field, th_0, m_0, pat_fn, sink_idx)
            settlements.append(s_step)

            # Full evolution for residual
            restore_ground_state(field, th_0, m_0)
            locked = apply_boundary(field, pat_fn)
            field.evolve(steps=steps, dt=dt, locked=locked)
            th_res, m_res = field.get_state()
            final_triplet = encode_triplet(m_res, sink_idx)

            # Deform ground state
            th_0, m_0 = deform_ground_state(th_0, m_0, th_res, m_res)

        library[lbl]            = final_triplet
        settlement_history[lbl] = settlements
        settle_str = " → ".join(str(s) for s in settlements)
        print(f"  {lbl}  triplet={final_triplet.round(4)}  "
              f"settlement: {settle_str}  λ_eff={field.lambda_eff:.4f}")

    lbls   = list(library.keys())
    dists  = [np.linalg.norm(library[lbls[i]] - library[lbls[j]])
              for i in range(len(lbls)) for j in range(i+1, len(lbls))]
    spread = np.array(list(library.values())).std()
    print(f"\n  Mean pairwise separation: {np.mean(dists):.6f}")
    print(f"  Triplet spread:           {spread:.6f}")
    return library, th_0, m_0, settlement_history

# ── Recognition test ──────────────────────────────────────────────────────────
def test_recognition(field, theta_0, mass_0, sink_idx, library,
                     steps=320, dt=0.05):
    print("\n=== Recognition test ===")
    results = {}
    for true_lbl, pat_fn in PATTERNS.items():
        restore_ground_state(field, theta_0, mass_0)
        locked = apply_boundary(field, pat_fn)
        field.evolve(steps=steps, dt=dt, locked=locked)
        _, m    = field.get_state()
        triplet = encode_triplet(m, sink_idx)
        pred, distances = decode_triplet(triplet, library)
        correct = pred == true_lbl
        results[true_lbl] = correct
        print(f"  {true_lbl} → {pred}  {'✓' if correct else '✗'}"
              f"  λ_eff={field.lambda_eff:.4f}")
        print(f"    triplet:   {triplet.round(5)}")
        print(f"    distances: { {k: round(v,5) for k,v in distances.items()} }")
    n = sum(results.values())
    print(f"\n  Result: {n}/5 correct")
    return results

# ── Reports ───────────────────────────────────────────────────────────────────
def print_triplet_space(library, field, sink_idx):
    print("\n=== Triplet space (trinary basis) ===")
    sink_pos = [field.nodes[i].position.round(3).tolist() for i in sink_idx]
    print(f"  Sinks R³: {sink_pos}")
    print(f"  {'Pat':<6} {'w_s0':>9} {'w_s1':>9} {'w_s2':>9}")
    print(f"  {'-'*36}")
    for lbl, t in sorted(library.items()):
        print(f"  {lbl:<6} {t[0]:>9.5f} {t[1]:>9.5f} {t[2]:>9.5f}")

def print_settlement_report(settlement_history):
    print("\n=== Attractor depth (steps to convergence) ===")
    n = max(len(v) for v in settlement_history.values())
    header = "  ".join(f"{'Pass '+str(i+1):>8}" for i in range(n))
    print(f"  {'Pat':<6} {header}  {'Δ':>6}")
    print(f"  {'-'*60}")
    for lbl, hist in sorted(settlement_history.items()):
        vals  = "  ".join(f"{s:>8}" for s in hist)
        delta = hist[0] - hist[-1]
        print(f"  {lbl:<6} {vals}  {delta:>+6}")

# ── Save / load ───────────────────────────────────────────────────────────────
SAVE_PATH = os.path.expanduser("~/mati_saves/field_v20_3.json")

def save_state(field, sink_idx, library, mass_profile, results,
               theta_0, settlement_history):
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    state = {
        "version": "20.3",
        "field": {
            "N": field.N, "N_boundary": N_BOUNDARY, "SPACE_L": SPACE_L,
            "E_0": E_0, "l_0": l_0, "g_eff": g_eff, "beta": beta,
            "alpha": alpha, "alpha_gs": alpha_gs,
            "XI": XI, "SIGMA": SIGMA,
            "dynamics":      "(1+lambda_eff)*l0^2*nabla2(theta) + carrier_inflow",
            "lambda_eff":    "alpha * <|grad_theta|^2> * vol_element",
            "ground_update": "theta_0 += alpha_gs * sin(theta_res - theta_0)",
        },
        "nodes":              [n.to_dict() for n in field.nodes],
        "sink_indices":       [int(i) for i in sink_idx],
        "sink_positions":     [field.nodes[i].position.tolist() for i in sink_idx],
        "triplet_library":    {k: v.tolist() for k, v in library.items()},
        "mass_profile":       mass_profile.tolist(),
        "theta_0_evolved":    theta_0.tolist(),
        "settlement_history": settlement_history,
        "test_results":       results
    }
    with open(SAVE_PATH, 'w') as f:
        json.dump(state, f, indent=2)
    print(f"\n  Saved → {SAVE_PATH}")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("╔══════════════════════════════════════╗")
    print("║     Mati phase-field engine v20.3   ║")
    print("║     θ(x): R³ → S¹  phase field      ║")
    print("║     Dynamic field — mobile nodes     ║")
    print("╚══════════════════════════════════════╝\n")
    print(f"  Nodes: {N_NODES}  Boundary: {N_BOUNDARY}  SPACE_L: {SPACE_L}")
    print(f"  E₀={E_0}  ℓ₀={l_0}  g_eff={g_eff}  β={beta}")
    print(f"  ξ={XI}  σ={SIGMA}  α={alpha:.4f}  α_gs={alpha_gs:.6f}\n")

    # 1. Ground state
    field   = make_field()
    theta_0, mass_0 = compute_ground_state(steps=200)

    # 2. Sinks
    sink_idx, mass_profile = identify_sinks(field, theta_0, mass_0,
                                            n_probes=30, steps=320)

    # 3. Learn
    library, theta_0_ev, mass_0_ev, settlement_history = \
        learn_patterns(field, theta_0, mass_0, sink_idx,
                       steps=320, n_presentations=3)

    # 4. Report
    print_triplet_space(library, field, sink_idx)
    print_settlement_report(settlement_history)

    # 5. Recognize
    results = test_recognition(field, theta_0_ev, mass_0_ev,
                               sink_idx, library, steps=320)

    # 6. Save
    save_state(field, sink_idx, library, mass_profile, results,
               theta_0_ev, settlement_history)

    print("\n  v20.3 complete.")
