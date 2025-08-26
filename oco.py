"""
FTRL vs FTL vs SMART vs empirical_g_SMART
- SMART's threshold g(T) is estimated from a near-minimax backup algorithm (FTRL),
  using adversarial-ish sequences, per the SMART framework.
- Comparator for regret is the best constant action in hindsight from a soft-margin SVM.
"""

from __future__ import annotations

import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Dict, Tuple
from functools import partial

# ==============================================================
# Utilities
# ==============================================================

def clip_unit(q):
    return np.clip(q, -1.0, 1.0)

def norm2(x: np.ndarray) -> float:
    return float(np.linalg.norm(x))

def project_to_ball(x: np.ndarray, R: float) -> np.ndarray:
    n = norm2(x)
    return x if n == 0.0 or n <= R else (R / n) * x

def normalized_hinge(q: float, y: float) -> float:
    # (1/2)*|q - y|, with y in {±1}, q ∈ [-1,1]
    return 0.5 * abs(q - y)

def regret_increment(q_pred: float, y_true: float, q_comp: float) -> float:
    return normalized_hinge(q_pred, y_true) - normalized_hinge(q_comp, y_true)

# ==============================================================
# Actions: FTL and FTRL
# ==============================================================

def action_ftl(theta: np.ndarray, R: float = 1.0) -> np.ndarray:
    n = norm2(theta)
    return np.zeros_like(theta) if n == 0.0 else -R * theta / n

def action_ftrl(theta: np.ndarray, t: int, R: float = 1.0, eta0: float = math.sqrt(2)) -> np.ndarray:
    """
    Gradient step then project (FTRL-style with L2 regularization proxy).
    η_t = eta0 / sqrt(t).
    """
    eta_t = eta0 / math.sqrt(max(1, t))
    x = -eta_t * theta
    return project_to_ball(x, R)

# ==============================================================
# Online simulation (FTL, FTRL), regret vs SVM comparator
# ==============================================================

def simulate_alg(z: np.ndarray,
                 y: np.ndarray,
                 u_comp: np.ndarray,
                 *,
                 alg: str = "FTRL",
                 R: float = 1.0,
                 eta0: float = math.sqrt(2)) -> float:
    T, d = z.shape
    theta = np.zeros(d)
    cum_reg = 0.0

    for t in range(1, T + 1):
        x_t = action_ftrl(theta, t, R, eta0) if alg.upper() == "FTRL" else action_ftl(theta, R)

        q    = clip_unit(float(z[t-1] @ x_t))
        q_uc = clip_unit(float(z[t-1] @ u_comp))
        cum_reg += regret_increment(q, float(y[t-1]), q_uc)

        # subgradient wrt x via q=z·x
        grad_q = 0.5 * np.sign(q - y[t-1])
        theta += grad_q * z[t-1]

    return cum_reg

# ==============================================================
# SMART (single switch) with proper reset of ALG_WC
# ==============================================================

def simulate_SMART_like(z: np.ndarray,
                        y: np.ndarray,
                        u_comp: np.ndarray,
                        theta_thresh: float,
                        *,
                        R: float = 1.0,
                        eta0: float = math.sqrt(2)) -> float:
    """
    Start with FTL. If the estimated FTL regret crosses θ, switch once to ALG_WC=FTRL.
    IMPORTANT: Reset FTRL's internal state at the switch (as in SMART analysis).
    """
    T, d = z.shape
    theta_ftl  = np.zeros(d)
    theta_ftrl = np.zeros(d)   # will be reset at switch
    switched   = False

    ftl_reg    = 0.0
    total_reg  = 0.0

    for t in range(1, T + 1):
        x_ftl  = action_ftl(theta_ftl, R)
        x_ftrl = action_ftrl(theta_ftrl, t, R, eta0)
        x_t    = x_ftl if not switched else x_ftrl

        q    = clip_unit(float(z[t-1] @ x_t))
        q_uc = clip_unit(float(z[t-1] @ u_comp))
        total_reg += regret_increment(q, float(y[t-1]), q_uc)

        # gradient at played action
        grad_q = 0.5 * np.sign(q - y[t-1])
        g_vec  = grad_q * z[t-1]

        # Always update the FTL "surrogate" state (to track Σ_FTL)
        theta_ftl += g_vec

        # Only update FTRL AFTER we've switched (reset semantics)
        if switched:
            theta_ftrl += g_vec

        # Track anytime FTL regret (estimator for SMART)
        if not switched:
            q_ftl   = clip_unit(float(z[t-1] @ x_ftl))
            ftl_reg += regret_increment(q_ftl, float(y[t-1]), q_uc)
            if ftl_reg >= theta_thresh:
                switched   = True
                theta_ftrl = np.zeros(d)  # reset when switching

    return total_reg

def simulate_SMART(z, y, u_comp, *, R=1.0, eta0=math.sqrt(2)) -> float:
    T = z.shape[0]
    return simulate_SMART_like(z, y, u_comp, theta_thresh=math.sqrt(2*T), R=R, eta0=eta0)

def simulate_empirical_g_SMART(z, y, u_comp, theta_emp, *, R=1.0, eta0=math.sqrt(2)) -> float:
    return simulate_SMART_like(z, y, u_comp, theta_thresh=theta_emp, R=R, eta0=eta0)

# ==============================================================
# Soft-margin SVM (Pegasos) for hindsight comparator
# ==============================================================

def pegasos_svm(X, y, *, lam, iters=None, batch_size=None, R=1.0, seed=0, epochs=100):
    N, d = X.shape
    batch_size = batch_size or min(100, N)
    if iters is None:
        iters = max(1, int(epochs * N / batch_size))  # ~epochs passes
    rng = np.random.default_rng(seed)
    w = np.zeros(d)

    for t in range(1, iters + 1):
        idx = rng.integers(0, N, size=batch_size)
        Xb, yb = X[idx], y[idx]
        margin = yb * (Xb @ w)
        viol   = margin < 1.0

        eta = 1.0 / (lam * t)
        if np.any(viol):
            w = (1 - eta * lam) * w + (eta / batch_size) * (Xb[viol].T @ yb[viol])
        else:
            w = (1 - eta * lam) * w

        w = project_to_ball(w, R)
    return w

def best_action_in_hindsight_svm(z: np.ndarray, y: np.ndarray, *, R: float = 1.0, C: float = 1.0, seed: int = 0) -> np.ndarray:
    N = len(y)
    lam = 1.0 / max(C * N, 1.0)
    return pegasos_svm(z, y, lam=lam, R=R, seed=seed)

# ==============================================================
# Streams (revamped 5-case suite)
# ==============================================================

def random_iid_sequence(T: int, d: int = 5, R: float = 1.0):
    z = np.random.randn(T, d)
    z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1.0) * R
    u = np.random.randn(d)
    mm = np.max(np.abs(z @ u))
    if mm > 1.0:
        u = u / mm
    y = np.sign(z @ u); y[y == 0] = 1
    return z, y.astype(float), u

def noisy_iid_sequence(T: int, d: int = 5, R: float = 1.0, p: float = 0.1):
    """Massart-like label noise on top of separable i.i.d."""
    z = np.random.randn(T, d)
    z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1.0) * R
    u = np.random.randn(d)
    mm = np.max(np.abs(z @ u))
    if mm > 1.0:
        u = u / mm
    y = np.sign(z @ u); y[y == 0] = 1
    flips = np.random.rand(T) < p
    y[flips] *= -1
    return z, y.astype(float), u

def flip_sequence(T: int, d: int = 5, R: float = 1.0):
    z = np.zeros((T, d)); z[:, 0] = R
    y = np.array([1.0 if t % 2 else -1.0 for t in range(1, T + 1)], dtype=float)
    u = np.zeros(d)
    return z, y, u

def orthogonal_hard_sequence(T: int, R: float = 1.0):
    z = np.zeros((T, T))
    for t in range(T):
        z[t, t] = R
    y = np.ones(T, dtype=float)
    u = np.ones(T) / math.sqrt(T)
    return z, y, u



# --- UPDATED CASE SET ---
CASES = {
    "Random i.i.d. (separable)":        random_iid_sequence,
    "Noisy i.i.d. (Massart 10%)":       partial(noisy_iid_sequence, p=0.10),
    "Noisy i.i.d. (Massart 40%)":       partial(noisy_iid_sequence, p=0.40),
    "Label flips (worst-case)":         flip_sequence,
    # "Orthogonal (hard)":                orthogonal_hard_sequence,
}

# Optional: case-specific averaging to reduce variance where needed
RUNS_BY_TITLE = {
    "Random i.i.d. (separable)":       30,
    "Noisy i.i.d. (Massart 10%)":      30,
    "Noisy i.i.d. (Massart 40%)":      30,
    "Label flips (worst-case)":         1,
    # "Orthogonal (hard)":                1,
}

# ==============================================================
# Empirical g(T) from the worst-case (near-minimax) backup alg.
# ==============================================================

def empirical_worst_case_thresholds(T_grid: np.ndarray,
                                    *,
                                    runs: int = 5,
                                    R: float = 1.0,
                                    C: float = 1.0,
                                    base_seed: int = 0) -> Dict[int, float]:
    """
    Approximate g(T) as the maximum regret of ALG_WC = FTRL across adverse families.
    Families: label flips, orthogonal hard. (Add more if desired.)
    """
    adv_families = [flip_sequence, orthogonal_hard_sequence]
    g_emp: Dict[int, float] = {}

    for T in T_grid:
        worst = 0.0
        for fam in adv_families:
            for r in range(runs):
                np.random.seed(base_seed + 1337 * (int(T) + r))
                z, y, _ = fam(int(T), R=R) if fam is orthogonal_hard_sequence else fam(int(T), R=R)
                u_star = best_action_in_hindsight_svm(z, y, R=R, C=C, seed=base_seed + 7919 * (int(T) + r))
                reg_ftrl = simulate_alg(z, y, u_star, alg="FTRL", R=R)
                worst = max(worst, reg_ftrl)
        g_emp[int(T)] = worst
    return g_emp

# ==============================================================
# Evaluation
# ==============================================================

def evaluate_stream(seq_fn: Callable[..., Tuple[np.ndarray, np.ndarray, np.ndarray]],
                    T_grid: np.ndarray,
                    g_emp: Dict[int, float],
                    *,
                    runs: int = 1,
                    R: float = 1.0,
                    C: float = 1.0,
                    base_seed: int = 0):
    r_ftrl, r_ftl, r_smart, r_emp = [], [], [], []
    for T in T_grid:
        ftrl = ftl = smart = emp = 0.0
        for r in range(runs):
            np.random.seed(base_seed + 2025 * (int(T) + r))
            z, y, _ = seq_fn(int(T), R=R) if seq_fn is orthogonal_hard_sequence else seq_fn(int(T), R=R)
            u_star = best_action_in_hindsight_svm(z, y, R=R, C=C, seed=base_seed + 101 * (int(T) + r))
            ftrl  += simulate_alg(z, y, u_star, alg="FTRL", R=R)
            ftl   += simulate_alg(z, y, u_star, alg="FTL",  R=R)
            smart += simulate_SMART(z, y, u_star, R=R)
            emp   += simulate_empirical_g_SMART(z, y, u_star, g_emp[int(T)], R=R)
        r_ftrl.append(ftrl / runs)
        r_ftl.append(ftl  / runs)
        r_smart.append(smart / runs)
        r_emp.append(emp / runs)
    return r_ftrl, r_ftl, r_smart, r_emp

# ==============================================================
# Main
# ==============================================================

np.random.seed(0)

# Experiment settings
R        = 1.0       # action set radius
C        = 1.0       # SVM "C" via λ = 1/(C*N)
T_grid   = np.arange(100, 1100, 100)

# 1) Empirically estimate g(T) from FTRL on adversarial families (near-minimax)
g_emp = empirical_worst_case_thresholds(T_grid, runs=5, R=R, C=C)

# 2) Plot empirical g(T) vs theoretical sqrt(2T)
plt.figure(figsize=(7.5, 5.0))
g_vals = [g_emp[int(T)] for T in T_grid]
theory = [math.sqrt(2 * T) for T in T_grid]
plt.plot(T_grid, g_vals,  marker="o", label="Empirical g(T) from FTRL (worst-case families)")
plt.plot(T_grid, theory,  marker="x", label=r"Theory $\sqrt{2T}$ (scale comparator)")
plt.title("Empirical worst-case g(T) for SMART (ALG_WC = FTRL)")
plt.xlabel("T rounds")
plt.ylabel("g(T)")
plt.legend()
plt.tight_layout()
plt.savefig('empirical_g_T.png', dpi=300, bbox_inches='tight')
plt.close()

# 3) Evaluate across revamped 4 streams
rows, cols = 2, 2
fig, axes = plt.subplots(rows, cols, figsize=(12, 9))
axes = axes.flatten()

for idx, (title, seq_fn) in enumerate(CASES.items()):
    ax = axes[idx]
    runs = RUNS_BY_TITLE.get(title, 1)
    ftrl, ftl, smart, emp = evaluate_stream(seq_fn, T_grid, g_emp, runs=runs, R=R, C=C)
    ax.plot(T_grid, ftrl,  label="FTRL")
    ax.plot(T_grid, ftl,   label="FTL")
    ax.plot(T_grid, smart, label="SMART (√2T)")
    ax.plot(T_grid, emp,   label="SMART (empirical g from FTRL)")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("T rounds")
    ax.set_ylabel("Cumulative regret")
    ax.legend()

for j in range(len(CASES), rows * cols):
    axes[j].axis('off')

fig.suptitle("Four algorithms across four streams (comparator = SVM hindsight)", fontsize=14)
fig.tight_layout()
plt.savefig('algorithm_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
