"""
FTL vs FTRL vs SMART vs empirical_g_SMART — Regression version
- Loss:      squared loss  (1/2)*(q - y)^2  with q = z·x
- Comparator: ridge regression in hindsight (L2), then projected to ‖x‖ ≤ R
- SMART:     same single-switch design with reset; g(T) estimated from near-worst-case families

This file mirrors the classification scaffold so you can compare apples-to-apples.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Callable, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

# ==============================================================
# Utilities
# ==============================================================

def norm2(x: np.ndarray) -> float:
    return float(np.linalg.norm(x))

def project_to_ball(x: np.ndarray, R: float) -> np.ndarray:
    n = norm2(x)
    return x if n == 0.0 or n <= R else (R / n) * x

def half_square_loss(q: float, y: float) -> float:
    return 0.5 * (q - y) ** 2

def regret_increment(q_pred: float, y_true: float, q_comp: float) -> float:
    return half_square_loss(q_pred, y_true) - half_square_loss(q_comp, y_true)

# ==============================================================
# Actions: FTL and FTRL on linearized losses
# ==============================================================

def action_ftl(theta: np.ndarray, R: float = 1.0) -> np.ndarray:
    """Follow-the-leader on linearized losses: argmin_{‖x‖≤R} ⟨θ, x⟩ = -R θ/‖θ‖ (or 0)."""
    n = norm2(theta)
    return np.zeros_like(theta) if n == 0.0 else -R * theta / n

def action_ftrl(theta: np.ndarray, t: int, R: float = 1.0, eta0: float = math.sqrt(2)) -> np.ndarray:
    """
    Simple FTRL proxy: gradient step on cumulative subgradients then project.
    η_t = eta0 / sqrt(t).
    """
    eta_t = eta0 / math.sqrt(max(1, t))
    x = -eta_t * theta
    return project_to_ball(x, R)

# ==============================================================
# Online simulation (FTL, FTRL), regret vs ridge comparator
# ==============================================================

def simulate_alg(z: np.ndarray,
                 y: np.ndarray,
                 u_comp: np.ndarray,
                 *,
                 alg: str = "FTRL",
                 R: float = 1.0,
                 eta0: float = math.sqrt(2)) -> float:
    """
    Plays ALG ∈ {FTRL, FTL}. Loss = (1/2)*(z_t·x - y_t)^2.
    Comparator prediction uses u_comp (ridge in hindsight).
    """
    T, d = z.shape
    theta = np.zeros(d)
    cum_reg = 0.0

    for t in range(1, T + 1):
        x_t = action_ftrl(theta, t, R, eta0) if alg.upper() == "FTRL" else action_ftl(theta, R)

        q    = float(z[t-1] @ x_t)
        q_uc = float(z[t-1] @ u_comp)
        cum_reg += regret_increment(q, float(y[t-1]), q_uc)

        # gradient wrt x via q = z·x
        grad_q = (q - y[t-1])              # d/dq (1/2)(q - y)^2 = (q - y)
        theta += grad_q * z[t-1]           # accumulate linearized gradient

    return cum_reg

# ==============================================================
# SMART (single switch) with reset of ALG_WC state
# ==============================================================

def simulate_SMART_like(z: np.ndarray,
                        y: np.ndarray,
                        u_comp: np.ndarray,
                        theta_thresh: float,
                        *,
                        R: float = 1.0,
                        eta0: float = math.sqrt(2)) -> float:
    """
    Start with FTL. If estimated FTL regret crosses θ, switch once to ALG_WC=FTRL.
    Reset FTRL's internal state at switch.
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

        q    = float(z[t-1] @ x_t)
        q_uc = float(z[t-1] @ u_comp)
        total_reg += regret_increment(q, float(y[t-1]), q_uc)

        # gradient at played action
        grad_q = (q - y[t-1])
        g_vec  = grad_q * z[t-1]

        # Always update FTL surrogate
        theta_ftl += g_vec

        # Update FTRL only after switching (reset semantics)
        if switched:
            theta_ftrl += g_vec

        # Track anytime FTL regret (estimator for SMART)
        if not switched:
            q_ftl   = float(z[t-1] @ x_ftl)
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
# Ridge regression (closed-form) for hindsight comparator
# ==============================================================

def ridge_regression(X: np.ndarray,
                     y: np.ndarray,
                     *,
                     lam: float,
                     R: float = 1.0) -> np.ndarray:
    """
    w* = argmin_w Σ (1/2)(z·w - y)^2 + (lam/2)||w||^2   (closed-form),
    then projected to ‖w‖ ≤ R.
    """
    d = X.shape[1]
    A = X.T @ X + lam * np.eye(d)
    b = X.T @ y
    w = np.linalg.solve(A, b)
    return project_to_ball(w, R)

def best_action_in_hindsight_ridge(z: np.ndarray, y: np.ndarray, *, R: float = 1.0, C: float = 1.0) -> np.ndarray:
    """
    Mirror the classification file's C/λ relation: λ = 1/(C*N).
    """
    N = len(y)
    lam = 1.0 / max(C * N, 1.0)
    return ridge_regression(z, y, lam=lam, R=R)

# ==============================================================
# Streams (regression-flavored)
# ==============================================================

def _normalize_features(z: np.ndarray, R: float) -> np.ndarray:
    z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1.0) * R
    return z

def random_iid_sequence(T: int, d: int = 5, R: float = 1.0, sigma: float = 0.05):
    """
    Linear signal with small Gaussian noise, clipped targets to [-1,1] for comparability.
    Ensures predictions q=z·x stay within [-1,1] when R=1.
    """
    z = _normalize_features(np.random.randn(T, d), R)
    u = np.random.randn(d)
    # scale u so that max|z·u| ≤ 1
    mm = np.max(np.abs(z @ u))
    if mm > 1.0:
        u = u / mm
    y = z @ u + sigma * np.random.randn(T)
    y = np.clip(y, -1.0, 1.0)
    return z, y.astype(float), u

def heteroscedastic_sequence(T: int, d: int = 5, R: float = 1.0, sigma0: float = 0.03):
    """
    Noise grows over time; first easy then harder.
    """
    z = _normalize_features(np.random.randn(T, d), R)
    u = np.random.randn(d)
    mm = np.max(np.abs(z @ u))
    if mm > 1.0:
        u = u / mm
    scales = np.linspace(1.0, 3.0, T)
    y = z @ u + (sigma0 * scales) * np.random.randn(T)
    y = np.clip(y, -1.0, 1.0)
    return z, y.astype(float), u

def alternating_targets_sequence(T: int, d: int = 5, R: float = 1.0, sigma: float = 0.02):
    """
    Constant feature direction, targets alternate +1/-1 (adversarial-ish).
    """
    z = np.zeros((T, d)); z[:, 0] = R
    y = np.array([1.0 if t % 2 else -1.0 for t in range(1, T + 1)], dtype=float)
    y += sigma * np.random.randn(T)
    y = np.clip(y, -1.0, 1.0)
    u = np.zeros(d)  # hindsight ridge will try to sit near e1
    return z, y, u

def orthogonal_hard_sequence(T: int, R: float = 1.0):
    """
    Orthogonal features; constant target.
    """
    z = np.zeros((T, T))
    for t in range(T):
        z[t, t] = R
    y = np.ones(T, dtype=float)
    u = np.ones(T) / math.sqrt(T)
    return z, y, u

def phase_change_sequence(T: int, d: int = 5, R: float = 1.0, frac_easy: float = 0.6, sigma_easy: float = 0.02):
    """
    First k rounds: easy i.i.d. linear; remainder: alternating targets on constant feature.
    """
    k = max(1, int(T * frac_easy))
    z1, y1, u = random_iid_sequence(k, d=d, R=R, sigma=sigma_easy)
    z2 = np.zeros((T - k, d)); z2[:, 0] = R
    y2 = np.array([1.0 if t % 2 else -1.0 for t in range(1, T - k + 1)], dtype=float)
    z = np.vstack([z1, z2]); y = np.concatenate([y1, y2])
    return z, y, u

# --- Case set (mirrors the classification file’s layout) ---
CASES = {
    "Random i.i.d. (linear + noise)":     random_iid_sequence,
    "Heteroscedastic (easy→hard)":         heteroscedastic_sequence,
    "Phase change: linear→alt (60/40)":    partial(phase_change_sequence, frac_easy=0.6),
    "Alternating targets (adversarial)":   alternating_targets_sequence,
    "Orthogonal (hard)":                   orthogonal_hard_sequence,
}

RUNS_BY_TITLE = {
    "Random i.i.d. (linear + noise)":    30,
    "Heteroscedastic (easy→hard)":       30,
    "Phase change: linear→alt (60/40)":  10,
    "Alternating targets (adversarial)":  1,
    "Orthogonal (hard)":                  1,
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
    Families: alternating targets, orthogonal hard. (Extend as desired.)
    """
    adv_families = [alternating_targets_sequence, orthogonal_hard_sequence]
    g_emp: Dict[int, float] = {}

    for T in T_grid:
        worst = 0.0
        for fam in adv_families:
            for r in range(runs):
                np.random.seed(base_seed + 1337 * (int(T) + r))
                z, y, _ = fam(int(T), R=R) if fam is orthogonal_hard_sequence else fam(int(T), R=R)
                u_star = best_action_in_hindsight_ridge(z, y, R=R, C=C)
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
                    * ,
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
            u_star = best_action_in_hindsight_ridge(z, y, R=R, C=C)
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

if __name__ == "__main__":
    np.random.seed(0)

    # Experiment settings
    R        = 1.0       # action set radius
    C        = 1.0       # ridge “C” via λ = 1/(C*N)
    T_grid   = np.arange(100, 1100, 100)

    # 1) Empirically estimate g(T) from FTRL on adversarial families
    g_emp = empirical_worst_case_thresholds(T_grid, runs=5, R=R, C=C)

    # 2) Plot empirical g(T) vs theoretical sqrt(2T) (back-of-the-envelope)
    plt.figure(figsize=(7.5, 5.0))
    g_vals = [g_emp[int(T)] for T in T_grid]
    theory = [math.sqrt(2 * T) for T in T_grid]
    plt.plot(T_grid, g_vals,  marker="o", label="Empirical g(T) from FTRL (worst-case families)")
    plt.plot(T_grid, theory,  marker="x", label=r"Theory $\sqrt{2T}$ (rough scale)")
    plt.title("Empirical worst-case g(T) for SMART (ALG_WC = FTRL) — Regression")
    plt.xlabel("T rounds")
    plt.ylabel("g(T)")
    plt.legend()
    plt.tight_layout()

    # 3) Evaluate across five streams
    rows, cols = 3, 2
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

    fig.suptitle("Four algorithms across five regression streams (comparator = ridge in hindsight)", fontsize=14)
    fig.tight_layout()
    plt.show()
