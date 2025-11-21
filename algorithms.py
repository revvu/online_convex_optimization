from __future__ import annotations

import math
from typing import Dict

import numpy as np
from tqdm import tqdm


def _normalized_hinge(q: float, y: float) -> float:
    return 0.5 * abs(q - y)  # y ∈ {±1}, q ∈ [-1,1]

def _action_ftl(theta: np.ndarray, out: np.ndarray) -> None:
    n = np.linalg.norm(theta)
    out[:] = 0.0 if n == 0.0 else -(1.0 / n) * theta

def _action_ftrl(theta: np.ndarray, t: int, eta0: float, out: np.ndarray) -> None:
    out[:] = -(eta0 / math.sqrt(max(1, t))) * theta
    n = np.linalg.norm(out)
    if n > 1.0:
        out *= 1.0 / n


# ==============================================================
# Online simulation (FTL, FTRL), regret vs comparator
# ==============================================================

def simulate_alg(z: np.ndarray,
                 y: np.ndarray,
                 alg_flag: int,  # 0 = FTRL, 1 = FTL
                 eta0: float) -> float:
    T, d = z.shape
    theta = np.zeros(d, dtype=np.float32)
    x_t = np.zeros(d, dtype=np.float32)
    cum_loss = 0.0

    for t in range(T):
        if alg_flag == 0:
            _action_ftrl(theta, t + 1, eta0, x_t)
        else:
            _action_ftl(theta, x_t)

        q = float(np.dot(z[t], x_t))
        y_t = float(y[t])
        cum_loss += _normalized_hinge(q, y_t)

        diff = q - y_t
        grad_q = 0.5 if diff > 0.0 else -0.5 if diff < 0.0 else 0.0
        theta += grad_q * z[t]

    _action_ftl(theta, x_t)
    q_all = z @ x_t
    comp_loss = np.sum(0.5 * np.abs(q_all - y))
    return cum_loss - comp_loss


# ==============================================================
# SMART (single switch)
# ==============================================================

def _compute_gradient(q_pred: float, y_true: float) -> float:
    diff = q_pred - y_true
    return 0.5 if diff > 0.0 else -0.5 if diff < 0.0 else 0.0

def simulate_SMART_like(z: np.ndarray,
                        y: np.ndarray,
                        theta_thresh: float,
                        eta0: float) -> float:
    """
    Single-switch SMART: start with FTL, switch to FTRL when FTL's
    cumulative regret exceeds `theta_thresh`.
    Returns: total_loss - comparator_loss.
    """
    T, d = z.shape

    # Parameters
    theta_ftl  = np.zeros(d, dtype=np.float32)
    theta_ftrl = np.zeros(d, dtype=np.float32)

    # Scratch vectors
    x = np.zeros(d, dtype=np.float32)  # action used to play (FTL or FTRL)
    s = np.zeros(d, dtype=np.float32)  # best constant action estimate

    switched = False
    ftl_loss = 0.0
    total_loss = 0.0

    for t in range(T):
        zt = z[t]
        yt = float(y[t])

        # --- Always update FTL parameters (used for switch test) ---
        _action_ftl(theta_ftl, x)
        pred_ftl = float(zt @ x)
        theta_ftl += _compute_gradient(pred_ftl, yt) * zt
        loss_ftl = _normalized_hinge(pred_ftl, yt)
        ftl_loss += loss_ftl

        # --- Play either FTL (pre-switch) or FTRL (post-switch) ---
        if switched:
            _action_ftrl(theta_ftrl, t + 1, eta0, x)
            pred = float(zt @ x)
            total_loss += _normalized_hinge(pred, yt)
            theta_ftrl += _compute_gradient(pred, yt) * zt
        else:
            total_loss += loss_ftl

            # Check switch condition using best constant action so far
            _action_ftl(theta_ftl, s)
            q_hist = z[:t+1] @ s
            s_loss = np.sum(0.5 * np.abs(q_hist - y[:t+1]))
            if ftl_loss - s_loss >= theta_thresh:
                switched = True

    # Final comparator loss (with final s from FTL)
    _action_ftl(theta_ftl, s)
    q_all = z @ s
    comp_loss = np.sum(0.5 * np.abs(q_all - y))

    return total_loss - comp_loss


def simulate_SMART(z: np.ndarray, y: np.ndarray, *, eta0: float = math.sqrt(2)) -> float:
    T = z.shape[0]
    return simulate_SMART_like(z, y, theta_thresh=math.sqrt(2*T), eta0=eta0)

def simulate_empirical_g_SMART(z: np.ndarray, y: np.ndarray, theta_emp: float, *, eta0: float = math.sqrt(2)) -> float:
    return simulate_SMART_like(z, y, theta_thresh=theta_emp, eta0=eta0)


# ==============================================================
# Empirical g(T) for random sequences
# ==============================================================

def empirical_worst_case_thresholds(
    T_grid: np.ndarray,
    *,
    runs: int = 5,
    base_seed: int = 0,
) -> Dict[int, float]:
    """
    For each horizon T in `T_grid`, sample `runs` i.i.d. sequences (z, y),
    run FTRL, and record the maximum regret observed.

    Returns:
        Dict mapping T -> max_regret over `runs`.
    """
    g_emp: Dict[int, float] = {}

    for T_val in tqdm(T_grid, desc="Estimating g(T) on random sequences"):
        T = int(T_val)
        max_regret = 0.0

        for r in range(runs):
            gen = _rng(base_seed, T, r)

            # z: rows clipped to unit norm
            z = gen.standard_normal((T, 5)).astype(np.float32, copy=False)
            norms = np.linalg.norm(z, axis=1, keepdims=True).astype(np.float32, copy=False)
            z *= (1.0 / np.maximum(norms, 1.0))

            # labels y ∈ {−1, +1}
            y = gen.choice([-1.0, 1.0], size=T).astype(np.float32, copy=False)

            reg = simulate_alg(z, y, alg_flag=0, eta0=math.sqrt(2))
            if reg > max_regret:
                max_regret = reg

        g_emp[T] = max_regret

    return g_emp

# ==============================================================
# RNG helper (simple, reproducible)
# ==============================================================

def _rng(base_seed: int, T: int, run: int) -> np.random.Generator:
    # Independent reproducible stream per (T, run)
    ss = np.random.SeedSequence([base_seed, T, run])
    return np.random.Generator(np.random.PCG64(ss))
