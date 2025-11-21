from __future__ import annotations

import math
from typing import Dict

import numpy as np
from numba import njit
from tqdm import tqdm


@njit(cache=True)
def _dot(a: np.ndarray, b: np.ndarray) -> float:
    total = 0.0
    for i in range(a.shape[0]):
        total += a[i] * b[i]
    return total


@njit(cache=True)
def _normalized_hinge(q: float, y: float) -> float:
    diff = q - y
    if diff < 0.0:
        diff = -diff
    return 0.5 * diff


@njit(cache=True)
def _compute_gradient(q_pred: float, y_true: float) -> float:
    diff = q_pred - y_true
    if diff > 0.0:
        return 0.5
    if diff < 0.0:
        return -0.5
    return 0.0


@njit(cache=True)
def _action_ftl(theta: np.ndarray, out: np.ndarray) -> None:
    norm_sq = 0.0
    d = theta.shape[0]
    for j in range(d):
        norm_sq += theta[j] * theta[j]
    if norm_sq == 0.0:
        for j in range(d):
            out[j] = 0.0
        return
    scale = -(1.0 / math.sqrt(norm_sq))
    for j in range(d):
        out[j] = scale * theta[j]


@njit(cache=True)
def _action_ftrl(theta: np.ndarray, t: int, eta0: float, out: np.ndarray) -> None:
    d = theta.shape[0]
    scale = -(eta0 / math.sqrt(max(1, t)))
    for j in range(d):
        out[j] = scale * theta[j]
    norm_sq = 0.0
    for j in range(d):
        norm_sq += out[j] * out[j]
    if norm_sq <= 1.0:
        return
    norm = math.sqrt(norm_sq)
    factor = 1.0 / norm
    for j in range(d):
        out[j] *= factor


@njit(cache=True)
def _total_comparator_loss(z: np.ndarray, y: np.ndarray, action: np.ndarray) -> float:
    total = 0.0
    T = z.shape[0]
    for i in range(T):
        pred = _dot(z[i], action)
        total += _normalized_hinge(pred, y[i])
    return total


@njit(cache=True)
def _comparator_loss_prefix(z: np.ndarray, y: np.ndarray, action: np.ndarray, length: int) -> float:
    total = 0.0
    for i in range(length):
        pred = _dot(z[i], action)
        total += _normalized_hinge(pred, y[i])
    return total


@njit(cache=True)
def _simulate_alg_core(z: np.ndarray,
                       y: np.ndarray,
                       alg_flag: int,
                       eta0: float) -> float:
    T = z.shape[0]
    d = z.shape[1]
    theta = np.zeros(d)
    x_t = np.zeros(d)
    cum_loss = 0.0

    for t in range(T):
        if alg_flag == 0:
            _action_ftrl(theta, t + 1, eta0, x_t)
        else:
            _action_ftl(theta, x_t)

        q = _dot(z[t], x_t)
        y_t = y[t]
        cum_loss += _normalized_hinge(q, y_t)

        grad_q = _compute_gradient(q, y_t)
        for j in range(d):
            theta[j] += grad_q * z[t, j]

    _action_ftl(theta, x_t)
    comp_loss = _total_comparator_loss(z, y, x_t)
    return cum_loss - comp_loss


@njit(cache=True)
def _simulate_SMART_like_core(z: np.ndarray,
                              y: np.ndarray,
                              theta_thresh: float,
                              eta0: float) -> float:
    T = z.shape[0]
    d = z.shape[1]

    theta_ftl = np.zeros(d)
    theta_ftrl = np.zeros(d)

    x = np.zeros(d)
    s = np.zeros(d)

    switched = False
    ftl_loss = 0.0
    total_loss = 0.0

    for t in range(T):
        zt = z[t]
        yt = y[t]

        _action_ftl(theta_ftl, x)
        pred_ftl = _dot(zt, x)
        grad_ftl = _compute_gradient(pred_ftl, yt)
        for j in range(d):
            theta_ftl[j] += grad_ftl * zt[j]
        loss_ftl = _normalized_hinge(pred_ftl, yt)
        ftl_loss += loss_ftl

        if switched:
            _action_ftrl(theta_ftrl, t + 1, eta0, x)
            pred = _dot(zt, x)
            total_loss += _normalized_hinge(pred, yt)
            grad = _compute_gradient(pred, yt)
            for j in range(d):
                theta_ftrl[j] += grad * zt[j]
        else:
            total_loss += loss_ftl
            _action_ftl(theta_ftl, s)
            s_loss = _comparator_loss_prefix(z, y, s, t + 1)
            if ftl_loss - s_loss >= theta_thresh:
                switched = True

    _action_ftl(theta_ftl, s)
    comp_loss = _total_comparator_loss(z, y, s)
    return total_loss - comp_loss


# ==============================================================
# Online simulation (FTL, FTRL), regret vs comparator
# ==============================================================

def simulate_alg(z: np.ndarray,
                 y: np.ndarray,
                 alg_flag: int,
                 eta0: float) -> float:
    z_arr = np.ascontiguousarray(z, dtype=np.float64)
    y_arr = np.ascontiguousarray(y, dtype=np.float64)
    return float(_simulate_alg_core(z_arr, y_arr, int(alg_flag), float(eta0)))


# ==============================================================
# SMART (single switch)
# ==============================================================

def simulate_SMART_like(z: np.ndarray,
                        y: np.ndarray,
                        theta_thresh: float,
                        eta0: float) -> float:
    """
    Single-switch SMART: start with FTL, switch to FTRL when FTL's
    cumulative regret exceeds `theta_thresh`.
    Returns: total_loss - comparator_loss.
    """
    z_arr = np.ascontiguousarray(z, dtype=np.float64)
    y_arr = np.ascontiguousarray(y, dtype=np.float64)
    return float(_simulate_SMART_like_core(z_arr, y_arr, float(theta_thresh), float(eta0)))


def simulate_SMART(z: np.ndarray, y: np.ndarray, *, eta0: float = math.sqrt(2)) -> float:
    T = z.shape[0]
    return simulate_SMART_like(z, y, theta_thresh=math.sqrt(2 * T), eta0=eta0)


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
            z = gen.standard_normal((T, 5)).astype(np.float64, copy=False)
            norms = np.linalg.norm(z, axis=1, keepdims=True).astype(np.float64, copy=False)
            z *= (1.0 / np.maximum(norms, 1.0))

            # labels y ∈ {−1, +1}
            y = gen.choice([-1.0, 1.0], size=T).astype(np.float64, copy=False)

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
