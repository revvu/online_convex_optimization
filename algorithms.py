"""
Online Convex Optimization Algorithms

This module contains implementations of various online learning algorithms:
- FTRL (Follow the Regularized Leader)
- FTL (Follow the Leader) 
- SMART (Switching algorithm)
- Supporting utilities and comparators

Speed-optimized with Numba JIT compilation for performance.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np

# ------------------------ Acceleration toggles ------------------------
USE_NUMBA = True  # set False to disable JIT
NUMBA_THREADS = None  # e.g., set to an int to control threads

try:
    if USE_NUMBA:
        from numba import njit, prange, set_num_threads
        if NUMBA_THREADS is not None:
            set_num_threads(NUMBA_THREADS)
    else:
        raise ImportError
except Exception:
    # Fallback: define no-op decorators
    def njit(*args, **kwargs):
        def deco(f): return f
        return deco
    def prange(x): return range(x)


# ==============================================================
# Utilities (Numba-friendly)
# ==============================================================

@njit(cache=True, fastmath=True)
def _clip_unit_scalar(q: float) -> float:
    if q > 1.0:
        return 1.0
    if q < -1.0:
        return -1.0
    return q

@njit(cache=True, fastmath=True)
def _norm2(x: np.ndarray) -> float:
    s = 0.0
    for i in range(x.size):
        s += float(x[i]) * float(x[i])
    return math.sqrt(s)

@njit(cache=True, fastmath=True)
def _project_to_ball(x: np.ndarray, R: float) -> None:
    n = _norm2(x)
    if n > R and n > 0.0:
        scale = R / n
        for i in range(x.size):
            x[i] *= scale

@njit(cache=True, fastmath=True)
def _normalized_hinge(q: float, y: float) -> float:
    # y in {±1}, q ∈ [-1,1]; equals 0 if correct sign at ±1, 1 if wrong sign at ∓1
    diff = q - y
    if diff < 0:
        diff = -diff
    return 0.5 * diff

@njit(cache=True, fastmath=True)
def _regret_increment(q_pred: float, y_true: float, q_comp: float) -> float:
    return _normalized_hinge(q_pred, y_true) - _normalized_hinge(q_comp, y_true)

@njit(cache=True, fastmath=True)
def _action_ftl(theta: np.ndarray, R: float, out: np.ndarray) -> None:
    n = _norm2(theta)
    if n == 0.0:
        for i in range(theta.size):
            out[i] = 0.0
    else:
        s = -R / n
        for i in range(theta.size):
            out[i] = s * theta[i]

@njit(cache=True, fastmath=True)
def _action_ftrl(theta: np.ndarray, t: int, R: float, eta0: float, out: np.ndarray) -> None:
    eta_t = eta0 / math.sqrt(float(max(1, t)))
    for i in range(theta.size):
        out[i] = -eta_t * theta[i]
    _project_to_ball(out, R)


# ==============================================================
# Online simulation (FTL, FTRL), regret vs comparator
# ==============================================================

@njit(cache=True, fastmath=True)
def simulate_alg(z: np.ndarray,
                 y: np.ndarray,
                 u_comp: np.ndarray,
                 alg_flag: int,  # 0 = FTRL, 1 = FTL
                 R: float,
                 eta0: float) -> float:
    T, d = z.shape
    theta = np.zeros(d, dtype=np.float32)
    x_t   = np.zeros(d, dtype=np.float32)

    # Precompute q_uc[t] = clip(z[t]·u_comp)
    q_uc_vec = np.empty(T, dtype=np.float32)
    for t in range(T):
        s = 0.0
        for j in range(d):
            s += float(z[t, j]) * float(u_comp[j])
        q_uc_vec[t] = _clip_unit_scalar(s)

    cum_reg = 0.0
    for t in range(T):
        if alg_flag == 0:
            _action_ftrl(theta, t + 1, R, eta0, x_t)
        else:
            _action_ftl(theta, R, x_t)

        s = 0.0
        for j in range(d):
            s += float(z[t, j]) * float(x_t[j])
        q = _clip_unit_scalar(s)

        cum_reg += _regret_increment(q, float(y[t]), float(q_uc_vec[t]))

        diff = q - float(y[t])
        grad_q = 0.0
        if diff > 0.0:
            grad_q = 0.5
        elif diff < 0.0:
            grad_q = -0.5

        for j in range(d):
            theta[j] += grad_q * float(z[t, j])

    return cum_reg


# ==============================================================
# Hindsight comparator: best fixed linear predictor
# ==============================================================

@njit(cache=True, fastmath=True)
def comparator_ftl_hindsight(z: np.ndarray, y: np.ndarray, R: float) -> np.ndarray:
    T, d = z.shape
    s = np.zeros(d, dtype=np.float32)
    for t in range(T):
        yt = float(y[t])
        for j in range(d):
            s[j] += yt * float(z[t, j])
    n = _norm2(s)
    if n == 0.0:
        return s
    scale = R / n
    for j in range(d):
        s[j] *= scale
    return s


# ==============================================================
# SMART (single switch) with proper reset of ALG_WC
# ==============================================================

@njit(cache=True, fastmath=True)
def simulate_SMART_like(z: np.ndarray,
                        y: np.ndarray,
                        u_comp: np.ndarray,
                        theta_thresh: float,
                        R: float,
                        eta0: float) -> float:
    T, d = z.shape
    theta_ftl  = np.zeros(d, dtype=np.float32)
    theta_ftrl = np.zeros(d, dtype=np.float32)
    x_ftl      = np.zeros(d, dtype=np.float32)
    x_ftrl     = np.zeros(d, dtype=np.float32)
    switched   = False

    q_uc_vec = np.empty(T, dtype=np.float32)
    for t in range(T):
        s = 0.0
        for j in range(d):
            s += float(z[t, j]) * float(u_comp[j])
        q_uc_vec[t] = _clip_unit_scalar(s)

    ftl_reg    = 0.0
    total_reg  = 0.0

    for t in range(T):
        _action_ftl(theta_ftl, R, x_ftl)
        _action_ftrl(theta_ftrl, t + 1, R, eta0, x_ftrl)
        x_t = x_ftl if not switched else x_ftrl

        s = 0.0
        for j in range(d):
            s += float(z[t, j]) * float(x_t[j])
        q    = _clip_unit_scalar(s)
        q_uc = float(q_uc_vec[t])

        total_reg += _regret_increment(q, float(y[t]), q_uc)

        diff = q - float(y[t])
        grad_q = 0.0
        if diff > 0.0:
            grad_q = 0.5
        elif diff < 0.0:
            grad_q = -0.5

        for j in range(d):
            g = grad_q * float(z[t, j])
            theta_ftl[j] += g
            if switched:
                theta_ftrl[j] += g

        if not switched:
            s2 = 0.0
            for j in range(d):
                s2 += float(z[t, j]) * float(x_ftl[j])
            q_ftl = _clip_unit_scalar(s2)
            ftl_reg += _regret_increment(q_ftl, float(y[t]), q_uc)
            if ftl_reg >= theta_thresh:
                switched = True
                for j in range(d):
                    theta_ftrl[j] = 0.0

    return total_reg


def simulate_SMART(z: np.ndarray, y: np.ndarray, u_comp: np.ndarray, *, R: float = 1.0, eta0: float = math.sqrt(2)) -> float:
    T = z.shape[0]
    return simulate_SMART_like(z, y, u_comp, theta_thresh=math.sqrt(2*T), R=R, eta0=eta0)

def simulate_empirical_g_SMART(z: np.ndarray, y: np.ndarray, u_comp: np.ndarray, theta_emp: float, *, R: float = 1.0, eta0: float = math.sqrt(2)) -> float:
    return simulate_SMART_like(z, y, u_comp, theta_thresh=theta_emp, R=R, eta0=eta0)


# ==============================================================
# Empirical g(T) for random sequences (random z, random y)
# ==============================================================

def empirical_worst_case_thresholds(T_grid: np.ndarray,
                                    *,
                                    runs: int = 5,
                                    R: float = 1.0,
                                    base_seed: int = 0) -> Dict[int, float]:
    # For each T, sample `runs` i.i.d. random sequences (z, y) and
    # take the maximum FTRL regret against the best fixed comparator.
    from tqdm import tqdm
    
    g_emp: Dict[int, float] = {}
    for T in tqdm(T_grid, desc="Estimating g(T) on random sequences"):
        max_regret = 0.0
        for r in range(runs):
            gen = _mix_seed(base_seed + r, int(T), 13 + r)
            z = gen.standard_normal((int(T), 5)).astype(np.float32)
            norms = np.linalg.norm(z, axis=1, keepdims=True).astype(np.float32)
            norms = np.maximum(norms, 1.0)
            z = z / norms * R
            y = gen.choice([-1.0, 1.0], size=int(T)).astype(np.float32)
            u_star = comparator_ftl_hindsight(z, y, R)
            reg = simulate_alg(z, y, u_star, alg_flag=0, R=R, eta0=math.sqrt(2))
            if reg > max_regret:
                max_regret = reg
        g_emp[int(T)] = max_regret
    return g_emp


# ==============================================================
# RNG helpers (fast, reproducible)
# ==============================================================

def _mix_seed(run_seed: int, T: int, salt: int = 0) -> np.random.Generator:
    ss = np.random.SeedSequence(entropy=(run_seed, T, salt, 0xA5A55A5A))
    return np.random.Generator(np.random.PCG64(ss))
