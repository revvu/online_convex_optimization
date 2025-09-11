"""
Online Convex Optimization Algorithms (Numba-optimized, incremental comparator)

- FTRL (Follow the Regularized Leader)
- FTL (Follow the Leader)
- SMART (single-switch) with incremental/batched comparator loss
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
from tqdm import tqdm
from numba import njit

# ==============================================================
# Low-level float32 helpers
# ==============================================================

@njit(fastmath=True, cache=True)
def _norm2_f32(x: np.ndarray) -> float:
    s = 0.0
    for i in range(x.size):
        v = x[i]
        s += v * v
    return math.sqrt(s)

@njit(fastmath=True, cache=True)
def _dot_f32(a: np.ndarray, b: np.ndarray) -> float:
    s = 0.0
    for i in range(a.size):
        s += a[i] * b[i]
    return s

@njit(fastmath=True, cache=True)
def _clip_unit(q: float) -> float:
    if q < -1.0:
        return -1.0
    if q > 1.0:
        return 1.0
    return q

# ==============================================================
# Core primitives
# ==============================================================

@njit(fastmath=True, cache=True)
def _project_to_ball(x: np.ndarray, R: float) -> None:
    n = _norm2_f32(x)
    if n > R and n > 0.0:
        s = R / n
        for i in range(x.size):
            x[i] *= s

@njit(fastmath=True, cache=True)
def _normalized_hinge(q: float, y: float) -> float:
    # y in {1, -1}, q in [-1,1]
    d = q - y
    if d < 0:
        d = -d
    return 0.5 * d

@njit(fastmath=True, cache=True)
def _action_ftl(theta: np.ndarray, R: float, out: np.ndarray) -> None:
    n = _norm2_f32(theta)
    s = 0.0 if n == 0.0 else -R / n
    for i in range(theta.size):
        out[i] = s * theta[i]

@njit(fastmath=True, cache=True)
def _action_ftrl(theta: np.ndarray, t: int, R: float, eta0: float, out: np.ndarray) -> None:
    tt = float(t)
    if tt < 1.0:
        tt = 1.0
    eta_t = eta0 / math.sqrt(tt)
    for i in range(theta.size):
        out[i] = -eta_t * theta[i]
    _project_to_ball(out, R)

# ==============================================================
# Hindsight comparator (exact, whole sequence)
# ==============================================================

@njit(fastmath=True, cache=True)
def comparator_ftl_hindsight(z: np.ndarray, y: np.ndarray, R: float) -> Tuple[np.ndarray, float]:
    T, d = z.shape
    s_raw = np.zeros(d, dtype=np.float32)
    for t in range(T):
        yt = y[t]
        for j in range(d):
            s_raw[j] += yt * z[t, j]
    n = _norm2_f32(s_raw)
    if n > 0.0:
        sc = R / n
        for j in range(d):
            s_raw[j] *= sc

    s = s_raw  # now normalized
    cum_loss = 0.0
    for t in range(T):
        q = _clip_unit(_dot_f32(z[t], s))
        dlt = q - y[t]
        if dlt < 0:
            dlt = -dlt
        cum_loss += 0.5 * dlt
    return s, cum_loss

# ==============================================================
# Online simulation (FTL, FTRL)
# ==============================================================

@njit(fastmath=True, cache=True)
def simulate_alg(z: np.ndarray,
                 y: np.ndarray,
                 u_comp: np.ndarray,  # API compatibility
                 alg_flag: int,  # 0 = FTRL, 1 = FTL
                 R: float,
                 eta0: float) -> float:
    T, d = z.shape
    theta = np.zeros(d, dtype=np.float32)
    x_t = np.zeros(d, dtype=np.float32)
    cum_loss = 0.0

    for t in range(T):
        if alg_flag == 0:
            _action_ftrl(theta, t + 1, R, eta0, x_t)
        else:
            _action_ftl(theta, R, x_t)

        q = _clip_unit(_dot_f32(z[t], x_t))
        yt = y[t]

        dlt = q - yt
        if dlt < 0:
            dlt = -dlt
        cum_loss += 0.5 * dlt

        # grad wrt q
        diff = q - yt
        if diff > 0.0:
            grad_q = 0.5
        elif diff < 0.0:
            grad_q = -0.5
        else:
            grad_q = 0.0

        for j in range(d):
            theta[j] += grad_q * z[t, j]

    # comparator loss
    _, comp_loss = comparator_ftl_hindsight(z, y, R)
    return cum_loss - comp_loss

# ==============================================================
# SMART: incremental/batched comparator
# ==============================================================

@njit(fastmath=True, cache=True)
def _compute_gradient(q_pred: float, y_true: float) -> float:
    diff = q_pred - y_true
    if diff > 0.0:
        return 0.5
    elif diff < 0.0:
        return -0.5
    else:
        return 0.0

@njit(fastmath=True, cache=True)
def _normalize_into(s_raw: np.ndarray, R: float, out: np.ndarray) -> None:
    n = _norm2_f32(s_raw)
    if n > 0.0:
        sc = R / n
        for j in range(s_raw.size):
            out[j] = sc * s_raw[j]
    else:
        for j in range(s_raw.size):
            out[j] = 0.0

@njit(fastmath=True, cache=True)
def _sum_hinge_from_cache(dot_cache: np.ndarray, y: np.ndarray, upto: int) -> float:
    # uses un-clipped dot_cache; clips on the fly
    L = 0.0
    for i in range(upto + 1):
        q = dot_cache[i]
        if q < -1.0:
            q = -1.0
        elif q > 1.0:
            q = 1.0
        dlt = q - y[i]
        if dlt < 0:
            dlt = -dlt
        L += 0.5 * dlt
    return L

@njit(fastmath=True, cache=True)
def simulate_SMART_like(z: np.ndarray,
                        y: np.ndarray,
                        u_comp: np.ndarray,      # API compatibility
                        theta_thresh: float,
                        R: float,
                        eta0: float,
                        comp_update_stride: int = 16) -> float:
    """
    SMART-like algorithm with a single switch.
    Comparator loss up to t is maintained incrementally using:
      - s_raw prefix sum and normalized s_t
      - cached inner products updated by delta_s every K steps
    """
    T, d = z.shape

    theta_ftl = np.zeros(d, dtype=np.float32)
    theta_ftrl = np.zeros(d, dtype=np.float32)
    x_ftl = np.zeros(d, dtype=np.float32)
    x_ftrl = np.zeros(d, dtype=np.float32)
    has_switched = False

    # Comparator state
    s_raw = np.zeros(d, dtype=np.float32)            # running sum of y*z
    s_prev = np.zeros(d, dtype=np.float32)           # previous normalized comparator
    s_curr = np.zeros(d, dtype=np.float32)           # current normalized comparator
    dot_cache = np.zeros(T, dtype=np.float32)        # Z[:t] @ s_prev (unclipped)
    comp_loss_cached = 0.0                           # L_comp up to last update
    last_update_t = -1

    ftl_loss = 0.0
    total_loss = 0.0

    for t in range(T):
        # --- actions
        _action_ftl(theta_ftl, R, x_ftl)
        _action_ftrl(theta_ftrl, t + 1, R, eta0, x_ftrl)
        action = x_ftl if not has_switched else x_ftrl

        # --- play, observe loss, update thetas
        q = _clip_unit(_dot_f32(z[t], action))
        yt = y[t]
        dlt = q - yt
        if dlt < 0:
            dlt = -dlt
        total_loss += 0.5 * dlt

        grad_q = _compute_gradient(q, yt)
        for j in range(d):
            g = grad_q * z[t, j]
            theta_ftl[j] += g
            if has_switched:
                theta_ftrl[j] += g

        # --- FTL loss tracking (exact, cheap)
        q_ftl = _clip_unit(_dot_f32(z[t], x_ftl))
        dftl = q_ftl - yt
        if dftl < 0:
            dftl = -dftl
        ftl_loss += 0.5 * dftl

        # --- Update running comparator direction s_t
        for j in range(d):
            s_raw[j] += yt * z[t, j]
        # s_curr = normalized(s_raw)
        _normalize_into(s_raw, R, s_curr)

        # --- Update comparator cache periodically
        need_update = ((t - last_update_t) >= comp_update_stride) or (t == 0)

        if need_update:
            # delta_s = s_curr - s_prev
            for j in range(d):
                s_curr_j = s_curr[j]
                delta = s_curr_j - s_prev[j]
                s_prev[j] = s_curr_j  # move prev -> curr for next round
                s_curr[j] = delta     # temporarily store delta in s_curr
            # Now s_curr stores delta_s

            # Update cached dots for all i in [0, t] by adding z[i]·delta_s
            for i in range(t + 1):
                inc = 0.0
                zi = z[i]
                for j in range(d):
                    inc += zi[j] * s_curr[j]
                dot_cache[i] += inc

            # Restore s_curr back to the actual comparator (copy from s_prev)
            for j in range(d):
                s_curr[j] = s_prev[j]

            # Compute comparator loss up to t from cache
            comp_loss_cached = _sum_hinge_from_cache(dot_cache, y, t)
            last_update_t = t
        else:
            # Fast path: extend cache with the new sample using previous s_prev
            # dot_cache[t] = z[t] @ s_prev
            dot_cache[t] = _dot_f32(z[t], s_prev)
            # (Use comp_loss_cached as comparator loss proxy until next update)
            # This slightly delays switch but avoids O(t·d) every step.

        # --- Switch condition using the cached/last comparator loss
        ftl_regret = ftl_loss - comp_loss_cached
        if (not has_switched) and (ftl_regret >= theta_thresh):
            has_switched = True
            for j in range(d):
                theta_ftrl[j] = 0.0

    # Final comparator loss (exact) for the return value
    _, final_comp_loss = comparator_ftl_hindsight(z, y, R)
    return total_loss - final_comp_loss

# Public wrappers (preserve API)
def simulate_SMART(z: np.ndarray, y: np.ndarray, u_comp: np.ndarray, *,
                   R: float = 1.0,
                   eta0: float = math.sqrt(2),
                   comp_update_stride: int = 16) -> float:
    T = z.shape[0]
    return float(simulate_SMART_like(z.astype(np.float32),
                                     y.astype(np.float32),
                                     u_comp.astype(np.float32) if u_comp is not None else np.zeros(z.shape[1], np.float32),
                                     theta_thresh=math.sqrt(2.0 * float(T)),
                                     R=float(R),
                                     eta0=float(eta0),
                                     comp_update_stride=int(comp_update_stride)))

def simulate_empirical_g_SMART(z: np.ndarray, y: np.ndarray, u_comp: np.ndarray, theta_emp: float, *,
                               R: float = 1.0,
                               eta0: float = math.sqrt(2),
                               comp_update_stride: int = 16) -> float:
    return float(simulate_SMART_like(z.astype(np.float32),
                                     y.astype(np.float32),
                                     u_comp.astype(np.float32) if u_comp is not None else np.zeros(z.shape[1], np.float32),
                                     theta_thresh=float(theta_emp),
                                     R=float(R),
                                     eta0=float(eta0),
                                     comp_update_stride=int(comp_update_stride)))

# ==============================================================
# Empirical g(T)
# ==============================================================

def empirical_worst_case_thresholds(T_grid: np.ndarray,
                                    *,
                                    runs: int = 5,
                                    R: float = 1.0,
                                    base_seed: int = 0) -> Dict[int, float]:
    g_emp: Dict[int, float] = {}
    for T in tqdm(T_grid, desc="Estimating g(T) on random sequences"):
        T_int = int(T)
        max_regret = 0.0
        for r in range(runs):
            gen = _mix_seed(base_seed + r, T_int, 13 + r)
            z = gen.standard_normal((T_int, 5)).astype(np.float32)

            # project to ball of radius R
            for t in range(T_int):
                nrm = math.sqrt(float((z[t] * z[t]).sum()))
                if nrm < 1.0:
                    nrm = 1.0
                z[t] *= (R / nrm)

            y = gen.choice(np.array([-1.0, 1.0], dtype=np.float32), size=T_int).astype(np.float32)
            u_star, _ = comparator_ftl_hindsight(z, y, float(R))
            reg = simulate_alg(z, y, u_star, alg_flag=0, R=float(R), eta0=math.sqrt(2.0))
            if reg > max_regret:
                max_regret = float(reg)
        g_emp[T_int] = float(max_regret)
    return g_emp

# ==============================================================
# RNG helper
# ==============================================================

def _mix_seed(run_seed: int, T: int, salt: int = 0) -> np.random.Generator:
    ss = np.random.SeedSequence(entropy=(run_seed, T, salt, 0xA5A55A5A))
    return np.random.Generator(np.random.PCG64(ss))

# ==============================================================
# Self-test
# ==============================================================

if __name__ == "__main__":
    T, d = 5000, 5
    rng = _mix_seed(0, T, 7)
    z = rng.standard_normal((T, d)).astype(np.float32)
    for t in range(T):
        nrm = math.sqrt(float((z[t] * z[t]).sum()))
        if nrm > 1.0:
            z[t] /= nrm
    y = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=T).astype(np.float32)

    u_star, _ = comparator_ftl_hindsight(z, y, 1.0)
    print("Regret(FTRL):", float(simulate_alg(z, y, u_star, alg_flag=0, R=1.0, eta0=math.sqrt(2.0))))
    print("Regret(FTL):  ", float(simulate_alg(z, y, u_star, alg_flag=1, R=1.0, eta0=math.sqrt(2.0))))
    print("Regret(SMART, stride=16):",
          float(simulate_SMART(z, y, u_star, R=1.0, eta0=math.sqrt(2.0), comp_update_stride=16)))
