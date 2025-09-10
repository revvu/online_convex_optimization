"""
FTRL vs FTL vs SMART vs empirical_g_SMART

Speed-optimized (no change to runs/replicates/T):
- Numba JIT for hot loops (simulate_alg, simulate_SMART_like, helpers)
- float32 arrays for data and parameters
- Modern RNG (np.random.Generator + SeedSequence)
- Same behavior, faster execution

Includes:
- "Two leaders (√t gap)" deterministic sequence (blocks) — FTL ≫ FTRL.
- "Stochastic √t-gap (Bernoulli drift)" — cleaner, non-contrived analogue.
- "Two leaders (switching, no drift)" — simple alternating leaders with fixed blocks.
- Diagnostic plot for the deterministic √t-gap sequence.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

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
# RNG helpers (fast, reproducible)
# ==============================================================

def _mix_seed(run_seed: int, T: int, salt: int = 0) -> np.random.Generator:
    ss = np.random.SeedSequence(entropy=(run_seed, T, salt, 0xA5A55A5A))
    return np.random.Generator(np.random.PCG64(ss))


# ==============================================================
# Adversarial families used for g(T)
# ==============================================================

def flip_sequence(T: int, d: int = 5, R: float = 1.0):
    z = np.zeros((T, d), dtype=np.float32); z[:, 0] = R
    y = np.array([1.0 if t % 2 else -1.0 for t in range(1, T + 1)], dtype=np.float32)
    u = np.zeros(d, dtype=np.float32)
    return z, y, u

# ==============================================================
# Two leaders (switching, no drift)
#   Alternate fixed-length blocks of +1 and -1: +1…+1, -1…-1, +1…+1, ...
#   No growing lead; tests adaptation to leader switches without contrived drift.
# ==============================================================

def switching_two_leaders_sequence(T: int, *, block_len: int = 20, d: int = 5, R: float = 1.0):
    y = np.empty(T, dtype=np.float32)
    sign = 1.0
    idx = 0
    while idx < T:
        run = min(block_len, T - idx)
        y[idx:idx+run] = sign
        idx += run
        sign = -sign
    z = np.zeros((T, d), dtype=np.float32); z[:, 0] = R
    u = np.zeros(d, dtype=np.float32)
    return z, y, u


# ==============================================================
# Stream builders (fixed task per run; fresh sequences per T; replicates)
# ==============================================================

def make_random_iid_stream(*, d: int = 5, R: float = 1.0, run_seed: int = 0):
    gen_u = _mix_seed(run_seed, 0, 11)
    u = gen_u.standard_normal(d).astype(np.float32)
    n = float(np.linalg.norm(u))
    if n > 0:
        u /= n

    def sample(T: int, rep: int = 0):
        gen = _mix_seed(run_seed, T, 13 + rep)
        z = gen.standard_normal((T, d)).astype(np.float32)
        norms = np.linalg.norm(z, axis=1, keepdims=True).astype(np.float32)
        norms = np.maximum(norms, 1.0)
        z = z / norms * R
        y = np.sign(z @ u).astype(np.float32); y[y == 0.0] = 1.0
        return z, y, u
    return sample

def make_noisy_iid_stream(*, p: float, d: int = 5, R: float = 1.0, run_seed: int = 0):
    gen_u = _mix_seed(run_seed, 0, 21)
    u = gen_u.standard_normal(d).astype(np.float32)
    n = float(np.linalg.norm(u))
    if n > 0:
        u /= n

    def sample(T: int, rep: int = 0):
        gen = _mix_seed(run_seed, T, 23 + rep)
        z = gen.standard_normal((T, d)).astype(np.float32)
        norms = np.linalg.norm(z, axis=1, keepdims=True).astype(np.float32)
        norms = np.maximum(norms, 1.0)
        z = z / norms * R
        y = np.sign(z @ u).astype(np.float32); y[y == 0.0] = 1.0
        flips = gen.random(T) < p
        y[flips] *= -1.0
        return z, y, u
    return sample

def make_flip_stream(*, d: int = 5, R: float = 1.0, run_seed: int = 0):
    def sample(T: int, rep: int = 0):
        return flip_sequence(T, d=d, R=R)
    return sample

def make_switching_two_leaders_stream(*, block_len: int = 20, d: int = 5, R: float = 1.0, run_seed: int = 0):
    def sample(T: int, rep: int = 0):
        return switching_two_leaders_sequence(T, block_len=block_len, d=d, R=R)
    return sample


# --- CASE SET (builders) ---
CASES: Dict[str, Callable[..., Callable[[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]]]] = {
    "Random i.i.d. (separable)":              lambda *, run_seed, R: make_random_iid_stream(d=5, R=R, run_seed=run_seed),
    "Massart noise 10%":             lambda *, run_seed, R: make_noisy_iid_stream(p=0.10, d=5, R=R, run_seed=run_seed),
    "Label flips":               lambda *, run_seed, R: make_flip_stream(d=5, R=R, run_seed=run_seed),
    "Switching leaders":      lambda *, run_seed, R: make_switching_two_leaders_stream(block_len=20, d=5, R=R, run_seed=run_seed),
}

# Case-specific averaging controls
RUNS_BY_TITLE = {
    "Random i.i.d. (separable)":              48,
    "Massart noise 10%":             48,
    "Label flips":                1,
    "Switching leaders":       1,   # deterministic
}
REPLICATES_BY_TITLE = {
    "Random i.i.d. (separable)":              16,
    "Massart noise 10%":             20,
    "Label flips":                1,
    "Switching leaders":       1,
}


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
# Evaluation (means and 95% CIs across runs; with replicates per T)
# ==============================================================

def evaluate_stream_with_stats(stream_builder: Callable[..., Callable[[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]]],
                               T_grid: np.ndarray,
                               g_emp: Dict[int, float],
                               *,
                               runs: int = 1,
                               replicates: int = 1,
                               R: float = 1.0,
                               base_seed: int = 0,
                               stream_name: str = ""):
    keys = ["FTRL", "FTL", "SMART", "EMP"]
    by_T: Dict[str, list[list[float]]] = {k: [[] for _ in range(len(T_grid))] for k in keys}

    run_iter = tqdm(range(runs),
                    desc=f"Evaluating stream: {stream_name} (runs={runs}, reps/T={replicates})",
                    position=0)
    for run in run_iter:
        run_seed = base_seed + 2025 * (run + 1)
        sampler = stream_builder(run_seed=run_seed, R=R)

        t_iter = tqdm(T_grid,
                      desc=f"  Run {run+1}/{runs}: sequence lengths",
                      leave=False,
                      position=1)
        for ti, T in enumerate(t_iter):
            reps_vals = {k: [] for k in keys}
            for rep in range(replicates):
                z, y, _ = sampler(int(T), rep=rep)
                u_star = comparator_ftl_hindsight(z, y, R)

                reps_vals["FTRL"].append(
                    simulate_alg(z, y, u_star, alg_flag=0, R=R, eta0=math.sqrt(2))
                )
                reps_vals["FTL"].append(
                    simulate_alg(z, y, u_star, alg_flag=1, R=R, eta0=math.sqrt(2))
                )
                reps_vals["SMART"].append(
                    simulate_SMART(z, y, u_star, R=R)
                )
                reps_vals["EMP"].append(
                    simulate_empirical_g_SMART(z, y, u_star, float(g_emp[int(T)]), R=R)
                )

            for k in keys:
                by_T[k][ti].append(float(np.mean(reps_vals[k])))

    means: Dict[str, list[float]] = {k: [] for k in keys}
    cis:   Dict[str, list[float]] = {k: [] for k in keys}
    for k in keys:
        for vals in by_T[k]:
            arr = np.array(vals, dtype=float)
            mu = float(np.mean(arr))
            if arr.size > 1:
                sem = float(np.std(arr, ddof=1)) / math.sqrt(arr.size)
                ci = 1.96 * sem
            else:
                ci = 0.0
            means[k].append(mu)
            cis[k].append(ci)

    stats = {k: (np.array(means[k]), np.array(cis[k])) for k in keys}
    return stats


def _plot_with_ci(ax, x, mean: np.ndarray, ci: np.ndarray, label: str):
    line, = ax.plot(x, mean, label=label)
    if np.any(ci > 0.0):
        ax.fill_between(x, mean - ci, mean + ci, alpha=0.2, linewidth=0, color=line.get_color())


# ==============================================================
# Main
# ==============================================================

if __name__ == "__main__":
    # Experiment settings
    R        = 1.0
    T_grid   = np.arange(100, 1100, 100, dtype=int)

    # 1) Empirically estimate g(T) from FTRL on adversarial families (near-minimax)
    g_emp = empirical_worst_case_thresholds(T_grid, runs=1000, R=R)

    # 2) Plot empirical g(T) vs theoretical sqrt(2T)
    plt.figure(figsize=(7.5, 5.0))
    g_vals = [g_emp[int(T)] for T in T_grid]
    theory = [math.sqrt(2 * int(T)) for T in T_grid]
    theory_pi = [math.sqrt(int(T) / (math.pi)) for T in T_grid]
    plt.plot(T_grid, g_vals,  marker="o", label="Empirical g(T)")
    plt.plot(T_grid, theory_pi, linestyle='--', label=r"$\sqrt{T/\pi}$")
    plt.plot(T_grid, theory,  marker="x", label=r"$\sqrt{2T}$")
    plt.title("Empirical worst-case g(T) for SMART (ALG_WC = FTRL)", fontsize=18)
    plt.xlabel("T rounds", fontsize=16)
    plt.ylabel("g(T)", fontsize=16)
    plt.legend(prop={'size': 14})
    plt.tight_layout()
    plt.savefig('empirical_g_T.png', dpi=600, bbox_inches='tight')
    plt.close()

    # 3) Evaluate across streams with per-T replicates
    n_cases = len(CASES)
    cols = 2
    rows = int(math.ceil(n_cases / cols))
    fig_width = 12
    fig_height = 4.0 * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    axes = axes.flatten()

    for idx, (title, builder) in enumerate(CASES.items()):
        ax = axes[idx]
        runs       = RUNS_BY_TITLE.get(title, 1)
        replicates = REPLICATES_BY_TITLE.get(title, 1)

        stats = evaluate_stream_with_stats(builder, T_grid, g_emp,
                                           runs=runs, replicates=replicates, R=R,
                                           stream_name=title)
        _plot_with_ci(ax, T_grid, *stats["FTRL"], label="FTRL")
        _plot_with_ci(ax, T_grid, *stats["FTL"],  label="FTL")
        _plot_with_ci(ax, T_grid, *stats["SMART"], label="SMART (√2T)")
        _plot_with_ci(ax, T_grid, *stats["EMP"],   label="SMART (empirical g)")

        ax.set_title(f"{title} (runs={runs}, reps/T={replicates})", fontsize=16)
        ax.set_xlabel("T rounds", fontsize=14)
        ax.set_ylabel("Cumulative regret", fontsize=14)
        ax.legend(prop={'size': 12})

    for j in range(len(CASES), rows * cols):
        axes[j].axis('off')

    fig.suptitle("Online Linear Binary Classification", fontsize=20)
    fig.tight_layout()
    plt.savefig('algorithm_comparison.png', dpi=600, bbox_inches='tight')
    plt.close()
