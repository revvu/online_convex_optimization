"""
Driver script for Online Convex Optimization experiments.

Compares FTRL, FTL, SMART (theoretical g), and SMART (empirical g) across
sequence types, reporting means and 95% CIs and producing two figures:

- empirical_g_T.png       : Empirical g(T) vs. theoretical references
- algorithm_comparison.png: Algorithm comparison across sequence families
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from algorithms import (
    empirical_worst_case_thresholds,
    simulate_alg,
    simulate_SMART,
    simulate_empirical_g_SMART,
)
from sequence_generation import CASES, RUNS_BY_TITLE, REPLICATES_BY_TITLE


# -------------------------
# Types & simple utilities
# -------------------------

Float32Array = NDArray[np.float32]
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

SamplerFn = Callable[[int, int], Tuple[Float32Array, Float32Array, Float32Array]]
StreamBuilder = Callable[..., SamplerFn]
Stats = Dict[str, Tuple[FloatArray, FloatArray]]  # algo -> (mean, ci)

ALGO_KEYS = ("FTRL", "FTL", "SMART", "EMP")
CI_Z = 1.96  # 95% normal CI


@dataclass(frozen=True)
class ExperimentConfig:
    T_grid: IntArray = np.arange(100, 1100, 100, dtype=int)
    base_seed: int = 0
    g_runs: int = 1000  # runs for empirical g(T) estimation


def _progress(iterable: Iterable, **kwargs):
    """Thin wrapper on tqdm for consistent styling."""
    return tqdm(iterable, dynamic_ncols=True, **kwargs)


def _sem(x: FloatArray) -> float:
    n = x.size
    if n <= 1:
        return 0.0
    return float(np.std(x, ddof=1) / math.sqrt(n))


# -------------------------
# Evaluation
# -------------------------

def evaluate_stream_with_stats(
    stream_builder: StreamBuilder,
    T_grid: IntArray,
    g_emp: Mapping[int, float],
    *,
    runs: int = 1,
    replicates: int = 1,
    base_seed: int = 0,
    stream_name: str = "",
) -> Stats:
    """
    Evaluate algorithms over a sequence generator with per-T replicates.
    Returns mean regret and 95% CI for each algorithm at each T.
    """
    # by_T[algo][ti] -> list of run means (each run mean is avg over replicates)
    by_T: Dict[str, list[list[float]]] = {k: [[] for _ in range(len(T_grid))] for k in ALGO_KEYS}

    for run in _progress(range(runs), desc=f"{stream_name:>24} | runs={runs}, reps/T={replicates}", position=0):
        run_seed = base_seed + 2025 * (run + 1)
        sampler = stream_builder(run_seed=run_seed)

        for ti, T in enumerate(_progress(T_grid, desc="  sequence lengths", leave=False, position=1)):
            rep_vals = {k: [] for k in ALGO_KEYS}
            T_int = int(T)

            theta_emp = float(g_emp[T_int])

            for rep in range(replicates):
                z, y, _ = sampler(T_int, rep=rep)

                # Keep the simulation calls visually aligned for quick scan
                rep_vals["FTRL"].append(
                    simulate_alg(z, y, alg_flag=0, eta0=math.sqrt(2))
                )
                rep_vals["FTL"].append(
                    simulate_alg(z, y, alg_flag=1, eta0=math.sqrt(2))
                )
                rep_vals["SMART"].append(
                    simulate_SMART(z, y)
                )
                rep_vals["EMP"].append(
                    simulate_empirical_g_SMART(z, y, theta_emp)
                )

            for k in ALGO_KEYS:
                by_T[k][ti].append(float(np.mean(rep_vals[k])))

    # Convert to means and 95% CIs across runs (already averaged over replicates)
    stats: Stats = {}
    for k in ALGO_KEYS:
        means, cis = [], []
        for vals in by_T[k]:
            arr = np.asarray(vals, dtype=float)
            mu = float(np.mean(arr)) if arr.size else 0.0
            ci = CI_Z * _sem(arr) if arr.size > 1 else 0.0
            means.append(mu)
            cis.append(ci)
        stats[k] = (np.array(means, dtype=float), np.array(cis, dtype=float))

    return stats


# -------------------------
# Plotting helpers
# -------------------------

def _plot_with_ci(ax: plt.Axes, x: IntArray, mean: FloatArray, ci: FloatArray, label: str) -> None:
    (line,) = ax.plot(x, mean, label=label)
    if np.any(ci > 0.0):
        ax.fill_between(x, mean - ci, mean + ci, alpha=0.2, linewidth=0, color=line.get_color())


def plot_empirical_g(T_grid: IntArray, g_emp: Mapping[int, float], *, out_path: str = "empirical_g_T.png") -> None:
    """Plot empirical g(T) against √(T/π) and √(2T)."""
    plt.figure(figsize=(7.5, 5.0))
    g_vals = [float(g_emp[int(T)]) for T in T_grid]
    theory_sqrt_2T = [math.sqrt(2 * int(T)) for T in T_grid]
    theory_sqrt_T_over_pi = [math.sqrt(int(T) / math.pi) for T in T_grid]

    plt.plot(T_grid, g_vals, marker="o", label="Empirical g(T)")
    plt.plot(T_grid, theory_sqrt_T_over_pi, linestyle="--", label=r"$\sqrt{T/\pi}$")
    plt.plot(T_grid, theory_sqrt_2T, marker="x", label=r"$\sqrt{2T}$")

    plt.title("Empirical worst-case g(T) for SMART (ALG_WC = FTRL)", fontsize=18)
    plt.xlabel("T rounds", fontsize=16)
    plt.ylabel("g(T)", fontsize=16)
    plt.legend(prop={"size": 14})
    plt.tight_layout()
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close()


def plot_comparisons(
    T_grid: IntArray,
    stats_by_case: Dict[str, Stats],
    *,
    out_path: str = "algorithm_comparison.png",
) -> None:
    """Plot algorithm comparisons per sequence case with 95% CIs."""
    n_cases = len(stats_by_case)
    cols = 2
    rows = int(math.ceil(n_cases / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(12, 4.0 * rows))
    axes = axes.flatten()

    for idx, (title, stats) in enumerate(stats_by_case.items()):
        ax = axes[idx]
        _plot_with_ci(ax, T_grid, *stats["FTRL"], label="FTRL")
        _plot_with_ci(ax, T_grid, *stats["FTL"], label="FTL")
        _plot_with_ci(ax, T_grid, *stats["SMART"], label="SMART (√2T)")
        _plot_with_ci(ax, T_grid, *stats["EMP"], label="SMART (empirical g)")

        runs = RUNS_BY_TITLE.get(title, 1)
        reps = REPLICATES_BY_TITLE.get(title, 1)
        ax.set_title(f"{title} (runs={runs}, reps/T={reps})", fontsize=16)
        ax.set_xlabel("T rounds", fontsize=14)
        ax.set_ylabel("Cumulative regret", fontsize=14)
        ax.legend(prop={"size": 12})

    # Hide any unused axes
    for j in range(n_cases, rows * cols):
        axes[j].axis("off")

    fig.suptitle("Online Linear Binary Classification", fontsize=20)
    fig.tight_layout()
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close()


# -------------------------
# Main
# -------------------------

def main() -> None:
    cfg = ExperimentConfig()

    # 1) Empirical g(T) from near-minimax adversarial families (ALG_WC = FTRL)
    g_emp = empirical_worst_case_thresholds(cfg.T_grid, runs=cfg.g_runs)

    # 2) Plot empirical g(T) vs references
    plot_empirical_g(cfg.T_grid, g_emp, out_path="empirical_g_T.png")

    # 3) Evaluate across sequence families (per-T replicates)
    stats_by_case: Dict[str, Stats] = {}
    for title, builder in CASES.items():
        stats_by_case[title] = evaluate_stream_with_stats(
            builder,
            cfg.T_grid,
            g_emp,
            runs=RUNS_BY_TITLE.get(title, 1),
            replicates=REPLICATES_BY_TITLE.get(title, 1),
            base_seed=cfg.base_seed,
            stream_name=title,
        )

    plot_comparisons(cfg.T_grid, stats_by_case, out_path="algorithm_comparison.png")


if __name__ == "__main__":
    main()
