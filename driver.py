"""
Driver script for Online Convex Optimization experiments

This module contains the main execution logic, evaluation functions, and plotting
for comparing FTRL, FTL, SMART, and empirical_g_SMART algorithms across different
sequence types.

Usage:
    python driver.py

This will generate:
- empirical_g_T.png: Plot of empirical g(T) vs theoretical bounds
- algorithm_comparison.png: Comparison of algorithms across different sequence types
"""

import math
from typing import Callable, Dict

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from algorithms import (
    empirical_worst_case_thresholds,
    simulate_alg,
    simulate_SMART,
    simulate_empirical_g_SMART,
    comparator_ftl_hindsight
)
from sequence_generation import CASES, RUNS_BY_TITLE, REPLICATES_BY_TITLE


# ==============================================================
# Evaluation (means and 95% CIs across runs; with replicates per T)
# ==============================================================

def evaluate_stream_with_stats(stream_builder: Callable[..., Callable[[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]]],
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
# Main execution
# ==============================================================

def main():
    """Main execution function for the online convex optimization experiments."""
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


if __name__ == "__main__":
    main()
