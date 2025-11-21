"""
Driver script for Online Convex Optimization experiments using the exact
FTL solver backed by cvxpy.

The script mirrors the structure of `fast_driver.py`, but evaluates
algorithms provided by `exact_ftl.py`. It produces two figures:

- empirical_g_T_exact.png        : Empirical g(T) vs. theoretical references
- algorithm_comparison_exact.png : Regret comparison for exact FTL variants
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from algorithms import _rng
from exact_ftl import (
    ExactFTLNoClip,
    compute_prefix_actions,
    replay_exact_ftl,
    run_ftrl,
)
from sequence_generation import CASES, REPLICATES_BY_TITLE, RUNS_BY_TITLE


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
SamplerFn = Callable[[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]]
Stats = Dict[str, Tuple[FloatArray, FloatArray]]

CI_Z = 1.96  # 95% normal CI


@dataclass(frozen=True)
class ExperimentConfig:
    T_grid: IntArray = field(default_factory=lambda: np.arange(100, 1100, 100, dtype=np.int64))
    base_seed: int = 0
    g_runs: int = 200  # fewer runs than fast driver; each call solves a convex program
    norm: str = "l2"
    solver: str | None = None
    solver_opts: Mapping[str, object] | None = None
    reuse_grid: Sequence[int] = (1,)


def _progress(iterable: Iterable, **kwargs):
    """Consistent tqdm styling."""
    return tqdm(iterable, dynamic_ncols=True, **kwargs)


def _sem(x: FloatArray) -> float:
    n = x.size
    if n <= 1:
        return 0.0
    return float(np.std(x, ddof=1) / math.sqrt(n))


def empirical_worst_case_thresholds(
    T_grid: IntArray,
    *,
    runs: int,
    base_seed: int,
    norm: str,
    solver: str | None,
    solver_opts: Mapping[str, object] | None,
) -> Dict[int, float]:
    """Empirical g(T) using the exact FTL module's FTRL baseline."""
    g_emp: Dict[int, float] = {}

    solver_cache: Dict[int, ExactFTLNoClip] = {}
    solver_opts_dict = None if solver_opts is None else dict(solver_opts)

    for T_val in tqdm(T_grid, desc="Estimating g(T) with exact FTRL"):
        T = int(T_val)
        max_regret = 0.0

        solver_ftl = solver_cache.get(T)
        if solver_ftl is None:
            solver_ftl = ExactFTLNoClip(
                d=5,
                T_max=T,
                norm=norm,
                solver=solver,
                solver_opts=solver_opts_dict,
            )
            solver_cache[T] = solver_ftl

        for run_idx in range(runs):
            gen = _rng(base_seed, T, run_idx)

            z = gen.standard_normal((T, 5)).astype(np.float64, copy=False)
            norms = np.linalg.norm(z, axis=1, keepdims=True)
            z *= 1.0 / np.maximum(norms, 1.0)

            y = gen.choice([-1.0, 1.0], size=T).astype(np.float64, copy=False)

            reg = run_ftrl(
                z,
                y,
                eta0=math.sqrt(2),
                norm=norm,
                solver=solver,
                solver_opts=solver_opts_dict,
                comparator_solver=solver_ftl,
            ).regret
            if reg > max_regret:
                max_regret = float(reg)

        g_emp[T] = max_regret

    return g_emp


def evaluate_stream_with_stats(
    stream_builder: Callable[..., SamplerFn],
    T_grid: IntArray,
    *,
    runs: int,
    replicates: int,
    cfg: ExperimentConfig,
    stream_name: str,
) -> Stats:
    ftl_labels = {1: "FTL (exact)"}
    algo_labels = ["FTRL", "FTL (exact)"]
    by_T: Dict[str, list[list[float]]] = {
        label: [[] for _ in range(len(T_grid))]
        for label in algo_labels
    }

    solver_cache: Dict[
        Tuple[int, int, str | None, str | None, Tuple[Tuple[str, object], ...] | None],
        ExactFTLNoClip,
    ] = {}
    solver_opts_key = None if cfg.solver_opts is None else tuple(sorted(cfg.solver_opts.items()))

    for run_idx in _progress(
        range(runs),
        desc=f"{stream_name:>24} | runs={runs}, reps/T={replicates}",
        position=0,
    ):
        run_seed = cfg.base_seed + 2025 * (run_idx + 1)
        sampler = stream_builder(run_seed=run_seed)

        for ti, T in enumerate(
            _progress(T_grid, desc="  sequence lengths", leave=False, position=1)
        ):
            T_int = int(T)
            rep_vals = {label: [] for label in algo_labels}

            for rep in range(replicates):
                z, y, _ = sampler(T_int, rep=rep)
                z_arr = np.ascontiguousarray(z, dtype=np.float64)
                y_arr = np.ascontiguousarray(y, dtype=np.float64)
                d = z_arr.shape[1]

                key = (d, T_int, cfg.norm, cfg.solver, solver_opts_key)
                solver_ftl = solver_cache.get(key)
                if solver_ftl is None:
                    solver_ftl = ExactFTLNoClip(
                        d=d,
                        T_max=T_int,
                        norm=cfg.norm,
                        solver=cfg.solver,
                        solver_opts=None if cfg.solver_opts is None else dict(cfg.solver_opts),
                    )
                    solver_cache[key] = solver_ftl

                actions = compute_prefix_actions(solver_ftl, z_arr, y_arr)

                ftrl_res = run_ftrl(
                    z_arr,
                    y_arr,
                    eta0=math.sqrt(2),
                    norm=cfg.norm,
                    comparator_action=actions[-1],
                )
                rep_vals["FTRL"].append(float(ftrl_res.regret))

                res_ftl = replay_exact_ftl(
                    z_arr,
                    y_arr,
                    actions,
                )
                rep_vals["FTL (exact)"].append(float(res_ftl.regret))

            for label in algo_labels:
                by_T[label][ti].append(float(np.mean(rep_vals[label])))

    stats: Stats = {}
    for label in algo_labels:
        means, cis = [], []
        for vals in by_T[label]:
            arr = np.asarray(vals, dtype=float)
            mu = float(np.mean(arr)) if arr.size else 0.0
            ci = CI_Z * _sem(arr) if arr.size > 1 else 0.0
            means.append(mu)
            cis.append(ci)
        stats[label] = (np.array(means, dtype=float), np.array(cis, dtype=float))

    return stats


def _plot_with_ci(ax: plt.Axes, x: IntArray, mean: FloatArray, ci: FloatArray, label: str) -> None:
    (line,) = ax.plot(x, mean, label=label)
    if np.any(ci > 0.0):
        ax.fill_between(x, mean - ci, mean + ci, alpha=0.2, linewidth=0, color=line.get_color())


def plot_empirical_g(T_grid: IntArray, g_emp: Mapping[int, float], *, out_path: str) -> None:
    plt.figure(figsize=(7.5, 5.0))
    g_vals = [float(g_emp[int(T)]) for T in T_grid]
    theory_sqrt_2T = [math.sqrt(2 * int(T)) for T in T_grid]
    theory_sqrt_T_over_pi = [math.sqrt(int(T) / math.pi) for T in T_grid]

    plt.plot(T_grid, g_vals, marker="o", label="Empirical g(T)")
    plt.plot(T_grid, theory_sqrt_T_over_pi, linestyle="--", label=r"$\sqrt{T/\pi}$")
    plt.plot(T_grid, theory_sqrt_2T, marker="x", label=r"$\sqrt{2T}$")

    plt.title("Empirical worst-case g(T) for SMART (exact FTRL)", fontsize=18)
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
    out_path: str,
) -> None:
    n_cases = len(stats_by_case)
    cols = 2
    rows = int(math.ceil(n_cases / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(12, 4.0 * rows))
    axes = axes.flatten()

    for idx, (title, stats) in enumerate(stats_by_case.items()):
        ax = axes[idx]
        for label, (mean, ci) in stats.items():
            _plot_with_ci(ax, T_grid, mean, ci, label=label)

        runs = RUNS_BY_TITLE.get(title, 1)
        reps = REPLICATES_BY_TITLE.get(title, 1)
        ax.set_title(f"{title} (runs={runs}, reps/T={reps})", fontsize=16)
        ax.set_xlabel("T rounds", fontsize=14)
        ax.set_ylabel("Cumulative regret", fontsize=14)
        ax.legend(prop={"size": 12})

    for j in range(n_cases, rows * cols):
        axes[j].axis("off")

    fig.suptitle("Exact FTL vs FTRL", fontsize=20)
    fig.tight_layout()
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close()


def main() -> None:
    cfg = ExperimentConfig()

    g_emp = empirical_worst_case_thresholds(
        cfg.T_grid,
        runs=cfg.g_runs,
        base_seed=cfg.base_seed,
        norm=cfg.norm,
        solver=cfg.solver,
        solver_opts=cfg.solver_opts,
    )

    plot_empirical_g(cfg.T_grid, g_emp, out_path="empirical_g_T_exact.png")

    stats_by_case: Dict[str, Stats] = {}
    for title, builder in CASES.items():
        stats_by_case[title] = evaluate_stream_with_stats(
            builder,
            cfg.T_grid,
            runs=RUNS_BY_TITLE.get(title, 1),
            replicates=REPLICATES_BY_TITLE.get(title, 1),
            cfg=cfg,
            stream_name=title,
        )

    plot_comparisons(cfg.T_grid, stats_by_case, out_path="algorithm_comparison_exact.png")


if __name__ == "__main__":
    main()
