from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Optional, Tuple

import numpy as np

from numba import njit
import cvxpy as cp  # type: ignore

# ==============================================================
# Small JIT helpers (mirrors fast_algorithms.py style)
# ==============================================================

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
def _grad_normalized_hinge(q: float, y: float) -> float:
    diff = q - y
    if diff > 0.0:
        return 0.5
    if diff < 0.0:
        return -0.5
    return 0.0

@njit(cache=True)
def _vec_norm2(x: np.ndarray) -> float:
    s = 0.0
    for i in range(x.shape[0]):
        s += x[i] * x[i]
    return math.sqrt(s)


# --------------------------------------------------------------
# Array helpers
# --------------------------------------------------------------

def _ensure_float64_contiguous(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.float64 and arr.flags["C_CONTIGUOUS"]:
        return arr
    return np.ascontiguousarray(arr, dtype=np.float64)


# ==============================================================
# Exact FTL (no clip): single reusable CVXPY problem
# ==============================================================

class ExactFTLNoClip:
    """
    Build once, solve many: exact FTL with a unit norm-ball constraint.
    Mask w \in {0,1}^Tmax selects the prefix.
    """
    def __init__(
        self,
        d: int,
        T_max: int,
        *,
        norm: Literal["l2", "linf", "l1"] = "l2",
        solver: Optional[str] = None,
        solver_opts: Optional[dict] = None,
    ) -> None:
        self.d = int(d)
        self.T_max = int(T_max)
        self.norm = norm
        self.solver = solver
        self.solver_opts = {} if solver_opts is None else dict(solver_opts)

        # Parameters (filled each round)
        self.Z = cp.Parameter((self.T_max, self.d))
        self.y = cp.Parameter(self.T_max)
        self.w = cp.Parameter(self.T_max, nonneg=True)  # 1 for rows we include; 0 else

        # Variables
        self.x = cp.Variable(self.d)
        s = cp.Variable(self.T_max, nonneg=True)

        # Residuals for all rows; w weights active prefix in objective
        r = self.Z @ self.x - self.y
        cons = [s >= r, s >= -r]

        if self.norm == "l2":
            cons += [cp.norm2(self.x) <= 1.0]    # SOCP
        elif self.norm == "linf":
            cons += [cp.norm_inf(self.x) <= 1.0] # LP
        elif self.norm == "l1":
            cons += [cp.norm1(self.x) <= 1.0]    # LP
        else:
            raise ValueError("norm must be one of {'l2','linf','l1'}")

        obj = cp.Minimize(0.5 * self.w @ s)
        self._prob = cp.Problem(obj, cons)

        # Reusable buffers for parameter updates
        self._Z_buf = np.zeros((self.T_max, self.d), dtype=np.float64)
        self._y_buf = np.zeros(self.T_max, dtype=np.float64)
        self._w_buf = np.zeros(self.T_max, dtype=np.float64)
        self._last_length = 0

        self.Z.value = self._Z_buf
        self.y.value = self._y_buf
        self.w.value = self._w_buf

        self._solve_current()

    def _solve_current(self) -> np.ndarray:
        if self.solver is None:
            self._prob.solve(warm_start=True)
        else:
            self._prob.solve(solver=self.solver, warm_start=True, **self.solver_opts)

        if self._prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"Exact FTL (no clip) failed: status={self._prob.status}")

        return np.asarray(self.x.value, dtype=np.float64)

    def _set_prefix(self, z_source: np.ndarray, y_source: np.ndarray, length: int) -> None:
        t = int(length)
        if t < 0 or t > self.T_max:
            raise ValueError("length must be between 0 and T_max inclusive")

        if t > 0:
            np.copyto(self._Z_buf[:t], z_source[:t])
            np.copyto(self._y_buf[:t], y_source[:t])
            self._w_buf[:t] = 1.0
        if t < self._last_length:
            tail = slice(t, self._last_length)
            self._Z_buf[tail] = 0.0
            self._y_buf[tail] = 0.0
            self._w_buf[tail] = 0.0

        self._last_length = t
        self.Z.value = self._Z_buf
        self.y.value = self._y_buf
        self.w.value = self._w_buf

    def solve_prefix_from_full(
        self,
        z_full: np.ndarray,
        y_full: np.ndarray,
        length: int,
    ) -> np.ndarray:
        """Solve using the first ``length`` rows of ``z_full`` / ``y_full``."""
        z_src = _ensure_float64_contiguous(z_full)
        y_src = _ensure_float64_contiguous(y_full)
        self._set_prefix(z_src, y_src, length)
        return self._solve_current()

    def solve_prefix(self, z_prefix: np.ndarray, y_prefix: np.ndarray) -> np.ndarray:
        """Backward-compatible wrapper that accepts prefix arrays directly."""
        z_src = _ensure_float64_contiguous(z_prefix)
        y_src = _ensure_float64_contiguous(y_prefix)
        t, d = z_src.shape
        if d != self.d:
            raise ValueError(f"Expected {self.d}-dimensional data, got {d}")
        if t > self.T_max:
            raise ValueError("prefix longer than T_max")
        self._set_prefix(z_src, y_src, t)
        return self._solve_current()


# ==============================================================
# Baseline FTRL (fast path for comparisons)
# ==============================================================

def _action_ftrl(theta: np.ndarray, t: int, eta0: float, out: np.ndarray) -> None:
    """
    x_t = - (eta0 / sqrt(max(1,t))) * theta, projected to l2-ball of radius 1.
    Hot path mirrors fast_algorithms.py.
    """
    d = theta.shape[0]
    scale = -(eta0 / math.sqrt(max(1, t)))
    out[:] = scale * theta
    nrm = _vec_norm2(out)
    if nrm > 1.0:
        out *= (1.0 / nrm)


# ==============================================================
# Run loop + comparator
# ==============================================================

@dataclass
class RunResult:
    cum_loss: float
    regret: float
    comp_loss: float
    x_last: np.ndarray

def _comparator_loss(z: np.ndarray, y: np.ndarray, x: np.ndarray) -> float:
    """0.5 * sum |z x - y| under the unit-norm constraint."""
    r = z @ x - y
    return 0.5 * float(np.abs(r).sum())


def _simulate_ftrl(
    z_arr: np.ndarray,
    y_arr: np.ndarray,
    *,
    eta0: float,
    comparator_action: Optional[np.ndarray] = None,
    comparator_solver: Optional[ExactFTLNoClip] = None,
    norm: Literal["l2", "linf", "l1"] = "l2",
    solver_name: Optional[str] = None,
    solver_opts: Optional[dict] = None,
) -> RunResult:
    T, d = z_arr.shape
    theta = np.zeros(d, dtype=np.float64)
    x_t = np.zeros(d, dtype=np.float64)
    cum_loss = 0.0

    for t in range(T):
        _action_ftrl(theta, t + 1, eta0, x_t)
        q = _dot(z_arr[t], x_t)
        y_t = y_arr[t]
        cum_loss += _normalized_hinge(q, y_t)

        grad = _grad_normalized_hinge(q, y_t)
        theta += grad * z_arr[t]

    if comparator_action is None:
        solver = comparator_solver
        if solver is None:
            solver = ExactFTLNoClip(
                d=d,
                T_max=T,
                norm=norm,
                solver=solver_name,
                solver_opts=solver_opts,
            )
        comp_vec = solver.solve_prefix_from_full(z_arr, y_arr, T)
    else:
        comp_vec = _ensure_float64_contiguous(comparator_action)

    comp_loss = _comparator_loss(z_arr, y_arr, comp_vec)
    regret = cum_loss - comp_loss

    return RunResult(
        cum_loss=float(cum_loss),
        regret=float(regret),
        comp_loss=float(comp_loss),
        x_last=x_t.copy(),
    )


def compute_prefix_actions(
    solver: ExactFTLNoClip,
    z: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Return exact FTL solutions for every prefix length 0..T."""
    z_arr = _ensure_float64_contiguous(z)
    y_arr = _ensure_float64_contiguous(y)
    T, d = z_arr.shape
    if solver.d != d:
        raise ValueError(f"Solver dimension {solver.d} incompatible with data dimension {d}")
    if solver.T_max < T:
        raise ValueError("Solver T_max is smaller than sequence length")

    actions = np.zeros((T + 1, d), dtype=np.float64)
    # Prefix length 0 -> zero vector
    if d > 0:
        actions[0].fill(0.0)

    for length in range(1, T + 1):
        actions[length] = solver.solve_prefix_from_full(z_arr, y_arr, length)

    return actions


def replay_exact_ftl(
    z: np.ndarray,
    y: np.ndarray,
    actions: np.ndarray,
) -> RunResult:
    z_arr = _ensure_float64_contiguous(z)
    y_arr = _ensure_float64_contiguous(y)
    T, d = z_arr.shape
    if actions.shape != (T + 1, d):
        raise ValueError("actions must have shape (T+1, d)")

    cum_loss = 0.0

    for t in range(T):
        x_t = actions[t]
        q = _dot(z_arr[t], x_t)
        cum_loss += _normalized_hinge(q, y_arr[t])

    comp_action = actions[T]
    comp_loss = _comparator_loss(z_arr, y_arr, comp_action)
    regret = cum_loss - comp_loss

    return RunResult(
        cum_loss=float(cum_loss),
        regret=float(regret),
        comp_loss=float(comp_loss),
        x_last=actions[T].copy(),
    )


def simulate(
    z: np.ndarray,
    y: np.ndarray,
    *,
    algo: Literal["ftrl", "ftl_exact"] = "ftl_exact",
    eta0: float = 1.0,
    norm: Literal["l2", "linf", "l1"] = "l2",
    solver: Optional[str] = None,
    solver_opts: Optional[dict] = None,
    ftl_solver: Optional[ExactFTLNoClip] = None,
    comparator_solver: Optional[ExactFTLNoClip] = None,
    prefix_actions: Optional[np.ndarray] = None,
    comparator_action: Optional[np.ndarray] = None,
) -> RunResult:
    """Unified front-end used by the convenience wrappers."""
    z_arr = _ensure_float64_contiguous(z)
    y_arr = _ensure_float64_contiguous(y)

    if algo == "ftl_exact":
        solver_obj = ftl_solver
        if solver_obj is None:
            T, d = z_arr.shape
            solver_obj = ExactFTLNoClip(
                d=d,
                T_max=T,
                norm=norm,
                solver=solver,
                solver_opts=solver_opts,
            )
        actions = prefix_actions
        if actions is None:
            actions = compute_prefix_actions(solver_obj, z_arr, y_arr)
        return replay_exact_ftl(z_arr, y_arr, actions)

    if algo == "ftrl":
        solver_for_comp = comparator_solver
        if solver_for_comp is None and comparator_action is None:
            T, d = z_arr.shape
            solver_for_comp = ExactFTLNoClip(
                d=d,
                T_max=T,
                norm=norm,
                solver=solver,
                solver_opts=solver_opts,
            )
        return _simulate_ftrl(
            z_arr,
            y_arr,
            eta0=eta0,
            comparator_action=comparator_action,
            comparator_solver=solver_for_comp,
            norm=norm,
            solver_name=solver,
            solver_opts=solver_opts,
        )

    raise ValueError("algo must be either 'ftrl' or 'ftl_exact'")


# ==============================================================
# Convenience wrappers
# ==============================================================

def run_ftrl(
    z: np.ndarray,
    y: np.ndarray,
    *,
    eta0: float = 1.0,
    norm: Literal["l2", "linf", "l1"] = "l2",
    solver: Optional[str] = None,
    solver_opts: Optional[dict] = None,
    comparator_solver: Optional[ExactFTLNoClip] = None,
    comparator_action: Optional[np.ndarray] = None,
) -> RunResult:
    return simulate(
        z,
        y,
        algo="ftrl",
        eta0=eta0,
        norm=norm,
        solver=solver,
        solver_opts=solver_opts,
        comparator_solver=comparator_solver,
        comparator_action=comparator_action,
    )


def run_ftl_exact(
    z: np.ndarray,
    y: np.ndarray,
    *,
    norm: Literal["l2", "linf", "l1"] = "l2",
    solver: Optional[str] = None,
    solver_opts: Optional[dict] = None,
    ftl_solver: Optional[ExactFTLNoClip] = None,
    prefix_actions: Optional[np.ndarray] = None,
    return_actions: bool = False,
) -> RunResult | Tuple[RunResult, np.ndarray]:
    z_arr = _ensure_float64_contiguous(z)
    y_arr = _ensure_float64_contiguous(y)

    solver_obj = ftl_solver
    if solver_obj is None:
        T, d = z_arr.shape
        solver_obj = ExactFTLNoClip(
            d=d,
            T_max=T,
            norm=norm,
            solver=solver,
            solver_opts=solver_opts,
        )

    actions = prefix_actions if prefix_actions is not None else compute_prefix_actions(solver_obj, z_arr, y_arr)
    result = replay_exact_ftl(z_arr, y_arr, actions)

    if return_actions:
        return result, actions
    return result


# ==============================================================
# Smoke test
# ==============================================================

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T, d = 200, 10
    z = rng.normal(size=(T, d))
    z /= np.maximum(1.0, np.linalg.norm(z, axis=1, keepdims=True))
    w_true = rng.normal(size=d)
    w_true /= np.linalg.norm(w_true)
    y = np.sign(z @ w_true + 0.2 * rng.normal(size=T))

    print("FTL exact (l2)...")
    res_ftl = run_ftl_exact(z, y, norm="l2", solver="ECOS")
    print(f"cum_loss={res_ftl.cum_loss:.4f}  regret={res_ftl.regret:.4f}  comp={res_ftl.comp_loss:.4f}")

    print("FTRL baseline...")
    res_ftrl = run_ftrl(z, y, eta0=1.0)
    print(f"cum_loss={res_ftrl.cum_loss:.4f}  regret={res_ftrl.regret:.4f}  comp={res_ftrl.comp_loss:.4f}")
