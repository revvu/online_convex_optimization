"""
Sequence Generation for Online Convex Optimization

This module contains functions for generating various types of sequences
and stream builders for online learning experiments:

- Random i.i.d. sequences
- Noisy i.i.d. sequences (Massart noise)
- Label flip sequences
- Switching two-leaders sequences
- Stream builders for reproducible experiments
"""

from __future__ import annotations
from typing import Callable, Dict, Tuple
import numpy as np
from algorithms import _rng


# ==============================================================
# Adversarial families used for g(T)
# ==============================================================

def flip_sequence(T: int, d: int = 5):
    z = np.zeros((T, d), dtype=np.float32); z[:, 0] = 1.0
    y = np.array([1.0 if t % 2 else -1.0 for t in range(1, T + 1)], dtype=np.float32)
    u = np.zeros(d, dtype=np.float32)
    return z, y, u

# ==============================================================
# Two leaders (switching, no drift)
#   Alternate fixed-length blocks of +1 and -1: +1…+1, -1…-1, +1…+1, ...
#   No growing lead; tests adaptation to leader switches without contrived drift.
# ==============================================================

def switching_two_leaders_sequence(T: int, *, block_len: int = 20, d: int = 5):
    y = np.empty(T, dtype=np.float32)
    sign = 1.0
    idx = 0
    while idx < T:
        run = min(block_len, T - idx)
        y[idx:idx+run] = sign
        idx += run
        sign = -sign
    z = np.zeros((T, d), dtype=np.float32); z[:, 0] = 1.0
    u = np.zeros(d, dtype=np.float32)
    return z, y, u


# ==============================================================
# Stream builders (fixed task per run; fresh sequences per T; replicates)
# ==============================================================

def make_random_iid_stream(*, d: int = 5, run_seed: int = 0):
    gen_u = _rng(run_seed, 0, 11)
    u = gen_u.standard_normal(d).astype(np.float32, copy=False)
    n = float(np.linalg.norm(u))
    if n > 0:
        u /= n

    def sample(T: int, rep: int = 0):
        gen = _rng(run_seed, T, 13 + rep)
        z = gen.standard_normal((T, d)).astype(np.float32, copy=False)
        norms = np.linalg.norm(z, axis=1, keepdims=True).astype(np.float32, copy=False)
        np.maximum(norms, 1.0, out=norms)
        z *= (1.0 / norms)
        y = np.sign(z @ u).astype(np.float32, copy=False)
        y[y == 0.0] = 1.0
        return z, y, u
    return sample

def make_noisy_iid_stream(*, p: float, d: int = 5, run_seed: int = 0):
    gen_u = _rng(run_seed, 0, 21)
    u = gen_u.standard_normal(d).astype(np.float32, copy=False)
    n = float(np.linalg.norm(u))
    if n > 0:
        u /= n

    def sample(T: int, rep: int = 0):
        gen = _rng(run_seed, T, 23 + rep)
        z = gen.standard_normal((T, d)).astype(np.float32, copy=False)
        norms = np.linalg.norm(z, axis=1, keepdims=True).astype(np.float32, copy=False)
        np.maximum(norms, 1.0, out=norms)
        z *= (1.0 / norms)
        y = np.sign(z @ u).astype(np.float32, copy=False)
        y[y == 0.0] = 1.0
        flips = gen.random(T) < p
        y[flips] *= -1.0
        return z, y, u
    return sample

def make_flip_stream(*, d: int = 5, run_seed: int = 0):
    def sample(T: int, rep: int = 0):
        return flip_sequence(T, d=d)
    return sample

def make_switching_two_leaders_stream(*, block_len: int = 20, d: int = 5, run_seed: int = 0):
    def sample(T: int, rep: int = 0):
        return switching_two_leaders_sequence(T, block_len=block_len, d=d)
    return sample


# --- CASE SET (builders) ---
CASES: Dict[str, Callable[..., Callable[[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]]]] = {
    "Random i.i.d. (separable)":              lambda *, run_seed: make_random_iid_stream(d=5, run_seed=run_seed),
    "Massart noise 10%":             lambda *, run_seed: make_noisy_iid_stream(p=0.10, d=5, run_seed=run_seed),
    "Label flips":               lambda *, run_seed: make_flip_stream(d=5, run_seed=run_seed),
    "Switching leaders":      lambda *, run_seed: make_switching_two_leaders_stream(block_len=20, d=5, run_seed=run_seed),
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
