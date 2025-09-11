"""
Online Convex Optimization Algorithms

This module contains implementations of various online learning algorithms:
- FTRL (Follow the Regularized Leader)
- FTL (Follow the Leader) 
- SMART (Switching algorithm)
- Supporting utilities and comparators

"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
from tqdm import tqdm


def _project_to_ball(x: np.ndarray, R: float) -> None:
    n = np.linalg.norm(x)
    if n > R and n > 0.0:
        x *= R / n

def _normalized_hinge(q: float, y: float) -> float:
    # y in {1, -1}, q in [-1,1]; normalized hinge loss
    return 0.5 * abs(q - y)

def _action_ftl(theta: np.ndarray, R: float, out: np.ndarray) -> None:
    n = np.linalg.norm(theta)
    s = 0.0 if n == 0.0 else -R / n
    for i in range(theta.size):
        out[i] = s * theta[i]

def _action_ftrl(theta: np.ndarray, t: int, R: float, eta0: float, out: np.ndarray) -> None:
    eta_t = eta0 / math.sqrt(max(1.0, float(t)))
    out[:] = -eta_t * theta
    _project_to_ball(out, R)


# ==============================================================
# Online simulation (FTL, FTRL), regret vs comparator
# ==============================================================

def simulate_alg(z: np.ndarray,
                 y: np.ndarray,
                 u_comp: np.ndarray,
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

        q = np.clip(np.dot(z[t], x_t), -1.0, 1.0)
        y_t = float(y[t])
        cum_loss += _normalized_hinge(q, y_t)

        diff = q - y_t
        grad_q = 0.5 if diff > 0.0 else -0.5 if diff < 0.0 else 0.0
        theta += grad_q * z[t]

    # Calculate comparator loss using comparator_ftl_hindsight
    _, comp_loss = comparator_ftl_hindsight(z, y, R)
    
    # Return regret (algorithm loss - comparator loss)
    return cum_loss - comp_loss


# ==============================================================
# Hindsight comparator: best fixed linear predictor
# ==============================================================

def comparator_ftl_hindsight(z: np.ndarray, y: np.ndarray, R: float) -> tuple[np.ndarray, float]:
    T, d = z.shape
    # Vectorized computation: s = sum(y[t] * z[t] for t in range(T))
    s = np.sum(y[:, np.newaxis] * z, axis=0).astype(np.float32)
    n = np.linalg.norm(s)
    if n > 0.0:
        s *= R / n
    
    # Calculate cumulative loss of playing strategy s throughout the sequence
    # Vectorized computation for speed
    q_all = np.clip(z @ s, -1.0, 1.0)  # All dot products at once
    cum_loss = np.sum(0.5 * np.abs(q_all - y))  # Vectorized loss calculation
    
    return s, cum_loss


# ==============================================================
# SMART (single switch) with proper reset of ALG_WC
# ==============================================================


def _compute_gradient(q_pred: float, y_true: float) -> float:
    """Compute gradient for normalized hinge loss."""
    diff = q_pred - y_true
    return 0.5 if diff > 0.0 else -0.5 if diff < 0.0 else 0.0

def _comparator_loss_up_to_t(z: np.ndarray, y: np.ndarray, R: float, t: int) -> float:
    """Compute comparator loss up to time t (inclusive)."""
    if t == 0:
        return 0.0
    
    # Compute best fixed strategy up to time t
    s = np.sum(y[:t+1, np.newaxis] * z[:t+1], axis=0).astype(np.float32)
    n = np.linalg.norm(s)
    if n > 0.0:
        s *= R / n
    
    # Compute cumulative loss of strategy s up to time t
    q_all = np.clip(z[:t+1] @ s, -1.0, 1.0)
    return np.sum(0.5 * np.abs(q_all - y[:t+1]))

def simulate_SMART_like(z: np.ndarray,
                        y: np.ndarray,
                        u_comp: np.ndarray,
                        theta_thresh: float,
                        R: float,
                        eta0: float) -> float:
    """
    Simulate SMART-like algorithm with single switch from FTL to FTRL.
    
    The algorithm starts with FTL and switches to FTRL when FTL loss
    exceeds the threshold. After switching, FTRL parameters are reset.
    """
    T, d = z.shape
    
    # Initialize algorithm parameters
    theta_ftl = np.zeros(d, dtype=np.float32)   # FTL parameter vector
    theta_ftrl = np.zeros(d, dtype=np.float32)  # FTRL parameter vector
    x_ftl = np.zeros(d, dtype=np.float32)       # FTL action vector
    x_ftrl = np.zeros(d, dtype=np.float32)      # FTRL action vector
    has_switched = False                        # Switch flag
    
    # Initialize loss tracking
    ftl_loss = 0.0
    total_loss = 0.0
    comp_loss_so_far = 0.0  # Comparator loss up to current time
    
    # Main algorithm loop
    for t in range(T):
        # Compute actions for both algorithms
        _action_ftl(theta_ftl, R, x_ftl)
        _action_ftrl(theta_ftrl, t + 1, R, eta0, x_ftrl)
        
        # Select action based on switch status
        current_action = x_ftl if not has_switched else x_ftrl
        
        # Compute prediction and loss
        prediction_dot = np.dot(z[t], current_action)
        prediction = np.clip(prediction_dot, -1.0, 1.0)
        
        total_loss += _normalized_hinge(prediction, float(y[t]))
        
        # Compute gradient and update parameters
        gradient = _compute_gradient(prediction, float(y[t]))
        
        # Vectorized gradient update
        gradient_vector = gradient * z[t]
        theta_ftl += gradient_vector
        if has_switched:
            theta_ftrl += gradient_vector
        
        # Check for switch condition (only if not already switched)
        if not has_switched:
            # Compute FTL prediction for loss calculation
            ftl_dot = np.dot(z[t], x_ftl)
            ftl_prediction = np.clip(ftl_dot, -1.0, 1.0)
            ftl_loss += _normalized_hinge(ftl_prediction, float(y[t]))
            
            # Compute comparator loss up to time t
            comp_loss_so_far = _comparator_loss_up_to_t(z, y, R, t)
            
            # Switch to FTRL if FTL regret exceeds threshold
            ftl_regret = ftl_loss - comp_loss_so_far
            if ftl_regret >= theta_thresh:
                has_switched = True
                # Reset FTRL parameters after switch
                theta_ftrl.fill(0.0)
    
    # Calculate final comparator loss for return value
    _, final_comp_loss = comparator_ftl_hindsight(z, y, R)
    
    # Return regret (algorithm loss - comparator loss)
    return total_loss - final_comp_loss


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
            u_star, _ = comparator_ftl_hindsight(z, y, R)
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
