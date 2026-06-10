"""
metrics.py — Canonical metric functions for all experiments.

Functions defined here are imported everywhere; never redefine.
"""

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon
import logging

logger = logging.getLogger(__name__)


def feasibility_rate(solutions: list) -> float:
    """
    Calculate fraction of feasible solutions (no H_assign or H_overlap violations).

    Args:
        solutions: List of dicts with 'feasible' key (bool)

    Returns:
        Float in [0, 1]
    """
    if not solutions:
        return np.nan
    feasible_count = sum(1 for s in solutions if s.get("feasible", False))
    return feasible_count / len(solutions)


def rpd_vs_gurobi(obj_solver: float, obj_gurobi: float) -> float:
    """
    Relative Percent Deviation vs Gurobi optimum (absolute).

    Formula: RPD = 100 * (obj_solver - obj_gurobi) / obj_gurobi

    Args:
        obj_solver: Objective from solver (LH or SA)
        obj_gurobi: Optimal objective from Gurobi

    Returns:
        Percentage (positive = worse than optimal, 0 = optimal)
    """
    if obj_gurobi == 0 or np.isnan(obj_gurobi) or np.isinf(obj_gurobi):
        return np.nan
    if np.isnan(obj_solver) or np.isinf(obj_solver):
        return np.nan
    return 100.0 * (obj_solver - obj_gurobi) / obj_gurobi


def rpd_head_to_head(obj_lh: float, obj_sa: float) -> float:
    """
    RPD LeapHybrid vs SA (head-to-head comparison).

    Formula: RPD = 100 * (obj_LH - obj_SA) / obj_SA

    Args:
        obj_lh: LeapHybrid objective
        obj_sa: SA objective

    Returns:
        Percentage (negative = LH better than SA)
    """
    if obj_sa == 0 or np.isnan(obj_sa) or np.isinf(obj_sa):
        return np.nan
    if np.isnan(obj_lh) or np.isinf(obj_lh):
        return np.nan
    return 100.0 * (obj_lh - obj_sa) / obj_sa


def energy_gap(energy_feasible: float, energy_infeasible: float) -> float:
    """
    Gap between best feasible and best infeasible QUBO energy.

    Larger gap = clearer separation between feasible and infeasible regions.

    Args:
        energy_feasible: Best energy among feasible solutions
        energy_infeasible: Best energy among infeasible solutions

    Returns:
        Difference (gap > 0 is good for separability)
    """
    if np.isnan(energy_feasible) or np.isnan(energy_infeasible):
        return np.nan
    return energy_feasible - energy_infeasible


def q_density(n_vars: int, n_offdiag_nonzero: int) -> float:
    """
    Density of QUBO matrix Q (fraction of off-diagonal couplers present).

    Formula: density = n_offdiag_nonzero / (n_vars * (n_vars - 1))

    Args:
        n_vars: Number of variables
        n_offdiag_nonzero: Number of non-zero off-diagonal entries

    Returns:
        Density in [0, 1]
    """
    if n_vars <= 1:
        return 0.0
    max_offdiag = n_vars * (n_vars - 1)
    if max_offdiag == 0:
        return 0.0
    return n_offdiag_nonzero / max_offdiag


def mann_whitney_u_test(sample1: list, sample2: list) -> dict:
    """
    Mann-Whitney U test (independent samples).

    Args:
        sample1, sample2: Lists of numeric values

    Returns:
        Dict with keys: u_statistic, p_value, effect_size
    """
    try:
        u_stat, p_val = mannwhitneyu(sample1, sample2, alternative="two-sided")
        n1, n2 = len(sample1), len(sample2)
        # Effect size: r = Z / sqrt(N)
        from scipy.stats import norm
        z_score = norm.ppf(1 - p_val / 2)  # Approximate
        r = z_score / np.sqrt(n1 + n2) if (n1 + n2) > 0 else np.nan
        return {
            "u_statistic": u_stat,
            "p_value": p_val,
            "effect_size_r": r,
        }
    except Exception as e:
        logger.warning(f"Mann-Whitney U test failed: {e}")
        return {"u_statistic": np.nan, "p_value": np.nan, "effect_size_r": np.nan}


def wilcoxon_test(sample1: list, sample2: list) -> dict:
    """
    Wilcoxon signed-rank test (paired samples).

    Args:
        sample1, sample2: Lists of paired numeric values

    Returns:
        Dict with keys: statistic, p_value
    """
    try:
        # Paired: difference = sample1 - sample2
        diff = np.array(sample1) - np.array(sample2)
        stat, p_val = wilcoxon(diff, alternative="two-sided")
        return {
            "statistic": stat,
            "p_value": p_val,
        }
    except Exception as e:
        logger.warning(f"Wilcoxon test failed: {e}")
        return {"statistic": np.nan, "p_value": np.nan}
