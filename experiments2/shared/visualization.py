"""
visualization.py — Plot functions (one per plot, return fig, never print/show).

Each function returns a matplotlib Figure object ready to save.
No plt.show() calls; user saves figures explicitly.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Set style for consistency
sns.set_style("whitegrid")
sns.color_palette("husl")


def plot_gurobi_wall(df: pd.DataFrame, n_star: Optional[int] = None) -> plt.Figure:
    """
    Gurobi solution time vs problem size N (log scale on y-axis).

    Args:
        df: DataFrame with columns ['N', 'wall_time_s', ...]
        n_star: Optional, mark classical wall with vertical line

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by N, calculate mean ± std
    grouped = df.groupby('N')['wall_time_s'].agg(['mean', 'std', 'count'])
    grouped['se'] = grouped['std'] / np.sqrt(grouped['count'])

    x = grouped.index
    y = grouped['mean']
    yerr = grouped['se']

    ax.errorbar(x, y, yerr=yerr, fmt='o-', capsize=5, label='Gurobi (mean ± SE)', linewidth=2)
    ax.set_yscale('log')
    ax.set_xlabel('Problem Size (N vessels)', fontsize=12)
    ax.set_ylabel('Wall Time (seconds)', fontsize=12)
    ax.set_title('Gurobi Scaling: Classical Wall Detection', fontsize=14, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)

    if n_star is not None:
        ax.axvline(x=n_star, color='red', linestyle='--', linewidth=2, label=f'N* = {n_star}')

    ax.legend()
    fig.tight_layout()
    return fig


def plot_mip_gap_vs_n(df: pd.DataFrame, n_star: Optional[int] = None) -> plt.Figure:
    """
    MIP gap (%) at time limit vs problem size N.

    Args:
        df: DataFrame with columns ['N', 'mip_gap_pct', ...]
        n_star: Optional, mark classical wall

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    grouped = df.groupby('N')['mip_gap_pct'].agg(['mean', 'std', 'count'])
    grouped['se'] = grouped['std'] / np.sqrt(grouped['count'])

    x = grouped.index
    y = grouped['mean']
    yerr = grouped['se']

    ax.errorbar(x, y, yerr=yerr, fmt='s-', capsize=5, label='MIP Gap (mean ± SE)', linewidth=2, color='orange')
    ax.axhline(y=10.0, color='red', linestyle='--', linewidth=1.5, label='Gap Threshold (10%)')
    ax.set_xlabel('Problem Size (N vessels)', fontsize=12)
    ax.set_ylabel('MIP Gap (%)', fontsize=12)
    ax.set_title('MIP Gap at Time Limit', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    if n_star is not None:
        ax.axvline(x=n_star, color='red', linestyle='--', linewidth=2, label=f'N* = {n_star}')

    ax.legend()
    fig.tight_layout()
    return fig


def plot_feasibility_heatmap(df: pd.DataFrame, size_label: str) -> plt.Figure:
    """
    Heatmap of feasibility_rate vs (alpha, beta) for a given size.

    Args:
        df: DataFrame with columns ['alpha', 'beta', 'feasible', ...]
            filtered to a single size
        size_label: Size label for title

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Pivot: rows=alpha, cols=beta, values=feasibility_rate
    pivot = df.groupby(['alpha', 'beta'])['feasible'].mean().unstack()

    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', vmin=0, vmax=1,
                cbar_kws={'label': 'Feasibility Rate'}, ax=ax)
    ax.set_title(f'Feasibility Rate vs (α, β) — Size {size_label}', fontsize=12, fontweight='bold')
    ax.set_xlabel('β', fontsize=11)
    ax.set_ylabel('α', fontsize=11)
    fig.tight_layout()
    return fig


def plot_feasibility_curve(df: pd.DataFrame) -> plt.Figure:
    """
    Feasibility rate vs alpha (line per size).

    Args:
        df: DataFrame with columns ['alpha', 'size_label', 'feasible', ...]

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for size in df['size_label'].unique():
        df_size = df[df['size_label'] == size]
        grouped = df_size.groupby('alpha')['feasible'].mean()
        ax.plot(grouped.index, grouped.values, marker='o', label=f'Size {size}', linewidth=2)

    ax.axhline(y=0.95, color='red', linestyle='--', linewidth=1.5, label='Target (95%)')
    ax.set_xlabel('α (Penalty Scale)', fontsize=12)
    ax.set_ylabel('Feasibility Rate', fontsize=12)
    ax.set_title('Feasibility vs α (Lagrange Calibration)', fontsize=14, fontweight='bold')
    ax.set_ylim([0.8, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_solution_quality_boxplot(df: pd.DataFrame, size_label: str) -> plt.Figure:
    """
    Box plots: objective value per solver for a given size.

    Args:
        df: DataFrame with columns ['solver', 'obj_value', 'feasible', ...]
            filtered to a single size and feasible=True
        size_label: Size label for title

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    df_feas = df[df['feasible'] == True]
    solvers = df_feas['solver'].unique()

    data_by_solver = [df_feas[df_feas['solver'] == s]['obj_value'].dropna().values for s in solvers]

    bp = ax.boxplot(data_by_solver, labels=solvers, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen', 'lightyellow']):
        patch.set_facecolor(color)

    ax.set_ylabel('Schedule Cost (Tardiness)', fontsize=12)
    ax.set_xlabel('Solver', fontsize=12)
    ax.set_title(f'Solution Quality — Size {size_label}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    return fig


def plot_rpd_vs_gurobi_by_size(df: pd.DataFrame) -> plt.Figure:
    """
    RPD (%) vs Gurobi optimum, aggregated by size (mean ± std).

    Args:
        df: DataFrame with columns ['N', 'size_label', 'solver', 'rpd_vs_gurobi', ...]

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    size_order = df['size_label'].unique()  # consistent x-axis regardless of which solvers are present

    for solver in df['solver'].unique():
        if solver == 'Gurobi':
            continue  # Skip Gurobi (RPD=0 trivially)

        df_solver = df[df['solver'] == solver]
        grouped = df_solver.groupby('size_label')['rpd_vs_gurobi'].agg(['mean', 'std', 'count'])
        grouped['se'] = grouped['std'] / np.sqrt(grouped['count'])

        x = range(len(grouped))
        ax.errorbar(x, grouped['mean'], yerr=grouped['se'], marker='o', label=solver, linewidth=2, capsize=5)

    ax.axhline(y=0.0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax.set_xticks(range(len(size_order)))
    ax.set_xticklabels(size_order)
    ax.set_xlabel('Problem Size', fontsize=12)
    ax.set_ylabel('RPD vs Gurobi (%)', fontsize=12)
    ax.set_title('Solution Quality: RPD vs Optimum', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    fig.tight_layout()
    return fig


def plot_feasibility_by_solver(df: pd.DataFrame) -> plt.Figure:
    """
    Feasibility rate vs N (problem size) by solver.

    Args:
        df: DataFrame with columns ['N', 'solver', 'feasible', ...]

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for solver in df['solver'].unique():
        df_solver = df[df['solver'] == solver]
        grouped = df_solver.groupby('N')['feasible'].mean()
        ax.plot(grouped.index, grouped.values, marker='o', label=solver, linewidth=2)

    ax.set_xlabel('Problem Size (N vessels)', fontsize=12)
    ax.set_ylabel('Feasibility Rate', fontsize=12)
    ax.set_title('Feasibility by Solver vs Problem Size', fontsize=14, fontweight='bold')
    ax.set_ylim([0.7, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_energy_gap_by_size(df: pd.DataFrame) -> plt.Figure:
    """
    Energy gap (separation feasible/infeasible) by size (LH only).

    Args:
        df: DataFrame with columns ['size_label', 'energy_gap', ...]
            (LeapHybrid results only)

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    grouped = df.groupby('size_label')['energy_gap'].agg(['mean', 'std', 'count'])
    grouped['se'] = grouped['std'] / np.sqrt(grouped['count'])

    x = range(len(grouped))
    ax.bar(x, grouped['mean'], yerr=grouped['se'], capsize=5, alpha=0.7, color='skyblue')
    ax.set_xticks(x)
    ax.set_xticklabels(grouped.index)
    ax.set_ylabel('Energy Gap (feasible - infeasible)', fontsize=12)
    ax.set_xlabel('Problem Size', fontsize=12)
    ax.set_title('QUBO Energy Gap (LeapHybrid) — Separability Indicator', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    return fig


def plot_convergence_tardiness(df: pd.DataFrame, size_label: str, solver: str) -> plt.Figure:
    """
    Tardiness vs iteration (left y-axis).

    Args:
        df: DataFrame with columns ['k', 'size_label', 'solver', 'total_weighted_tardiness', ...]
            filtered to size_label and solver
        size_label: Size label
        solver: Solver name

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    df_filt = df[(df['size_label'] == size_label) & (df['solver'] == solver)]
    grouped = df_filt.groupby('k')['total_weighted_tardiness'].mean()

    ax.plot(grouped.index, grouped.values, marker='o', linewidth=2, color='blue', label='Tardiness')
    ax.set_xlabel('Iteration (k)', fontsize=12)
    ax.set_ylabel('Total Weighted Tardiness', fontsize=12, color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Convergence: Tardiness vs Iteration — {size_label} ({solver})',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_convergence_violations(df: pd.DataFrame, size_label: str, solver: str) -> plt.Figure:
    """
    Number of violations vs iteration (right y-axis).

    Args:
        df: DataFrame with columns ['k', 'size_label', 'solver', 'n_violations', ...]
        size_label: Size label
        solver: Solver name

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    df_filt = df[(df['size_label'] == size_label) & (df['solver'] == solver)]
    grouped = df_filt.groupby('k')['n_violations'].mean()

    ax.plot(grouped.index, grouped.values, marker='s', linewidth=2, color='red', label='Violations')
    ax.set_xlabel('Iteration (k)', fontsize=12)
    ax.set_ylabel('Number of Violations', fontsize=12, color='red')
    ax.tick_params(axis='y', labelcolor='red')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Convergence: Violations vs Iteration — {size_label} ({solver})',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_q_density_stability(df: pd.DataFrame, size_label: str) -> plt.Figure:
    """
    Q matrix density vs iteration k (with ±2% tolerance band).

    Args:
        df: DataFrame with columns ['k', 'size_label', 'q_density_k', ...]

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    df_filt = df[df['size_label'] == size_label]
    grouped = df_filt.groupby('k')['q_density_k'].mean()

    # Get baseline (k=0)
    baseline = grouped.iloc[0] if len(grouped) > 0 else 0
    tolerance_zone_upper = baseline * 1.02
    tolerance_zone_lower = baseline * 0.98

    ax.plot(grouped.index, grouped.values, marker='o', linewidth=2, color='green', label='Q Density')
    ax.axhline(y=baseline, color='gray', linestyle='--', linewidth=1, label='Baseline (k=0)')
    ax.fill_between(grouped.index, tolerance_zone_lower, tolerance_zone_upper, alpha=0.2, color='green',
                     label='Tolerance ±2%')
    ax.set_xlabel('Iteration (k)', fontsize=12)
    ax.set_ylabel('Q Density', fontsize=12)
    ax.set_title(f'Q Matrix Density Stability — {size_label}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_convergence_iterations_by_size(df: pd.DataFrame) -> plt.Figure:
    """
    Iterations to convergence (k_conv) by size and solver (bar plot).

    Args:
        df: DataFrame with columns ['size_label', 'solver', 'converged']
            (or explicit k_conv column)

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    sizes = sorted(df['size_label'].unique())
    solvers = df['solver'].unique()

    x = np.arange(len(sizes))
    width = 0.35

    for i, solver in enumerate(solvers):
        df_solver = df[df['solver'] == solver]
        grouped = df_solver.groupby('size_label')['k_conv' if 'k_conv' in df.columns else 'k'].mean()
        values = [grouped.get(s, 0) for s in sizes]
        ax.bar(x + i * width, values, width, label=solver, alpha=0.8)

    ax.set_ylabel('Iterations to Convergence', fontsize=12)
    ax.set_xlabel('Problem Size', fontsize=12)
    ax.set_title('Hybrid Loop Convergence Speed', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(sizes)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    return fig


def plot_scaling_crossover(df_gurobi: pd.DataFrame, df_lh: pd.DataFrame,
                          n_star: int, n_star_star: Optional[float] = None,
                          ic_lower: Optional[float] = None, ic_upper: Optional[float] = None) -> plt.Figure:
    """
    CENTRAL FIGURE: Gurobi + LeapHybrid scaling curves with crossover projection.

    Args:
        df_gurobi: Gurobi data ['N', 'wall_time_s']
        df_lh: LeapHybrid data ['N', 'wall_time_s']
        n_star: Classical wall position
        n_star_star: Projected crossover point
        ic_lower, ic_upper: Confidence interval for crossover

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Visual x-limit: data range × 1.5, never driven by extrapolated IC
    x_max_data = df_gurobi['N'].max() if not df_gurobi.empty else 100
    x_min_data = df_gurobi['N'].min() if not df_gurobi.empty else 0
    x_lim = x_max_data * 1.5
    data_range = x_max_data - x_min_data if x_max_data > x_min_data else x_max_data

    # Gurobi: mean ± std
    if not df_gurobi.empty:
        grouped_gurobi = df_gurobi.groupby('N')['wall_time_s'].agg(['mean', 'std', 'count'])
        grouped_gurobi['se'] = grouped_gurobi['std'] / np.sqrt(grouped_gurobi['count'])
        x_gurobi = grouped_gurobi.index
        y_gurobi = grouped_gurobi['mean']
        yerr_gurobi = grouped_gurobi['se']
        ax.errorbar(x_gurobi, y_gurobi, yerr=yerr_gurobi, fmt='o-', capsize=5,
                    label='Gurobi', linewidth=2.5, markersize=8)

    # LeapHybrid: constant line (mean ± std) extended to visual x_lim
    if not df_lh.empty:
        mean_lh = df_lh['wall_time_s'].mean()
        std_lh = df_lh['wall_time_s'].std()
        x_range = np.linspace(x_min_data, x_lim, 100)
        ax.plot(x_range, [mean_lh] * len(x_range), 'g-', linewidth=2.5, label='LeapHybrid')
        ax.fill_between(x_range, mean_lh - std_lh, mean_lh + std_lh, alpha=0.2, color='green')

    # Classical wall
    ax.axvline(x=n_star, color='red', linestyle='--', linewidth=2, label=f'Classical Wall (N* = {n_star})')

    # Crossover — annotate real value in label even if IC is clipped visually
    if n_star_star is not None:
        ax.axvline(x=n_star_star, color='purple', linestyle=':', linewidth=2,
                   label=f'Crossover (N** ≈ {n_star_star:.1f})')
        if ic_lower is not None and ic_upper is not None:
            ic_width = ic_upper - ic_lower
            # Only draw span if IC is reasonably tight (< 10× data range); otherwise just annotate label
            if ic_width < 10 * data_range:
                ic_vis_low = max(ic_lower, 0)
                ic_vis_high = min(ic_upper, x_lim)
                ax.axvspan(ic_vis_low, ic_vis_high, alpha=0.15, color='purple',
                           label=f'Crossover IC [{ic_lower:.1f}, {ic_upper:.1f}]')
            else:
                # IC too wide to be meaningful visually — show in legend only
                ax.plot([], [], color='purple', alpha=0.3, linewidth=8,
                        label=f'Crossover IC [{ic_lower:.1f}, {ic_upper:.1f}] (extrapolado)')

    ax.set_xlim(left=max(0, x_min_data * 0.9), right=x_lim)
    ax.set_yscale('log')
    ax.set_xlabel('Problem Size (N vessels)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Wall Time (seconds)', fontsize=13, fontweight='bold')
    ax.set_title('Scaling & Crossover: Gurobi vs LeapHybrid', fontsize=15, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=10, loc='best')
    fig.tight_layout()
    return fig


def plot_qubo_sensitivity_curve(df: pd.DataFrame) -> plt.Figure:
    """
    RPD vs QUBO size (N_vars) for fixed N but varying horizon/density.

    Args:
        df: DataFrame with columns ['n_vars', 'solver', 'rpd_vs_gurobi', ...]

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for solver in df['solver'].unique():
        df_solver = df[df['solver'] == solver]
        grouped = df_solver.groupby('n_vars')['rpd_vs_gurobi'].agg(['mean', 'std', 'count'])
        grouped['se'] = grouped['std'] / np.sqrt(grouped['count'])

        x = grouped.index
        ax.errorbar(x, grouped['mean'], yerr=grouped['se'], marker='o', label=solver, linewidth=2, capsize=5)

    ax.set_xlabel('QUBO Size (Number of Variables)', fontsize=12)
    ax.set_ylabel('RPD vs Gurobi (%)', fontsize=12)
    ax.set_title('QUBO Size Sensitivity (N=8 fixed, ρ variable)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig
