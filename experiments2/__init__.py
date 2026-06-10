"""
experiments2 — Hybrid Quantum-Classical Experiments v6.0

Modular framework for executing the 6-experiment pipeline:
  1. exp01_classical_wall.py     — Gurobi stress test (find N*)
  2. exp02_lagrange_calibration.py — SA sweep + LH validation (find α*, β*)
  3. exp03_solution_quality.py    — LH vs Gurobi vs SA (quality comparison)
  4. exp04_scaling_crossover.py   — Regression + crossover projection (N**)
  5. exp05_hybrid_convergence.py  — Iterative loop stability (k_conv, q_density)
  6. exp06_qubo_sensitivity.py   — QUBO size effects (optional)

Design principles:
  - All instances stored in data/instances.xlsx (generate once, reuse everywhere)
  - Results saved to results/{gurobi,SA,LeapHybrid}/*.xlsx
  - Flexible cell structure: N compute cells + 1 cell per plot
  - No modifications to src/config.py
"""

__version__ = "0.6.0"
