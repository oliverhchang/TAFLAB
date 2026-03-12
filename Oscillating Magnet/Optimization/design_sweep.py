"""
design_sweep.py
---------------
Parameter sweeps and Optimization for the WEC design.

Three levels of search:
    1. 1D / 2D parameter sweeps   — understand sensitivity of each parameter
    2. Grid search                 — brute-force over a coarse design space
    3. Gradient-free Optimization  — scipy differential evolution / Nelder-Mead
                                     for finding the true optimum

All sweeps use the fast frequency-domain evaluator from objective.py.
The best design found is then verified with the full time-domain pipeline.

Typical workflow
----------------
    1. Run sensitivity_sweep()   to understand which parameters matter most
    2. Run grid_search_2d()      on the top 2 parameters (k_eff, magnet_diameter)
    3. Run optimize()            to find the global optimum
    4. Run verify_optimum()      to confirm with full time-domain simulation
"""

import numpy as np
import copy
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import warnings
warnings.filterwarnings("ignore")

# Add parent directory to path for imports
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import WECConfig
from optimization.objective import (
    objective, objective_result, ObjectiveResult,
    vector_to_config, config_to_vector, default_design_vector,
    DESIGN_BOUNDS, PARAM_NAMES, N_PARAMS,
    normalized_to_si, si_to_normalized, fast_power_estimate,
    print_design_bounds,
)


# ---------------------------------------------------------------------------
# Output containers
# ---------------------------------------------------------------------------
@dataclass
class SweepResult1D:
    param_name: str
    param_values: np.ndarray
    P_avg_mW: np.ndarray
    f_n_Hz: np.ndarray
    feasible: np.ndarray
    best_value: float
    best_P_avg_mW: float


@dataclass
class SweepResult2D:
    param_x: str
    param_y: str
    x_values: np.ndarray
    y_values: np.ndarray
    P_grid: np.ndarray          # shape (n_y, n_x) in mW
    feasibility_grid: np.ndarray
    best_x: float
    best_y: float
    best_P_mW: float


@dataclass
class OptimizationResult:
    method: str
    x_opt: np.ndarray           # optimal design vector (SI)
    P_avg_opt_mW: float         # optimal average power (mW)
    n_evaluations: int
    elapsed_s: float
    convergence_history: List[float]   # objective value per iteration
    final_result: ObjectiveResult      # full result at optimum
    verified_P_avg_mW: Optional[float] = None  # from full time-domain check


# ---------------------------------------------------------------------------
# 1D sensitivity sweep
# ---------------------------------------------------------------------------
def sensitivity_sweep(
    base_cfg: WECConfig,
    param_name: str,
    n_points: int = 40,
    Gamma_rms: float = 0.05,
) -> SweepResult1D:
    """
    Sweep one parameter while holding all others at their default values.
    Shows how P_avg responds to that parameter alone.
    """
    assert param_name in DESIGN_BOUNDS, f"Unknown parameter: {param_name}"

    lo, hi, desc, unit = DESIGN_BOUNDS[param_name]
    param_idx  = PARAM_NAMES.index(param_name)
    x0         = default_design_vector(base_cfg)
    values     = np.linspace(lo, hi, n_points)

    P_arr      = np.zeros(n_points)
    fn_arr     = np.zeros(n_points)
    feasible   = np.zeros(n_points, dtype=bool)

    for i, v in enumerate(values):
        x = x0.copy()
        x[param_idx] = v
        try:
            res = objective_result(x, base_cfg, mode="fast", Gamma_rms=Gamma_rms)
            P_arr[i]    = res.P_avg_mW if res.feasible else np.nan
            fn_arr[i]   = res.f_n_Hz
            feasible[i] = res.feasible
        except Exception:
            P_arr[i] = np.nan

    valid = ~np.isnan(P_arr)
    if valid.any():
        best_idx = np.nanargmax(P_arr)
        best_val = values[best_idx]
        best_P   = P_arr[best_idx]
    else:
        best_val = x0[param_idx]
        best_P   = 0.0

    return SweepResult1D(
        param_name=param_name,
        param_values=values,
        P_avg_mW=P_arr,
        f_n_Hz=fn_arr,
        feasible=feasible,
        best_value=best_val,
        best_P_avg_mW=best_P,
    )


def run_all_sensitivity_sweeps(
    base_cfg: WECConfig,
    Gamma_rms: float = 0.05,
    save_path: Optional[str] = None,
) -> Dict[str, SweepResult1D]:
    """Run 1D sensitivity sweep for every parameter and plot results."""
    print("\n--- Running sensitivity sweeps for all parameters ---")
    results = {}

    for name in PARAM_NAMES:
        print(f"  Sweeping {name}...", end=" ", flush=True)
        res = sensitivity_sweep(base_cfg, name, n_points=40, Gamma_rms=Gamma_rms)
        results[name] = res
        print(f"best = {res.best_value:.4g}  → P = {res.best_P_avg_mW:.3f} mW")

    plot_sensitivity_sweeps(results, base_cfg, save_path=save_path)
    return results


# ---------------------------------------------------------------------------
# 2D grid search
# ---------------------------------------------------------------------------
def grid_search_2d(
    base_cfg: WECConfig,
    param_x: str,
    param_y: str,
    n_x: int = 30,
    n_y: int = 30,
    Gamma_rms: float = 0.05,
) -> SweepResult2D:
    """
    2D grid search over two parameters.
    All other parameters held at default values.
    """
    assert param_x in DESIGN_BOUNDS and param_y in DESIGN_BOUNDS

    lox, hix = DESIGN_BOUNDS[param_x][:2]
    loy, hiy = DESIGN_BOUNDS[param_y][:2]
    idx_x = PARAM_NAMES.index(param_x)
    idx_y = PARAM_NAMES.index(param_y)

    x0      = default_design_vector(base_cfg)
    x_vals  = np.linspace(lox, hix, n_x)
    y_vals  = np.linspace(loy, hiy, n_y)

    P_grid   = np.full((n_y, n_x), np.nan)
    feas_grid = np.zeros((n_y, n_x), dtype=bool)

    total = n_x * n_y
    done  = 0
    t0    = time.time()

    for j, yv in enumerate(y_vals):
        for i, xv in enumerate(x_vals):
            x = x0.copy()
            x[idx_x] = xv
            x[idx_y] = yv
            try:
                res = objective_result(x, base_cfg, mode="fast", Gamma_rms=Gamma_rms)
                if res.feasible:
                    P_grid[j, i]    = res.P_avg_mW
                    feas_grid[j, i] = True
            except Exception:
                pass
            done += 1

        # Progress
        elapsed = time.time() - t0
        rate    = done / elapsed if elapsed > 0 else 1
        eta     = (total - done) / rate
        print(f"  {done}/{total} ({done/total*100:.0f}%)  ETA: {eta:.0f}s", end="\r")

    print(f"\n  Done in {time.time()-t0:.1f}s")

    best_flat = np.nanargmax(P_grid)
    best_j, best_i = np.unravel_index(best_flat, P_grid.shape)

    return SweepResult2D(
        param_x=param_x, param_y=param_y,
        x_values=x_vals, y_values=y_vals,
        P_grid=P_grid, feasibility_grid=feas_grid,
        best_x=x_vals[best_i], best_y=y_vals[best_j],
        best_P_mW=float(np.nanmax(P_grid)),
    )


# ---------------------------------------------------------------------------
# Gradient-free Optimization
# ---------------------------------------------------------------------------
def optimize(
    base_cfg: WECConfig,
    method: str = "differential_evolution",
    max_evals: int = 500,
    Gamma_rms: float = 0.05,
    seed: int = 42,
    verbose: bool = True,
) -> OptimizationResult:
    """
    Optimize the WEC design using gradient-free methods.

    Methods
    -------
    "differential_evolution" : global optimizer, best for multimodal problems
    "nelder_mead"            : local optimizer, fast, good for refinement
    "dual_annealing"         : simulated annealing variant, good global search

    All methods operate on the NORMALIZED [0,1] design space.
    """
    from scipy.optimize import differential_evolution, minimize, dual_annealing

    history = []
    n_evals = [0]
    t_start = time.time()

    def _obj_normalized(x_norm):
        x_si = normalized_to_si(x_norm)
        val  = objective(x_si, base_cfg, mode="fast",
                        Gamma_rms=Gamma_rms, penalty_weight=10.0)
        history.append(-val * 1000)   # store as P_avg mW (positive)
        n_evals[0] += 1
        if verbose and n_evals[0] % 50 == 0:
            best_so_far = max(history) if history else 0.0
            print(f"    eval {n_evals[0]:4d}  best P_avg = {best_so_far:.3f} mW")
        return val

    bounds_norm = [(0.0, 1.0)] * N_PARAMS

    if verbose:
        print(f"\n  Optimizing with {method} (max {max_evals} evals)...")

    if method == "differential_evolution":
        result = differential_evolution(
            _obj_normalized,
            bounds=bounds_norm,
            maxiter=max_evals // (15 * N_PARAMS) + 1,
            popsize=15,
            seed=seed,
            tol=1e-6,
            polish=True,
        )
        x_opt_norm = result.x

    elif method == "nelder_mead":
        x0_norm = si_to_normalized(default_design_vector(base_cfg))
        result  = minimize(
            _obj_normalized, x0_norm, method="Nelder-Mead",
            options={"maxfev": max_evals, "xatol": 1e-5, "fatol": 1e-8},
        )
        x_opt_norm = result.x

    elif method == "dual_annealing":
        result = dual_annealing(
            _obj_normalized,
            bounds=bounds_norm,
            maxfun=max_evals,
            seed=seed,
        )
        x_opt_norm = result.x

    else:
        raise ValueError(f"Unknown method: {method}")

    x_opt_si   = normalized_to_si(np.clip(x_opt_norm, 0, 1))
    final_res  = objective_result(x_opt_si, base_cfg, mode="fast",
                                  Gamma_rms=Gamma_rms)
    elapsed    = time.time() - t_start

    if verbose:
        print(f"\n  Optimization complete in {elapsed:.1f}s ({n_evals[0]} evals)")
        print(f"  Optimal design:")
        print(final_res.summary())
        print("\n  Parameter values at optimum:")
        for name, val in final_res.params.items():
            lo, hi, _, unit = DESIGN_BOUNDS[name]
            print(f"    {name:<20} = {val:.4g} {unit}")

    return OptimizationResult(
        method=method,
        x_opt=x_opt_si,
        P_avg_opt_mW=final_res.P_avg_mW,
        n_evaluations=n_evals[0],
        elapsed_s=elapsed,
        convergence_history=history,
        final_result=final_res,
    )


# ---------------------------------------------------------------------------
# Verify optimum with full time-domain simulation
# ---------------------------------------------------------------------------
def verify_optimum(
    opt_result: OptimizationResult,
    base_cfg: WECConfig,
    wave_mode: str = "jonswap",
) -> OptimizationResult:
    """
    Verify the optimized design with the full nonlinear time-domain pipeline.
    Replaces the fast estimate with the true physics-based P_avg.
    """
    from optimization.objective import full_power_evaluation

    print("\n--- Verifying optimum with full time-domain simulation ---")
    cfg = vector_to_config(opt_result.x_opt, base_cfg)

    P_avg, P_peak, dyn, em = full_power_evaluation(cfg, wave_mode=wave_mode)

    print(f"  Fast estimate  P_avg = {opt_result.P_avg_opt_mW:.3f} mW")
    print(f"  Full sim       P_avg = {P_avg*1000:.3f} mW")
    print(f"  Full sim       P_peak = {P_peak*1000:.3f} mW")

    opt_result.verified_P_avg_mW = P_avg * 1000
    return opt_result


# ---------------------------------------------------------------------------
# Pareto front: power vs. magnet mass
# ---------------------------------------------------------------------------
def pareto_sweep(
    base_cfg: WECConfig,
    Gamma_rms: float = 0.05,
    n_mass_points: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Trace the Pareto front of Power vs. Magnet Mass.
    For each mass level, optimize remaining parameters to maximize power.

    Returns (mass_g_array, P_avg_mW_array)
    """
    from scipy.optimize import minimize

    print("\n--- Computing Pareto front: Power vs. Magnet Mass ---")

    d_range = np.linspace(
        DESIGN_BOUNDS["magnet_diameter"][0],
        DESIGN_BOUNDS["magnet_diameter"][1],
        n_mass_points,
    )

    masses   = []
    P_values = []
    idx_d    = PARAM_NAMES.index("magnet_diameter")

    for d in d_range:
        # Fix magnet diameter, optimize remaining parameters
        def _obj_fixed_mass(x_norm_reduced):
            x_norm_full        = np.ones(N_PARAMS) * 0.3
            x_norm_full[idx_d] = (d - DESIGN_BOUNDS["magnet_diameter"][0]) / (
                DESIGN_BOUNDS["magnet_diameter"][1] - DESIGN_BOUNDS["magnet_diameter"][0]
            )
            # Overwrite remaining free parameters
            free_idx = [i for i in range(N_PARAMS) if i != idx_d]
            for fi, xi in zip(free_idx, x_norm_reduced):
                x_norm_full[fi] = xi

            x_si = normalized_to_si(x_norm_full)
            return objective(x_si, base_cfg, mode="fast", Gamma_rms=Gamma_rms)

        x0_red = np.ones(N_PARAMS - 1) * 0.3
        bounds = [(0.0, 1.0)] * (N_PARAMS - 1)
        res    = minimize(_obj_fixed_mass, x0_red, method="Nelder-Mead",
                         options={"maxfev": 200})

        # Compute mass
        cfg_tmp = copy.deepcopy(base_cfg)
        cfg_tmp.magnet.diameter = d
        cfg_tmp.magnet.__post_init__()
        masses.append(cfg_tmp.magnet.mass * 1000)
        P_values.append(-res.fun * 1000)
        print(f"  d={d*1000:.0f}mm  m={masses[-1]:.1f}g  P={P_values[-1]:.2f}mW")

    return np.array(masses), np.array(P_values)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_sensitivity_sweeps(
    results: Dict[str, SweepResult1D],
    base_cfg: WECConfig,
    save_path: Optional[str] = None,
):
    n = len(results)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = axes.flatten()

    for i, (name, res) in enumerate(results.items()):
        ax = axes[i]
        lo, hi, desc, unit = DESIGN_BOUNDS[name]

        valid = ~np.isnan(res.P_avg_mW)
        ax.plot(res.param_values[valid], res.P_avg_mW[valid],
                color="steelblue", lw=2)
        ax.axvline(res.best_value, color="darkorange", ls="--",
                   label=f"best={res.best_value:.3g}")
        ax.fill_between(
            res.param_values[valid], 0, res.P_avg_mW[valid],
            alpha=0.15, color="steelblue"
        )
        ax.set_title(f"{name}", fontweight="bold", fontsize=9)
        ax.set_xlabel(f"{unit}", fontsize=8)
        ax.set_ylabel("P_avg (mW)", fontsize=8)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.4)
        ax.tick_params(labelsize=7)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("1D Sensitivity Sweeps — Effect of Each Parameter on P_avg",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


def plot_grid_search_2d(
    result: SweepResult2D,
    save_path: Optional[str] = None,
):
    lo_x, hi_x, _, unit_x = DESIGN_BOUNDS[result.param_x]
    lo_y, hi_y, _, unit_y = DESIGN_BOUNDS[result.param_y]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Power heatmap
    P_plot = np.where(result.feasibility_grid, result.P_grid, np.nan)
    im = axes[0].pcolormesh(
        result.x_values, result.y_values, P_plot,
        cmap="viridis", shading="auto",
    )
    axes[0].plot(result.best_x, result.best_y, "r*", ms=14,
                label=f"Optimum ({result.best_P_mW:.2f} mW)")
    plt.colorbar(im, ax=axes[0], label="P_avg (mW)")
    axes[0].set_xlabel(f"{result.param_x} ({unit_x})")
    axes[0].set_ylabel(f"{result.param_y} ({unit_y})")
    axes[0].set_title("Average Power P_avg (mW)", fontweight="bold")
    axes[0].legend(fontsize=9)

    # Feasibility mask
    axes[1].pcolormesh(
        result.x_values, result.y_values,
        result.feasibility_grid.astype(float),
        cmap="RdYlGn", shading="auto", vmin=0, vmax=1,
    )
    axes[1].plot(result.best_x, result.best_y, "b*", ms=14, label="Optimum")
    axes[1].set_xlabel(f"{result.param_x} ({unit_x})")
    axes[1].set_ylabel(f"{result.param_y} ({unit_y})")
    axes[1].set_title("Feasibility (green = feasible)", fontweight="bold")
    axes[1].legend(fontsize=9)

    fig.suptitle(
        f"2D Grid Search: {result.param_x} vs {result.param_y}",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


def plot_convergence(
    opt_result: OptimizationResult,
    save_path: Optional[str] = None,
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    evals = np.arange(1, len(opt_result.convergence_history) + 1)
    history = np.array(opt_result.convergence_history)

    # Raw history
    axes[0].plot(evals, history, color="steelblue", lw=0.8, alpha=0.6)
    axes[0].set_xlabel("Function Evaluations")
    axes[0].set_ylabel("P_avg (mW)")
    axes[0].set_title("Optimization History (all evals)", fontweight="bold")
    axes[0].grid(True, alpha=0.4)

    # Running best
    running_best = np.maximum.accumulate(history)
    axes[1].plot(evals, running_best, color="crimson", lw=2)
    axes[1].axhline(opt_result.P_avg_opt_mW, color="darkorange", ls="--",
                    label=f"Final: {opt_result.P_avg_opt_mW:.3f} mW")
    if opt_result.verified_P_avg_mW is not None:
        axes[1].axhline(opt_result.verified_P_avg_mW, color="green", ls=":",
                        label=f"Verified: {opt_result.verified_P_avg_mW:.3f} mW")
    axes[1].set_xlabel("Function Evaluations")
    axes[1].set_ylabel("Best P_avg (mW)")
    axes[1].set_title("Running Best", fontweight="bold")
    axes[1].legend(fontsize=10); axes[1].grid(True, alpha=0.4)

    fig.suptitle(
        f"Optimization Convergence — {opt_result.method}  "
        f"({opt_result.n_evaluations} evals, {opt_result.elapsed_s:.1f}s)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


def plot_pareto(
    masses: np.ndarray,
    P_values: np.ndarray,
    save_path: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(masses, P_values, "o-", color="steelblue", lw=2, ms=7)

    # Annotate the 1W target
    ax.axhline(1000, color="red", ls="--", lw=1.5, label="1W target")
    ax.fill_between(masses, P_values, 1000,
                    where=P_values >= 1000, alpha=0.2, color="green",
                    label="Meets target")
    ax.set_xlabel("Magnet Mass (g)", fontsize=12)
    ax.set_ylabel("Max P_avg (mW)", fontsize=12)
    ax.set_title("Pareto Front: Power vs. Magnet Mass", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.4)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Print Optimization summary report
# ---------------------------------------------------------------------------
def print_optimization_report(opt: OptimizationResult, base_cfg: WECConfig):
    cfg = vector_to_config(opt.x_opt, base_cfg)
    m   = cfg.magnet.mass
    k   = cfg.spring.k_eff
    f_n = np.sqrt(k / m) / (2 * np.pi)

    print("\n" + "="*60)
    print("  OPTIMIZATION REPORT")
    print("="*60)
    print(f"  Method          : {opt.method}")
    print(f"  Evaluations     : {opt.n_evaluations}")
    print(f"  Time            : {opt.elapsed_s:.1f} s")
    print(f"  Fast P_avg      : {opt.P_avg_opt_mW:.3f} mW")
    if opt.verified_P_avg_mW:
        print(f"  Verified P_avg  : {opt.verified_P_avg_mW:.3f} mW")
    print(f"  Target          : 1000.0 mW  (1 W)")
    print(f"  Gap to target   : {1000 - opt.P_avg_opt_mW:.1f} mW")
    print("\n  Optimal Parameters:")
    for name, val in opt.final_result.params.items():
        lo, hi, desc, unit = DESIGN_BOUNDS[name]
        print(f"    {name:<22} = {val:.4g} {unit:<10}  ({desc})")
    print(f"\n  Implied system:")
    print(f"    Magnet mass    = {m*1000:.1f} g")
    print(f"    Natural freq   = {f_n:.4f} Hz")
    print(f"    Wave freq      = {base_cfg.wave.f_peak:.4f} Hz")
    print(f"    Freq ratio     = {base_cfg.wave.omega_peak / (2*np.pi*f_n):.3f}")
    print(f"    Coil R_total   = {cfg.coil.coil_resistance * cfg.coil.n_coils:.2f} Ω")
    print("="*60)


# ---------------------------------------------------------------------------
# Quick validation
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = WECConfig()
    print(cfg.summary())

    # 1. Sensitivity sweeps
    sweep_results = run_all_sensitivity_sweeps(
        cfg, Gamma_rms=0.05, save_path="opt_sensitivity.png"
    )

    # 2. 2D grid search: k_eff vs magnet diameter (the two most important)
    print("\n--- 2D Grid Search: k_eff vs magnet_diameter ---")
    grid = grid_search_2d(cfg, "k_eff", "magnet_diameter", n_x=25, n_y=25)
    print(f"  Best: k_eff={grid.best_x:.2f} N/m, d={grid.best_y*1000:.1f}mm → {grid.best_P_mW:.3f} mW")
    plot_grid_search_2d(grid, save_path="opt_grid_search.png")

    # 3. Global Optimization
    opt = optimize(cfg, method="differential_evolution", max_evals=400, verbose=True)
    plot_convergence(opt, save_path="opt_convergence.png")
    print_optimization_report(opt, cfg)

    # 4. Pareto front (lighter analysis)
    masses, P_vals = pareto_sweep(cfg, n_mass_points=12)
    plot_pareto(masses, P_vals, save_path="opt_pareto.png")