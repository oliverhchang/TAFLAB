"""
main.py
-------
Top-level runner for the WEC analytical model.
Coordinates the physics and analysis packages into a full Wave-to-Wire simulation.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Standardized package imports referencing your specific folder structure
from physics.config import WECConfig
from physics.wave_input import generate_wave_excitation, plot_wave_excitation, spectrum_stats
from physics.dynamics import (
    solve_nonlinear, solve_linear, array_excitation,
    sinusoidal_excitation, compute_power
)
from physics.electromagnetics import (
    build_coil_geometry, compute_flux_profile,
    build_flux_interpolants, solve_electromagnetics,
    compute_c_L, plot_flux_profile, plot_em_result
)
from analysis.frequency_response import (
    compute_transfer_function, print_resonance_report,
    plot_frequency_response
)

# ---------------------------------------------------------------------------
def run_validation(cfg: WECConfig):
    """Mode 1: Sinusoidal excitation, compare linear vs nonlinear solver."""
    print("\n" + "="*55)
    print("  MODE: Validation (Sinusoid, Linear vs Nonlinear)")
    print("="*55)

    t = cfg.wave.t_eval
    Omega = cfg.wave.omega_peak
    cfg.platform.Omega = Omega

    alpha_f, _, alpha_ddot_f = sinusoidal_excitation(cfg.platform.alpha_0, Omega)

    # Run solvers to cross-validate Lagrangian derivation
    result_nl  = solve_nonlinear(cfg, alpha_f, alpha_ddot_f, t)
    result_lin = solve_linear(cfg, alpha_f, alpha_ddot_f, t)

    power_nl  = compute_power(result_nl,  cfg)
    power_lin = compute_power(result_lin, cfg)

    print(f"\n  Nonlinear P_avg: {power_nl['P_avg_load']*1000:.3f} mW")
    print(f"  Linear    P_avg: {power_lin['P_avg_load']*1000:.3f} mW")

# ---------------------------------------------------------------------------
def run_jonswap(cfg: WECConfig):
    """Mode 2: Full stochastic JONSWAP pipeline from ocean wave to mW power."""
    print("\n" + "="*55)
    print("  MODE: JONSWAP Stochastic Pipeline")
    print("="*55)

    # 1. Generate stochastic ocean excitation
    ws = generate_wave_excitation(cfg, mode="jonswap", seed=42)
    plot_wave_excitation(ws, save_path="output_wave.png")

    # 2. Run Nonlinear Dynamics
    alpha_f, _, alpha_ddot_f = array_excitation(ws.t, ws.alpha)
    dyn_result = solve_nonlinear(cfg, alpha_f, alpha_ddot_f, ws.t)

    # 3. Electromagnetic Conversion
    em_result = solve_electromagnetics(cfg, dyn_result.t, dyn_result.x, dyn_result.x_dot)
    plot_em_result(em_result, save_path="output_em.png")

    print(f"\n  *** JONSWAP P_avg: {em_result.P_avg*1000:.3f} mW ***")

# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="WEC Analytical Model")
    parser.add_argument("--mode", choices=["validate", "jonswap", "sweep", "all"], default="validate")
    args = parser.parse_args()

    cfg = WECConfig()

    if args.mode in ("validate", "all"):
        run_validation(cfg)
    if args.mode in ("jonswap", "all"):
        run_jonswap(cfg)
    if args.mode in ("sweep", "all"):
        fr = compute_transfer_function(cfg)
        print_resonance_report(fr, cfg)

if __name__ == "__main__":
    main()