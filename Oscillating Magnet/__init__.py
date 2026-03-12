"""
main.py
-------
Top-level runner for the WEC analytical model.
Coordinates the physics, analysis, and optimization sub-packages.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

# 1. Import from the 'physics' folder
from physics.config import WECConfig
from physics.wave_input import generate_wave_excitation, spectrum_stats
from physics.dynamics import solve_nonlinear, sinusoidal_excitation, compute_power
from physics.electromagnetics import solve_electromagnetics

# 2. Import from the 'analysis' folder
from analysis.frequency_response import compute_transfer_function, print_resonance_report

# 3. Import from the 'Optimization' folder (Note the capital 'O' as per your screenshot)
# Ensure you have an __init__.py in this folder as well
# from Optimization.design_sweep import run_optimization

def main():
    parser = argparse.ArgumentParser(description="WEC Analytical Model")
    parser.add_argument(
        "--mode", choices=["validate", "jonswap", "sweep", "all"],
        default="validate", help="Run mode"
    )
    args = parser.parse_args()

    # Initialize the Master Config from physics.config
    cfg = WECConfig()
    print(cfg.summary())

    if args.mode in ("validate", "all"):
        # Run validation using modules from the physics package
        t = cfg.wave.t_eval
        alpha_f, _, alpha_ddot_f = sinusoidal_excitation(cfg.platform.alpha_0, cfg.wave.omega_peak)
        result = solve_nonlinear(cfg, alpha_f, alpha_ddot_f, t)
        print(f"Validation Complete. P_avg: {compute_power(result, cfg)['P_avg_load']*1000:.2f} mW")

    if args.mode in ("sweep", "all"):
        # Run frequency analysis from the analysis package
        fr = compute_transfer_function(cfg)
        print_resonance_report(fr, cfg)

    print("\n  Simulation Complete.")

if __name__ == "__main__":
    main()