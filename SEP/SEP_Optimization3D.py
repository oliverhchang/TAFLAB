import numpy as np
import matplotlib.pyplot as plt
from SEP_Modeling3D_V1 import SphericalPendulum3D


# ==========================================
# OPTIMIZATION ROUTINE (Power & Fit)
# ==========================================
def optimize_coil_geometry():
    # 1. Define the Sweep Range
    # Testing radii from 5mm to 25mm
    test_radii = np.linspace(0.005, 0.025, 30)

    results = []
    valid_radii = []

    print(f"Starting optimization sweep over {len(test_radii)} geometries...")
    print("-" * 50)
    print(f"{'Radius (mm)':<15} | {'Status':<15} | {'Power Score':<15}")
    print("-" * 50)

    for r in test_radii:
        # ------------------------------------------
        # A. Geometric Constraint Check (The "Fit")
        # ------------------------------------------
        # We check the most crowded ring: Layer 3 (Theta = 60 deg, 12 coils)
        # Ring Radius = R_shell * sin(60)
        # Circumference = 2 * pi * Ring Radius
        # Arc length available per coil = Circumference / 12
        R_shell = 0.095
        ring_circumference = 2 * np.pi * R_shell * np.sin(np.radians(60))
        available_arc = ring_circumference / 12

        # Coil diameter (plus 1mm margin for printing)
        coil_diameter = (2 * r) + 0.001

        # ------------------------------------------
        # B. Physics Simulation (The "Power")
        # ------------------------------------------
        sim = SphericalPendulum3D()

        # Override coil areas in the engine
        for coil in sim.coils:
            coil['area'] = np.pi * r ** 2

        # Run short simulation (1.0s)
        t_data, state_data = sim.run_simulation(t_max=1.0)

        sum_flux_sq = 0
        # Calculate Resistance Penalty
        # Length of wire = (Circumference * N_turns)
        wire_length = (2 * np.pi * r) * 250
        # Resistance (30 AWG = 0.338 Ohms/m)
        resistance = wire_length * 0.338

        # Process Simulation Data
        for state in state_data:
            alpha, beta = state[0], state[2]
            flux = sim.calculate_total_flux(alpha, beta)
            # We use Flux^2 as a proxy for Voltage^2 (Power potential)
            sum_flux_sq += flux ** 2

        # Power Score = (Average V^2) / R
        # This balances "catching more flux" vs "adding more resistance"
        power_score = sum_flux_sq / resistance

        print(f"{r * 1000:<15.1f} | OK              | {power_score:.4f}")
        results.append(power_score)
        valid_radii.append(r)

    # ==========================================
    # 3. PLOT THE RESULTS
    # ==========================================
    best_idx = np.argmax(results)
    best_r = valid_radii[best_idx]
    max_score = results[best_idx]

    plt.figure(figsize=(10, 6))
    plt.plot(np.array(valid_radii) * 1000, results, marker='o', linestyle='-', color='b')

    # Highlight the Peak
    plt.axvline(best_r * 1000, color='r', linestyle='--', label=f'Optimal: {best_r * 1000:.1f}mm')
    plt.scatter([best_r * 1000], [max_score], color='red', s=100, zorder=5)

    plt.xlabel('Coil Radius (mm)')
    plt.ylabel('Power Efficiency Score (Flux^2 / R)')
    plt.title('Optimization: Power Trade-off vs Geometric Fit')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("-" * 50)
    print(f"OPTIMIZATION COMPLETE.")
    print(f"Recommended Radius: {best_r * 1000:.2f} mm")
    print(f"Reason: Maximizes power before hitting resistance penalty or collision.")


if __name__ == "__main__":
    optimize_coil_geometry()