import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# ==========================================
# 1. SHARED CONSTANTS & MATERIAL PROPERTIES
# ==========================================
Br = 1.45  # Remanence in Tesla
Hcb_ka = 900  # Normal Coercivity in kA/m
Hcb_si = Hcb_ka * 1000  # Coercivity in A/m
Bs = 1.6  # Saturation in Tesla
mu0 = 4 * np.pi * 1e-7

# Fitting parameter for the tanh model (used in Graph 2)
a = Hcb_si / np.arctanh(Br / Bs)


# Function to calculate Permeance Coefficient (Pc) for a cylinder
def get_pc(diameter, thickness):
    r = diameter / 2.0
    l = thickness
    ratio = l / r
    return ratio * np.sqrt(1 + ratio)


# ==========================================
# GRAPH 1: Linear Approximation & 40x20mm
# ==========================================
Pc_val = get_pc(40, 20)  # Automatically calculates to ~1.41

# H axis range (0 to 1100 kA/m)
H_ka_1 = np.linspace(0, 1100, 500)
H_si_1 = H_ka_1 * 1000

# Normal B-H curve approximation (Linear until Hcb)
B_normal = Br * (1 - H_ka_1 / Hcb_ka)
B_normal = np.maximum(B_normal, 0)  # Clamp at zero

# Load line equation: B = mu0 * Pc * H
B_load_line = mu0 * Pc_val * H_si_1

# Find intersection algebraically for linear model
H_op_si_1 = Br / (mu0 * Pc_val + (Br / Hcb_si))
B_op_1 = mu0 * Pc_val * H_op_si_1

fig1 = plt.figure(figsize=(10, 7))
plt.plot(H_ka_1, B_normal, 'b-', linewidth=3, label='N52 Normal Demagnetization Curve')
plt.plot(H_ka_1, B_load_line, 'g--', linewidth=2, label=f'Load Line ($P_c$ = {Pc_val:.2f})')
plt.plot(H_op_si_1 / 1000, B_op_1, 'ro', markersize=10,
         label=f'Operating Point\n({B_op_1:.2f}T, {H_op_si_1 / 1000:.0f} kA/m)')

# Highlight the "Knee" Danger Zone (Approx 850 kA/m for N52)
plt.axvspan(Hcb_ka - 50, 1100, color='red', alpha=0.1, label='Irreversible Loss Zone (Below Knee)')

plt.title("Rigorous N52 B-H Curve & Load Line Analysis", fontsize=16, fontweight='bold')
plt.xlabel("Demagnetizing Field Strength |H| (kA/m)", fontsize=12)
plt.ylabel("Magnetic Flux Density B (Tesla)", fontsize=12)
plt.xlim(0, 1100)
plt.ylim(0, 1.6)
plt.grid(True, which='both', linestyle='--', alpha=0.5)

plt.annotate('Remanence $B_r$', xy=(0, Br), xytext=(50, Br + 0.05), arrowprops=dict(facecolor='black', shrink=0.05))
plt.annotate('Coercivity $H_{cb}$', xy=(Hcb_ka, 0), xytext=(Hcb_ka - 150, 0.1),
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.legend(loc='upper right', fontsize=10)
plt.tight_layout()
plt.savefig('n52_bh_curve_analysis.png')

# ==========================================
# GRAPH 2: Tanh Model & Multiple Geometries
# ==========================================
# Geometries to test (Diameter, Thickness, Label)
geometries = [
    (40, 20, "40x20mm (Target)"),
    (25, 5, "25x5mm (Thin Disc)"),
    (25, 25, "25x25mm (Tall Cylinder)"),
    (50, 10, "50x10mm (Wide Disc)")
]

# H range for plot (kA/m)
H_plot_ka_2 = np.linspace(-1200, 200, 1000)
H_plot_si_2 = H_plot_ka_2 * 1000

# Loop branch using tanh model
B_upper = Bs * np.tanh((H_plot_si_2 + Hcb_si) / a)

fig2 = plt.figure(figsize=(12, 8))
plt.plot(H_plot_ka_2, B_upper, 'b-', linewidth=3, alpha=0.3, label='N52 Demagnetization Curve')

colors = ['red', 'orange', 'green', 'purple']


# Objective function for fsolve
def find_intersection(h_val, pc_val):
    return Bs * np.tanh((h_val + Hcb_si) / a) - (-mu0 * pc_val * h_val)


print("\n--- Magnet Geometry Results ---")
for i, (d, t, label) in enumerate(geometries):
    pc = get_pc(d, t)

    # Plot Load Line
    H_load_ka = np.linspace(-1200, 0, 100)
    H_load_si = H_load_ka * 1000
    B_load = -mu0 * pc * H_load_si
    plt.plot(H_load_ka, B_load, color=colors[i], linestyle='--', alpha=0.7, label=f'{label}: $P_c$={pc:.2f}')

    # Find and plot intersection
    h_op = fsolve(find_intersection, -500000, args=(pc,))[0]
    b_op = -mu0 * pc * h_op

    plt.plot(h_op / 1000, b_op, 'o', color=colors[i], markersize=8)
    plt.annotate(f"{b_op:.2f}T", (h_op / 1000, b_op), textcoords="offset points", xytext=(0, 10), ha='center',
                 fontweight='bold', color=colors[i])

    print(f"{label}: Pc={pc:.2f}, B_op={b_op:.2f}T, H_op={h_op / 1000:.1f} kA/m")

# Mark the Knee (Danger Zone)
plt.axvspan(-1200, -850, color='red', alpha=0.1, label='Irreversible Loss Zone (Below Knee)')

plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.title("Comparison of Common Magnet Geometries on N52 Curve", fontsize=16, fontweight='bold')
plt.xlabel("Demagnetizing Field Strength H (kA/m)", fontsize=12)
plt.ylabel("Magnetic Flux Density B (Tesla)", fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.xlim(-1200, 100)
plt.ylim(0, 1.6)
plt.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig('geometry_comparison_bh.png')

# Show both plots at the very end
plt.show()