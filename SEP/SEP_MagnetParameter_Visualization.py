import numpy as np
import matplotlib.pyplot as plt

# --- physics Formulas ---
def b_field_cylinder(z, D, H, Br=1.26):
    """Calculate on-axis B-field of a cylinder at distance z."""
    R = D / 2.0
    term1 = (H + z) / np.sqrt(R**2 + (H + z)**2)
    term2 = z / np.sqrt(R**2 + z**2)
    return (Br / 2.0) * (term1 - term2)

def b_field_sphere(z, D, Br=1.26):
    """Calculate on-axis B-field of a sphere at distance z from its surface."""
    r = D / 2.0
    # Field outside a sphere falls off as (r / distance_from_center)^3
    return (2.0 / 3.0) * Br * (r / (r + z))**3

# --- Test Conditions ---
Br_grade = 1.26       # Tesla (N40 Grade NdFeB)
air_gap = 0.002       # 2mm air gap (plastic bowl + tape clearance)

# Parameter arrays to sweep (from 5mm to 50mm)
dimensions_mm = np.linspace(5, 50, 100)
dimensions_m = dimensions_mm / 1000.0

# --- 1. Sphere Radius Sweep ---
# How does sphere diameter affect field at a 2mm gap?
B_sphere = b_field_sphere(air_gap, dimensions_m, Br_grade)

# --- 2. Cylinder Height Sweep ---
# Keep Diameter fixed at 20mm, sweep Height
fixed_D = 0.040
B_cyl_height = b_field_cylinder(air_gap, fixed_D, dimensions_m, Br_grade)

# --- 3. Cylinder Diameter Sweep ---
# Keep Height fixed at 20mm, sweep Diameter
fixed_H = 0.020
B_cyl_diam = b_field_cylinder(air_gap, dimensions_m, fixed_H, Br_grade)

# --- Plotting ---
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Sphere Diameter
axs[0].plot(dimensions_mm, B_sphere, color='red', lw=2)
axs[0].set_title("Sphere: Changing Diameter")
axs[0].set_xlabel("Sphere Diameter (mm)")
axs[0].set_ylabel("B-Field at 2mm Gap (Tesla)")
axs[0].grid(True, alpha=0.5)

# Plot 2: Cylinder Height
axs[1].plot(dimensions_mm, B_cyl_height, color='blue', lw=2)
axs[1].set_title("Cylinder: Changing Height\n(Fixed 40mm Diameter)")
axs[1].set_xlabel("Cylinder Height (mm)")
axs[1].grid(True, alpha=0.5)
# Add a line showing where Height = Diameter
axs[1].axvline(x=20, color='gray', linestyle='--', label='Height = Diameter')
axs[1].legend()

# Plot 3: Cylinder Diameter
axs[2].plot(dimensions_mm, B_cyl_diam, color='green', lw=2)
axs[2].set_title("Cylinder: Changing Diameter\n(Fixed 20mm Height)")
axs[2].set_xlabel("Cylinder Diameter (mm)")
axs[2].grid(True, alpha=0.5)

plt.tight_layout()
plt.show()