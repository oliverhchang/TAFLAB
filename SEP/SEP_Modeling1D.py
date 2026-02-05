import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import simpson

# ==========================================
# 1. ASSUMPTIONS & CONSTANTS
# ==========================================
# --- MAGNET (Source) ---
# Shape: Cylinder
Br = 0.6  # Remanence (Tesla)
R_mag = 0.020  # Magnet Radius (20mm)
D_mag = 0.020  # Magnet Thickness (20mm)

# --- COIL (Harvester) ---
# Shape: Pancake / Donut (Distributed Model)
# We assume the coil is a rectangular cross-section ring
ID = 0.036  # Inner Diameter (m)
OD = 0.045  # Outer Diameter (m)
H_coil = 0.008  # Coil Height/Thickness (m)
N_turns = 500  # Total Turn Count

# Material Properties (30 AWG)
rho_cu = 1.68e-8
area_30awg = 0.0509e-6

# --- GEOMETRY & MECHANICS ---
gap = 0.007  # Air Gap (m): Distance from Magnet Face to Top of Coil
L = 0.2  # Pendulum Length (m)
theta_0 = 45  # Release Angle (degrees)
M_total = 0.168  # Total Moving Mass (kg)
g = 9.81

# ==========================================
# 2. RESISTANCE CALCULATION
# ==========================================
# Average radius is still useful for resistance approx
r_avg = (ID + OD) / 4  # Average Radius
len_wire = N_turns * 2 * np.pi * r_avg
R_coil = (rho_cu * len_wire) / area_30awg
print(f"Calculated Coil Resistance: {R_coil:.2f} Ohms")


# ==========================================
# 3. PHYSICS MODELS (Distributed Mesh)
# ==========================================
def calc_Bz(z):
    """
    Biot-Savart Law for Cylinder on Axis.
    Calculates B-field strength (Tesla) at distance z from magnet face.
    """
    term1 = (D_mag + z) / np.sqrt(R_mag ** 2 + (D_mag + z) ** 2)
    term2 = z / np.sqrt(R_mag ** 2 + z ** 2)
    return (Br / 2) * (term1 - term2)


def get_distributed_flux(angles_deg):
    """
    Calculates total flux by summing contributions from a 5x5 grid
    representing the coil's cross-section.
    """
    rads = np.radians(angles_deg)

    # Grid Settings
    RADIAL_STEPS = 10
    AXIAL_STEPS = 10

    # Create the grid points inside the copper cross-section
    # r_layers: from Inner Radius to Outer Radius
    r_layers = np.linspace(ID / 2, OD / 2, RADIAL_STEPS)
    # z_layers: from Top of coil (0) to Bottom of coil (H_coil)
    z_layers = np.linspace(0, H_coil, AXIAL_STEPS)

    # Turns per grid point
    turns_per_point = N_turns / (RADIAL_STEPS * AXIAL_STEPS)

    # Base vertical distance from magnet to coil plane (Pendulum Geometry)
    # This is the "Air Gap" varying with swing angle
    z_pendulum = np.sqrt(gap ** 2 + (L * np.sin(rads)) ** 2)

    total_flux = np.zeros_like(angles_deg)

    # --- MESH INTEGRATION ---
    for r in r_layers:
        for z_offset in z_layers:
            # 1. Distance to THIS specific wire layer
            z_actual = z_pendulum + z_offset

            # 2. Calculate B-field on axis
            B_center = calc_Bz(z_actual)

            # 3. Off-Axis Correction (The "Falloff Factor")
            # Approximates that field weakens at the outer edges of the pancake
            falloff_factor = 1 / (1 + (r / R_mag) ** 2)
            B_effective = B_center * falloff_factor

            # 4. Flux through this specific loop (B * Area)
            loop_area = np.pi * r ** 2
            loop_flux = B_effective * loop_area

            # 5. Add to total
            total_flux += loop_flux * turns_per_point

    return total_flux


# Fitted Model for smooth derivative
def flux_model(alpha, a, b, c):
    return a / np.sqrt(b * alpha ** 2 + c)


# ==========================================
# 4. SIMULATION ROUTINE
# ==========================================

# A. Generate Flux Data
angles = np.linspace(0, theta_0, 100)
flux_data = get_distributed_flux(angles)  # <--- Uses new Mesh Model

# B. Fit Curve to eliminate noise before differentiation
popt, _ = curve_fit(flux_model, angles, flux_data, p0=[1, 1, 1])
a, b, c = popt

# C. Calculate Dynamics (Velocity)
rads = np.radians(angles)
rad_0 = np.radians(theta_0)
# Velocity (omega) = sqrt( 2g/L * (cos(theta) - cos(theta_0)) )
omega = np.sqrt((2 * g / L) * (np.cos(rads) - np.cos(rad_0)))

# D. Calculate Flux Gradient (K = dPhi/dTheta)
# Derivative of a/sqrt(b*x^2 + c) is: -0.5 * a * (b*x^2+c)^(-1.5) * 2bx
d_flux_deg = -0.5 * a * (b * angles ** 2 + c) ** (-1.5) * (2 * b * angles)
K_vals = d_flux_deg * (180 / np.pi)  # Convert to Flux per Radian

# E. Calculate Outputs
# Voltage = -N * (dPhi/dt) -- note: N is already baked into flux_data sum
# But we need to use the distributed flux sum derivative directly.
# Since flux_data IS the total flux for all turns, we don't multiply by N_turns again here.
voltage = -1 * K_vals * omega

# Power = V^2 / (2 * R) (Matched Load Assumption)
power = (voltage ** 2) / (2 * R_coil)

# ==========================================
# 5. ENERGY & EFFICIENCY ANALYSIS
# ==========================================

# A. Mechanical Input (PE = mgh)
h_lift = L * (1 - np.cos(rad_0))
E_input_mech = M_total * g * h_lift

# B. Electrical Output (Integral P dt)
d_theta = np.abs(np.diff(rads))
avg_omega = (omega[1:] + omega[:-1]) / 2
avg_omega[avg_omega < 1e-6] = 1e-6  # Avoid div/0
dt_steps = d_theta / avg_omega
time_array = np.cumsum(np.concatenate(([0], dt_steps)))

E_output_elec = simpson(y=power, x=time_array)

# C. Efficiency
efficiency = (E_output_elec / E_input_mech) * 100

print("-" * 30)
print(f"DISTRIBUTED MESH SIMULATION (1/4 Swing)")
print("-" * 30)
print(f"Peak Voltage:      {np.max(np.abs(voltage)):.2f} V")
print(f"Peak Voltage (After Diode Loss):      {np.max(np.abs(voltage)) - 1.4:.2f} V")
print(f"Peak Power:        {np.max(power) * 1000:.2f} mW")
print("-" * 30)
print(f"Mechanical Input:  {E_input_mech * 1000:.2f} mJ")
print(f"Electrical Output: {E_output_elec * 1000:.2f} mJ")
print("-" * 30)

# ==========================================
# 6. PLOTTING
# ==========================================
fig, ax1 = plt.subplots(figsize=(8, 5))

color = 'tab:red'
ax1.set_xlabel('Swing Angle (Degrees)')
ax1.set_ylabel('Power Output (mW)', color=color)
ax1.plot(angles, power * 1000, color=color, linewidth=2)
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Voltage (V)', color=color)
ax2.plot(angles, np.abs(voltage), color=color, linestyle='--')
ax2.tick_params(axis='y', labelcolor=color)

plt.title(f"Distributed Model: Power & Voltage")
fig.tight_layout()
plt.show()