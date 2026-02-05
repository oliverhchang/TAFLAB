import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm  # <--- FIXED: Was missing
from mpl_toolkits.mplot3d import Axes3D  # <--- FIXED: Needed for 3D projection
from scipy.optimize import curve_fit

# ============================================================
# 1. CORE PHYSICS ENGINE
# ============================================================
N_QUAD = 12  # Gauss-Legendre order for radial flux quadrature

def _bz_on_axis(Br, R_mag, D_mag, z):
    """
    Axial field on the symmetry axis of a uniformly axially-magnetised
    disk magnet. Vectorised.
    """
    z = np.asarray(z, dtype=float)
    t1 = (D_mag + z) / np.sqrt(R_mag ** 2 + (D_mag + z) ** 2)
    t2 = z / np.sqrt(R_mag ** 2 + z ** 2)
    return (Br / 2.0) * (t1 - t2)


def _flux_through_turn(r_turn, z, Br, R_mag, D_mag):
    """
    Flux through one coaxial loop using Gauss-Legendre quadrature.
    """
    nodes, weights = np.polynomial.legendre.leggauss(N_QUAD)
    # Map [-1,1] -> [0, r_turn]
    r_prime = 0.5 * r_turn * (nodes + 1.0)
    half = 0.5 * r_turn

    z_eff = np.sqrt(z ** 2 + r_prime ** 2)
    Bz = _bz_on_axis(Br, R_mag, D_mag, z_eff)
    return half * np.dot(weights, Bz * 2.0 * np.pi * r_prime)


def run_sim(N_turns, ID, OD, H_coil, AWG_val):
    """
    Simulates peak battery charging power.
    Uses Threshold Model: Power is only generated if Voltage > 4.1V.
    """
    # ---- Constants ----
    Br = 0.6;
    R_mag = 0.020;
    D_mag = 0.020
    gap = 0.007;
    L_pend = 0.2;
    theta_0 = 45.0;
    g = 9.81
    rho_cu = 1.68e-8

    # ---- Battery Constants ----
    V_BATT = 3.7  # LiPo Nominal Voltage
    V_DIODE = 0.4  # Schottky Bridge Drop
    V_THRESH = V_BATT + V_DIODE  # ~4.1V required to push current

    # ---- Wire & Resistance ----
    # Protect against divide by zero or negative AWG math
    if AWG_val > 50 or AWG_val < 1: return 0.0

    d_m = 0.127e-3 * (92 ** ((36.0 - AWG_val) / 39.0))
    A_wire = np.pi * (d_m / 2.0) ** 2
    r_avg = (ID + OD) / 4.0
    R_coil = rho_cu * (N_turns * 2.0 * np.pi * r_avg) / A_wire

    # ---- Coil Mesh ----
    R_STEPS = 5;
    Z_STEPS = 5
    r_layers = np.linspace(ID / 2.0, OD / 2.0, R_STEPS)
    z_layers = np.linspace(0.0, H_coil, Z_STEPS)
    turns_per_cell = N_turns / (R_STEPS * Z_STEPS)

    # ---- Pendulum Sweep ----
    angles = np.linspace(0, theta_0, 50)
    rads = np.radians(angles)
    z_pendulum = np.sqrt(gap ** 2 + (L_pend * np.sin(rads)) ** 2)

    # ---- Flux Calculation ----
    flux_data = np.zeros_like(angles)
    for r_turn in r_layers:
        for z_off in z_layers:
            z_act = z_pendulum + z_off
            phi = np.array([_flux_through_turn(r_turn, z, Br, R_mag, D_mag)
                            for z in z_act])
            flux_data += phi * turns_per_cell

    # ---- Analysis ----
    def model(x, a, b, c):
        # Safety: Ensure term inside sqrt is non-negative
        val = b * x ** 2 + c
        return a / np.sqrt(np.abs(val) + 1e-9)

    try:
        # Bounds enforce positive b and c to prevent sqrt errors
        popt, _ = curve_fit(
            model,
            angles,
            flux_data,
            p0=[flux_data[0], 1.0, 1e-4],
            bounds=([-np.inf, 0, 0], [np.inf, np.inf, np.inf]),
            maxfev=2000
        )
        a, b, c = popt

        # Derivative (dPhi/dt)
        # d/dx of a/sqrt(bx^2+c) -> -abx / (bx^2+c)^(3/2)
        # Note: Previous code had a factor of 0.5 and 2 which cancel out
        denom = (b * angles ** 2 + c) ** 1.5
        d_flux_deg = -a * b * angles / (denom + 1e-9)
        d_flux_rad = d_flux_deg * (180.0 / np.pi)

        # Velocity
        cos_diff = np.maximum(np.cos(rads) - np.cos(np.radians(theta_0)), 0.0)
        omega = np.sqrt(2.0 * g / L_pend * cos_diff)

        # EMF (Open Circuit Voltage)
        emf = np.abs(d_flux_rad * omega)
        peak_emf = np.max(emf)

        # ---- BATTERY LOGIC ----
        if peak_emf <= V_THRESH:
            return 0.0  # Voltage too low to charge
        else:
            I_charging = (peak_emf - V_THRESH) / R_coil
            P_delivered = I_charging * V_BATT
            return P_delivered * 1000.0  # mW

    except Exception:
        return 0.0


# ============================================================
# 2. STANDARD SWEEP SETUP (1D Graphs)
# ============================================================
BASE_ID = 0.036
BASE_OD = 0.045
BASE_H = 0.008
BASE_N = 1500
BASE_AWG = 34

# Calculate Bobbin Constant (Total Copper Area)
_d_base = 0.127e-3 * (92 ** ((36.0 - BASE_AWG) / 39.0))
BOBBIN_AREA = BASE_N * np.pi * (_d_base / 2.0) ** 2


def _turns_for_awg(awg):
    d = 0.127e-3 * (92 ** ((36.0 - awg) / 39.0))
    return BOBBIN_AREA / (np.pi * (d / 2.0) ** 2)


def _awg_for_turns(n):
    req_d = 2.0 * np.sqrt(BOBBIN_AREA / (n * np.pi))
    return 36.0 - 39.0 * np.log(req_d / 0.127e-3) / np.log(92.0)


# ============================================================
# 3. RUNNING 1D EXPERIMENTS
# ============================================================
print("Running 1D optimization sweeps...")

# A - Turns Sweep
n_range = np.linspace(100, 2500, 25)
p_turns = []
for n in n_range:
    awg = _awg_for_turns(n)
    p_turns.append(run_sim(int(n), BASE_ID, BASE_OD, BASE_H, awg))

# B - AWG Sweep
awg_range = np.arange(24, 42, 1)
p_awg = []
for awg in awg_range:
    n = _turns_for_awg(awg)
    p_awg.append(run_sim(int(n), BASE_ID, BASE_OD, BASE_H, awg))

# C - Radius Sweep
RADIAL_DEPTH = BASE_OD - BASE_ID
r_center_range = np.linspace(0.010, 0.060, 20)
p_radial = []
for rc in r_center_range:
    id_c = 2.0 * (rc - RADIAL_DEPTH / 2.0)
    od_c = 2.0 * (rc + RADIAL_DEPTH / 2.0)
    if id_c > 0.002:
        p_radial.append(run_sim(BASE_N, id_c, od_c, BASE_H, BASE_AWG))
    else:
        p_radial.append(0.0)

# D - OD Sweep
od_range = np.linspace(BASE_ID + 0.003, 0.080, 20)
p_od = []
for od in od_range:
    p_od.append(run_sim(BASE_N, BASE_ID, od, BASE_H, BASE_AWG))

# E - Height Sweep
h_range = np.linspace(0.002, 0.030, 20)
p_h = []
for h in h_range:
    p_h.append(run_sim(BASE_N, BASE_ID, BASE_OD, h, BASE_AWG))

# ============================================================
# 4. PLOTTING 1D GRAPHS
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Generator Optimization: 3.7V Battery Charging\n(Threshold Model: V > 4.1V required)", fontsize=14)
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# Plot A: Turns
axes[0, 0].plot(n_range, p_turns, 'tab:blue', marker='o')
axes[0, 0].set_title("Effect of Turn Count")
axes[0, 0].set_ylabel("Charging Power (mW)")
axes[0, 0].set_xlabel("Turns")
axes[0, 0].grid(True, alpha=0.3)
idx = np.argmax(p_turns)
axes[0, 0].axvline(n_range[idx], color='r', linestyle='--')
axes[0, 0].text(n_range[idx], max(p_turns) * 0.5, f"Peak: {int(n_range[idx])}", rotation=90, color='r', va='center')

# Plot B: AWG
axes[0, 1].plot(awg_range, p_awg, 'tab:orange', marker='s')
axes[0, 1].set_title("Effect of Wire Gauge")
axes[0, 1].set_xlabel("AWG")
axes[0, 1].grid(True, alpha=0.3)
idx = np.argmax(p_awg)
axes[0, 1].axvline(awg_range[idx], color='r', linestyle='--')
axes[0, 1].text(awg_range[idx], max(p_awg) * 0.5, f"Peak: {int(awg_range[idx])}", rotation=90, color='r', va='center')

# Plot C: Center Radius
axes[0, 2].plot(r_center_range * 1000, p_radial, 'tab:green')
axes[0, 2].set_title("Coil Position (Center Radius)")
axes[0, 2].set_xlabel("Radius (mm)")
axes[0, 2].grid(True, alpha=0.3)

# Plot D: OD
axes[1, 0].plot(od_range * 1000, p_od, 'tab:purple')
axes[1, 0].set_title("Outer Diameter")
axes[1, 0].set_xlabel("OD (mm)")
axes[1, 0].grid(True, alpha=0.3)

# Plot E: Height
axes[1, 1].plot(h_range * 1000, p_h, 'tab:red')
axes[1, 1].set_title("Coil Thickness")
axes[1, 1].set_xlabel("Height (mm)")
axes[1, 1].grid(True, alpha=0.3)

# Info Box
axes[1, 2].axis("off")
axes[1, 2].text(0, 0.5,
                "INSIGHTS:\n\n1. VOLTAGE WALL:\n   Low turns = 0 mW because\n   Voltage < 4.1V.\n\n2. PEAK:\n   High turns needed to\n   open diodes. Too many\n   turns = high resistance.",
                fontsize=10, family='monospace')

plt.tight_layout()
plt.show()

# ============================================================
# 5. ADVANCED 3D VISUALIZATION
# ============================================================
print("Generating Advanced 3D Interaction Maps...")


# --- GRAPH 1: GEOMETRY OPTIMIZATION (ID vs OD) ---
def plot_geometry_heatmap():
    res = 20
    ids = np.linspace(0.004, 0.040, res)  # 4mm to 40mm ID
    ods = np.linspace(0.030, 0.080, res)  # 30mm to 80mm OD

    ID_Grid, OD_Grid = np.meshgrid(ids, ods)
    Power_Grid = np.zeros_like(ID_Grid)

    for i in range(res):
        for j in range(res):
            cur_id = ID_Grid[i, j]
            cur_od = OD_Grid[i, j]

            if cur_id >= (cur_od - 0.005):
                Power_Grid[i, j] = 0
            else:
                cur_n = 1500
                bobbin_w = (cur_od - cur_id) / 2
                bobbin_area = bobbin_w * 0.008
                req_d = 2.0 * np.sqrt(bobbin_area / (cur_n * np.pi))
                cur_awg = 36.0 - 39.0 * np.log(req_d / 0.127e-3) / np.log(92.0)
                Power_Grid[i, j] = run_sim(cur_n, cur_id, cur_od, 0.008, cur_awg)

    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes()
    cp = ax.contourf(ID_Grid * 1000, OD_Grid * 1000, Power_Grid, 100, cmap='viridis')
    cbar = fig.colorbar(cp)
    cbar.set_label('Charging Power (mW)', rotation=270, labelpad=15)

    ax.set_title('ID vs OD Optimization (Turns=1500)', fontsize=14)
    ax.set_xlabel('Inner Diameter (mm)', fontsize=12)
    ax.set_ylabel('Outer Diameter (mm)', fontsize=12)

    max_idx = np.unravel_index(np.argmax(Power_Grid), Power_Grid.shape)
    best_id = ID_Grid[max_idx] * 1000
    best_od = OD_Grid[max_idx] * 1000
    ax.plot(best_id, best_od, 'ro', markersize=10)
    ax.text(best_id + 2, best_od, f"PEAK: {int(np.max(Power_Grid))}mW", color='white', fontweight='bold')
    plt.show()


# --- GRAPH 2: DESIGN LANDSCAPE (Turns vs Thickness) ---
def plot_design_surface():
    res = 20
    turns = np.linspace(500, 2500, res)
    heights = np.linspace(0.002, 0.025, res)

    T_Grid, H_Grid = np.meshgrid(turns, heights)
    P_Grid = np.zeros_like(T_Grid)

    FIXED_ID = 0.006
    FIXED_OD = 0.046

    for i in range(res):
        for j in range(res):
            cur_n = T_Grid[i, j]
            cur_h = H_Grid[i, j]

            bobbin_w = (FIXED_OD - FIXED_ID) / 2
            bobbin_area = bobbin_w * cur_h
            req_d = 2.0 * np.sqrt(bobbin_area / (cur_n * np.pi))
            cur_awg = 36.0 - 39.0 * np.log(req_d / 0.127e-3) / np.log(92.0)
            P_Grid[i, j] = run_sim(cur_n, FIXED_ID, FIXED_OD, cur_h, cur_awg)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(T_Grid, H_Grid * 1000, P_Grid, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax.set_title('Turns vs. Coil Thickness', fontsize=14)
    ax.set_xlabel('Number of Turns', fontsize=11)
    ax.set_ylabel('Thickness (mm)', fontsize=11)
    ax.set_zlabel('Power (mW)', fontsize=11)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.view_init(elev=30, azim=225)
    plt.show()


# Execute 3D Plots
plot_geometry_heatmap()
plot_design_surface()