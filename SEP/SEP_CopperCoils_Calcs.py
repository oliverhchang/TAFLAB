import numpy as np

# ================= CONFIGURATION =================
# Magnet (20mm Radius, 20mm Thick, N52)
Br = 0.6  # Remanence (Tesla)
R_mag = 0.020
D_mag = 0.020
Gap = 0.007

# Pendulum Physics
L_pend = 0.2  # Length (m)
Theta_0 = 45.0  # Swing Amplitude (degrees)
g = 9.81

# Battery & Circuit
V_BATT = 3.7  # LiPo Voltage
V_DIODE = 0.4  # Rectifier Drop
V_THRESH = V_BATT + V_DIODE  # Min voltage to charge

# Coil Parameters (Enter your design here)
N_TURNS = 1500
ID = 0.036
OD = 0.045
H_COIL = 0.008
AWG = 34


# ================= PHYSICS ENGINE =================

def get_bz_on_axis(z):
    """Calculates axial magnetic field at distance z."""
    # Standard formula for cylindrical magnet on axis
    term1 = (D_mag + z) / np.sqrt(R_mag ** 2 + (D_mag + z) ** 2)
    term2 = z / np.sqrt(R_mag ** 2 + z ** 2)
    return (Br / 2.0) * (term1 - term2)


def calculate_resistance():
    """Calculates coil resistance based on AWG and geometry."""
    # Wire Diameter formula
    d_wire = 0.127e-3 * (92 ** ((36.0 - AWG) / 39.0))
    A_wire = np.pi * (d_wire / 2.0) ** 2

    # Average Turn Length
    r_avg = (ID + OD) / 4.0
    len_total = N_TURNS * (2 * np.pi * r_avg)

    rho_copper = 1.68e-8
    return rho_copper * len_total / A_wire


def run_simulation():
    # 1. Setup Time and Motion
    # Create an array of angles from 0 to max amplitude
    angles_deg = np.linspace(0, Theta_0, 100)
    angles_rad = np.radians(angles_deg)

    # Calculate pendulum velocity at each angle (Conservation of Energy)
    # Omega = sqrt( 2g/L * (cos(theta) - cos(theta_max)) )
    delta_h = np.cos(angles_rad) - np.cos(np.radians(Theta_0))
    omega = np.sqrt(2 * g / L_pend * np.maximum(delta_h, 0))

    # 2. Calculate Magnetic Flux Coupling
    # We slice the coil into layers to get an accurate average flux
    flux_profile = np.zeros_like(angles_deg)

    # Create mesh of coil cross-section
    r_layers = np.linspace(ID / 2, OD / 2, 5)
    z_layers = np.linspace(0, H_COIL, 5)
    turns_per_cell = N_TURNS / 25.0  # 5x5 grid

    # Distance from magnet to coil center at every angle
    # z_dist = sqrt( gap^2 + (L * sin(theta))^2 )
    z_pendulum = np.sqrt(Gap ** 2 + (L_pend * np.sin(angles_rad)) ** 2)

    # Sum flux for every layer
    for r in r_layers:
        area = np.pi * r ** 2
        for z_off in z_layers:
            # Total Z distance = Pendulum Arc + Z-offset in coil
            z_total = z_pendulum + z_off

            # B-field at this distance
            B_field = get_bz_on_axis(z_total)

            # Flux = B * Area
            flux_profile += (B_field * area) * turns_per_cell

    # 3. Calculate Voltage (EMF)
    # V = -dPhi/dt = -(dPhi/dTheta) * (dTheta/dt)
    d_flux_d_theta = np.gradient(flux_profile, angles_rad)
    emf_profile = np.abs(d_flux_d_theta * omega)

    peak_voltage = np.max(emf_profile)
    R_coil = calculate_resistance()

    # 4. Power Calculation (Threshold Model)
    if peak_voltage > V_THRESH:
        # Ohm's Law: I = (V_gen - V_batt) / R
        current = (peak_voltage - V_THRESH) / R_coil
        power_mW = (current * V_BATT) * 1000.0
    else:
        current = 0.0
        power_mW = 0.0

    return peak_voltage, R_coil, power_mW


# ================= EXECUTION =================

volts, ohms, mw = run_simulation()

print(f"Coil Config: {N_TURNS} Turns | {AWG} AWG")
print(f"Coil Resistance: {ohms:.2f} Ohms")
print(f"Peak Voltage:    {volts:.2f} V")
print(f"Threshold Req:   {V_THRESH:.2f} V")