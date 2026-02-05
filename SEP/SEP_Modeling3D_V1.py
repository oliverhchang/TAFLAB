import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. CONSTANTS & PARAMETERS
# ==========================================
# Physics
MU_0 = 4 * np.pi * 1e-7
g = 9.81

# Optimized Geometry (Phase 2 Results)
R_pendulum = 0.25  # Radius of pendulum (m)
R_shell = 0.25 + 0.013  # 10mm gap
Coil_Radius = 0.0205  # OPTIMIZED RADIUS (20.5 mm)

# Electrical Components
N_turns = 500  # Turns per coil
Wire_Gauge = '30AWG'  # Standard 30 AWG
Diode_Drop = 0.3  # Schottky Diode Drop (e.g., 1N5817)
Capacitance = 470e-6  # 470 uF smoothing capacitor
Load_R = 26  # Load Resistor (Ohms) - Matched to ~22 Ohm Internal R

# Calculate Internal Coil Resistance (The "Source Impedance")
# 30 AWG is approx 0.338 Ohms/meter
Wire_Resistivity = 0.338
Wire_Length = N_turns * 2 * np.pi * Coil_Radius
Internal_R = Wire_Length * Wire_Resistivity

print(f"--- SYSTEM PARAMETERS ---")
print(f"Coil Internal Resistance: {Internal_R:.2f} Ohms")
print(f"Load Resistance:          {Load_R:.2f} Ohms")
print(f"-------------------------")


# ==========================================
# 2. PHYSICS ENGINE (3D Dipole Model)
# ==========================================
class SphericalPendulum3D:
    def __init__(self):
        # Quadrilateral Array (36 Magnets)
        self.magnets = self.generate_quadrilateral_array()
        # Optimized Coil Array (12 coils at 60-deg ring)
        self.coils = self.generate_coil_ring(theta_deg=60, count=12)

        # Magnet Properties (N40)
        self.Br = 1.26
        self.mag_vol = np.pi * (0.015 ** 2) * 0.010
        self.m_dipole = (self.Br / MU_0) * self.mag_vol

    def generate_quadrilateral_array(self):
        magnets = []
        # Layers: (Count, Polarity, Theta)
        layers = [(4, 1, 20), (8, -1, 40), (12, 1, 60), (12, -1, 80)]
        for num, pol, theta_deg in layers:
            theta = np.radians(theta_deg)
            for i in range(num):
                phi = np.radians(i * (360 / num))
                magnets.append({'r': R_pendulum, 'theta': theta, 'phi': phi,
                                'pol': pol, 'vec': self.sph_to_cart_vec(theta, phi) * pol})
        return magnets

    def generate_coil_ring(self, theta_deg, count):
        coils = []
        theta = np.radians(theta_deg)
        for i in range(count):
            phi = np.radians(i * (360 / count))
            pos = self.sph_to_cart_pos(R_shell, theta, phi)
            coils.append({'pos': pos, 'norm': -pos / np.linalg.norm(pos),
                          'area': np.pi * Coil_Radius ** 2})
        return coils

    # --- Math Helpers ---
    def sph_to_cart_pos(self, r, theta, phi):
        return np.array([r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), -r * np.cos(theta)])

    def sph_to_cart_vec(self, theta, phi):
        return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), -np.cos(theta)])

    def get_rotation_matrix(self, alpha, beta):
        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta), np.sin(beta)
        Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
        Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
        return np.dot(Ry, Rx)

    # --- Flux Calculation ---
    def calculate_flux(self, alpha, beta):
        R_mat = self.get_rotation_matrix(alpha, beta)
        total_flux = 0

        # Rotate Magnets
        rotated_mags_pos = []
        rotated_mags_vec = []
        for mag in self.magnets:
            p0 = self.sph_to_cart_pos(mag['r'], mag['theta'], mag['phi'])
            rotated_mags_pos.append(np.dot(R_mat, p0))
            rotated_mags_vec.append(np.dot(R_mat, mag['vec']) * self.m_dipole)

        # Sum Flux through Coils
        for coil in self.coils:
            coil_flux = 0
            for i in range(len(self.magnets)):
                r_vec = coil['pos'] - rotated_mags_pos[i]
                dist = np.linalg.norm(r_vec)
                # Optimization: Ignore magnets too far away (>6cm)
                if dist > 0.06: continue

                # Dipole Field B formula
                r_hat = r_vec / dist
                m = rotated_mags_vec[i]
                B = (MU_0 / (4 * np.pi * dist ** 3)) * (3 * np.dot(m, r_hat) * r_hat - m)

                coil_flux += np.dot(B, coil['norm']) * coil['area']

            # Add absolute flux (Simulating rectification per coil or ideal series)
            total_flux += abs(coil_flux)

            # Realistic Coupling Factor (Flux Leakage)
        return total_flux * 0.7

    # ==========================================


# 3. ELECTRONICS SIMULATION (UPDATED)
# ==========================================
def simulate_circuit(t_array, flux_array):
    dt = t_array[1] - t_array[0]

    # A. Calculate Raw EMF (Faraday's Law: V = -N * dPhi/dt)
    dPhi_dt = np.gradient(flux_array, dt)
    v_open_circuit = -N_turns * dPhi_dt

    # B. Full-Bridge Rectifier + Voltage Divider (The "Real Physics" Update)
    # 1. Diode Loss: The source must overcome the diode drop first
    v_source_rect = np.abs(v_open_circuit) - (2 * Diode_Drop)
    v_source_rect = np.maximum(v_source_rect, 0)  # Clip negative values

    # 2. Voltage Divider: V_load = V_source * (R_load / (R_load + R_internal))
    # This accounts for voltage lost as heat inside the copper coil
    v_load_input = v_source_rect * (Load_R / (Load_R + Internal_R))

    # C. Smoothing Capacitor Logic
    v_cap = np.zeros_like(v_load_input)
    current_v = 0

    for i in range(len(t_array)):
        # Charging Phase
        if v_load_input[i] > current_v:
            current_v = v_load_input[i]
        # Discharging Phase (RC Decay)
        else:
            decay = np.exp(-dt / (Load_R * Capacitance))
            current_v = current_v * decay

        v_cap[i] = current_v

    # D. Power Calculation (P = V^2 / R)
    power = (v_cap ** 2) / Load_R

    return v_open_circuit, v_load_input, v_cap, power


# ==========================================
# 4. MAIN RUNNER
# ==========================================
if __name__ == "__main__":
    print("Initializing 3D Model...")
    sim = SphericalPendulum3D()

    # 1. Simulate Dynamics (1.5 seconds)
    print("Simulating Swing Dynamics...")
    t = np.linspace(0, 1.5, 300)
    alpha = np.radians(20) * np.cos(2 * np.pi * 1.0 * t)  # Pitch
    beta = np.radians(5) * np.sin(2 * np.pi * 1.0 * t)  # Roll

    # 2. Calculate Flux
    print("Calculating Magnetic Flux...")
    flux = []
    for i in range(len(t)):
        f = sim.calculate_flux(alpha[i], beta[i])
        flux.append(f)
    flux = np.array(flux)

    # 3. Simulate Electronics
    print("Simulating Electronics...")
    v_oc, v_in, v_dc, p_out = simulate_circuit(t, flux)

    # 4. Plot Results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot 1: Voltage Stages
    ax1.plot(t, v_oc, label='Open Circuit EMF (Ideal)', color='gray', alpha=0.3, linestyle='--')
    ax1.plot(t, v_in, label='Load Input (After Diodes & Divider)', color='orange')
    ax1.plot(t, v_dc, label=f'Battery/Cap Voltage', color='blue', linewidth=2)
    ax1.set_title(f'Voltage Regulation (Internal R={Internal_R:.1f}$\Omega$ vs Load R={Load_R}$\Omega$)')
    ax1.set_ylabel('Voltage (V)')
    ax1.legend(loc='upper right')
    ax1.grid(True)

    # Plot 2: Usable Power
    ax2.plot(t, p_out * 1000, color='green', fillstyle='full')
    ax2.fill_between(t, p_out * 1000, color='green', alpha=0.2)
    ax2.set_title(f'Real Power Output (Load: {Load_R} Ohms)')
    ax2.set_ylabel('Power (mW)')
    ax2.set_xlabel('Time (s)')
    ax2.grid(True)

    # Calculate Stats
    avg_power = np.mean(p_out) * 1000
    total_energy = np.trapz(p_out, t) * 1000  # mJ

    print("-" * 40)
    print(f"SIMULATION RESULTS (Realistic Physics)")
    print(f"Peak Open Circuit V: {np.max(np.abs(v_oc)):.2f} V")
    print(f"Peak Load Voltage:   {np.max(v_dc):.2f} V (Expected Drop due to Divider)")
    print(f"Average Power:       {avg_power:.2f} mW")
    print(f"Total Energy:        {total_energy:.2f} mJ")
    print("-" * 40)

    plt.tight_layout()
    plt.show()