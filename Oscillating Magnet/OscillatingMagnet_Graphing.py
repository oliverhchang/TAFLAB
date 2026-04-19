import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline


class AnalyticalWEC:
    def __init__(self, num_magnets=4, R_load=None):
        # --- Physical Constants ---
        self.mu0 = 4 * np.pi * 1e-7
        self.g = 9.81

        # --- Magnet Parameters (Linear Alternating Array) ---
        self.num_magnets = num_magnets
        self.Br = 1.45  # Remanence [T] (N52)
        self.R_m = 0.0125  # Magnet radius [m] (25mm diameter)
        self.d_m = 0.025  # Magnet thickness [m]
        self.mag_spacing = 0.026  # Center-to-center spacing
        self.M_single = 0.092
        self.M = self.M_single * self.num_magnets + 0.050  # Total sled mass (kg)

        # --- Coil & Rectifier Parameters (Parallel Topology) ---
        self.num_coils_per_side = 5
        self.total_coils = self.num_coils_per_side * 2
        self.coil_spacing = 0.046
        self.R_in = 0.008
        self.R_out = 0.036
        self.d_c = 0.00635

        self.N_turns = 1300
        self.R_int_per_coil = 36.0

        # Parallel Internal Resistance (~5.2 Ohms equivalent)
        self.R_int = self.R_int_per_coil / self.total_coils

        # Set Load Resistance (or default to impedance matched)
        self.R_load = R_load if R_load is not None else self.R_int
        self.V_drop = 0.6  # Schottky diode drop

        # --- Geometry & Mechanical Damping (Wang et al. Eq 34) ---
        self.air_gap = 0.0012
        self.z_dist = (self.d_m / 2) + self.air_gap + (self.d_c / 2)
        self.tilt_angle = np.deg2rad(35.0)
        self.track_length = 0.40

        # Mechanical Damping separated into Viscous and Equivalent Coulomb Friction
        # (Replacing the arbitrary aggregated 0.70 with the formal breakdown)
        self.C_m = 0.25  # Viscous mechanical sliding damping (C_m)
        self.C_f = 0.45  # Equivalent damping coefficient from Coulomb friction (C_f)
        self.C_mech = self.C_m + self.C_f  # Total equivalent mechanical damping

    def calculate_Bz_vectorized(self, r_p, z_p):
        n_theta, n_z = 50, 15
        theta_q = np.linspace(0, 2 * np.pi, n_theta)
        z_q = np.linspace(-self.d_m / 2, self.d_m / 2, n_z)
        TH, ZQ = np.meshgrid(theta_q, z_q)
        eps = 1e-6
        dist_sq = (r_p - self.R_m * np.cos(TH)) ** 2 + \
                  (self.R_m * np.sin(TH)) ** 2 + \
                  (z_p - ZQ) ** 2 + eps

        integrand = self.R_m * (r_p * np.cos(TH) - self.R_m) / (dist_sq ** 1.5)
        integral = np.trapezoid(np.trapezoid(integrand, theta_q, axis=1), z_q, axis=0)
        return -(self.Br / (4 * np.pi)) * integral

    def get_individual_Ks(self, u):
        Ks = np.zeros(self.total_coils)
        coil_positions = (np.arange(self.num_coils_per_side) - (self.num_coils_per_side - 1) / 2) * self.coil_spacing
        mag_offsets = (np.arange(self.num_magnets) - (self.num_magnets - 1) / 2) * self.mag_spacing

        r_pts = np.linspace(self.R_in, self.R_out, 12)
        dx_eps = 0.0005

        for c_idx, c_pos in enumerate(coil_positions):
            k_coil = 0
            for m_idx, m_off in enumerate(mag_offsets):
                polarity = 1 if m_idx % 2 == 0 else -1
                m_pos = u + m_off
                dx = m_pos - c_pos

                bz_vals = np.array(
                    [self.calculate_Bz_vectorized(np.sqrt(dx ** 2 + r ** 2), self.z_dist) for r in r_pts])
                phi_1 = np.trapezoid(bz_vals * 2 * np.pi * r_pts, r_pts) * self.N_turns

                dx_shifted = dx + dx_eps
                bz_vals_2 = np.array(
                    [self.calculate_Bz_vectorized(np.sqrt(dx_shifted ** 2 + r ** 2), self.z_dist) for r in r_pts])
                phi_2 = np.trapezoid(bz_vals_2 * 2 * np.pi * r_pts, r_pts) * self.N_turns

                k_coil += polarity * (phi_2 - phi_1) / dx_eps

            Ks[c_idx] = k_coil
            Ks[c_idx + self.num_coils_per_side] = k_coil

        return Ks

    def generate_lookup_table(self):
        num_pts = 60
        u_pts = np.linspace(-self.track_length / 2, self.track_length / 2, num_pts)
        k_pts = []
        for i, u in enumerate(u_pts):
            k_pts.append(self.get_individual_Ks(u))

        self.u_table = u_pts
        self.k_table = np.array(k_pts)
        self.k_spline = CubicSpline(self.u_table, self.k_table, axis=0, extrapolate=True)

    def get_K(self, u):
        return self.k_spline(u)

    def solve_parallel_network(self, V_induced_array):
        V_abs = np.maximum(np.abs(V_induced_array) - self.V_drop, 0)
        V_sorted = np.sort(V_abs)[::-1]

        V_L = 0
        if V_sorted[0] > 0:
            for N in range(1, len(V_abs) + 1):
                V_L_test = np.sum(V_sorted[:N]) / self.R_int_per_coil / (1 / self.R_load + N / self.R_int_per_coil)
                if N == len(V_abs) or V_L_test >= V_sorted[N]:
                    V_L = V_L_test
                    break

        I_array = np.maximum(0, V_abs - V_L) / self.R_int_per_coil
        I_directed = I_array * np.sign(V_induced_array)
        return I_directed, V_L

    def system_dynamics(self, t, y):
        u, v = y
        if u > self.track_length / 2:
            return [0.0, 0.0]

        # 1. Driving Force: Gravity component down the 30 deg incline
        F_gravity = self.M * self.g * np.sin(self.tilt_angle)

        K_array = self.get_K(u)
        V_induced = K_array * v
        I_directed, _ = self.solve_parallel_network(V_induced)

        # 2. Electromagnetic Damping (Lorentz Force)
        # Wang et al. Eq 32 defines this for a standard loop: C_e = K^2 / (R_int + R_load)
        # For a parallel array with diode voltage drops, the rigorous instantaneous equivalent is:
        F_em = -np.sum(K_array * I_directed)

        # 3. Mechanical Damping (Wang et al. Eq 34)
        # Formulates the friction as equivalent mechanical damping: F_mech = -(C_m + C_f) * v
        F_mech = -self.C_mech * v

        # Newton's Second Law
        dvdt = (F_gravity + F_em + F_mech) / self.M
        return [v, dvdt]

    def simulate_slide(self, recalculate_map=True):
        if recalculate_map or not hasattr(self, 'k_spline'):
            self.generate_lookup_table()

        y0 = [-self.track_length / 2, 0.001]
        t_span = (0, 2.5)
        t_eval = np.linspace(0, 2.5, 2000)

        def hit_end(t, y):
            return (self.track_length / 2) - y[0]

        hit_end.terminal = True
        hit_end.direction = -1

        sol = solve_ivp(self.system_dynamics, t_span, y0, method='LSODA',
                        t_eval=t_eval, events=hit_end)

        if sol.status == 1:
            valid_points = len(sol.t) - 1
            u_sim = sol.y[0][:valid_points]
            v_sim = sol.y[1][:valid_points]

            pad_len = len(t_eval) - valid_points
            if pad_len > 0:
                u_sim = np.pad(u_sim, (0, pad_len), 'constant', constant_values=self.track_length / 2)
                v_sim = np.pad(v_sim, (0, pad_len), 'constant', constant_values=0.0)
            t_sim = t_eval
        else:
            u_sim, v_sim, t_sim = sol.y[0], sol.y[1], sol.t

        k_sim = np.array([self.get_K(pos) for pos in u_sim])
        v_induced_sim = k_sim * v_sim[:, np.newaxis]
        p_load = np.zeros(len(t_sim))

        for i in range(len(t_sim)):
            _, V_L = self.solve_parallel_network(v_induced_sim[i])
            p_load[i] = (V_L ** 2) / self.R_load

        return p_load


# --- HELPER FUNCTION ---
def get_avg_active_power(p_load):
    active_mask = p_load > 1e-7
    active_p = p_load[active_mask]
    return np.mean(active_p) * 1000 if len(active_p) > 0 else 0.0


# =====================================================================
# MAIN SWEEP EXECUTION
# =====================================================================
if __name__ == "__main__":
    print("==================================================")
    print(" WEC PARAMETRIC SWEEP ANALYSIS (32 AWG, Parallel)")
    print("  * Including C_mech (Mechanical Damping) *")
    print("==================================================\n")

    # -------------------------------------------------------------
    # SWEEP 1: Number of Magnets (Fixed 10 Ohm Load)
    # -------------------------------------------------------------
    magnet_counts = [1, 2, 3, 4]
    power_vs_magnets = []

    print("--- Running Sweep 1: Number of Magnets (Load = 10Ω) ---")
    for m in magnet_counts:
        print(f"Simulating {m} magnet(s)... (Calculating spatial fields)")
        wec_mag = AnalyticalWEC(num_magnets=m, R_load=10.0)
        p_load = wec_mag.simulate_slide(recalculate_map=True)
        avg_p = get_avg_active_power(p_load)
        power_vs_magnets.append(avg_p)
        print(f"   -> Avg Active Power: {avg_p:.2f} mW")

    # -------------------------------------------------------------
    # SWEEP 2: Load Resistance (Fixed 4 Magnets)
    # -------------------------------------------------------------
    resistances = [1, 2.5, 5, 10, 15, 20, 30, 40, 50]
    power_vs_resistance = []

    print("\n--- Running Sweep 2: Load Resistance (Magnets = 4) ---")
    wec_res = AnalyticalWEC(num_magnets=4)
    print("Pre-calculating magnetic spatial fields for 4 magnets (Just once)...")
    wec_res.generate_lookup_table()

    for r in resistances:
        print(f"Simulating Load = {r} Ohms...", end='\r')
        wec_res.R_load = r
        p_load = wec_res.simulate_slide(recalculate_map=False)
        avg_p = get_avg_active_power(p_load)
        power_vs_resistance.append(avg_p)
    print(f"Simulating Load = {resistances[-1]} Ohms... Done!      \n")

    # -------------------------------------------------------------
    # EXPERIMENTAL DATA (Added from Physical Testing)
    # -------------------------------------------------------------
    mag_data = {
        'Test_ID': ['#1.1', '#1.2', '#2.1', '#2.2', '#2.3', '#2.4', '#3.1', '#3.2', '#3.3', '#3.4', '#3.5', '#4.1',
                    '#4.2', '#4.3'],
        'Magnets': [1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4],
        'AvgActivePower': [22.644, 38.322, 71.465, 61.057, np.nan, 57.291, 131.809, 91.479, 73.165, 73.165, 73.165,
                           193.969, 175.815, 192.010],
        'SE_ActivePower': [0, 0, 3.9173, 2.4043, 0, 2.1612, 5.4045, 3.0579, 2.5949, 2.5949, 2.5949, 6.2005, 6.5544,
                           8.2911],

    }

    # Sourced from the previous testing regarding Resistance vs Power
    res_data = {
        'Resistance': [1, 10, 20, 30, 30, 40, 40],
        'AvgActivePower': [67.341, 135.302, 178.601, 129.497, 184.936, 145.587, 159.329],
        'SE_ActivePower': [2.229, 4.245, 6.808, 4.617, 7.486, 5.368, 7.701]
    }

    # -------------------------------------------------------------
    # GRAPH PLOTTING AND DISPLAY
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Graph 1: Power vs Magnets ---
    # Analytical Curve
    axes[0].plot(magnet_counts, power_vs_magnets, 'b-', lw=2, label='Analytical (Avg Active Power)')

    # Experimental Points: Avg Active Power
    m_x = np.array(mag_data['Magnets'])
    p_act = np.array(mag_data['AvgActivePower'])
    se_act = np.array(mag_data['SE_ActivePower'])
    valid_act = ~np.isnan(p_act) & (p_act > 0)
    axes[0].errorbar(m_x[valid_act], p_act[valid_act], yerr=se_act[valid_act],
                     fmt='sg', alpha=0.8, markersize=6, capsize=4, label='Exp: Avg Active Power')


    axes[0].set_title('Avg Power vs. Number of Magnets\n(Fixed 10$\Omega$ Load)', fontsize=12)
    axes[0].set_xlabel('Number of Magnets')
    axes[0].set_ylabel('Power (mW)')
    axes[0].set_xticks(magnet_counts)
    axes[0].grid(True, alpha=0.4)
    axes[0].legend()

    # --- Graph 2: Power vs Resistance ---
    # Analytical Curve
    axes[1].plot(resistances, power_vs_resistance, 'r-', lw=2, label='Analytical (Avg Active Power)')

    # Experimental Points: Avg Active Power
    r_x = np.array(res_data['Resistance'])
    r_p = np.array(res_data['AvgActivePower'])
    r_se = np.array(res_data['SE_ActivePower'])
    axes[1].errorbar(r_x, r_p, yerr=r_se, fmt='sg', alpha=0.8, markersize=6, capsize=4, label='Exp: Avg Active Power')

    # Add a vertical line to show where the Equivalent Internal Resistance is
    axes[1].axvline(x=5.2, color='k', linestyle='--', alpha=0.5, label='Internal Imp. (5.2$\Omega$)')

    axes[1].set_title('Avg Power vs. Load Resistance\n(Fixed 4 Magnets)', fontsize=12)
    axes[1].set_xlabel('Load Resistance ($\Omega$)')
    axes[1].set_ylabel('Power (mW)')
    axes[1].grid(True, alpha=0.4)
    axes[1].legend()

    plt.tight_layout()
    plt.show()