import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline


class AnalyticalWEC:
    def __init__(self, num_magnets=4):
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
        self.M = self.M_single * self.num_magnets  # Total sled mass (kg)

        # --- Coil & Rectifier Parameters (Parallel Topology) ---
        self.num_coils_per_side = 5
        self.total_coils = self.num_coils_per_side * 2
        self.coil_spacing = 0.046
        self.R_in = 0.006
        self.R_out = 0.018
        self.d_c = 0.00635

        self.N_turns = 1200
        self.R_int_per_coil = 36.0

        self.R_int = self.R_int_per_coil / self.total_coils

        self.R_load = 10

        self.V_drop = 1.2

        # --- Geometry/Air Gap ---
        self.air_gap = 0.0012
        # Z-distance from magnet center to coil center
        self.z_dist = (self.d_m / 2) + self.air_gap + (self.d_c / 2)

        # --- Linear Track Parameters ---
        self.tilt_angle = np.deg2rad(45.0)
        self.track_length = 0.40  # Total slide distance [m] (400mm)
        self.zeta_i = 0.0  # Frictionless (Zero internal damping)

    def calculate_Bz_vectorized(self, r_p, z_p):
        """Vectorized Biot-Savart for a single cylindrical magnet."""
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
        """Calculates individual coupling K for every single coil independently."""
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

            # Assign symmetric top and bottom coils to the array
            Ks[c_idx] = k_coil
            Ks[c_idx + self.num_coils_per_side] = k_coil

        return Ks

    def generate_lookup_table(self):
        print(f"Mapping Individual Coils ({int(self.track_length * 1000)}mm Track, 32 AWG)...")
        num_pts = 60
        u_pts = np.linspace(-self.track_length / 2, self.track_length / 2, num_pts)
        k_pts = []
        for i, u in enumerate(u_pts):
            print(f"  Mapping Progress: {int((i / num_pts) * 100)}%...", end='\r')
            k_pts.append(self.get_individual_Ks(u))

        self.u_table = u_pts
        self.k_table = np.array(k_pts)  # Shape: (60, 10)

        # Interpolate all 10 coils simultaneously
        self.k_spline = CubicSpline(self.u_table, self.k_table, axis=0, extrapolate=True)
        print("\nMapping complete.")

    def get_K(self, u):
        """Returns array of 10 K-values for current position."""
        return self.k_spline(u)

    def solve_parallel_network(self, V_induced_array):
        """
        Nodal solver for 10 parallel rectified sources feeding a common load.
        Only coils generating enough voltage to overcome V_L + V_drop will conduct.
        """
        # Subtract diode drop. If less than 0, diode blocks current.
        V_abs = np.maximum(np.abs(V_induced_array) - self.V_drop, 0)
        V_sorted = np.sort(V_abs)[::-1]

        V_L = 0
        if V_sorted[0] > 0:
            # Iteratively find how many coils are "active" and driving the load
            for N in range(1, len(V_abs) + 1):
                V_L_test = np.sum(V_sorted[:N]) / self.R_int_per_coil / (1 / self.R_load + N / self.R_int_per_coil)
                if N == len(V_abs) or V_L_test >= V_sorted[N]:
                    V_L = V_L_test
                    break

        # Calculate current drawn from each coil based on the shared load voltage
        I_array = np.maximum(0, V_abs - V_L) / self.R_int_per_coil

        # Re-apply directional sign so electromagnetic force opposes motion
        I_directed = I_array * np.sign(V_induced_array)
        return I_directed, V_L

    def system_dynamics(self, t, y):
        u, v = y
        if u > self.track_length / 2:
            return [0.0, 0.0]

        F_gravity = self.M * self.g * np.sin(self.tilt_angle)

        # Get individual coupling factors and raw induced voltages
        K_array = self.get_K(u)
        V_induced = K_array * v

        # Solve the non-linear parallel electrical circuit
        I_directed, _ = self.solve_parallel_network(V_induced)

        # Calculate total electromagnetic braking drag across all coils
        F_em = -np.sum(K_array * I_directed)

        dvdt = (F_gravity + F_em) / self.M
        return [v, dvdt]

    def simulate_slide(self):
        self.generate_lookup_table()
        y0 = [-self.track_length / 2, 0.001]

        # High drag from parallel matching might slow the sled down
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
            u_sim = sol.y[0]
            v_sim = sol.y[1]
            t_sim = sol.t

        # Post-Processing: Reconstruct voltages and power from the track history
        k_sim = np.array([self.get_K(pos) for pos in u_sim])
        v_induced_sim = k_sim * v_sim[:, np.newaxis]

        p_load = np.zeros(len(t_sim))
        v_load_sim = np.zeros(len(t_sim))

        for i in range(len(t_sim)):
            _, V_L = self.solve_parallel_network(v_induced_sim[i])
            v_load_sim[i] = V_L
            p_load[i] = (V_L ** 2) / self.R_load

        # Extract the highest generating coil voltage for visualization
        max_v_induced = np.max(np.abs(v_induced_sim), axis=1)

        return t_sim, u_sim, v_sim, p_load, v_load_sim, max_v_induced


if __name__ == "__main__":
    magnets_to_test = 4
    wec = AnalyticalWEC(num_magnets=magnets_to_test)
    t, u, vel, p, v_load, v_ind = wec.simulate_slide()

    # -------------------------------------------------------------
    # CONSOLE SUMMARY PRINTED FIRST
    # -------------------------------------------------------------
    print(f"\n--- Linear Slide Summary (32 AWG, Parallel Rectifiers) ---")
    print(f"Track Length:    {wec.track_length * 1000:.0f} mm")
    print(f"Parallel Load:   {wec.R_load:.1f} Ohm")
    print(f"Diode Drop:      {wec.V_drop:.1f} V per coil")
    print(f"Peak Load DC:    {np.max(v_load):.2f} V")
    print(f"Max Power:      {np.max(p) * 1000:.2f} mW")
    print(f"Final Speed:     {np.max(vel):.2f} m/s (Peak), {vel[-1]:.2f} m/s (End)")

    active_mask = p > 1e-7
    active_p = p[active_mask]
    active_v_load = v_load[active_mask]
    active_v_ind = v_ind[active_mask]

    avg_active_p = np.mean(active_p) * 1000 if len(active_p) > 0 else 0.0
    avg_active_v_load = np.mean(active_v_load) if len(active_v_load) > 0 else 0.0
    avg_active_v_ind = np.mean(active_v_ind) if len(active_v_ind) > 0 else 0.0

    print(f"Avg Active Power: {avg_active_p:.2f} mW (excluding idle time)")
    print(f"Avg Raw Coil Volts:{avg_active_v_ind:.2f} V (Peak Coil)")
    print(f"Avg Sensor Volts: {avg_active_v_load:.2f} V (DC Load)")

    # -------------------------------------------------------------
    # GRAPH PLOTTING AND DISPLAY
    # -------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    axes[0].plot(t, u * 1000, 'b', lw=1.5, label='Sled Position (mm)')
    axes[0].set_ylabel('Displacement (mm)')

    axes[1].plot(t, v_ind, 'r--', lw=1.0, alpha=0.5, label='Max Raw Coil Voltage')
    axes[1].plot(t, v_load, 'r', lw=1.5, label='Rectified Load Voltage (DC)')
    axes[1].set_ylabel('Voltage (V)')

    axes[2].plot(t, p * 1000, 'g', lw=1.5, label='Power to Load (mW)')
    axes[2].set_ylabel('Power (mW)')
    axes[2].set_xlabel('Time (s)')

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')

    plt.suptitle(f'Parallel Configuration (32 AWG): {wec.num_magnets} Magnets ({wec.R_load:.1f}$\\Omega$ Load)',
                 fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()