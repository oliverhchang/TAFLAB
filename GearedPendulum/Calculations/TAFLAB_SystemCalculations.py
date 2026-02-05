import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import itertools
from mpl_toolkits.mplot3d import Axes3D


# =========================================================================
#  CLASS 1: WEC Parameter Calculator
# =========================================================================
class WECParameterCalculator:
    """
    Calculates the key physical parameters for a keel-mounted pendulum
    Wave Energy Converter (WEC) based on its design specifications.
    """

    def __init__(self, keel_length_m, pendulum_mass_kg, load_resistor_ohm, pendulum_arm_length_m):
        # --- Store Adjustable Inputs ---
        self.keel_length_m = keel_length_m
        self.pendulum_mass_kg = pendulum_mass_kg
        self.load_resistor_ohm = load_resistor_ohm
        self.pendulum_arm_length_m = pendulum_arm_length_m

        # --- FIXED PHYSICAL PARAMETERS ---
        self.boat_mass_kg = 12.0
        self.boat_radius_m = 0.090
        self.boat_length_m = 0.630
        self.plywood_density_kg_m3 = 600.0
        self.keel_width_m = 8 * 0.0254
        self.keel_thickness_m = 0.75 * 0.0254
        self.pendulum_material_density_kg_m3 = 7850.0
        self.pendulum_radius_m = 0.05
        self.motor_holding_torque_Nm = 1.26
        self.motor_rated_current_A = 2.8
        self.motor_phase_resistance_ohm = 0.9
        self.gravity_m_s2 = 9.81

        # --- HYDRODYNAMIC ESTIMATES ---
        self.hydro_damping_coeff_k = 0.5
        self.hydro_damping_coeff_p = 0.5

        # --- Perform all calculations ---
        self._calculate_all_params()

    def _calculate_all_params(self):
        """Master function to run all calculation methods."""
        self._calculate_boat_params()
        self._calculate_keel_params()
        self._calculate_pendulum_params()
        self._calculate_generator_params()

    def _calculate_boat_params(self):
        m = self.boat_mass_kg
        r = self.boat_radius_m
        self.boat_moment_of_inertia_kgm2 = 0.5 * m * r ** 2

    def _calculate_keel_params(self):
        self.keel_mass_kg = (self.plywood_density_kg_m3 *
                             self.keel_length_m *
                             self.keel_width_m *
                             self.keel_thickness_m)
        m = self.keel_mass_kg
        L = self.keel_length_m
        self.keel_moment_of_inertia_kgm2 = (1 / 3) * m * L ** 2
        self.keel_com_dist_l_k1 = L / 2.0

    def _calculate_pendulum_params(self):
        m = self.pendulum_mass_kg
        rho = self.pendulum_material_density_kg_m3
        r = self.pendulum_radius_m
        arm = self.pendulum_arm_length_m
        volume = m / rho
        self.pendulum_length_m = volume / (np.pi * r ** 2)
        L = self.pendulum_length_m
        I_cm = (1 / 4) * m * r ** 2 + (1 / 12) * m * L ** 2
        self.pendulum_com_dist_l_p2 = arm + (L / 2)
        self.pendulum_moment_of_inertia_kgm2 = I_cm + m * self.pendulum_com_dist_l_p2 ** 2

    def _calculate_generator_params(self):
        self.motor_torque_constant_Kt = self.motor_holding_torque_Nm / self.motor_rated_current_A
        self.motor_back_emf_constant_Ke = self.motor_torque_constant_Kt
        self.pto_damping_coefficient_B_PTO = (self.motor_torque_constant_Kt * self.motor_back_emf_constant_Ke) / \
                                             (self.motor_phase_resistance_ohm + self.load_resistor_ohm)


# =========================================================================
#  CLASS 2: WEC Dynamic Simulator
# =========================================================================
class WECSimulator:
    """
    Performs a dynamic simulation of the WEC using the Lagrangian method.
    """

    def __init__(self, wec_params, wave_amplitude_rad, wave_frequency_rad_s):
        self.params = wec_params
        self.wave_A = wave_amplitude_rad
        self.wave_w = wave_frequency_rad_s
        self.sol = None

    def _equations_of_motion(self, t, y):
        th1, th2, th1_dot, th2_dot = y
        m_k = self.params.keel_mass_kg
        I_k_pivot = self.params.keel_moment_of_inertia_kgm2
        m_p = self.params.pendulum_mass_kg
        I_p_pivot = self.params.pendulum_moment_of_inertia_kgm2
        l_k1 = self.params.keel_com_dist_l_k1
        L_k = self.params.keel_length_m
        l_p2 = self.params.pendulum_com_dist_l_p2
        g = self.params.gravity_m_s2
        B_k_hydro = self.params.hydro_damping_coeff_k
        B_p_hydro = self.params.hydro_damping_coeff_p
        B_PTO = self.params.pto_damping_coefficient_B_PTO

        phi = self.wave_A * np.sin(self.wave_w * t)
        phi_ddot = -self.wave_A * (self.wave_w ** 2) * np.sin(self.wave_w * t)
        c2 = np.cos(th2)
        s1 = np.sin(phi + th1)
        s2 = np.sin(phi + th1 + th2)

        M11 = I_k_pivot + m_p * L_k ** 2
        M12 = m_p * L_k * l_p2 * c2
        M21 = M12
        M22 = I_p_pivot
        M = np.array([[M11, M12], [M21, M22]])

        h = -m_p * L_k * l_p2 * np.sin(th2)
        C11 = h * th2_dot
        C12 = h * (th1_dot + th2_dot)
        C21 = -h * th1_dot
        C22 = 0
        C = np.array([[C11, C12], [C21, C22]])

        G1 = (m_k * l_k1 + m_p * L_k) * g * s1 + m_p * l_p2 * g * s2
        G2 = m_p * l_p2 * g * s2
        G = np.array([G1, G2])

        F1 = -(M11 * phi_ddot + M12 * phi_ddot)
        F2 = -(M21 * phi_ddot + M22 * phi_ddot)
        F = np.array([F1, F2])

        Q1 = -B_k_hydro * th1_dot
        Q2 = -B_p_hydro * th2_dot - B_PTO * th2_dot
        Q = np.array([Q1, Q2])

        q_dot = np.array([th1_dot, th2_dot])
        try:
            M_inv = np.linalg.inv(M)
            q_ddot = M_inv @ (F + Q - C @ q_dot - G)
        except np.linalg.LinAlgError:
            q_ddot = np.zeros(2)

        return [th1_dot, th2_dot, q_ddot[0], q_ddot[1]]

    def run_simulation(self, t_span, t_eval, initial_conditions):
        self.sol = solve_ivp(
            fun=self._equations_of_motion,
            t_span=t_span,
            y0=initial_conditions,
            t_eval=t_eval,
            method='RK45',
            dense_output=True
        )

    def calculate_phase_metric(self, transient_time_s=10):
        """
        Calculates a metric for how out-of-phase the keel and pendulum are.
        A value of -1 is perfectly out-of-phase.
        A value of +1 is perfectly in-phase.
        The goal is to find the design that minimizes this value.
        """
        if self.sol is None: return 0

        transient_mask = self.sol.t > transient_time_s
        if not np.any(transient_mask):
            transient_mask = self.sol.t >= 0

        theta1_dot = self.sol.y[2, transient_mask]
        theta2_dot = self.sol.y[3, transient_mask]

        if theta1_dot.size > 0 and theta2_dot.size > 0:
            # The product of the signs is -1 when velocities are opposite.
            # The mean of this product is our phase metric.
            phase_metric = np.mean(np.sign(theta1_dot) * np.sign(theta2_dot))
        else:
            phase_metric = 0  # No motion, so no phase relationship

        return phase_metric


# =========================================================================
#  CLASS 3: WEC Analysis and Optimization Suite
# =========================================================================
class WECAnalysisSuite:
    """
    Orchestrates parametric sweeps and visualizes results to find
    optimal WEC design parameters for phase opposition.
    """

    def __init__(self, keel_lengths, pendulum_masses, load_resistors, pendulum_arm_lengths, wave_params, sim_params):
        self.keel_lengths = keel_lengths
        self.pendulum_masses = pendulum_masses
        self.load_resistors = load_resistors
        self.pendulum_arm_lengths = pendulum_arm_lengths
        self.wave_params = wave_params
        self.sim_params = sim_params
        self.results_df = None

    def run_parametric_sweep(self):
        """
        Iterates through all combinations of design parameters, runs a
        simulation for each, and stores the resulting phase metric.
        """
        param_combinations = list(itertools.product(
            self.keel_lengths,
            self.pendulum_masses,
            self.load_resistors,
            self.pendulum_arm_lengths
        ))
        results = []

        print(f"Starting parametric sweep for {len(param_combinations)} combinations...")

        for i, (k_len, p_mass, l_res, p_arm_len) in enumerate(param_combinations):
            print(f"  Running case {i + 1}/{len(param_combinations)}: "
                  f"Keel={k_len}m, Arm={p_arm_len}m, Mass={p_mass}kg, Resistor={l_res}Ω")

            wec_design = WECParameterCalculator(k_len, p_mass, l_res, p_arm_len)
            simulator = WECSimulator(wec_design, **self.wave_params)
            simulator.run_simulation(**self.sim_params)
            phase_metric = simulator.calculate_phase_metric()

            results.append({
                'keel_length': k_len,
                'pendulum_mass': p_mass,
                'load_resistor': l_res,
                'pendulum_arm_length': p_arm_len,
                'phase_metric': phase_metric
            })

        self.results_df = pd.DataFrame(results)
        print("Parametric sweep complete.")

    def plot_optimization_results(self):
        """
        Visualizes the results of the parametric sweep as a series of heatmaps.
        """
        if self.results_df is None:
            print("Error: Must run parametric sweep first.")
            return

        g = sns.FacetGrid(
            self.results_df,
            row="keel_length",
            col="pendulum_arm_length",
            margin_titles=True,
            height=4
        )

        def draw_heatmap(*args, **kwargs):
            data = kwargs.pop('data')
            d = data.pivot(index='pendulum_mass', columns='load_resistor', values='phase_metric')
            # Annotate with 2 decimal places, use a diverging colormap
            sns.heatmap(d, annot=True, fmt=".2f", cmap="coolwarm_r", cbar=True, vmin=-1, vmax=1, **kwargs)

        g.map_dataframe(draw_heatmap)
        g.fig.suptitle('Phase Metric vs. Design Parameters (-1 is optimal)', y=1.03, fontsize=16)
        g.set_axis_labels('Load Resistor (Ω)', 'Pendulum Mass (kg)')
        g.set_titles(row_template="Keel Length = {row_name} m", col_template="Arm Length = {col_name} m")
        g.fig.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

    def plot_best_case_details(self):
        """
        Finds the design with the best phase opposition and plots its
        detailed dynamic behavior.
        """
        if self.results_df is None or self.results_df.empty:
            print("Error: Must run parametric sweep first.")
            return

        # Find the case with the MINIMUM phase metric (closest to -1)
        best_case = self.results_df.loc[self.results_df['phase_metric'].idxmin()]
        print("\n--- Best Performing Design (for Phase Opposition) ---")
        print(best_case)

        wec_design = WECParameterCalculator(
            best_case['keel_length'],
            best_case['pendulum_mass'],
            best_case['load_resistor'],
            best_case['pendulum_arm_length']
        )
        simulator = WECSimulator(wec_design, **self.wave_params)
        simulator.run_simulation(**self.sim_params)

        t = simulator.sol.t

        phi = np.rad2deg(self.wave_params['wave_amplitude_rad'] * np.sin(self.wave_params['wave_frequency_rad_s'] * t))
        th1 = np.rad2deg(simulator.sol.y[0, :])
        th2 = np.rad2deg(simulator.sol.y[1, :])

        fig, ax = plt.subplots(1, 1, figsize=(12, 6), constrained_layout=True)
        fig.suptitle(f'Detailed Dynamics of Best Phase Case (Metric: {best_case["phase_metric"]:.3f})', fontsize=16)

        ax.plot(t, phi, label=r'Boat Roll ($\phi$)', color='k', linestyle='--')
        ax.plot(t, th1, label=r'Keel Angle rel. to Boat ($\theta_1$)')
        ax.plot(t, th2, label=r'Pendulum Angle rel. to Keel ($\theta_2$)')
        ax.set_title('Angle Time Series (Phase Analysis)')
        ax.set_ylabel('Angle (degrees)')
        ax.set_xlabel('Time (s)')
        ax.grid(True)
        ax.legend()
        ax.text(0.01, 0.05, "Out-of-phase motion between $\\theta_1$ and $\\theta_2$ is desired.",
                transform=ax.transAxes, style='italic')

        plt.show()

    def plot_3d_surface(self):
        """
        Creates a 3D surface plot of phase metric vs. arm length and mass
        for the best combination of other parameters.
        """
        if self.results_df is None or self.results_df.empty:
            print("Error: Must run parametric sweep first.")
            return

        # Find the best fixed parameters (keel length and resistor)
        best_case = self.results_df.loc[self.results_df['phase_metric'].idxmin()]
        best_resistor = best_case['load_resistor']
        best_keel_length = best_case['keel_length']

        print(
            f"\nGenerating 3D plot for best fixed parameters: Keel Length = {best_keel_length}m, Resistor = {best_resistor}Ω")

        plot_df = self.results_df[
            (self.results_df['load_resistor'] == best_resistor) &
            (self.results_df['keel_length'] == best_keel_length)
            ]

        X, Y = np.meshgrid(
            plot_df['pendulum_arm_length'].unique(),
            plot_df['pendulum_mass'].unique()
        )

        Z = plot_df.pivot(
            index='pendulum_mass',
            columns='pendulum_arm_length',
            values='phase_metric'
        ).values

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        # Use a diverging colormap, centered at 0.
        surf = ax.plot_surface(X, Y, Z, cmap='coolwarm_r', edgecolor='none', vmin=-1, vmax=1)

        ax.set_title(f'Phase Metric vs. Arm Length and Mass\n(Keel={best_keel_length}m, Resistor={best_resistor}Ω)',
                     fontsize=16)
        ax.set_xlabel('Pendulum Arm Length (m)')
        ax.set_ylabel('Pendulum Mass (kg)')
        ax.set_zlabel('Phase Metric (-1 is optimal)')
        fig.colorbar(surf, shrink=0.5, aspect=5, label='Phase Metric')

        plt.show()


if __name__ == '__main__':
    # ===================================================================
    # --- 1. DEFINE PARAMETER RANGES FOR OPTIMIZATION ---
    # ===================================================================
    num_mass_points = 10
    num_arm_length_points = 10

    keel_lengths_to_test = [1.0]
    pendulum_masses_to_test = np.linspace(5.0, 15.0, num_mass_points)
    resistors_to_test = [5.0, 10.0, 15.0]
    pendulum_arm_lengths_to_test = np.linspace(0.25, 2.0, num_arm_length_points)

    # ===================================================================
    # --- 2. DEFINE WAVE & SIMULATION CONDITIONS (Fixed for the sweep) ---
    # ===================================================================
    wave_params = {
        'wave_amplitude_rad': np.deg2rad(15.0),
        # EDIT: Changed the wave period from 4.0 seconds to 2.0 seconds
        'wave_frequency_rad_s': (2 * np.pi) / 1  # 2-second period
    }

    sim_params = {
        't_span': [0, 60.0],
        't_eval': np.linspace(0, 60.0, num=60 * 30),
        'initial_conditions': [0.0, 0.0, 0.0, 0.0]
    }

    # ===================================================================
    # --- 3. RUN THE ANALYSIS SUITE ---
    # ===================================================================

    suite = WECAnalysisSuite(
        keel_lengths=keel_lengths_to_test,
        pendulum_masses=pendulum_masses_to_test,
        load_resistors=resistors_to_test,
        pendulum_arm_lengths=pendulum_arm_lengths_to_test,
        wave_params=wave_params,
        sim_params=sim_params
    )
    suite.run_parametric_sweep()
    suite.plot_optimization_results()
    suite.plot_best_case_details()
    suite.plot_3d_surface()
