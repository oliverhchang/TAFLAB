import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings('ignore')


# --- Data Loading (from your calibration script) ---
def load_test_data():
    """Load and clean the experimental data"""
    try:
        data_1ohm = pd.read_csv('1ohm_Data.csv')
        data_1ohm.columns = [col.strip().replace('(mA)', '_mA').replace('(mW)', '_mW').replace('(s)', '_s') for col in
                             data_1ohm.columns]
        data_10ohm = pd.read_csv('10ohm_Data.csv')
        data_10ohm.columns = [col.strip().replace('(mA)', '_mA').replace('(mW)', '_mW').replace('(s)', '_s') for col in
                              data_10ohm.columns]
    except FileNotFoundError as e:
        print(f"Error: {e}. Please make sure '1ohm_Data.csv' and '10ohm_Data.csv' are in the same directory.")
        return None, None

    # Process 1 ohm data
    data_1ohm_clean = data_1ohm[['Time_s', 'Current_mA', 'Power_mW', 'RPM']].copy()
    data_1ohm_clean = data_1ohm_clean.dropna()
    data_1ohm_clean = data_1ohm_clean[data_1ohm_clean['RPM'] > 10]
    data_1ohm_clean['Load_Resistance'] = 1.0
    data_1ohm_clean['Current_A'] = data_1ohm_clean['Current_mA'] / 1000
    data_1ohm_clean['Power_W'] = data_1ohm_clean['Power_mW'] / 1000

    # Process 10 ohm data
    data_10ohm_clean = data_10ohm[['Time_s', 'Current_mA', 'Power_mW', 'RPM']].copy()
    data_10ohm_clean = data_10ohm_clean.dropna()
    data_10ohm_clean = data_10ohm_clean[data_10ohm_clean['RPM'] > 10]
    data_10ohm_clean['Load_Resistance'] = 10.0
    data_10ohm_clean['Current_A'] = data_10ohm_clean['Current_mA'] / 1000
    data_10ohm_clean['Power_W'] = data_10ohm_clean['Power_mW'] / 1000

    return data_1ohm_clean, data_10ohm_clean


# --- Calibrated Generator Model (from your calibration script) ---
class StepperGenerator:
    """A physically-based model for a stepper motor acting as a generator."""

    def __init__(self):
        self.ke = 0.1
        self.phase_resistance = 2.0
        self.phase_inductance = 0.003
        self.steps_per_rev = 200

    def predict(self, load_resistance, rpm):
        """Predicts current and power, vectorized for numpy arrays."""
        v_generated = self.ke * rpm
        electrical_frequency = (rpm / 60) * (self.steps_per_rev / 2)
        inductive_reactance = 2 * np.pi * electrical_frequency * self.phase_inductance
        total_resistance = self.phase_resistance + load_resistance
        impedance = np.sqrt(total_resistance ** 2 + inductive_reactance ** 2)
        current = np.divide(v_generated, impedance, out=np.zeros_like(v_generated), where=impedance != 0)
        current = np.maximum(0, current)
        power_to_load = (current ** 2) * load_resistance
        return current, power_to_load


def calibrate_generator_model(data_1ohm, data_10ohm):
    """Calibrate the generator model parameters."""
    combined_data = pd.concat([data_1ohm, data_10ohm], ignore_index=True)
    valid_mask = (combined_data['RPM'] > 20) & (combined_data['RPM'] < 400)
    load_res = combined_data['Load_Resistance'][valid_mask].values
    rpm = combined_data['RPM'][valid_mask].values
    measured_power = combined_data['Power_W'][valid_mask].values
    generator = StepperGenerator()

    def objective_function(params):
        generator.ke, generator.phase_resistance, generator.phase_inductance = params
        _, pred_power = generator.predict(load_res, rpm)
        power_error = np.mean((pred_power - measured_power) ** 2)
        return power_error

    bounds = [(0.01, 0.5), (0.5, 10.0), (0.001, 0.05)]
    initial_params = [0.1, 3.0, 0.005]
    result = minimize(objective_function, initial_params, bounds=bounds, method='L-BFGS-B')
    if result.success:
        generator.ke, generator.phase_resistance, generator.phase_inductance = result.x
        print("\n✅ Calibration Successful!")
        print(f"  Back-EMF Constant (ke): {generator.ke:.4f} V/RPM")
        print(f"  Internal Resistance:    {generator.phase_resistance:.2f} Ω")
        print(f"  Inductance:             {generator.phase_inductance:.4f} H")
    else:
        print(f"❌ Optimization failed: {result.message}")
    return generator


# --- Theoretical Design Model (from your first script) ---
class TheoreticalModel:
    """Calculates the single, ideal operating point from the design script."""

    def __init__(self):
        # Using the key electrical constants from your theoretical script
        self.ke = 0.075  # V/RPM back-EMF constant
        self.V_charge = 5.0  # Target voltage
        self.R_int = 3.0  # Assumed stepper internal resistance

    def get_operating_point(self, R_load):
        """Calculates the target RPM and Power for a given external load."""
        V_gen = self.V_charge

        # Calculate theoretical power
        I_actual = V_gen / (self.R_int + R_load)
        P_elec = V_gen * I_actual

        # Calculate the RPM required to generate the target voltage
        rpm_motor = V_gen / self.ke

        return rpm_motor, P_elec


# --- Combined Plotting Function ---
def plot_comparison(calibrated_model, theoretical_model, data_1ohm, data_10ohm):
    """
    Overlays measured data, the calibrated model, and the theoretical target point.
    """
    datasets = [(data_1ohm, '1Ω Load'), (data_10ohm, '10Ω Load')]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Comparison: Theoretical vs. Calibrated vs. Measured', fontsize=18, fontweight='bold')

    for i, (data, label) in enumerate(datasets):
        load_r = data['Load_Resistance'].iloc[0]

        # --- Predictions ---
        # 1. Calibrated model curve
        rpm_range = np.linspace(0, data['RPM'].max() * 1.1, 200)
        calibrated_current, calibrated_power = calibrated_model.predict(load_r, rpm_range)

        # 2. Theoretical model single point
        theoretical_rpm, theoretical_power = theoretical_model.get_operating_point(load_r)

        # --- Plot Power vs RPM ---
        ax_power = axes[i, 0]
        # Plot measured data
        ax_power.scatter(data['RPM'], data['Power_W'], alpha=0.6, s=25, label='Measured Data')
        # Plot calibrated model
        ax_power.plot(rpm_range, calibrated_power, 'r-', linewidth=2.5, label='Calibrated Model')
        # Plot theoretical target point
        ax_power.plot(theoretical_rpm, theoretical_power, '*', c='gold', markersize=15, markeredgecolor='black',
                      label='Theoretical Target')

        ax_power.set_title(f'Power vs RPM ({label})', fontsize=14)
        ax_power.set_xlabel('RPM')
        ax_power.set_ylabel('Power (W)')
        ax_power.legend()
        ax_power.grid(True, linestyle=':')
        ax_power.set_xlim(left=0)
        ax_power.set_ylim(bottom=0)

        # --- Plot Current vs RPM ---
        ax_current = axes[i, 1]
        ax_current.scatter(data['RPM'], data['Current_A'], alpha=0.6, s=25, label='Measured Data')
        ax_current.plot(rpm_range, calibrated_current, 'r-', linewidth=2.5, label='Calibrated Model')

        ax_current.set_title(f'Current vs RPM ({label})', fontsize=14)
        ax_current.set_xlabel('RPM')
        ax_current.set_ylabel('Current (A)')
        ax_current.legend()
        ax_current.grid(True, linestyle=':')
        ax_current.set_xlim(left=0)
        ax_current.set_ylim(bottom=0)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    # Load experimental data
    data_1ohm, data_10ohm = load_test_data()

    if data_1ohm is not None and data_10ohm is not None:
        # Calibrate the physically-based generator model
        calibrated_model = calibrate_generator_model(data_1ohm, data_10ohm)

        # Instantiate the theoretical design model
        theoretical_model = TheoreticalModel()

        # Plot the comparison
        plot_comparison(calibrated_model, theoretical_model, data_1ohm, data_10ohm)
