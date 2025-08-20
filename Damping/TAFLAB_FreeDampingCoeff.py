import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def analyze_damping_data(filepath):
    """
    Loads experimental free decay data from a CSV file, calculates the
    damping ratio for each trial, and provides the final equation to
    calculate the damping coefficient.

    Args:
        filepath (str): The path to the CSV file containing the data.
    """
    file_path = "Wave Data _ Free Damping Testing Data - Sheet1.csv"
    try:
        # Load the CSV file. The header is on the first row (index 0).
        df = pd.read_csv(filepath, header=0)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return

    # The column names are on two levels. We can simplify them.
    # The structure is ('Trial 1', 'Time'), ('Trial 1', 'Angle'), etc.
    # Let's create single-level column names like 'Time_1', 'Angle_1'.
    df.columns = [f'{c[1]}_{c[0].split(" ")[1]}' for c in df.columns.str.split(',')]

    num_trials = 5
    results = []

    plt.figure(figsize=(15, 8))

    for i in range(1, num_trials + 1):
        # Select the time and angle columns for the current trial and drop any empty rows
        time = df[f'Time_{i}'].dropna()
        angle = df[f'Angle_{i}'].dropna()

        # --- Step 1: Find the positive peaks in the data ---
        # We use find_peaks to get the indices of the positive peaks.
        # The initial displacement at t=0 is the first peak.
        peak_indices, _ = find_peaks(angle, height=0)

        # Add the first data point (t=0) as the initial peak
        all_peak_indices = np.insert(peak_indices, 0, 0)

        # Get the time and angle values at these peak indices
        peak_times = time[all_peak_indices].values
        peak_amplitudes = angle[all_peak_indices].values

        # Plot the raw data and the detected peaks for visualization
        plt.plot(time, angle, '-', label=f'Trial {i} Data', alpha=0.6)
        plt.plot(peak_times, peak_amplitudes, 'x', label=f'Trial {i} Peaks', markersize=8)

        # --- Step 2: Calculate Logarithmic Decrement and Damping Ratio ---
        if len(peak_amplitudes) > 1:
            # We will use the first peak (A1) and the second peak (A2) for a stable calculation
            # N = 1 because we are comparing consecutive peaks.
            A1 = peak_amplitudes[0]
            A2 = peak_amplitudes[1]
            N = 1

            # Logarithmic Decrement (delta)
            delta = (1 / N) * np.log(A1 / A2)

            # Damping Ratio (zeta)
            zeta = delta / np.sqrt((2 * np.pi) ** 2 + delta ** 2)

            # --- Step 3: Calculate the Damped Period ---
            # The period is the time difference between consecutive peaks
            damped_period_Td = peak_times[1] - peak_times[0]

            results.append({'trial': i, 'zeta': zeta, 'damped_period_Td': damped_period_Td, 'delta': delta})

    plt.title('Free Vibration Decay Data and Detected Peaks')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (degrees)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # --- Step 4: Average the results and present the final calculation ---
    if not results:
        print("No valid peaks found to perform analysis.")
        return

    results_df = pd.DataFrame(results)
    print("--- Analysis Results per Trial ---")
    print(results_df.to_string(index=False))

    avg_zeta = results_df['zeta'].mean()
    avg_Td = results_df['damped_period_Td'].mean()

    print("\n" + "=" * 50)
    print("--- Averaged System Properties ---")
    print(f"Average Damping Ratio (ζ): {avg_zeta:.4f}")
    print(f"Average Damped Period (Td): {avg_Td:.3f} s")
    print("=" * 50)

    # --- Step 5: Final Calculation for the Damping Coefficient B ---
    # Calculate frequencies
    omega_d = (2 * np.pi) / avg_Td  # Damped natural frequency
    omega_n = omega_d / np.sqrt(1 - avg_zeta ** 2)  # Undamped natural frequency

    print("\n--- Final Damping Coefficient (B) Calculation ---")
    print("The final step is to calculate the damping coefficient 'B'.")
    print("This requires the Moment of Inertia 'I' of the pendulum used in the test.")
    print("\nUse the following formula:")
    print(f"B = 2 * ζ * I * ω_n")
    print("  = 2 * {avg_zeta:.4f} * I * {omega_n:.4f}")
    print(f"  = {2 * avg_zeta * omega_n:.4f} * I")
    print("\nTo get the final value for B (in N*m*s/rad), you must calculate 'I' for")
    print("your specific pendulum (mass and dimensions) and multiply it by the coefficient above.")
    print("=" * 50)


if __name__ == '__main__':
    # Make sure the CSV file is in the same directory as this script,
    # or provide the full path to the file.
    file_path = 'Wave Data _ Free Damping Testing Data - Sheet1.csv'
    analyze_damping_data(file_path)
