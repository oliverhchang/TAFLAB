import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt


def load_and_preprocess(filepath):
    """
    Loads the CSV file, converts units, and calculates
    time derivatives needed for analysis.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

    # Force columns to be numeric
    df['Timestamp_ms'] = pd.to_numeric(df['Timestamp_ms'], errors='coerce')
    cols_to_convert = ['Voltage_V', 'Current_mA', 'Power_mW',
                       'AccX_g', 'AccY_g', 'AccZ_g',
                       'Roll_deg', 'Pitch_deg', 'Yaw_deg']
    for col in cols_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=['Timestamp_ms'], inplace=True)
    df.fillna(0, inplace=True)

    # 1. Time calculations
    df['Time_s'] = df['Timestamp_ms'] / 1000.0
    df['delta_t_s'] = df['Time_s'].diff().fillna(0)

    # 2. Electrical calculations
    df['Power_W'] = df['Power_mW'] / 1000.0
    df['Energy_J'] = df['Power_W'] * df['delta_t_s']

    # 3. Motion calculations
    df['Velocity_Pitch_dps'] = df['Pitch_deg'].diff().fillna(0) / df['delta_t_s']
    df['Velocity_Roll_dps'] = df['Roll_deg'].diff().fillna(0) / df['delta_t_s']
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df['Acc_Total_g'] = np.sqrt(df['AccX_g'] ** 2 + df['AccY_g'] ** 2 + df['AccZ_g'] ** 2)

    return df


def identify_swings(df, motion_signal='Pitch_deg', min_height_deg=5, min_dist_samples=50):
    """
    Identifies individual swings by finding peaks in the motion signal.
    """
    peaks, _ = find_peaks(df[motion_signal], height=min_height_deg, distance=min_dist_samples)
    return peaks


def calculate_metrics(df, swing_peak_indices):
    """
    Calculates all key metrics from the preprocessed data.
    """
    metrics = {}

    # === 1. Electrical Performance ===
    metrics['Total_Energy_Generated_J'] = df['Energy_J'].sum()
    metrics['Average_Power_mW'] = df['Power_mW'].mean()
    metrics['Peak_Power_mW'] = df['Power_mW'].max()
    metrics['RMS_Voltage_V'] = np.sqrt(np.mean(df['Voltage_V'] ** 2))
    metrics['RMS_Current_mA'] = np.sqrt(np.mean(df['Current_mA'] ** 2))

    # === 2. Mechanical Motion ===
    metrics['Range_of_Motion_Pitch_deg'] = df['Pitch_deg'].max() - df['Pitch_deg'].min()
    metrics['Range_of_Motion_Roll_deg'] = df['Roll_deg'].max() - df['Roll_deg'].min()
    metrics['Average_Abs_Velocity_Pitch_dps'] = df['Velocity_Pitch_dps'].abs().mean()
    metrics['Average_Abs_Velocity_Roll_dps'] = df['Velocity_Roll_dps'].abs().mean()
    metrics['Peak_Acceleration_g'] = df['Acc_Total_g'].max()

    # === 3. Dominant Frequency (FFT) ===
    signal = df['Pitch_deg'].to_numpy()
    N = len(signal)
    sample_spacing = df['delta_t_s'].mean()

    if N > 0 and sample_spacing > 0:
        yf = rfft(signal)
        xf = rfftfreq(N, sample_spacing)
        peak_idx = np.argmax(np.abs(yf[1:])) + 1
        metrics['Dominant_Motion_Frequency_Hz'] = xf[peak_idx]
    else:
        metrics['Dominant_Motion_Frequency_Hz'] = np.nan

    # === 4. Energy per Swing ===
    energy_per_swing_list = []
    if len(swing_peak_indices) > 1:
        for i in range(len(swing_peak_indices) - 1):
            start_index = swing_peak_indices[i]
            end_index = swing_peak_indices[i + 1]
            energy_this_swing = df.iloc[start_index:end_index]['Energy_J'].sum()
            energy_per_swing_list.append(energy_this_swing)

        if energy_per_swing_list:
            metrics['Average_Energy_per_Swing_J'] = np.mean(energy_per_swing_list)
            n_swings = len(energy_per_swing_list)
            if n_swings > 1:
                std_dev = np.std(energy_per_swing_list, ddof=1)
                metrics['SE_Energy_per_Swing_J'] = std_dev / np.sqrt(n_swings)
            else:
                metrics['SE_Energy_per_Swing_J'] = np.nan
        else:
            metrics['Average_Energy_per_Swing_J'] = np.nan
            metrics['SE_Energy_per_Swing_J'] = np.nan
    else:
        metrics['Average_Energy_per_Swing_J'] = np.nan
        metrics['SE_Energy_per_Swing_J'] = np.nan

    # === 5. System Efficiency Proxy ===
    if metrics['Average_Abs_Velocity_Pitch_dps'] > 0:
        metrics['Efficiency_Proxy_J_per_dps'] = \
            metrics['Total_Energy_Generated_J'] / metrics['Average_Abs_Velocity_Pitch_dps']
    else:
        metrics['Efficiency_Proxy_J_per_dps'] = np.nan

    return metrics


def create_plots(df, swing_indices):
    """
    Generates and saves the three requested plots.
    """
    print("\nCreating plots...")

    # Get peak data for plotting
    peak_times = df.iloc[swing_indices]['Time_s']
    peak_pitches = df.iloc[swing_indices]['Pitch_deg']

    # Plot 1: Power (mW) vs. Time (s)
    plt.figure(figsize=(12, 6))
    plt.plot(df['Time_s'], df['Power_mW'], label='Power', color='blue', linewidth=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Power (mW)')
    plt.title('Instantaneous Power vs. Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('power_vs_time.png')
    plt.close()

    # Plot 2: Pitch (deg) vs. Time (s) (overlay swing peaks)
    plt.figure(figsize=(12, 6))
    plt.plot(df['Time_s'], df['Pitch_deg'], label='Pitch Angle', color='green', linewidth=1)
    plt.plot(peak_times, peak_pitches, 'x', color='red', markersize=8, label='Swing Peaks')
    plt.xlabel('Time (s)')
    plt.ylabel('Pitch (deg)')
    plt.title('Pitch Angle vs. Time (with Swing Peaks)')
    plt.legend()
    plt.grid(True)
    plt.savefig('pitch_vs_time_with_peaks.png')
    plt.close()

    # Plot 3: Scatter plot: Power (mW) vs. Abs. Pitch Velocity (deg/s)
    plt.figure(figsize=(10, 8))
    plt.scatter(df['Velocity_Pitch_dps'].abs(), df['Power_mW'], alpha=0.3, s=10)
    plt.xlabel('Absolute Pitch Velocity (deg/s)')
    plt.ylabel('Power (mW)')
    plt.title('Power Generation vs. Pitch Velocity')
    plt.grid(True)
    plt.savefig('power_vs_velocity_scatter.png')
    plt.close()

    print("Plots saved as 'power_vs_time.png', 'pitch_vs_time_with_peaks.png', and 'power_vs_velocity_scatter.png'")


# --- NEW FUNCTION ---
def print_sheets_row(results, filename):
    """
    Prints a header and data row (tab-separated) for
    easy copy-pasting into Google Sheets or Excel.
    """
    print("\n--- Google Sheets Data Row ---")

    # 1. Define the order of keys
    keys = [
        'Filename',
        'Total_Energy_Generated_J',
        'Average_Power_mW',
        'Peak_Power_mW',
        'RMS_Voltage_V',
        'RMS_Current_mA',
        'Dominant_Motion_Frequency_Hz',
        'Pitch_Range_of_Motion_deg',
        'Roll_Range_of_Motion_deg',
        'Average_Abs_Velocity_Pitch_dps',
        'Peak_Acceleration_g',
        'Average_Energy_per_Swing_J',
        'SE_Energy_per_Swing_J',
        'Efficiency_Proxy_J_per_dps'
    ]

    # 2. Print Header Row
    header_row = "\t".join(keys)
    print("Header (copy this first):")
    print(header_row)

    # 3. Create and Print Data Row
    # Use .get() to safely access keys, providing np.nan as default
    data_list = [
        filename,
        f"{results.get('Total_Energy_Generated_J', np.nan):.4f}",
        f"{results.get('Average_Power_mW', np.nan):.2f}",
        f"{results.get('Peak_Power_mW', np.nan):.2f}",
        f"{results.get('RMS_Voltage_V', np.nan):.2f}",
        f"{results.get('RMS_Current_mA', np.nan):.2f}",
        f"{results.get('Dominant_Motion_Frequency_Hz', np.nan):.3f}",
        f"{results.get('Pitch_Range_of_Motion_deg', np.nan):.2f}",
        f"{results.get('Roll_Range_of_Motion_deg', np.nan):.2f}",
        f"{results.get('Average_Abs_Velocity_Pitch_dps', np.nan):.2f}",
        f"{results.get('Peak_Acceleration_g', np.nan):.2f}",
        f"{results.get('Average_Energy_per_Swing_J', np.nan):.4f}",
        f"{results.get('SE_Energy_per_Swing_J', np.nan):.5f}",
        f"{results.get('Efficiency_Proxy_J_per_dps', np.nan):.4f}"
    ]

    data_row = "\t".join(data_list)
    print("\nData (copy this row):")
    print(data_row)
    print("--------------------------------")


# --- END NEW FUNCTION ---


def main():
    """
    Main function to run the full analysis pipeline.
    """
    # --- Configuration ---
    FILE_PATH = '20251020_BucketShakeTest.csv'  # <--- SET YOUR FILE PATH HERE

    # --- Swing Detection Tuning ---
    PRIMARY_MOTION_SIGNAL = 'Pitch_deg'
    MIN_SWING_HEIGHT_DEG = 5.0
    MIN_SWING_DISTANCE_SAMPLES = 25
    # ---------------------

    # 1. Load and process data
    print(f"Loading and processing {FILE_PATH}...")
    df = load_and_preprocess(FILE_PATH)

    if df is None:
        return

    # 2. Identify swings
    print("Identifying swings...")
    swing_indices = identify_swings(df,
                                    motion_signal=PRIMARY_MOTION_SIGNAL,
                                    min_height_deg=MIN_SWING_HEIGHT_DEG,
                                    min_dist_samples=MIN_SWING_DISTANCE_SAMPLES)

    print(f"Found {len(swing_indices)} swings.")

    # 3. Calculate all metrics
    print("Calculating metrics...")
    results = calculate_metrics(df, swing_indices)

    # 4. Display results
    print("\n--- PENDULUM WEC ANALYSIS RESULTS ---")

    print("\n## Electrical Performance ##")
    print(f"Total Energy Generated: {results['Total_Energy_Generated_J']:.4f} Joules")
    print(f"Average Power:          {results['Average_Power_mW']:.2f} mW")
    print(f"Peak Power:             {results['Peak_Power_mW']:.2f} mW")
    print(f"RMS Voltage:            {results['RMS_Voltage_V']:.2f} V")
    print(f"RMS Current:            {results['RMS_Current_mA']:.2f} mA")

    print("\n## Mechanical Motion ##")
    print(f"Dominant Motion Freq:   {results['Dominant_Motion_Frequency_Hz']:.3f} Hz")
    print(f"Pitch Range of Motion:  {results['Range_of_Motion_Pitch_deg']:.2f} degrees")
    print(f"Roll Range of Motion:   {results['Range_of_Motion_Roll_deg']:.2f} degrees")
    print(f"Avg. Abs. Pitch Velo:   {results['Average_Abs_Velocity_Pitch_dps']:.2f} deg/s")
    print(f"Peak Acceleration:      {results['Peak_Acceleration_g']:.2f} g")

    print("\n## System Performance ##")
    print(f"Average Energy per Swing: {results['Average_Energy_per_Swing_J']:.4f} Joules/swing")
    print(f"Std. Error (Energy/Swing): {results['SE_Energy_per_Swing_J']:.5f} Joules")
    print(f"Efficiency Proxy:         {results['Efficiency_Proxy_J_per_dps']:.4f} J per (deg/s)")
    print("---------------------------------------")

    # 5. Create Plots
    if df is not None:
        create_plots(df, swing_indices)

    # 6. Print Sheets Row --- ADDED THIS CALL ---
    print_sheets_row(results, FILE_PATH)


if __name__ == "__main__":
    main()