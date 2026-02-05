import pandas as pd
import numpy as np
from scipy.fft import rfft, rfftfreq


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
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
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


def identify_swings(df, power_threshold_mw):
    """
    Identifies swings as contiguous blocks of data where power is above a threshold.
    Returns a list of DataFrames, where each DataFrame is one "swing" (power pulse).
    """
    # 1. Find all rows where power is above the threshold
    df['is_active'] = df['Power_mW'] > power_threshold_mw

    # 2. Find contiguous blocks of 'True'
    # This clever trick creates a unique ID for each block
    df['block_id'] = (df['is_active'] != df['is_active'].shift()).cumsum()

    # 3. Filter down to only the active blocks
    active_blocks = df[df['is_active'] == True]

    if active_blocks.empty:
        return []

    # 4. Group by the block_id and return a list of these DataFrames
    grouped = active_blocks.groupby('block_id')
    swing_dfs = [group for _, group in grouped]

    return swing_dfs


def calculate_metrics(df, swing_dfs):
    """
    Calculates all key metrics from the preprocessed data.
    'swing_dfs' is a list of DataFrames, each representing one power pulse.
    """
    metrics = {}

    # === 1. Electrical Performance (Global) ===
    metrics['Total_Energy_Generated_J'] = df['Energy_J'].sum()
    metrics['Average_Power_Global_mW'] = df['Power_mW'].mean()  # Renamed for clarity
    metrics['Peak_Power_mW'] = df['Power_mW'].max()
    metrics['RMS_Voltage_V'] = np.sqrt(np.mean(df['Voltage_V'] ** 2))
    metrics['RMS_Current_mA'] = np.sqrt(np.mean(df['Current_mA'] ** 2))

    # === 2. Mechanical Motion (Global) ===
    metrics['Range_of_Motion_Pitch_deg'] = df['Pitch_deg'].max() - df['Pitch_deg'].min()
    metrics['Range_of_Motion_Roll_deg'] = df['Roll_deg'].max() - df['Roll_deg'].min()
    metrics['Average_Abs_Velocity_Pitch_dps'] = df['Velocity_Pitch_dps'].abs().mean()
    metrics['Average_Abs_Velocity_Roll_dps'] = df['Velocity_Roll_dps'].abs().mean()
    metrics['Peak_Acceleration_g'] = df['Acc_Total_g'].max()

    # === 3. Dominant Frequency (FFT) (Global) ===
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

    # === 4. Per-Swing Metrics ===
    energy_per_swing_list = []
    avg_power_per_swing_list = []

    for swing_df in swing_dfs:
        # Calculate total energy for this one swing
        energy_per_swing_list.append(swing_df['Energy_J'].sum())
        # Calculate the average power *during* this one swing
        avg_power_per_swing_list.append(swing_df['Power_mW'].mean())

    n_swings = len(energy_per_swing_list)
    metrics['Number_of_Swings'] = n_swings

    if n_swings > 0:
        # Average of all swing energies
        metrics['Average_Energy_per_Swing_J'] = np.mean(energy_per_swing_list)
        # Average of all swing average powers
        metrics['Average_Power_per_Swing_mW'] = np.mean(avg_power_per_swing_list)

        if n_swings > 1:
            # Calculate Standard Errors
            std_dev_energy = np.std(energy_per_swing_list, ddof=1)
            metrics['SE_Energy_per_Swing_J'] = std_dev_energy / np.sqrt(n_swings)

            std_dev_power = np.std(avg_power_per_swing_list, ddof=1)
            metrics['SE_Power_per_Swing_mW'] = std_dev_power / np.sqrt(n_swings)
        else:
            # Cannot calculate SE with only 1 swing
            metrics['SE_Energy_per_Swing_J'] = np.nan
            metrics['SE_Power_per_Swing_mW'] = np.nan
    else:
        # No swings found
        metrics['Average_Energy_per_Swing_J'] = np.nan
        metrics['SE_Energy_per_Swing_J'] = np.nan
        metrics['Average_Power_per_Swing_mW'] = np.nan
        metrics['SE_Power_per_Swing_mW'] = np.nan

    # === 5. System Efficiency Proxy (Global) ===
    if metrics['Average_Abs_Velocity_Pitch_dps'] > 0:
        metrics['Efficiency_Proxy_J_per_dps'] = \
            metrics['Total_Energy_Generated_J'] / metrics['Average_Abs_Velocity_Pitch_dps']
    else:
        metrics['Efficiency_Proxy_J_per_dps'] = np.nan

    return metrics


def print_full_results(results, filename):
    """
    Prints the verbose, human-readable results for a single file.
    """
    print(f"\n--- PENDULUM WEC ANALYSIS RESULTS FOR: {filename} ---")

    print("\n## Global Electrical Performance ##")
    print(f"Total Energy Generated: {results.get('Total_Energy_Generated_J', 0):.4f} Joules")
    print(f"Average Power (Global): {results.get('Average_Power_Global_mW', 0):.2f} mW")
    print(f"Peak Power:             {results.get('Peak_Power_mW', 0):.2f} mW")
    print(f"RMS Voltage:            {results.get('RMS_Voltage_V', 0):.2f} V")
    print(f"RMS Current:            {results.get('RMS_Current_mA', 0):.2f} mA")

    print("\n## Global Mechanical Motion ##")
    print(f"Dominant Motion Freq:   {results.get('Dominant_Motion_Frequency_Hz', 0):.3f} Hz")
    print(f"Pitch Range of Motion:  {results.get('Pitch_Range_of_Motion_deg', 0):.2f} degrees")
    print(f"Roll Range of Motion:   {results.get('Roll_Range_of_Motion_deg', 0):.2f} degrees")
    print(f"Avg. Abs. Pitch Velo:   {results.get('Average_Abs_Velocity_Pitch_dps', 0):.2f} deg/s")
    print(f"Peak Acceleration:      {results.get('Peak_Acceleration_g', 0):.2f} g")

    print("\n## Per-Swing System Performance ##")
    print(f"Number of Swings Found: {results.get('Number_of_Swings', 0)}")
    print(f"Average Energy per Swing: {results.get('Average_Energy_per_Swing_J', 0):.4f} Joules/swing")
    print(f"Std. Error (Energy/Swing): {results.get('SE_Energy_per_Swing_J', 0):.5f} Joules")
    print(f"Average Power per Swing:  {results.get('Average_Power_per_Swing_mW', 0):.2f} mW/swing")
    print(f"Std. Error (Power/Swing): {results.get('SE_Power_per_Swing_mW', 0):.3f} mW")
    print(f"Efficiency Proxy (Global): {results.get('Efficiency_Proxy_J_per_dps', 0):.4f} J per (deg/s)")
    print("---------------------------------------")


def get_sheets_header():
    """
    Returns the header row for the Google Sheet.
    """
    keys = [
        'Filename', 'Total_Energy_Generated_J', 'Average_Power_Global_mW', 'Peak_Power_mW',
        'RMS_Voltage_V', 'RMS_Current_mA', 'Dominant_Motion_Frequency_Hz',
        'Pitch_Range_of_Motion_deg', 'Roll_Range_of_Motion_deg',
        'Average_Abs_Velocity_Pitch_dps', 'Peak_Acceleration_g', 'Number_of_Swings',
        'Average_Energy_per_Swing_J', 'SE_Energy_per_Swing_J',
        'Average_Power_per_Swing_mW', 'SE_Power_per_Swing_mW', 'Efficiency_Proxy_J_per_dps'
    ]
    return "\t".join(keys)


def get_sheets_data_row(results, filename):
    """
    Returns a single tab-separated data row for Google Sheets.
    """
    data_list = [
        filename,
        f"{results.get('Total_Energy_Generated_J', np.nan):.4f}",
        f"{results.get('Average_Power_Global_mW', np.nan):.2f}",
        f"{results.get('Peak_Power_mW', np.nan):.2f}",
        f"{results.get('RMS_Voltage_V', np.nan):.2f}",
        f"{results.get('RMS_Current_mA', np.nan):.2f}",
        f"{results.get('Dominant_Motion_Frequency_Hz', np.nan):.3f}",
        f"{results.get('Pitch_Range_of_Motion_deg', np.nan):.2f}",
        f"{results.get('Roll_Range_of_Motion_deg', np.nan):.2f}",
        f"{results.get('Average_Abs_Velocity_Pitch_dps', np.nan):.2f}",
        f"{results.get('Peak_Acceleration_g', np.nan):.2f}",
        f"{results.get('Number_of_Swings', 0)}",
        f"{results.get('Average_Energy_per_Swing_J', np.nan):.4f}",
        f"{results.get('SE_Energy_per_Swing_J', np.nan):.5f}",
        f"{results.get('Average_Power_per_Swing_mW', np.nan):.2f}",
        f"{results.get('SE_Power_per_Swing_mW', np.nan):.3f}",
        f"{results.get('Efficiency_Proxy_J_per_dps', np.nan):.4f}"
    ]
    return "\t".join(data_list)


def main():
    """
    Main function to run the full analysis pipeline on multiple files.
    """

    # --- 1. CONFIGURE YOUR FILES HERE ---
    FILE_PATHS_TO_PROCESS = [
        '20251020_BucketShakeTest.csv',
        '/Users/oliverchang/PycharmProjects/TAFLAB/Data Analysis/1ohm/1ohm9400uF.csv',
        '/Users/oliverchang/PycharmProjects/TAFLAB/Data Analysis/5ohm/5ohm0uF.csv',
        '/Users/oliverchang/PycharmProjects/TAFLAB/Data Analysis/5ohm/5ohm2350uF.csv',
        '/Users/oliverchang/PycharmProjects/TAFLAB/Data Analysis/5ohm/5ohm4700uF.csv',
        '/Users/oliverchang/PycharmProjects/TAFLAB/Data Analysis/5ohm/5ohm14100uF.csv',
        '/Users/oliverchang/PycharmProjects/TAFLAB/Data Analysis/5ohm/5ohm9400uF.csv',
        '/Users/oliverchang/PycharmProjects/TAFLAB/Data Analysis/10ohm/10ohm1000uF.csv',
        '/Users/oliverchang/PycharmProjects/TAFLAB/Data Analysis/10ohm/10ohm1000uF2.csv',
        '/Users/oliverchang/PycharmProjects/TAFLAB/Data Analysis/10ohm/10ohm2350uF.csv',
        '/Users/oliverchang/PycharmProjects/TAFLAB/Data Analysis/10ohm/10ohm2350uF2.csv',
        '/Users/oliverchang/PycharmProjects/TAFLAB/Data Analysis/10ohm/10ohm2350uF3.csv',
        '/Users/oliverchang/PycharmProjects/TAFLAB/Data Analysis/10ohm/10ohm2350uF4.csv',
        '/Users/oliverchang/PycharmProjects/TAFLAB/Data Analysis/10ohm/10ohm2350uF5.csv',
        '/Users/oliverchang/PycharmProjects/TAFLAB/Data Analysis/10ohm/10ohm4700uF.csv',
        '/Users/oliverchang/PycharmProjects/TAFLAB/Data Analysis/10ohm/10ohm4700uF2.csv',
        '/Users/oliverchang/PycharmProjects/TAFLAB/Data Analysis/10ohm/10ohm4700uF3.csv',
        '/Users/oliverchang/PycharmProjects/TAFLAB/Data Analysis/10ohm/10ohm4700uF4.csv',
        '/Users/oliverchang/PycharmProjects/TAFLAB/Data Analysis/10ohm/10ohm4700uF5.csv',
        '/Users/oliverchang/PycharmProjects/TAFLAB/Data Analysis/10ohm/10ohm14100uF.csv',
        '/Users/oliverchang/PycharmProjects/TAFLAB/Data Analysis/10ohm/10ohm5700uF.csv',
        '/Users/oliverchang/PycharmProjects/TAFLAB/Data Analysis/10ohm/10ohm5700uF2.csv',
        '/Users/oliverchang/PycharmProjects/TAFLAB/Data Analysis/10ohm/10ohm5700uF3.csv',
        '/Users/oliverchang/PycharmProjects/TAFLAB/Data Analysis/10ohm/10ohm14100uF.csv',
        '/Users/oliverchang/PycharmProjects/TAFLAB/Data Analysis/10ohm/10ohm18800uF.csv',
        '/Users/oliverchang/PycharmProjects/TAFLAB/Data Analysis/10ohm/10ohm18800uF2.csv',
    ]
    # ------------------------------------

    # --- 2. CONFIGURE SWING DETECTION ---
    # A "swing" will be counted as any continuous period where
    # power is above this value. Tune this to ignore sensor noise.
    POWER_THRESHOLD_MW = 1.0
    # ------------------------------------

    print("Starting batch processing...")

    all_sheets_data_rows = []

    for file_path in FILE_PATHS_TO_PROCESS:
        print(f"\nProcessing {file_path}...")

        # 1. Load and process data
        df = load_and_preprocess(file_path)
        if df is None:
            print(f"Skipping file {file_path} due to loading error.")
            continue

        # 2. Identify swings based on power pulses
        swings_list_of_dfs = identify_swings(df, POWER_THRESHOLD_MW)
        print(f"Found {len(swings_list_of_dfs)} power-based swings.")

        # 3. Calculate all metrics
        results = calculate_metrics(df, swings_list_of_dfs)

        # 4. Display full results for this file
        print_full_results(results, file_path)

        # 5. Store the data row for the final table
        all_sheets_data_rows.append(get_sheets_data_row(results, file_path))

    # 6. Print the final summary table for Google Sheets
    print("\n\n--- GOOGLE SHEETS BATCH OUTPUT ---")
    print("Copy and paste the table below into Google Sheets:\n")

    print(get_sheets_header())  # Print the header row
    for row in all_sheets_data_rows:
        print(row)  # Print each data row

    print("\n----------------------------------")
    print("Batch processing complete.")


if __name__ == "__main__":
    main()