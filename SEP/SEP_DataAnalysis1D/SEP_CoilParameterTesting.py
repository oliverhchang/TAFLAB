import serial
import serial.tools.list_ports
import time
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import shutil  # <--- Added for saving files

# ================= CONFIGURATION =================
LIVE_FILENAME = 'SEP_CoilParameterTesting_LiveData.csv'  # Temporary buffer (always overwritten)
LOG_FILENAME = 'Experiment_Log.csv'  # Master summary file

# Physics Constants
PENDULUM_LENGTH = 0.2  # Meters
PENDULUM_MASS = 0.16  # Kilograms
STARTING_ANGLE = 90  # Degrees
GRAVITY = 9.81  # m/s^2

# Analysis Thresholds
POWER_THRESHOLD = 0.5  # mW
MIN_SWING_LENGTH = 5  # Samples


# =================================================

def get_arduino_port():
    """Auto-detects an Arduino/Serial port."""
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if "Arduino" in p.description or "USB" in p.description or "ACM" in p.description:
            return p.device
    return None


def record_data():
    """Reads Serial data and saves to CSV until Ctrl+C is pressed."""
    port = get_arduino_port()
    if not port:
        print("❌ No Arduino found! Check connection.")
        return None

    print(f"✅ Connected to {port}. Initializing...")

    try:
        # Note: Ensure this matches your Arduino Serial.begin() rate!
        ser = serial.Serial(port, 500000, timeout=1)
        time.sleep(2)  # Wait for Arduino reset
        ser.reset_input_buffer()

        print("\n🌊 RECORDING STARTED! Swing the pendulum now.")
        print("🛑 Press Ctrl+C to STOP recording and analyze data.\n")

        data_buffer = []
        start_time = time.time()

        with open(LIVE_FILENAME, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Voltage_V", "Power_mW"])

            while True:
                try:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if line and ',' in line:
                        parts = line.split(',')
                        if len(parts) == 2:
                            volts = float(parts[0])
                            power = float(parts[1])

                            writer.writerow([volts, power])
                            data_buffer.append(power)

                            # LIVE PRINTING (Every 50 samples)
                            if len(data_buffer) % 50 == 0:
                                print(f"   Capture: {volts:.2f} V | {power:.2f} mW")

                except ValueError:
                    continue

    except KeyboardInterrupt:
        end_time = time.time()
        duration = end_time - start_time
        samples = len(data_buffer)
        if ser.is_open: ser.close()

        print(f"\n🛑 Recording Stopped.")
        real_fs = samples / duration if duration > 0 else 100
        print(f"⚡ True Sampling Rate: {real_fs:.2f} Hz")
        return real_fs

    except Exception as e:
        print(f"Error: {e}")
        return None


def analyze_and_log(sampling_rate):
    """Analyzes data and prompts user to save results."""
    print("\n🔍 Analyzing Data...")

    try:
        df = pd.read_csv(LIVE_FILENAME)
    except FileNotFoundError:
        print("No data file found.")
        return

    # --- 1. Identify Active Region ---
    active_indices = df.index[df['Power_mW'] > POWER_THRESHOLD].tolist()

    if not active_indices:
        print("⚠️ No valid power spikes detected. Check connections or Threshold.")
        return

    first_idx = active_indices[0]
    last_idx = active_indices[-1]
    session_data = df.iloc[first_idx: last_idx + 1]

    # --- 2. Calculate Metrics ---
    # A. Energy
    rad_angle = np.radians(STARTING_ANGLE)
    height_change = PENDULUM_LENGTH * (1 - np.cos(rad_angle))
    input_energy_mJ = (PENDULUM_MASS * GRAVITY * height_change) * 1000

    total_energy_mJ = df['Power_mW'].sum() / sampling_rate
    efficiency_percent = (total_energy_mJ / input_energy_mJ) * 100

    # B. Power Stats
    peak_power = df['Power_mW'].max()
    window_duration_s = len(session_data) / sampling_rate
    window_avg_power = session_data['Power_mW'].mean()
    duty_cycle = (len(active_indices) / len(df)) * 100

    # --- 3. Report ---
    print("=" * 60)
    print("      REAL-WORLD PERFORMANCE REPORT")
    print("=" * 60)
    print(f"1. ENERGY & EFFICIENCY")
    print(f"   Mechanical Input (PE):   {input_energy_mJ:.2f} mJ")
    print(f"   Electrical Output:       {total_energy_mJ:.2f} mJ")
    print(f"   SYSTEM EFFICIENCY:       {efficiency_percent:.4f} %")
    print("-" * 60)
    print(f"2. POWER METRICS")
    print(f"   Peak Power:              {peak_power:.2f} mW")
    print(f"   Session Avg Power:       {window_avg_power:.2f} mW")
    print("-" * 60)
    print(f"3. TIMING")
    print(f"   True Sampling Rate:      {sampling_rate:.2f} Hz")
    print("=" * 60)

    # --- 4. Plotting ---
    plt.style.use('bmh')
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Full Session
    time_axis = df.index / sampling_rate
    ax1.plot(time_axis, df['Power_mW'], color='#95a5a6', alpha=0.3, label='Idle Noise')

    # Highlight Session
    session_time = session_data.index / sampling_rate
    ax1.plot(session_time, session_data['Power_mW'], color='#2980b9', label='Active Session')

    # Mark Peak
    peak_time = df['Power_mW'].idxmax() / sampling_rate
    ax1.plot(peak_time, peak_power, 'r*', markersize=15, label=f'Peak {peak_power:.1f}mW')

    ax1.set_title(f"Generator Output (Avg in window: {window_avg_power:.2f} mW)")
    ax1.set_ylabel("Power (mW)")
    ax1.set_xlabel("Time (s)")
    ax1.legend()
    plt.show(block=False)

    # --- 5. LOGGING & FILE SAVING ---
    save = input("\n💾 Save results? (y/n): ").strip().lower()
    if save == 'y':
        save_experiment_data(peak_power, window_avg_power, total_energy_mJ, efficiency_percent)


def save_experiment_data(peak, avg, energy, eff):
    """Saves raw data to a named file AND appends stats to master log."""

    # 1. Get User Inputs
    print("\n📝 Enter Coil Details:")
    gauge = input("   Wire Gauge (AWG): ").strip()
    turns = input("   Number of Turns: ").strip()
    inner_d = input("   Inner Diameter (mm): ").strip()
    notes = input("   Notes (Optional): ").strip()

    # 2. Generate Filenames
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Format: 30AWG_1000T_10mm_2023-10-27_15-30-00.csv
    # Replacing spaces in notes to keep filename clean
    clean_notes = notes.replace(" ", "-") if notes else "Data"
    raw_data_filename = f"{gauge}AWG_{turns}T_{inner_d}mm_{clean_notes}_{timestamp_str}.csv"

    # 3. SAVE RAW DATA (Copy the temp file to the new name)
    try:
        shutil.copy(LIVE_FILENAME, raw_data_filename)
        print(f"\n✅ Raw data saved as: {raw_data_filename}")
    except Exception as e:
        print(f"❌ Error saving raw data file: {e}")

    # 4. APPEND TO MASTER LOG
    log_header = ["Timestamp", "Gauge_AWG", "Turns", "ID_mm", "Peak_Power_mW",
                  "Window_Avg_Power_mW", "Output_Energy_mJ", "Efficiency_Pct", "Raw_File_Name", "Notes"]

    log_row = [timestamp_str, gauge, turns, inner_d, round(peak, 2),
               round(avg, 2), round(energy, 2), round(eff, 4), raw_data_filename, notes]

    try:
        file_exists = os.path.isfile(LOG_FILENAME)
        with open(LOG_FILENAME, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(log_header)
            writer.writerow(log_row)
        print(f"✅ Summary stats appended to: {LOG_FILENAME}")
    except Exception as e:
        print(f"❌ Error saving to log: {e}")


if __name__ == "__main__":
    print("1. Record New Data")
    print("2. Analyze Last Recording")
    choice = input("Select option (1/2): ").strip()

    if choice == '1':
        measured_rate = record_data()
        if measured_rate:
            analyze_and_log(measured_rate)
    else:
        # Default to 650Hz if analyzing old data
        rate_input = input("Enter sampling rate (default 650): ").strip()
        rate = float(rate_input) if rate_input else 650.0
        analyze_and_log(rate)

    input("\nPress Enter to exit...")