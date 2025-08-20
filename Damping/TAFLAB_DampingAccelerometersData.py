import serial
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
from scipy.signal import find_peaks

# --- 1. Configuration & System Properties ---

# -- Serial Port Settings --
SERIAL_PORT = '/dev/cu.usbmodem11101'
BAUD_RATE = 900000

# -- File Settings --
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
FILENAME = f'pendulum_data_{timestamp}.csv'

# -- Pendulum Physical Properties (as provided) --
g = 9.81  # gravity [m/s^2]
rho_sand = 1922  # wet sand [kg/m^3]
r = 0.1016  # radius [m] (4 in)
l = 8 * 0.0254  # length [m] (8 in)
d = 5.75 * 0.0254  # offset from shaft [m] (5.75 in)

# Calculate pendulum mass and inertia
volume_cyl = np.pi * r ** 2 * l
mass = rho_sand * volume_cyl
I_cm = 0.5 * mass * r ** 2
I_pendulum = I_cm + mass * d ** 2

# NOTE: J_total should include inertia of the motor and gearbox if they are significant.
J_total = I_pendulum

# --- 2. Data Storage ---
time_data = []
angle_data = []

# --- 3. Serial Connection & 4. Data Collection Loop ---
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Connected to {SERIAL_PORT} at {BAUD_RATE} bps.")
    time.sleep(2)
    print("\nStarting data collection...")
    print("Press Ctrl+C in the terminal to stop.")
    while True:
        try:
            line_of_data = ser.readline().decode('utf-8').strip()
            if line_of_data:
                time_ms, angle_deg = map(float, line_of_data.split(','))
                time_data.append(time_ms / 1000.0)
                angle_data.append(angle_deg)
        except (ValueError, IndexError):
            print(f"Warning: Could not parse line: '{line_of_data}'")
except KeyboardInterrupt:
    print("\nData collection stopped by user.")
finally:
    ser.close()
    print("Serial port closed.")

# --- 5. Data Analysis & Damping Calculation ---
if len(time_data) > 10:
    time_array = np.array(time_data)
    angle_array = np.array(angle_data)

    # Use 'distance' and 'prominence' to filter out noise without smoothing.
    # ADJUST THESE VALUES if peak detection is not accurate.
    peak_indices, _ = find_peaks(angle_array, distance=5, prominence=1)

    if len(peak_indices) < 2:
        print("\n--- Starting Damping Analysis ---")
        print("Error: Could not find at least two peaks.")
        print("TRY THIS: Decrease the 'distance' and 'prominence' values in the find_peaks() function.")
        # Plot the data anyway for visual inspection
        plt.figure(figsize=(12, 6))
        plt.plot(time_array, angle_array, 'b-', label='Raw Data (No Peaks Found)')
        plt.title("Data Visualization - Could Not Find Peaks")
        plt.xlabel("Time (s)")
        plt.ylabel("Angle (degrees)")
        plt.grid(True)
        plt.legend()
        plt.show()
    else:
        peak_angles = angle_array[peak_indices]
        peak_times = time_array[peak_indices]

        num_cycles = len(peak_indices) - 1
        delta = (1 / num_cycles) * np.log(peak_angles[0] / peak_angles[-1])
        zeta = delta / np.sqrt((2 * np.pi) ** 2 + delta ** 2)
        avg_damped_period = np.mean(np.diff(peak_times))
        omega_d = 2 * np.pi / avg_damped_period
        omega_n = omega_d / np.sqrt(1 - zeta ** 2)
        c_critical = 2 * J_total * omega_n
        damping_coefficient = zeta * c_critical

        # --- 6. Print Results (Formatted as requested) ---
        print("\n--- Analysis Results ---")
        print(f"Logarithmic Decrement (δ): {delta:.4f}")
        print(f"Damping Ratio (ζ): {zeta:.4f}")
        print(f"Avg. Damped Period (τd): {avg_damped_period:.4f} s")
        print(f"Damped Natural Frequency (ωd): {omega_d:.4f} rad/s")
        print(f"Undamped Natural Frequency (ωn): {omega_n:.4f} rad/s")
        print("\n--- Final Damping Coefficients ---")
        print(f"Critical Damping Constant (c_c): {c_critical:.4f} N-m-s/rad")
        print(f"Calculated Damping Coefficient (c): {damping_coefficient:.4f} N-m-s/rad")

        # --- 7. Save and Plot Data ---
        print("\nSaving and plotting data...")
        with open(FILENAME, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Time (s)', 'Angle (deg)'])
            writer.writerows(zip(time_data, angle_data))
        print(f"Data saved to {FILENAME}")

        plt.figure(figsize=(12, 6))
        plt.plot(time_array, angle_array, 'b-', linewidth=1, label='Experimental Data')
        plt.plot(peak_times, peak_angles, 'ro', label='Detected Peaks')
        plt.title(f"Pendulum Angle vs. Time (Damping Analysis)")
        plt.xlabel("Time (s)")
        plt.ylabel("Angle (degrees)")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        results_text = (f"Damping Ratio (ζ) = {zeta:.4f}\n"
                        f"Damping Coeff. (c) = {damping_coefficient:.4f} N-m-s/rad")
        plt.text(0.98, 0.98, results_text, transform=plt.gca().transAxes,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        plt.legend()
        plt.tight_layout()
        plt.show()
else:
    print("No data was collected. Cannot perform analysis.")