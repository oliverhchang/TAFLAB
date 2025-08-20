import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# === Configuration ===
resistances = [0.2, 0.25, 0.33, 0.5, 1.0]
data_dir = "./"  # Change if needed


# === Helpers ===
def parse_trial(filepath):
    """Parse CSV and return DataFrame with time, voltage, current, power."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Skip to line with "Time(s)"
    for idx, line in enumerate(lines):
        if line.strip().startswith("Time(s)"):
            data_lines = lines[idx + 1:]
            break

    # Parse numerical values
    rows = []
    for line in data_lines:
        parts = line.strip().split()
        if len(parts) == 4:
            try:
                t, v, c, p = map(float, parts)
                rows.append((t, v, c, p))
            except ValueError:
                continue

    df = pd.DataFrame(rows, columns=["Time", "Voltage", "Current", "Power"])
    return df


def average_power_per_swing(df):
    """Estimate average power per swing (from 0 -> peak -> 0)"""
    peaks, _ = find_peaks(df["Power"], height=1)
    swings = []

    for peak in peaks:
        # Backward to power ≈ 0
        start = peak
        while start > 0 and df["Power"].iloc[start] > 1:
            start -= 1
        # Forward to power ≈ 0
        end = peak
        while end < len(df) - 1 and df["Power"].iloc[end] > 1:
            end += 1

        if end > start:
            swing_power = df["Power"].iloc[start:end + 1]
            swings.append(swing_power.mean())

    return np.mean(swings) if swings else 0


# === Data Collection ===
avg_power_data = {}
swing_power_data = {}
trial_data = []

for R in resistances:
    file1 = f"{R}ohm_T1.csv"
    file2 = f"{R}ohm_T2.csv"
    files = [file1, file2]

    avg_powers = []
    swing_powers = []

    for file in files:
        filepath = os.path.join(data_dir, file)
        df = parse_trial(filepath)
        trial_data.append((R, file, df))

        avg_powers.append(df["Power"].mean())
        swing_powers.append(average_power_per_swing(df))

    avg_power_data[R] = (np.mean(avg_powers), np.std(avg_powers) / np.sqrt(len(avg_powers)))
    swing_power_data[R] = np.mean(swing_powers)

# === Plot 1: Avg Power vs Resistance (with error bars) ===
x = list(avg_power_data.keys())
y = [avg_power_data[r][0] for r in x]
yerr = [avg_power_data[r][1] for r in x]

plt.figure(figsize=(8, 5))
plt.errorbar(x, y, yerr=yerr, fmt='o-', capsize=5)
plt.xlabel("Resistance (Ohms)")
plt.ylabel("Average Power (mW)")
plt.title("Average Power vs Resistance")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Plot 2: Avg Power Per Swing vs Resistance ===
swing_y = [swing_power_data[r] for r in x]

plt.figure(figsize=(8, 5))
plt.plot(x, swing_y, 's--', color='orange')
plt.xlabel("Resistance (Ohms)")
plt.ylabel("Avg Power per Swing (mW)")
plt.title("Swing-Based Power vs Resistance")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Plot 3: Voltage, Current, Power vs Time for Each Trial ===
for R, fname, df in trial_data:
    plt.figure(figsize=(10, 4))
    plt.plot(df["Time"], df["Voltage"], label="Voltage (V)")
    plt.plot(df["Time"], df["Current"], label="Current (mA)")
    plt.plot(df["Time"], df["Power"], label="Power (mW)")
    plt.title(f"Trial at {R}Ω - {fname}")
    plt.xlabel("Time (s)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
