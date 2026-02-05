import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# === Configuration ===
data_dir = "/"  # Use '.' if script and CSVs are in same folder

# === File list (manually defined, Option 1) ===
file_names = [
    "0.2ohm_T1.csv", "0.2ohm_T2.csv",
    "0.25ohm_T1.csv", "0.25ohm_T2.csv",
    "0.33ohm_T1.csv", "0.33ohm_T2.csv",
    "0.5ohm_T1.csv", "0.5ohm_T2.csv",
    "1ohm_T1.csv", "1ohm_T2.csv"
]

# === Parse Function ===
def parse_trial(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        if line.strip().startswith("Time(s)"):
            data_lines = lines[idx + 1:]
            break

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

# === Swing Analysis ===
def average_power_per_swing(df):
    peaks, _ = find_peaks(df["Power"], height=1)
    swings = []

    for peak in peaks:
        # Backward to ~0
        start = peak
        while start > 0 and df["Power"].iloc[start] > 1:
            start -= 1
        # Forward to ~0
        end = peak
        while end < len(df) - 1 and df["Power"].iloc[end] > 1:
            end += 1

        if end > start:
            swing_power = df["Power"].iloc[start:end+1]
            swings.append(swing_power.mean())

    return np.mean(swings) if swings else 0

# === Organize data by resistance ===
from collections import defaultdict

resistance_data = defaultdict(list)   # {resistance: [DataFrames]}
avg_power_data = {}                   # {resistance: (mean, std_err)}
swing_power_data = {}                 # {resistance: mean swing power}
trial_data = []                       # [(resistance, filename, df)]

# === Process all files ===
for fname in file_names:
    try:
        filepath = os.path.join(data_dir, fname)
        df = parse_trial(filepath)

        # Extract resistance from filename, e.g., "0.25" from "0.25ohm_T1.csv"
        resistance = float(fname.split("ohm")[0])
        resistance_data[resistance].append(df)
        trial_data.append((resistance, fname, df))
    except Exception as e:
        print(f"⚠️ Error reading {fname}: {e}")

# === Compute averages and errors ===
for R, trials in resistance_data.items():
    avg_powers = [df["Power"].mean() for df in trials]
    swing_powers = [average_power_per_swing(df) for df in trials]

    avg_power_data[R] = (np.mean(avg_powers), np.std(avg_powers) / np.sqrt(len(avg_powers)))
    swing_power_data[R] = np.mean(swing_powers)

# === Plot 1: Avg Power vs Resistance with Error Bars ===
x = sorted(avg_power_data.keys())
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
plt.ylabel("Average Power per Swing (mW)")
plt.title("Swing-Based Power vs Resistance")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Plot 3: Time Plots for Each Trial ===
for R, fname, df in trial_data:
    plt.figure(figsize=(10, 4))
    plt.plot(df["Time"], df["Voltage"], label="Voltage (V)")
    plt.plot(df["Time"], df["Current"], label="Current (mA)")
    plt.plot(df["Time"], df["Power"], label="Power (mW)")
    plt.title(f"{fname} (R = {R} Ω)")
    plt.xlabel("Time (s)")
    plt.ylabel("Measured Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
