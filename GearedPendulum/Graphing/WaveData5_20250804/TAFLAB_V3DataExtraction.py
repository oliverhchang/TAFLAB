import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import re

# === Configuration ===
data_dir = "/"  # Adjust if needed
file_names = [
    "0.5ohm_1Capacitor.csv",
    "1ohm_1Capacitor.csv",
    "2ohm_1Capacitor.csv",
    "2.5ohm_1Capacitor.csv",
    "3.3ohm_1Capacitor.csv",
    "5ohm_1Capacitor.csv",
    "10ohm_1Capacitor.csv",
    "20ohm_1Capacitor.csv",
    "20.1ohm_1Capacitor.csv",

    "0.5ohm_5Capacitor.csv",
    "1ohm_5Capacitor.csv",
    "2ohm_5Capacitor.csv",
    "2.5ohm_5Capacitor.csv",
    "3.3ohm_5Capacitor.csv",
    "5ohm_5Capacitor.csv",
    "10ohm_5Capacitor.csv",
    "20ohm_5Capacitor.csv",
]

# === Results container ===
results = []

# === Process each file ===
for file_name in file_names:
    try:
        # --- Load and parse CSV ---
        df = pd.read_csv(data_dir + file_name)
        header_index = df[df["Data"].str.contains("Time\(s\)")].index[0]
        data_lines = df.loc[header_index + 1:, "Data"].dropna()
        split_data = data_lines.str.split(r"\t+", expand=True)
        split_data.columns = ["Time_s", "Voltage_V", "Current_mA", "Power_mW"]
        split_data = split_data.apply(pd.to_numeric, errors="coerce").dropna()

        time = split_data["Time_s"].values
        power = split_data["Power_mW"].values

        # --- Find valleys to define swings ---
        valley_indices, _ = find_peaks(-power, distance=5)
        swing_powers = []
        for i in range(len(valley_indices) - 1):
            start, end = valley_indices[i], valley_indices[i + 1]
            t_segment = time[start:end+1]
            p_segment = power[start:end+1]
            energy = np.trapezoid(p_segment, t_segment)
            avg_power = energy / (t_segment[-1] - t_segment[0])
            swing_powers.append(avg_power)

        # --- Statistics ---
        swings = len(valley_indices) - 1
        total_time_minutes = (time[-1] - time[0]) / 60
        frequency_spm = swings / total_time_minutes
        avg_power_per_swing = np.mean(swing_powers)
        stderr_power_per_swing = np.std(swing_powers, ddof=1) / np.sqrt(len(swing_powers))
        avg_power_30s = split_data[split_data["Time_s"] <= (split_data["Time_s"].iloc[0] + 30)]["Power_mW"].mean()

        # --- Extract resistor/capacitor values ---
        match = re.match(r"([0-9.]+)ohm_([0-9]+)Capacitor\.csv", file_name)
        resistor = float(match.group(1))
        capacitor = int(match.group(2))

        # --- Save results ---
        results.append({
            "File": file_name,
            "Resistor (Ohm)": resistor,
            "Capacitor": capacitor,
            "Frequency (swings/min)": frequency_spm,
            "Avg Power per Swing (mW)": avg_power_per_swing,
            "StdErr Power per Swing (mW)": stderr_power_per_swing,
            "Avg Power in 30s (mW)": avg_power_30s
        })

        print(f"✓ Processed {file_name}")

    except Exception as e:
        print(f"⚠️ Error processing {file_name}: {e}")

# === Convert to DataFrame and Save CSV ===
results_df = pd.DataFrame(results).sort_values(by=["Capacitor", "Resistor (Ohm)"])
results_df.to_csv("swing_analysis_results.csv", index=False)
print("\n✅ Results saved to 'swing_analysis_results.csv'.")

# === Split data by capacitor count ===
cap1_df = results_df[results_df["Capacitor"] == 1].sort_values(by="Resistor (Ohm)")
cap5_df = results_df[results_df["Capacitor"] == 5].sort_values(by="Resistor (Ohm)")

# === Plot 1: Avg Power per Swing vs Resistor (with StdErr) ===
plt.figure(figsize=(8, 5))
plt.errorbar(cap1_df["Resistor (Ohm)"], cap1_df["Avg Power per Swing (mW)"],
             yerr=cap1_df["StdErr Power per Swing (mW)"], fmt='o-', capsize=5, label="1 Capacitor")
plt.errorbar(cap5_df["Resistor (Ohm)"], cap5_df["Avg Power per Swing (mW)"],
             yerr=cap5_df["StdErr Power per Swing (mW)"], fmt='s-', capsize=5, label="5 Capacitors")
plt.xlabel("Resistor (Ohm)")
plt.ylabel("Avg Power per Swing (mW)")
plt.title("Avg Power per Swing vs Resistor")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("avg_power_per_swing_vs_resistor_split.png")
plt.show()

# === Plot 2: Avg Power in 30s vs Resistor ===
plt.figure(figsize=(8, 5))
plt.plot(cap1_df["Resistor (Ohm)"], cap1_df["Avg Power in 30s (mW)"], 'o-', label="1 Capacitor")
plt.plot(cap5_df["Resistor (Ohm)"], cap5_df["Avg Power in 30s (mW)"], 's-', color='orange', label="5 Capacitors")
plt.xlabel("Resistor (Ohm)")
plt.ylabel("Avg Power in First 30 Seconds (mW)")
plt.title("Avg Power in First 30s vs Resistor")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("avg_power_30s_vs_resistor_split.png")
plt.show()
