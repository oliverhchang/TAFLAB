import pandas as pd
import matplotlib.pyplot as plt
import os

# List of filenames to load
file_names = [
    "WaveData2/Wave Energy _ Testing Data - 1 ohm _ 4700 uF.csv",
    "WaveData2/Wave Energy _ Testing Data - 1 ohm _ 1000 uF.csv",
    "WaveData2/Wave Energy _ Testing Data - 10 ohm _ 4700 uF.csv",
    "WaveData2/Wave Energy _ Testing Data - 10 ohm _ 1000 uF.csv"
]
# Smoothing interval in seconds
smooth_seconds = 5

# Create plot
plt.figure(figsize=(12, 6))

for file in file_names:
    if os.path.exists(file):
        # Load and clean data
        df = pd.read_csv(file)
        df['Time(s)'] = pd.to_numeric(df['Time(s)'], errors='coerce')
        df['Power(mW)'] = pd.to_numeric(df['Power(mW)'], errors='coerce')
        df = df.dropna(subset=['Time(s)', 'Power(mW)']).sort_values('Time(s)').reset_index(drop=True)

        # Estimate time step and window size
        time_step = df['Time(s)'].diff().median()
        window_size = max(1, int(smooth_seconds / time_step))

        # Apply rolling average
        df['Smoothed Power'] = df['Power(mW)'].rolling(window=window_size, center=True).mean()

        # Plot with filename as label (without extension)
        label = os.path.splitext(os.path.basename(file))[0]
        plt.plot(df['Time(s)'], df['Smoothed Power'], label=label)

# Final plot formatting
plt.title(f'Smoothed Power vs. Time ({smooth_seconds}s Rolling Average)')
plt.xlabel('Time (s)')
plt.ylabel('Power (mW)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()