import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Extracted data from Magnet.pdf as provided in the turn history summary
data = {
    'Test_ID': ['#1.1', '#1.2', '#2.1', '#2.2', '#2.3', '#2.4', '#3.1', '#3.2', '#3.3', '#3.4', '#3.5', '#4.1', '#4.2', '#4.3'],
    'Magnets': [1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4],
    'AvgActivePower': [22.644, 38.322, 71.465, 61.057, np.nan, 57.291, 131.809, 91.479, 73.165, 73.165, 73.165, 193.969, 175.815, 192.010],
    'SE_ActivePower': [0, 0, 3.9173, 2.4043, 0, 2.1612, 5.4045, 3.0579, 2.5949, 2.5949, 2.5949, 6.2005, 6.5544, 8.2911],
    'AvgCyclePower': [0, 0, 0, 0, 145.943, 163.902, 283.282, 262.796, 178.861, 178.861, 178.861, 450.274, 449.112, 496.686],
    'SE_CyclePower': [0, 0, 0, 0, 8.1813, 40.3451, 51.2437, 39.7616, 27.2969, 27.2969, 27.2969, 66.2822, 71.7139, 74.2996]
}

df = pd.DataFrame(data)

# Create Plot 1: Average Active Power vs Number of Magnets
plt.figure(figsize=(10, 6))
mask1 = df['AvgActivePower'].notna()
plt.errorbar(df[mask1]['Magnets'], df[mask1]['AvgActivePower'], yerr=df[mask1]['SE_ActivePower'], fmt='o', capsize=5, label='Active Power (Individual Tests)', color='#2a9d8f')

# Trendline for Active Power
active_means = df[mask1].groupby('Magnets')['AvgActivePower'].mean()
plt.plot(active_means.index, active_means.values, '--', color='#264653', label='Mean Active Power Trend')

plt.xlabel('Number of Magnets')
plt.ylabel('Average Active Power (mW)')
plt.title('WEC Performance: Average Active Power vs. Number of Magnets')
plt.xticks([1, 2, 3, 4])
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('active_power_vs_magnets.png')
plt.close()

# Create Plot 2: Average Cycle Power vs Number of Magnets
plt.figure(figsize=(10, 6))
mask2 = df['AvgCyclePower'] > 0
plt.errorbar(df[mask2]['Magnets'], df[mask2]['AvgCyclePower'], yerr=df[mask2]['SE_CyclePower'], fmt='s', capsize=5, label='Cycle Power (Individual Tests)', color='#e63946')

# Trendline for Cycle Power
cycle_means = df[mask2].groupby('Magnets')['AvgCyclePower'].mean()
plt.plot(cycle_means.index, cycle_means.values, '--', color='#1d3557', label='Mean Cycle Power Trend')

plt.xlabel('Number of Magnets')
plt.ylabel('Average Cycle Power (mW)')
plt.title('WEC Performance: Average Cycle Power vs. Number of Magnets')
plt.xticks([1, 2, 3, 4])
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('cycle_power_vs_magnets.png')
plt.close()

# Export table to CSV
df.to_csv('extracted_magnet_test_data.csv', index=False)