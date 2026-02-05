import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. LOAD DATA FROM FILE
# ==========================================
# Make sure 'Experiment_Log.csv' is in the same folder as this script
filename = 'Experiment_Log.csv'

try:
    df = pd.read_csv(filename)
except FileNotFoundError:
    print(f"❌ Error: Could not find '{filename}'. Make sure the file exists.")
    exit()

# ==========================================
# 2. DATA PROCESSING
# ==========================================
# Create a readable "Configuration" label
df['Config'] = df.apply(
    lambda x: f"{int(x['Gauge_AWG'])}AWG_{int(x['Turns'])}T_{int(x['ID_mm'])}mm",
    axis=1
)

# Group by Configuration and calculate averages
summary = df.groupby('Config')[
    ['Peak_Power_mW', 'Window_Avg_Power_mW', 'Output_Energy_mJ', 'Efficiency_Pct']
].mean().reset_index()

# Sort by Efficiency (Best to Worst)
summary = summary.sort_values('Efficiency_Pct', ascending=False)

# ==========================================
# 3. PRINT RESULTS
# ==========================================
print("="*80)
print("             PERFORMANCE SUMMARY (Sorted by Efficiency)")
print("="*80)
print(summary.to_string(index=False))
print("\n" + "="*80)

# Identify the winner
if not summary.empty:
    winner = summary.iloc[0]
    print(f"🏆 WINNER: {winner['Config']}")
    print(f"   Efficiency: {winner['Efficiency_Pct']:.2f}%")
    print(f"   Peak Power: {winner['Peak_Power_mW']:.1f} mW")
else:
    print("No data found.")
print("="*80)

# ==========================================
# 4. VISUALIZATION
# ==========================================
sns.set_theme(style="whitegrid")
plt.figure(figsize=(14, 8))

# Subplot 1: Efficiency Comparison
plt.subplot(2, 2, 1)
sns.barplot(x='Efficiency_Pct', y='Config', data=df, errorbar=None, palette='viridis', hue='Config', legend=False)
plt.title('Average Efficiency (%) - Higher is Better')
plt.xlabel('Efficiency (%)')
plt.ylabel('')

# Subplot 2: Peak Power Comparison
plt.subplot(2, 2, 2)
sns.barplot(x='Peak_Power_mW', y='Config', data=df, errorbar=None, palette='magma', hue='Config', legend=False)
plt.title('Average Peak Power (mW) - Higher is Better')
plt.xlabel('Power (mW)')
plt.ylabel('')
plt.yticks([])

# Subplot 3: Energy Output Comparison
plt.subplot(2, 2, 3)
sns.barplot(x='Output_Energy_mJ', y='Config', data=df, errorbar=None, palette='plasma', hue='Config', legend=False)
plt.title('Average Output Energy (mJ) - Higher is Better')
plt.xlabel('Energy (mJ)')
plt.ylabel('')

plt.tight_layout()
plt.show()