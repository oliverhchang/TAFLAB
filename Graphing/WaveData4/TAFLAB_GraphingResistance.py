import pandas as pd
import matplotlib.pyplot as plt

# Updated data
data = {
    "Resistance": [0.2, 0.25, 0.33, 0.5, 1],
    "Overall Avg Power (mW)": [62.83141536, 62.29122761, 52.89055582, 75.30196982, 22.09927835],
    "Overall Std Error (mW)": [1.235709234, 1.26409729, 1.058341913, 1.3395199, 0.5209326855],
    "Avg Power per Swing (mW)": [94.42050894, 98.42202729, 87.39895833, 113.7347804, 38.41203162],
    "Std Error of Swing Avg (mW)": [2.867440163, 3.53709331, 2.886882142, 2.696173041, 1.490640668]
}

df = pd.DataFrame(data)

# Plot 1: Resistance vs Overall Avg Power with error bars
plt.figure(figsize=(10, 5))
plt.errorbar(df['Resistance'], df['Overall Avg Power (mW)'],
             yerr=df['Overall Std Error (mW)'], fmt='o', capsize=5, label='Overall Avg Power')
plt.title('Resistance vs Overall Avg Power (mW)')
plt.xlabel('Resistance (ohms)')
plt.ylabel('Overall Avg Power (mW)')
plt.grid(True)
plt.legend()
plt.show()

# Plot 2: Resistance vs Avg Power per Swing with error bars
plt.figure(figsize=(10, 5))
plt.errorbar(df['Resistance'], df['Avg Power per Swing (mW)'],
             yerr=df['Std Error of Swing Avg (mW)'], fmt='o', capsize=5, color='orange', label='Avg Power per Swing')
plt.title('Resistance vs Avg Power per Swing (mW)')
plt.xlabel('Resistance (ohms)')
plt.ylabel('Avg Power per Swing (mW)')
plt.grid(True)
plt.legend()
plt.show()
