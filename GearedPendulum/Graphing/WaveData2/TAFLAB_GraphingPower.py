import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = "Wave Energy _ Testing Data - 1 ohm _ 1000 uF.csv"
df = pd.read_csv(file_path)

# Extract time and power columns
time = df['Time(s)']
power = df['Power(mW)']

# Plot power vs time
plt.figure(figsize=(10, 5))
plt.plot(time, power, color='blue')
plt.title('Power Generated vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Power (mW)')
plt.grid(True)
plt.tight_layout()
plt.show()
