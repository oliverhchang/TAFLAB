import numpy as np
import matplotlib.pyplot as plt

# Create continuous array for AWG
awg = np.linspace(26, 36, 500)

# Coil scaling relationships
N = 1530 * (1.59 ** ((awg - 30) / 2))
R = 40 * (2.5 ** ((awg - 30) / 2))
V = 7.5 * (N / 1530)

# Rectifier drop
V_drop = 1.2

# Usable power
P_usable = np.where(V > V_drop, ((V - V_drop)**2) / R, 0)
P_usable_norm = P_usable / np.max(P_usable)

# Plot
fig, ax1 = plt.subplots()

# Usable power (primary axis)
ax1.plot(awg, P_usable_norm, label='Usable Power')
ax1.set_xlabel('Wire Gauge (AWG)')
ax1.set_ylabel('Normalized Power')
ax1.set_xticks(np.arange(26, 37, 2))
ax1.grid(True)

# Secondary axis
ax2 = ax1.twinx()
ax2.plot(awg, V, linestyle='--', label='Voltage (V)')
ax2.plot(awg, R / 10, linestyle=':', label='Resistance (R/10)')
ax2.set_ylabel('Voltage / Resistance')

# Rectifier threshold
ax2.axhline(V_drop, linestyle='-', label='1.2V Drop')

# Optimal point
best_awg = awg[np.argmax(P_usable_norm)]
ax1.axvline(best_awg, linestyle='-.')
ax1.plot(best_awg, 1.0, marker='o')

# Title
plt.title('Wire Gauge Optimization')

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2)

plt.tight_layout()
plt.show()