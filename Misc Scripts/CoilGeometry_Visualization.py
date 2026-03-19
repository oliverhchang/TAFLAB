import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

# ==========================================
# GRAPH 1: Radial Flux Profile
# ==========================================
r = np.linspace(0, 30, 500)
magnet_radius = 12.5

# Gaussian approximation of flux gradient
flux_gradient = np.exp(-((r - magnet_radius)**2) / (2 * 2.0**2))

plt.figure()
plt.plot(r, flux_gradient, linewidth=2, label='|dBz/dr|')
plt.axvspan(6.25, 18.75, alpha=0.2, label='Coil Region')
plt.axvline(12.5, linestyle='--', label='Mean Radius')

plt.title('Radial Flux Profile')
plt.xlabel('Radius (mm)')
plt.ylabel('Normalized Flux Gradient')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('graph1_radial_flux.png')


# ==========================================
# GRAPH 2: Power Figure of Merit
# ==========================================
OD_array = np.linspace(13, 60, 400)
r_in = 12.5 / 2.0  # 6.25 mm

power_fom = []

for od in OD_array:
    r_out = od / 2.0
    r_coil = np.linspace(r_in, r_out, 100)

    grad = np.exp(-((r_coil - magnet_radius)**2) / (2 * 2.0**2))
    voltage = np.trapz(grad, r_coil)

    resistance = (r_out**2 - r_in**2)

    power_fom.append((voltage**2) / resistance)

power_fom = np.array(power_fom)
power_fom = power_fom / np.max(power_fom)

plt.figure()
plt.plot(OD_array, power_fom, linewidth=2, label='Power (V^2 / R)')
plt.axvline(37.5, linestyle='--', label='Optimal OD')

plt.title('Power vs Outer Diameter')
plt.xlabel('Outer Diameter (mm)')
plt.ylabel('Normalized Power')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('graph2_power_fom.png')


# ==========================================
# GRAPH 3: Phase Cancellation
# ==========================================
heights = np.linspace(0, 30, 500)

radial_flux_distribution = np.cos(np.pi * heights / 25.0)
voltage_total = cumulative_trapezoid(radial_flux_distribution, heights, initial=0)
voltage_total = voltage_total / np.max(voltage_total)

plt.figure()
plt.plot(heights, voltage_total, linewidth=2, label='Induced Voltage')
plt.axvline(12.5, linestyle='--', label='Optimal Height')
plt.axvspan(12.5, 30, alpha=0.1, label='Cancellation Region')

plt.title('Phase Cancellation vs Height')
plt.xlabel('Height (mm)')
plt.ylabel('Normalized Voltage')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('graph3_phase_cancellation.png')

plt.show()