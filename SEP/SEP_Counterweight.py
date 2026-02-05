import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. FIXED PARAMETERS (Your Build)
# ==========================================
g = 9.81
Target_Freq = 0.5  # Hz

# -- Dimensions --
R_pendulum = 0.25  # Main Pendulum Radius (25 cm)
H_rod = 0.25  # Counterweight Rod Length (25 cm) - FIXED

# -- Mass --
Mass_Magnets = 36 * 0.168  # 6.048 kg
Mass_Shell = 1.50  # 1.5 kg Structure
M_pendulum = Mass_Magnets + Mass_Shell  # 7.55 kg

# -- Physics Approximations --
L_pendulum = R_pendulum / 2  # CoM of shell
J_pendulum = (2 / 3) * M_pendulum * (R_pendulum ** 2)  # Inertia

print(f"--- CONFIGURATION ---")
print(f"Pendulum Radius: {R_pendulum * 100:.0f} cm")
print(f"Counterweight Rod: {H_rod * 100:.0f} cm")
print(f"Total Hanging Mass: {M_pendulum:.2f} kg")
print("-" * 30)


# ==========================================
# 2. SOLVER (Find Exact Weight)
# ==========================================
def get_freq(m_cw):
    """Calculate Hz for a specific mass on the 25cm rod"""
    m_total = M_pendulum + m_cw

    # Effective Center of Mass
    net_moment = (M_pendulum * L_pendulum) - (m_cw * H_rod)
    if net_moment < 0: return None  # Unstable
    l_eff = net_moment / m_total

    # Total Inertia
    j_total = J_pendulum + (m_cw * H_rod ** 2)

    # Frequency
    return (1 / (2 * np.pi)) * np.sqrt((m_total * g * l_eff) / j_total)


# Solve for exact mass to hit 0.5 Hz
# (Using brute force sweep for simplicity and robustness)
masses = np.linspace(0, 5, 5000)
best_m = 0
min_error = 1.0

freqs = []
valid_masses = []

for m in masses:
    f = get_freq(m)
    if f is not None:
        valid_masses.append(m)
        freqs.append(f)
        if abs(f - Target_Freq) < min_error:
            min_error = abs(f - Target_Freq)
            best_m = m

# ==========================================
# 3. PLOTTING
# ==========================================
plt.figure(figsize=(10, 6))

# Plot the Tuning Curve
plt.plot(valid_masses, freqs, color='blue', linewidth=3, label='Tuning Curve')

# Mark the Target
plt.axhline(Target_Freq, color='red', linestyle='--', label='Target (0.5 Hz)')
plt.axvline(best_m, color='green', linestyle=':', label=f'Required: {best_m:.2f} kg')

# Add Dot
plt.plot(best_m, Target_Freq, 'ro', markersize=10)

# Annotate
plt.title(f'Tuning for 25cm Counterweight Rod', fontsize=14)
plt.xlabel('Counterweight Mass (kg)', fontsize=12)
plt.ylabel('Natural Frequency (Hz)', fontsize=12)
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.legend(fontsize=11)
plt.ylim(0, 1.0)
plt.tight_layout()
plt.show()

# ==========================================
# 4. REPORT
# ==========================================
print(f"\n>>> FINAL RESULT <<<")
print(f"To achieve 0.5 Hz with a 25cm rod:")
print(f"REQUIRED MASS: {best_m:.2f} kg  ({best_m * 2.2:.1f} lbs)")
print(
    f"STABILITY CHECK: The Center of Mass is {(M_pendulum * L_pendulum - best_m * H_rod) / (M_pendulum + best_m) * 100:.1f} cm below pivot.")
print("(Positive value = Stable. Negative = Flips upside down.)")