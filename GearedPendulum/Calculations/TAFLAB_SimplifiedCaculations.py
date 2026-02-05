import numpy as np
import matplotlib.pyplot as plt

# === Constants ===
g = 9.81  # gravity [m/s^2]
rho_sand = 1922  # wet sand [kg/m^3]
rho_steel = 7850  # 1008 CR Steel [kg/m^3]

# === Pendulum Geometry ===
r = 0.1016  # radius [m] (4 in)
l = 8 * 0.0254  # length [m] (8 in)
d = 5.75 * 0.0254  # offset from shaft [m] (5.75 in)
theta_deg = 37.5
theta_rad = np.radians(theta_deg)

# === Pendulum Properties ===
volume_cyl = np.pi * r ** 2 * l
mass = rho_sand * volume_cyl
I_cm = 0.5 * mass * r ** 2
I_pendulum = I_cm + mass * d ** 2
torque_available = mass * g * d * np.sin(theta_rad)
PE_max = mass * g * d * (1 - np.cos(theta_rad))
omega_max = np.sqrt(2 * PE_max / I_pendulum)
rpm_max = omega_max * 60 / (2 * np.pi)

# === Generator Specs ===
ke = 0.03  # per-phase peak back-EMF [V/RPM]
I_rotor = 6.8e-5  # motor inertia [kg·m²]
V_charge = 3  # Charge Voltage [V]
eta = 0.5  # IMPROVED efficiency (30% was too conservative)
battery_capacity_Ah = 5.0  # 10,000 mAh battery

# === Electrical Load Specs ===
R_int = 3.0  # Stepper internal resistance [ohms]
R_load = 10.0  # Match label in print statement

# === Electrical Power Calculation ===
I_actual = V_charge / (R_int + R_load)
V_oc = V_charge + I_actual * R_int  # Open-circuit voltage
rpm_motor = V_oc / ke  # CORRECTED speed calculation
omega_motor = 2 * np.pi * rpm_motor / 60
V_load = I_actual * R_load
P_elec = V_charge * I_actual  # Power delivered to battery
P_mech = P_elec / eta
I_charge = I_actual
time_to_charge_hr = battery_capacity_Ah / I_charge

# === Gear Ratio ===
ratio = rpm_motor / rpm_max  # Fixed hardware ratio

# === Flywheel Design ===
T_half = np.pi / omega_max  # Half-period of swing
E_target = P_mech * T_half * 2  # 50% safety factor
I_flywheel_needed = (2 * E_target) / (omega_motor ** 2)
thickness = 0.00635  # 0.25 in = 6.35 mm
R_fly = ((2 * I_flywheel_needed) / (rho_steel * np.pi * thickness)) ** 0.25
mass_flywheel = rho_steel * np.pi * R_fly ** 2 * thickness
E_flywheel = 0.5 * I_flywheel_needed * omega_motor ** 2

# === Startup Time ===
I_motor_side = I_flywheel_needed + I_rotor
I_total_pendulum_side = I_pendulum + (I_motor_side * ratio ** 2)
torque_avg = 0.5 * torque_available  # Average torque during swing
alpha_avg = torque_avg / I_total_pendulum_side
omega_total = np.sqrt(2 * PE_max / I_total_pendulum_side)
startup_time = omega_total / alpha_avg

# === Startup Torque Check ===
torque_continuous = P_mech / omega_motor
torque_load_reflected = torque_continuous * ratio
alpha_motor_needed = omega_motor / startup_time
alpha_pendulum_needed = alpha_motor_needed / ratio
torque_for_accel = I_total_pendulum_side * alpha_pendulum_needed * 1.5  # 50% safety margin
torque_total_needed = torque_for_accel + torque_load_reflected
margin = torque_available - torque_total_needed

# === Print Results ===
print("--- CALCULATIONS FOR NOMINAL R_load = 10.0 Ohms ---")
print("=== Pendulum Analysis ===")
print(f"Pendulum mass: {mass:.1f} kg")
print(f"Moment of inertia: {I_pendulum:.4f} kg·m²")
print(f"Static torque @ {theta_deg}°: {torque_available:.2f} Nm")
print(f"Max swing speed: {rpm_max:.1f} RPM")
print(f"Half stroke period: {T_half:.3f} s\n")

print("=== Generator + Load ===")
print(f"Load resistance: {R_load:.2f} ohms")
print(f"Current draw: {I_actual:.3f} A")
print(f"Total electrical power generated: {P_elec:.2f} W")
print(f"Mechanical power required: {P_mech:.2f} W")
print(f"Time to charge 10Ah battery: {time_to_charge_hr:.1f} hours")
print(f"Motor speed needed: {rpm_motor:.0f} RPM")
print(f"Continuous load torque: {torque_continuous:.3f} Nm\n")

print("=== Gearing & Flywheel ===")
print(f"Required Gear Ratio (pendulum: motor): 1:{ratio:.2f}")
print(f"Flywheel inertia needed: {I_flywheel_needed:.4f} kg·m²")
print(f"Flywheel mass (t={thickness * 1000:.1f} mm): {mass_flywheel:.2f} kg")
print(f"Flywheel radius: {R_fly * 100:.1f} cm")
print(f"Flywheel stored energy: {E_flywheel:.2f} J\n")

print("=== Startup Torque Check ===")
print(f"Calculated startup time: {startup_time:.3f} s")
print(f"System inertia (reflected to pendulum): {I_total_pendulum_side:.4f} kg·m²")
print(f"Torque for acceleration: {torque_for_accel:.3f} Nm")
print(f"Reflected load torque: {torque_load_reflected:.3f} Nm")
print(f"Total startup torque needed: {torque_total_needed:.3f} Nm")
print(f"Available pendulum torque: {torque_available:.3f} Nm")
print(f"Safety margin: {margin:.3f} Nm")

if margin > 0:
    print("✓ Should start successfully")
else:
    print("✗ Need more mass, longer arm, or steeper angle")

# === Sweep Load Resistance ===
sweep = True
if sweep:
    R_vals = np.linspace(0.5, 40, 200)
    margins = []
    power_vals = []
    torque_avg = 0.5 * torque_available
    fixed_ratio = ratio  # Use nominal gear ratio
    fixed_flywheel_inertia = I_flywheel_needed  # <-- CORRECTED LINE: The flywheel is a fixed physical part.

    for R in R_vals:
        # Recalculate electrical parameters
        I_actual = V_charge / (R_int + R)
        V_oc = V_charge + I_actual * R_int
        rpm_motor = V_oc / ke
        omega_motor = 2 * np.pi * rpm_motor / 60

        # Power calculations
        P_elec = V_charge * I_actual
        P_mech = P_elec / eta
        power_vals.append(P_elec)

        # Inertia calculations use the FIXED flywheel inertia
        # <-- CORRECTED BLOCK: Removed flywheel recalculation
        I_motor_side = fixed_flywheel_inertia + I_rotor
        I_total_pendulum_side = I_pendulum + (I_motor_side * fixed_ratio ** 2)

        # Startup dynamics
        omega_total = np.sqrt(2 * PE_max / I_total_pendulum_side)
        alpha_avg = torque_avg / I_total_pendulum_side
        startup_time = omega_total / alpha_avg

        # Torque requirements
        torque_continuous = P_mech / omega_motor
        torque_load_reflected = torque_continuous * fixed_ratio
        alpha_motor_needed = omega_motor / startup_time
        alpha_pendulum_needed = alpha_motor_needed / fixed_ratio
        torque_for_accel = I_total_pendulum_side * alpha_pendulum_needed * 1.3
        torque_total_needed = torque_for_accel + torque_load_reflected
        margin = torque_available - torque_total_needed
        margins.append(margin)

    # === Plot Results ===
    fig, ax1 = plt.subplots(figsize=(12, 7))
    color1 = 'C0'
    ax1.set_xlabel("Load Resistance (Ohms)")
    ax1.set_ylabel("Startup Torque Margin (Nm)", color=color1)
    ax1.plot(R_vals, margins, color=color1, label="Torque Margin")
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, linestyle=':')
    ax1.axhline(0, color='gray', linestyle='--', linewidth=1.5)

    ax2 = ax1.twinx()
    color2 = 'C1'
    ax2.set_ylabel("Electrical Power (W)", color=color2)
    ax2.plot(R_vals, power_vals, color=color2, label="Power Generated")
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title("System Performance vs. Load Resistance", fontweight='bold')
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
    fig.tight_layout()
    plt.show()