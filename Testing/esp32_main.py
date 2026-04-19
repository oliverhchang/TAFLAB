import serial
import matplotlib.pyplot as plt
import time
import csv
import os
import math

# --- CONFIGURATION ---
SERIAL_PORT = '/dev/cu.usbserial-0001'  # Ensure this matches your port
BAUD_RATE = 115200
NOISE_THRESHOLD = 0.5  # mW threshold to separate real movement from sensor noise

times = []
voltages = []
currents = []
powers = []

print(f"Connecting to {SERIAL_PORT} at High Speed...")

try:
    with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
        print("Connected! Roll the magnet. Press Ctrl+C to stop and graph.")

        time.sleep(0.5)
        ser.reset_input_buffer()

        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()

                if not line or line == "START":
                    continue

                try:
                    parts = line.split(',')
                    # Expecting: Time_ms, Voltage_V, Current_mA, Power_mW
                    if len(parts) == 4:
                        time_sec = float(parts[0]) / 1000.0
                        volts_val = float(parts[1])
                        curr_val = float(parts[2])
                        power_val = float(parts[3])

                        times.append(time_sec)
                        voltages.append(volts_val)
                        currents.append(curr_val)
                        powers.append(power_val)

                        # Print status every ~20 samples to avoid slowing down Python
                        if len(times) % 20 == 0:
                            print(
                                f"Sampling... {time_sec:.1f}s | {volts_val:.2f} V | {curr_val:.1f} mA | {power_val:.2f} mW")

                except ValueError:
                    pass

except KeyboardInterrupt:
    print(f"\nData collection stopped. Collected {len(times)} data points.")
except serial.SerialException as e:
    print(f"\nSerial Error: {e}")

# --- SAVE CSV TO DOWNLOADS FOLDER ---
if times:
    start_time = times[0]
    times = [t - start_time for t in times]

    downloads_path = os.path.expanduser("~/Downloads/wave_energy_data.csv")
    with open(downloads_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Time_s", "Voltage_V", "Current_mA", "Power_mW"])
        for row in zip(times, voltages, currents, powers):
            writer.writerow(row)
    print(f"CSV saved to: {downloads_path}")

# --- CALCULATE ACTIVE AVERAGE, SE, FREQUENCY & GENERATE GRAPHS ---
if times:
    # 1. Find all data points where power is above our noise floor
    active_indices = [i for i, p in enumerate(powers) if p > NOISE_THRESHOLD]

    # Create main figure: 3 stacked subplots sharing the same X (Time) axis
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10, 9))
    fig.suptitle("Wave Energy Converter: Full Electrical Profile", fontsize=16, fontweight='bold')

    # --- PLOT 1: VOLTAGE ---
    ax1.plot(times, voltages, color='#2a9d8f', linewidth=1.5, label='Voltage (V)')
    ax1.fill_between(times, voltages, color='#2a9d8f', alpha=0.2)
    ax1.set_ylabel("Voltage (V)", fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper right')

    # --- PLOT 2: CURRENT ---
    ax2.plot(times, currents, color='#e9c46a', linewidth=1.5, label='Current (mA)')
    ax2.fill_between(times, currents, color='#e9c46a', alpha=0.2)
    ax2.set_ylabel("Current (mA)", fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='upper right')

    # --- PLOT 3: POWER ---
    ax3.plot(times, powers, color='#e63946', linewidth=1.5, label='Power (mW)')
    ax3.fill_between(times, powers, color='#e63946', alpha=0.2)
    ax3.set_ylabel("Power (mW)", fontsize=11)
    ax3.set_xlabel("Elapsed Time (Seconds)", fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.7)

    # 2. Calculate Active Window, SE, and Frequency
    if active_indices:
        start_idx = active_indices[0]
        end_idx = active_indices[-1]

        active_powers = powers[start_idx:end_idx + 1]
        active_times = times[start_idx:end_idx + 1]
        n = len(active_powers)
        active_average = sum(active_powers) / n
        active_duration = times[end_idx] - times[start_idx]

        # --- IDENTIFY INDIVIDUAL CYCLES (0 → peak → 0) ---
        cycles = []
        current_cycle = []
        in_cycle = False

        for i, p in enumerate(powers):
            if p > NOISE_THRESHOLD:
                current_cycle.append(p)
                in_cycle = True
            elif p <= NOISE_THRESHOLD and in_cycle:
                # End of a cycle
                if len(current_cycle) > 2:  # avoid tiny noise blips
                    cycles.append(current_cycle)
                current_cycle = []
                in_cycle = False

        # Catch edge case if last cycle doesn't close
        if in_cycle and len(current_cycle) > 2:
            cycles.append(current_cycle)
        cycle_means = []
        cycle_ses = []

        for cycle in cycles:
            n_c = len(cycle)
            mean_c = sum(cycle) / n_c

            if n_c > 1:
                var_c = sum((p - mean_c) ** 2 for p in cycle) / (n_c - 1)
                std_c = math.sqrt(var_c)
                se_c = std_c / math.sqrt(n_c)
            else:
                se_c = 0

            cycle_means.append(mean_c)
            cycle_ses.append(se_c)
        # --- STANDARD ERROR OF POWER ---
        variance = sum((p - active_average) ** 2 for p in active_powers) / (n - 1) if n > 1 else 0
        std_dev = math.sqrt(variance)
        std_error = std_dev / math.sqrt(n)

        # --- FREQUENCY CALCULATION ---
        # Count how many times the power crosses above the active average
        pulse_count = 0
        is_above_avg = False
        pulse_start_times = []

        for i, p in enumerate(active_powers):
            if p > active_average and not is_above_avg:
                pulse_count += 1
                is_above_avg = True
                pulse_start_times.append(active_times[i])
            elif p < active_average and is_above_avg:
                is_above_avg = False

        # Calculate frequency (Pulses per second)
        pulse_frequency = pulse_count / active_duration if active_duration > 0 else 0

        # NOTE: If your magnet passes the coil TWICE per wave cycle (forward and back),
        # the true mechanical wave frequency is exactly half of the pulse frequency.
        wave_frequency = pulse_frequency / 2.0

        print("-" * 40)
        print("RESULTS: Active Power & Frequency")
        print(f"Active Duration:      {active_duration:.2f} seconds")
        print(f"Average Active Power: {active_average:.3f} mW")
        print(f"Std Dev of Power:     {std_dev:.3f} mW")
        print(f"Std Error of Power:   {std_error:.4f} mW  (n={n})")
        print(f"Pulse Count:          {pulse_count} passes")
        print(f"Pulse Frequency:      {pulse_frequency:.2f} Hz (magnet passes per sec)")
        print(f"Est. Wave Frequency:  {wave_frequency:.2f} Hz (full mechanical cycles)")
        if cycle_means:
            avg_cycle_power = sum(cycle_means) / len(cycle_means)
            avg_cycle_se = sum(cycle_ses) / len(cycle_ses)

            print(f"Avg Cycle Power:      {avg_cycle_power:.3f} mW")
            print(f"Avg Cycle SE:         {avg_cycle_se:.4f} mW")

        # Draw the average power line on the bottom plot
        ax3.axhline(y=active_average, color='#1d3557', linestyle='--', linewidth=2,
                    label=f'Active Avg ({active_average:.2f} ± {std_error:.3f} mW SE)')
        ax3.legend(loc='upper right')

        # Highlight the active window on ALL three graphs
        for ax in [ax1, ax2, ax3]:
            ax.axvspan(times[start_idx], times[end_idx], color='#f4a261', alpha=0.15)

        # --- FIGURE 2: CONSOLIDATED SINGLE PHASE ZOOM ---
        # 1. Identify the first continuous "active" hump
        phase_start_idx = active_indices[0]
        phase_end_idx = phase_start_idx

        # Find exactly where it drops back below the noise threshold
        for i in range(phase_start_idx, len(powers)):
            if powers[i] < NOISE_THRESHOLD:
                phase_end_idx = i
                break
        else:
            phase_end_idx = len(powers) - 1

        # 2. Extract window data with 10% padding
        pad = (times[phase_end_idx] - times[phase_start_idx]) * 0.1
        z_start, z_end = times[phase_start_idx] - pad, times[phase_end_idx] + pad

        z_indices = [i for i, t in enumerate(times) if z_start <= t <= z_end]
        z_t = [times[i] for i in z_indices]
        z_v = [voltages[i] for i in z_indices]
        z_p = [powers[i] for i in z_indices]

        # 3. Create only TWO subplots for clarity
        fig2, (zax1, zax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
        fig2.suptitle("Wave Energy Converter: Isolated Phase Analysis", fontsize=14, fontweight='bold')

        # Voltage subplot
        zax1.plot(z_t, z_v, color='#2a9d8f', linewidth=2, label="Voltage (V)")
        zax1.set_ylabel("Voltage (V)")
        zax1.grid(True, alpha=0.3)
        zax1.legend(loc='upper right')

        # Power subplot
        zax2.plot(z_t, z_p, color='#e63946', linewidth=2, label="Power (mW)")
        zax2.fill_between(z_t, z_p, color='#e63946', alpha=0.2)
        zax2.set_ylabel("Power (mW)")
        zax2.set_xlabel("Time (Seconds)")
        zax2.grid(True, alpha=0.3)
        zax2.legend(loc='upper right')

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)

    else:
        ax3.legend(loc='upper right')
        print("\nNo power spikes detected above threshold. Generator remained idle.")

    plt.figure(fig.number)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()
else:
    print("No data collected.")