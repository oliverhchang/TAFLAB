import serial
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
SERIAL_PORT = '/dev/cu.usbserial-0001'  # Ensure this matches your port
BAUD_RATE = 115200
NOISE_THRESHOLD = 0.5  # mW threshold to separate real movement from sensor noise

times = []
powers = []

print(f"Connecting to {SERIAL_PORT} at High Speed...")

try:
    with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
        print("Connected! Roll the magnet. Press Ctrl+C to stop and graph.")

        import time

        time.sleep(0.5)
        ser.reset_input_buffer()

        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()

                if not line or line == "START":
                    continue

                try:
                    parts = line.split(',')
                    if len(parts) == 4:
                        time_ms = float(parts[0])
                        power_val = float(parts[3])

                        time_sec = time_ms / 1000.0

                        times.append(time_sec)
                        powers.append(power_val)

                        if len(times) % 20 == 0:
                            print(f"Sampling... {time_sec:.1f}s | {power_val:.2f} mW")

                except ValueError:
                    pass

except KeyboardInterrupt:
    print(f"\nData collection stopped. Collected {len(times)} data points.")
except serial.SerialException as e:
    print(f"\nSerial Error: {e}")

# --- CALCULATE ACTIVE AVERAGE & GENERATE THE GRAPH ---
if times:
    start_time = times[0]
    times = [t - start_time for t in times]

    # 1. Find all data points where power is above our noise floor
    active_indices = [i for i, p in enumerate(powers) if p > NOISE_THRESHOLD]

    plt.figure(figsize=(10, 6))

    # Plot the raw power data
    plt.plot(times, powers, color='#e63946', linewidth=1.5, label='Harvested Power (mW)')
    plt.fill_between(times, powers, color='#e63946', alpha=0.2)

    # 2. Calculate the "Active Window"
    if active_indices:
        start_idx = active_indices[0]
        end_idx = active_indices[-1]

        # Slice the lists to only include the wave cycle period
        active_powers = powers[start_idx:end_idx + 1]
        active_average = sum(active_powers) / len(active_powers)
        active_duration = times[end_idx] - times[start_idx]

        print("-" * 40)
        print("RESULTS: Active Power Calculation")
        print(f"Active Duration:      {active_duration:.2f} seconds")
        print(f"Average Active Power: {active_average:.3f} mW")
        print("-" * 40)

        # Draw the average line
        plt.axhline(y=active_average, color='#1d3557', linestyle='--', linewidth=2,
                    label=f'Active Average ({active_average:.2f} mW)')

        # Highlight the active window on the graph in light yellow
        plt.axvspan(times[start_idx], times[end_idx], color='#f4a261', alpha=0.15,
                    label=f'Active Window ({active_duration:.1f}s)')
    else:
        print("\nNo power spikes detected above threshold. Generator remained idle.")

    # Formatting
    plt.title("Wave Energy Converter: Active Power Profile", fontsize=16, fontweight='bold')
    plt.xlabel("Elapsed Time (Seconds)", fontsize=12)
    plt.ylabel("Output Power (milliwatts)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right', fontsize=11)
    plt.tight_layout()
    plt.show()
else:
    print("No data collected.")