import signal
import sys
import serial
import matplotlib
matplotlib.use('TkAgg')  # macOS + PyCharm fix — forces a real window backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import csv
import os
import math
import numpy as np
from scipy.signal import find_peaks

# ── SIGTERM handler (PyCharm stop button sends SIGTERM, not KeyboardInterrupt) ─
def _sigterm_handler(signum, frame):
    raise KeyboardInterrupt

signal.signal(signal.SIGTERM, _sigterm_handler)


SERIAL_PORT     = '/dev/cu.usbserial-0001'   # Change to match your port
BAUD_RATE       = 115200
NOISE_THRESHOLD = 0.5    # mW  – separates real motion from sensor noise

# Raw storage lists
times    = []
voltages = []
currents = []
powers   = []
rolls    = []
pitches  = []
yaws     = []

print(f"Connecting to {SERIAL_PORT} at {BAUD_RATE} baud …")

try:
    with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2) as ser:
        print("Connected!  Roll the magnet.  Press Ctrl+C to stop and graph.")
        time.sleep(0.5)
        ser.reset_input_buffer()

        while True:
            # Blocking readline (timeout=2s) instead of polling in_waiting —
            # more reliable on macOS USB-serial drivers.
            line = ser.readline().decode('utf-8', errors='ignore').strip()

            if not line:
                print("  [waiting for data…]")
                continue

            print(f"RAW: {line}")   # ← print every raw line from the ESP32

            if line == "START":
                continue

                try:
                    parts = line.split(',')
                    # Expected: Time_ms, Voltage_V, Current_mA, Power_mW,
                    #           Roll_deg, Pitch_deg, Yaw_deg
                    #
                    # Guard: parts[0] must be a plain integer (milliseconds).
                    # Corrupt/partial lines at startup often start with a comma
                    # or a float fragment – this rejects them cleanly.
                    if len(parts) == 7 and parts[0].strip().lstrip('-').isdigit():
                        time_sec  = float(parts[0]) / 1000.0
                        volts_val = float(parts[1])
                        curr_val  = float(parts[2])
                        power_val = float(parts[3])
                        roll_val  = float(parts[4])
                        pitch_val = float(parts[5])
                        yaw_val   = float(parts[6])

                        times.append(time_sec)
                        voltages.append(volts_val)
                        currents.append(curr_val)
                        powers.append(power_val)
                        rolls.append(roll_val)
                        pitches.append(pitch_val)
                        yaws.append(yaw_val)

                        if len(times) % 20 == 0:
                            print(f"  {time_sec:.1f}s | "
                                  f"{volts_val:.3f} V | {curr_val:.1f} mA | "
                                  f"{power_val:.2f} mW | roll={roll_val:.1f}°")

                except ValueError:
                    pass  # malformed line – skip

except (KeyboardInterrupt, SystemExit):
    print(f"\nStopped.  {len(times)} samples collected.")
except serial.SerialException as e:
    print(f"\nSerial error: {e}")


# ── SAVE CSV ───────────────────────────────────────────────────────────────────
if times:
    start_time = times[0]
    times    = np.array([t - start_time for t in times])
    voltages = np.array(voltages)
    currents = np.array(currents)
    powers   = np.array(powers)
    rolls    = np.array(rolls)
    pitches  = np.array(pitches)
    yaws     = np.array(yaws)

    csv_path = os.path.expanduser("~/Downloads/wave_energy_data.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Time_s", "Voltage_V", "Current_mA", "Power_mW",
                         "Roll_deg", "Pitch_deg", "Yaw_deg"])
        for row in zip(times, voltages, currents, powers, rolls, pitches, yaws):
            writer.writerow(row)
    print(f"CSV saved → {csv_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS & PLOTTING
# ══════════════════════════════════════════════════════════════════════════════
if len(times) == 0:
    print("No data collected.  Exiting.")
else:
    # ── Active window ──────────────────────────────────────────────────────────
    active_mask    = powers > NOISE_THRESHOLD
    active_indices = np.where(active_mask)[0]

    # ── Colour palette ─────────────────────────────────────────────────────────
    C_VOLT  = '#2a9d8f'   # teal
    C_CURR  = '#e9c46a'   # amber
    C_PWR   = '#e63946'   # red
    C_ROLL  = '#6a4c93'   # purple
    C_AVG   = '#1d3557'   # navy
    C_SPAN  = '#f4a261'   # orange (active window highlight)

    # ══════════════════════════════════════════════════════════════════════════
    #  FIGURE 1 – Full electrical profile  +  roll overlay on the power panel
    # ══════════════════════════════════════════════════════════════════════════
    fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(11, 10))
    fig1.suptitle("Wave Energy Converter – Full Electrical Profile  (roll overlaid)",
                  fontsize=16, fontweight='bold')

    # Panel 1 – Voltage
    ax1.plot(times, voltages, color=C_VOLT, linewidth=1.5, label='Voltage (V)')
    ax1.fill_between(times, voltages, color=C_VOLT, alpha=0.2)
    ax1.set_ylabel("Voltage (V)", fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right')

    # Panel 2 – Current
    ax2.plot(times, currents, color=C_CURR, linewidth=1.5, label='Current (mA)')
    ax2.fill_between(times, currents, color=C_CURR, alpha=0.2)
    ax2.set_ylabel("Current (mA)", fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right')

    # Panel 3 – Power (primary) + Roll (twin y-axis)
    ax3.plot(times, powers, color=C_PWR, linewidth=1.5, label='Power (mW)')
    ax3.fill_between(times, powers, color=C_PWR, alpha=0.2)
    ax3.set_ylabel("Power (mW)", fontsize=11)
    ax3.set_xlabel("Elapsed Time (s)", fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.5)

    ax3r = ax3.twinx()
    ax3r.plot(times, rolls, color=C_ROLL, linewidth=1.2, linestyle='--',
              alpha=0.85, label='Roll (°)')
    ax3r.set_ylabel("Roll Angle (°)", fontsize=11, color=C_ROLL)
    ax3r.tick_params(axis='y', labelcolor=C_ROLL)

    # ── Active-window analysis ─────────────────────────────────────────────────
    if len(active_indices) > 0:
        start_idx = active_indices[0]
        end_idx   = active_indices[-1]

        active_powers = powers[start_idx : end_idx + 1]
        active_times  = times [start_idx : end_idx + 1]
        active_rolls  = rolls [start_idx : end_idx + 1]
        active_duration = float(times[end_idx] - times[start_idx])
        n = len(active_powers)

        # Basic statistics
        active_average = float(np.mean(active_powers))
        std_dev   = float(np.std(active_powers, ddof=1)) if n > 1 else 0.0
        std_error = std_dev / math.sqrt(n)

        # Roll statistics in active window
        roll_range   = float(active_rolls.max() - active_rolls.min())
        roll_rms     = float(np.sqrt(np.mean(active_rolls**2)))
        roll_mean    = float(np.mean(active_rolls))

        # Per-cycle statistics
        cycles, current_cycle, in_cycle = [], [], False
        for p in powers:
            if p > NOISE_THRESHOLD:
                current_cycle.append(p); in_cycle = True
            elif in_cycle:
                if len(current_cycle) > 2: cycles.append(current_cycle)
                current_cycle = []; in_cycle = False
        if in_cycle and len(current_cycle) > 2:
            cycles.append(current_cycle)

        cycle_means, cycle_ses = [], []
        for c in cycles:
            nc = len(c)
            m  = float(np.mean(c))
            se = float(np.std(c, ddof=1) / math.sqrt(nc)) if nc > 1 else 0.0
            cycle_means.append(m); cycle_ses.append(se)

        # Peak / frequency detection
        prominence_threshold = 0.35 * active_powers.max()
        peaks, _ = find_peaks(active_powers,
                               prominence=prominence_threshold,
                               distance=20)

        n_peaks         = len(peaks)
        pulse_frequency = n_peaks / active_duration if active_duration > 0 else 0.0
        wave_frequency  = pulse_frequency / 2.0

        peak_times  = active_times[peaks]
        peak_powers = active_powers[peaks]

        # ── Print results ──────────────────────────────────────────────────────
        print("\n" + "─" * 50)
        print("RESULTS – Active Power & Frequency")
        print(f"  Active Duration:      {active_duration:.2f} s")
        print(f"  Average Active Power: {active_average:.3f} mW")
        print(f"  Std Dev of Power:     {std_dev:.3f} mW")
        print(f"  Std Error of Power:   {std_error:.4f} mW  (n={n})")
        print(f"  Peaks Detected:       {n_peaks} pulses")
        print(f"  Pulse Frequency:      {pulse_frequency:.3f} Hz")
        print(f"  Est. Wave Frequency:  {wave_frequency:.3f} Hz")
        if cycle_means:
            print(f"  Avg Cycle Power:      {np.mean(cycle_means):.3f} mW")
            print(f"  Avg Cycle SE:         {np.mean(cycle_ses):.4f} mW")
        print("─" * 50)
        print("ROLL ANGLE (active window)")
        print(f"  Mean Roll:            {roll_mean:.2f}°")
        print(f"  Peak-to-Peak Range:   {roll_range:.2f}°")
        print(f"  RMS Roll:             {roll_rms:.2f}°")
        print("─" * 50)

        # ── Annotate Figure 1 ──────────────────────────────────────────────────
        ax3.axhline(y=active_average, color=C_AVG, linestyle='--', linewidth=2,
                    label=f'Active Avg ({active_average:.2f} ± {std_error:.3f} mW SE)')
        ax3.plot(peak_times, peak_powers, "v", color=C_AVG, ms=8, zorder=5,
                 label=f'{n_peaks} pulses | f_wave ≈ {wave_frequency:.3f} Hz')

        for ax in [ax1, ax2, ax3]:
            ax.axvspan(times[start_idx], times[end_idx],
                       color=C_SPAN, alpha=0.15, label='Active window')

        # Combined legend for twin-axis panel
        lines3,  labs3  = ax3.get_legend_handles_labels()
        lines3r, labs3r = ax3r.get_legend_handles_labels()
        ax3.legend(lines3 + lines3r, labs3 + labs3r, loc='upper right', fontsize=8)

        # ══════════════════════════════════════════════════════════════════════
        #  FIGURE 2 – Isolated first-pulse zoom  +  roll overlay
        # ══════════════════════════════════════════════════════════════════════
        phase_start_idx = int(active_indices[0])
        phase_end_idx   = phase_start_idx
        for i in range(phase_start_idx, len(powers)):
            if powers[i] < NOISE_THRESHOLD:
                phase_end_idx = i
                break
        else:
            phase_end_idx = len(powers) - 1

        pad     = (times[phase_end_idx] - times[phase_start_idx]) * 0.1
        z_start = times[phase_start_idx] - pad
        z_end   = times[phase_end_idx]   + pad
        z_mask  = (times >= z_start) & (times <= z_end)

        z_t = times[z_mask]
        z_v = voltages[z_mask]
        z_p = powers[z_mask]
        z_r = rolls[z_mask]

        fig2, axes2 = plt.subplots(3, 1, sharex=True, figsize=(11, 9))
        fig2.suptitle("Wave Energy Converter – Isolated Phase Analysis  (roll overlaid)",
                      fontsize=14, fontweight='bold')
        zax_v, zax_p, zax_r = axes2

        # Voltage
        zax_v.plot(z_t, z_v, color=C_VOLT, linewidth=2, label='Voltage (V)')
        zax_v.fill_between(z_t, z_v, color=C_VOLT, alpha=0.2)
        zax_v.set_ylabel("Voltage (V)")
        zax_v.grid(True, alpha=0.3)
        zax_v.legend(loc='upper right')

        # Power
        zax_p.plot(z_t, z_p, color=C_PWR, linewidth=2, label='Power (mW)')
        zax_p.fill_between(z_t, z_p, color=C_PWR, alpha=0.2)
        zax_p.set_ylabel("Power (mW)")
        zax_p.grid(True, alpha=0.3)

        zoom_peak_mask = (peak_times >= z_start) & (peak_times <= z_end)
        if zoom_peak_mask.any():
            zax_p.plot(peak_times[zoom_peak_mask], peak_powers[zoom_peak_mask],
                       "v", color=C_AVG, ms=10, zorder=5, label='Detected peaks')
        zax_p.legend(loc='upper right')

        # Roll angle (dedicated panel so it's easy to read)
        zax_r.plot(z_t, z_r, color=C_ROLL, linewidth=2, label='Roll (°)')
        zax_r.axhline(0, color='grey', linewidth=0.8, linestyle=':')
        zax_r.fill_between(z_t, z_r, 0, where=(z_r >= 0),
                           color=C_ROLL, alpha=0.15, label='Positive roll')
        zax_r.fill_between(z_t, z_r, 0, where=(z_r <  0),
                           color=C_CURR, alpha=0.15, label='Negative roll')
        zax_r.set_ylabel("Roll Angle (°)")
        zax_r.set_xlabel("Time (s)")
        zax_r.grid(True, alpha=0.3)
        zax_r.legend(loc='upper right')

        fig2.tight_layout()
        fig2.subplots_adjust(top=0.92)

        # ══════════════════════════════════════════════════════════════════════
        #  FIGURE 3 – Roll vs Power scatter  (phase-relationship diagnostic)
        # ══════════════════════════════════════════════════════════════════════
        fig3, ax_sc = plt.subplots(figsize=(7, 6))
        fig3.suptitle("Roll Angle vs Power  (active window)",
                      fontsize=14, fontweight='bold')

        sc = ax_sc.scatter(active_rolls, active_powers,
                           c=active_times, cmap='plasma',
                           s=8, alpha=0.6, zorder=3)
        cbar = fig3.colorbar(sc, ax=ax_sc, label='Time (s)')
        ax_sc.set_xlabel("Roll Angle (°)", fontsize=12)
        ax_sc.set_ylabel("Power (mW)",     fontsize=12)
        ax_sc.grid(True, linestyle='--', alpha=0.5)

        # Linear trend line
        if len(active_rolls) > 2:
            coeffs = np.polyfit(active_rolls, active_powers, 1)
            x_fit  = np.linspace(active_rolls.min(), active_rolls.max(), 200)
            ax_sc.plot(x_fit, np.polyval(coeffs, x_fit),
                       color=C_AVG, linewidth=2, linestyle='--',
                       label=f'Linear fit  slope={coeffs[0]:.3f} mW/°')
            ax_sc.legend()

        fig3.tight_layout()

    else:
        # No active window found – still show roll overlay
        lines3,  labs3  = ax3.get_legend_handles_labels()
        lines3r, labs3r = ax3r.get_legend_handles_labels()
        ax3.legend(lines3 + lines3r, labs3 + labs3r, loc='upper right')
        print("\nNo power spikes above threshold – generator remained idle.")

    fig1.tight_layout()
    fig1.subplots_adjust(top=0.92)
    plt.show()