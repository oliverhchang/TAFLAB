import signal
import sys
import serial
import matplotlib
matplotlib.use('MacOSX')  # macOS native backend — run from Terminal, not PyCharm
import matplotlib.pyplot as plt
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

# ── CONFIGURATION ──────────────────────────────────────────────────────────────
SERIAL_PORT     = '/dev/cu.usbserial-0001'
BAUD_RATE       = 115200
NOISE_THRESHOLD = 0.5    # mW

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
            line = ser.readline().decode('utf-8', errors='ignore').strip()

            if not line:
                print("  [waiting for data…]")
                continue

            print(f"RAW: {line}")

            if line == "START":
                continue

            try:
                parts = line.split(',')
                # Expected 7 columns: Time_ms, Voltage_V, Current_mA, Power_mW,
                #                     Roll_deg, Pitch_deg, Yaw_deg
                # parts[0] must be a plain integer (ms timestamp).
                # Corrupt/partial lines at startup are rejected by the isdigit check.
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
    active_mask    = powers > NOISE_THRESHOLD
    active_indices = np.where(active_mask)[0]

    C_VOLT = '#2a9d8f'
    C_CURR = '#e9c46a'
    C_PWR  = '#e63946'
    C_ROLL = '#6a4c93'
    C_AVG  = '#1d3557'
    C_SPAN = '#f4a261'

    # ══════════════════════════════════════════════════════════════════════════
    #  FIGURE 1 – Full electrical profile (3 stacked panels) + roll overlay
    # ══════════════════════════════════════════════════════════════════════════
    fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10, 9))
    fig1.suptitle("Wave Energy Converter: Full Electrical Profile",
                  fontsize=16, fontweight='bold')

    ax1.plot(times, voltages, color=C_VOLT, linewidth=1.5, label='Voltage (V)')
    ax1.fill_between(times, voltages, color=C_VOLT, alpha=0.2)
    ax1.set_ylabel("Voltage (V)", fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper right')

    ax2.plot(times, currents, color=C_CURR, linewidth=1.5, label='Current (mA)')
    ax2.fill_between(times, currents, color=C_CURR, alpha=0.2)
    ax2.set_ylabel("Current (mA)", fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='upper right')

    ax3.plot(times, powers, color=C_PWR, linewidth=1.5, label='Power (mW)')
    ax3.fill_between(times, powers, color=C_PWR, alpha=0.2)
    ax3.set_ylabel("Power (mW)", fontsize=11)
    ax3.set_xlabel("Elapsed Time (Seconds)", fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.7)

    # Roll overlaid on power panel (twin y-axis)
    ax3r = ax3.twinx()
    ax3r.plot(times, rolls, color=C_ROLL, linewidth=1.2, linestyle='--',
              alpha=0.85, label='Roll (°)')
    ax3r.set_ylabel("Roll Angle (°)", fontsize=11, color=C_ROLL)
    ax3r.tick_params(axis='y', labelcolor=C_ROLL)

    if len(active_indices) > 0:
        start_idx = active_indices[0]
        end_idx   = active_indices[-1]

        active_powers   = powers[start_idx : end_idx + 1]
        active_times    = times [start_idx : end_idx + 1]
        active_rolls    = rolls [start_idx : end_idx + 1]
        active_duration = float(times[end_idx] - times[start_idx])
        n = len(active_powers)

        active_average = float(np.mean(active_powers))
        std_dev        = float(np.std(active_powers, ddof=1)) if n > 1 else 0.0
        std_error      = std_dev / math.sqrt(n)

        roll_range = float(active_rolls.max() - active_rolls.min())
        roll_rms   = float(np.sqrt(np.mean(active_rolls**2)))
        roll_mean  = float(np.mean(active_rolls))

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

        prominence_threshold = 0.35 * active_powers.max()
        peaks, _ = find_peaks(active_powers,
                               prominence=prominence_threshold,
                               distance=20)

        n_peaks         = len(peaks)
        pulse_frequency = n_peaks / active_duration if active_duration > 0 else 0.0
        wave_frequency  = pulse_frequency / 2.0

        peak_times  = active_times[peaks]
        peak_powers = active_powers[peaks]

        print("-" * 45)
        print("RESULTS: Active Power & Frequency")
        print(f"  Active Duration:      {active_duration:.2f} s")
        print(f"  Average Active Power: {active_average:.3f} mW")
        print(f"  Std Dev of Power:     {std_dev:.3f} mW")
        print(f"  Std Error of Power:   {std_error:.4f} mW  (n={n})")
        print(f"  Peaks Detected:       {n_peaks} pulses")
        print(f"  Pulse Frequency:      {pulse_frequency:.3f} Hz  (pulses/sec)")
        print(f"  Est. Wave Frequency:  {wave_frequency:.3f} Hz  (full cycles)")
        if cycle_means:
            print(f"  Avg Cycle Power:      {np.mean(cycle_means):.3f} mW")
            print(f"  Avg Cycle SE:         {np.mean(cycle_ses):.4f} mW")
        print("-" * 45)
        print("ROLL ANGLE (active window)")
        print(f"  Mean Roll:            {roll_mean:.2f}°")
        print(f"  Peak-to-Peak Range:   {roll_range:.2f}°")
        print(f"  RMS Roll:             {roll_rms:.2f}°")
        print("-" * 45)

        ax3.axhline(y=active_average, color=C_AVG, linestyle='--', linewidth=2,
                    label=f'Active Avg ({active_average:.2f} ± {std_error:.3f} mW SE)')
        ax3.plot(peak_times, peak_powers, "v", color=C_AVG, ms=8, zorder=5,
                 label=f'{n_peaks} pulses | f_wave ≈ {wave_frequency:.3f} Hz')

        for ax in [ax1, ax2, ax3]:
            ax.axvspan(times[start_idx], times[end_idx],
                       color=C_SPAN, alpha=0.15, label='Active window')

        lines3,  labs3  = ax3.get_legend_handles_labels()
        lines3r, labs3r = ax3r.get_legend_handles_labels()
        ax3.legend(lines3 + lines3r, labs3 + labs3r, loc='upper right', fontsize=8)

        # ══════════════════════════════════════════════════════════════════════
        #  FIGURE 2 – Isolated first-pulse zoom + roll on twin axis
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

        fig2, (zax1, zax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
        fig2.suptitle("Wave Energy Converter: Isolated Phase Analysis",
                      fontsize=14, fontweight='bold')

        zax1.plot(z_t, z_v, color=C_VOLT, linewidth=2, label="Voltage (V)")
        zax1.set_ylabel("Voltage (V)")
        zax1.grid(True, alpha=0.3)
        zax1.legend(loc='upper right')

        zax2.plot(z_t, z_p, color=C_PWR, linewidth=2, label="Power (mW)")
        zax2.fill_between(z_t, z_p, color=C_PWR, alpha=0.2)
        zax2.set_ylabel("Power (mW)")
        zax2.set_xlabel("Time (Seconds)")
        zax2.grid(True, alpha=0.3)

        # Roll on twin axis
        zax2r = zax2.twinx()
        zax2r.plot(z_t, z_r, color=C_ROLL, linewidth=1.5, linestyle='--',
                   alpha=0.85, label='Roll (°)')
        zax2r.set_ylabel("Roll Angle (°)", color=C_ROLL)
        zax2r.tick_params(axis='y', labelcolor=C_ROLL)

        zoom_peak_mask = (peak_times >= z_start) & (peak_times <= z_end)
        if zoom_peak_mask.any():
            zax2.plot(peak_times[zoom_peak_mask], peak_powers[zoom_peak_mask],
                      "v", color=C_AVG, ms=10, zorder=5, label="Detected peaks")

        lines2,  labs2  = zax2.get_legend_handles_labels()
        lines2r, labs2r = zax2r.get_legend_handles_labels()
        zax2.legend(lines2 + lines2r, labs2 + labs2r, loc='upper right', fontsize=8)

        fig2.tight_layout()
        fig2.subplots_adjust(top=0.92)

        # ══════════════════════════════════════════════════════════════════════
        #  FIGURE 3 – Roll vs Power scatter (phase diagnostic)
        # ══════════════════════════════════════════════════════════════════════
        fig3, ax_sc = plt.subplots(figsize=(7, 6))
        fig3.suptitle("Roll Angle vs Power  (active window)",
                      fontsize=14, fontweight='bold')

        sc = ax_sc.scatter(active_rolls, active_powers,
                           c=active_times, cmap='plasma',
                           s=8, alpha=0.6, zorder=3)
        fig3.colorbar(sc, ax=ax_sc, label='Time (s)')
        ax_sc.set_xlabel("Roll Angle (°)", fontsize=12)
        ax_sc.set_ylabel("Power (mW)",     fontsize=12)
        ax_sc.grid(True, linestyle='--', alpha=0.5)

        if len(active_rolls) > 2:
            coeffs = np.polyfit(active_rolls, active_powers, 1)
            x_fit  = np.linspace(active_rolls.min(), active_rolls.max(), 200)
            ax_sc.plot(x_fit, np.polyval(coeffs, x_fit),
                       color=C_AVG, linewidth=2, linestyle='--',
                       label=f'Linear fit  slope={coeffs[0]:.3f} mW/°')
            ax_sc.legend()

        fig3.tight_layout()

        # ══════════════════════════════════════════════════════════════════════
        #  PER-CYCLE DATA: segment by positive-going roll zero-crossings
        #
        #  For each cycle between consecutive zero-crossings compute:
        #    • max_angle  – the highest roll angle reached in that cycle (°)
        #    • avg_power  – mean power over that cycle window (mW)
        #    • total_energy – sum(power) * dt  i.e. mW·s = mJ for that cycle
        # ══════════════════════════════════════════════════════════════════════

        # Positive-going zero crossings (roll: negative → positive)
        zc_indices = []
        for i in range(1, len(rolls)):
            if rolls[i - 1] < 0 and rolls[i] >= 0:
                zc_indices.append(i)

        cycle_max_angles  = []   # max |roll| per cycle
        cycle_avg_powers  = []   # mean power per cycle
        cycle_energies    = []   # total energy (mJ) per cycle
        cycle_mid_times   = []   # midpoint time for colour-coding

        for ci in range(len(zc_indices) - 1):
            i0 = zc_indices[ci]
            i1 = zc_indices[ci + 1]
            if i1 - i0 < 3:
                continue
            seg_roll  = rolls [i0:i1]
            seg_power = powers[i0:i1]
            seg_times = times [i0:i1]

            dt = float(np.mean(np.diff(seg_times))) if len(seg_times) > 1 else 0.01

            cycle_max_angles.append(float(np.max(np.abs(seg_roll))))
            cycle_avg_powers.append(float(np.mean(seg_power)))
            cycle_energies.append(float(np.sum(seg_power) * dt))   # mJ
            cycle_mid_times.append(float((seg_times[0] + seg_times[-1]) / 2))

        cycle_max_angles = np.array(cycle_max_angles)
        cycle_avg_powers = np.array(cycle_avg_powers)
        cycle_energies   = np.array(cycle_energies)
        cycle_mid_times  = np.array(cycle_mid_times)

        if len(cycle_max_angles) > 0:

            # ── Print per-cycle table ──────────────────────────────────────────
            print("\nPER-CYCLE SUMMARY")
            print(f"  {'Cycle':<8}{'Time (s)':<12}{'Max Angle (°)':<18}"
                  f"{'Avg Power (mW)':<18}{'Energy (mJ)'}")
            print("  " + "-" * 65)
            for ci in range(len(cycle_max_angles)):
                print(f"  {ci+1:<8}{cycle_mid_times[ci]:<12.2f}"
                      f"{cycle_max_angles[ci]:<18.2f}"
                      f"{cycle_avg_powers[ci]:<18.3f}"
                      f"{cycle_energies[ci]:.4f}")
            print("-" * 45)

            # ── Quadratic fit helper ───────────────────────────────────────────
            def quad_fit(x, y):
                if len(x) < 4:
                    return None, None
                try:
                    c = np.polyfit(x, y, 2)
                    x_fit = np.linspace(x.min(), x.max(), 300)
                    return c, x_fit
                except np.linalg.LinAlgError:
                    return None, None

            # ══════════════════════════════════════════════════════════════════
            #  FIGURE 4 – Average power per cycle vs max roll angle per cycle
            #
            #  Each dot = one wave cycle.
            #  X = peak angle reached in that cycle (how big the wave was).
            #  Y = average power generated during that cycle.
            #  Colour = time, so you can see if the relationship drifts.
            # ══════════════════════════════════════════════════════════════════
            fig4, ax4 = plt.subplots(figsize=(8, 6))
            fig4.suptitle("Average Power per Cycle vs Max Roll Angle per Cycle",
                          fontsize=14, fontweight='bold')

            sc4 = ax4.scatter(cycle_max_angles, cycle_avg_powers,
                              c=cycle_mid_times, cmap='plasma',
                              s=60, alpha=0.85, zorder=4, edgecolors='white',
                              linewidths=0.5, label='Individual cycles')
            fig4.colorbar(sc4, ax=ax4, label='Time into test (s)')

            # Annotate cycle numbers
            for ci, (xa, ya) in enumerate(zip(cycle_max_angles, cycle_avg_powers)):
                ax4.annotate(str(ci + 1), (xa, ya),
                             textcoords='offset points', xytext=(5, 4),
                             fontsize=7, color=C_AVG, alpha=0.8)

            # Quadratic trend
            c4, x4 = quad_fit(cycle_max_angles, cycle_avg_powers)
            if c4 is not None:
                ax4.plot(x4, np.polyval(c4, x4),
                         color=C_ROLL, linewidth=2.5, linestyle='--', zorder=5,
                         label=f'Quadratic fit  '
                               f'({c4[0]:.4f}θ²  {c4[1]:+.3f}θ  {c4[2]:+.2f})')

            ax4.set_xlabel("Max Roll Angle Reached in Cycle (°)", fontsize=12)
            ax4.set_ylabel("Average Power During Cycle (mW)", fontsize=12)
            ax4.set_xlim(left=0)
            ax4.set_ylim(bottom=0)
            ax4.grid(True, linestyle='--', alpha=0.5)
            ax4.legend(fontsize=9)
            fig4.tight_layout()

            # ══════════════════════════════════════════════════════════════════
            #  FIGURE 5 – Peak roll amplitude vs total energy per cycle
            #
            #  X = max angle per cycle  (same as Fig 4 x-axis)
            #  Y = total energy (mJ) generated in that cycle  = ∫P dt
            #
            #  This is the "how much energy did each wave actually deliver?"
            #  view across the whole test — a bigger wave should deliver more
            #  total energy even if average power is similar.
            # ══════════════════════════════════════════════════════════════════
            fig5, ax5 = plt.subplots(figsize=(8, 6))
            fig5.suptitle("Peak Roll Amplitude vs Energy Generated per Cycle\n"
                          "(whole-test relationship)",
                          fontsize=14, fontweight='bold')

            sc5 = ax5.scatter(cycle_max_angles, cycle_energies,
                              c=cycle_mid_times, cmap='viridis',
                              s=60, alpha=0.85, zorder=4, edgecolors='white',
                              linewidths=0.5, label='Individual cycles')
            fig5.colorbar(sc5, ax=ax5, label='Time into test (s)')

            for ci, (xa, ya) in enumerate(zip(cycle_max_angles, cycle_energies)):
                ax5.annotate(str(ci + 1), (xa, ya),
                             textcoords='offset points', xytext=(5, 4),
                             fontsize=7, color=C_AVG, alpha=0.8)

            # Quadratic trend
            c5, x5 = quad_fit(cycle_max_angles, cycle_energies)
            if c5 is not None:
                ax5.plot(x5, np.polyval(c5, x5),
                         color=C_PWR, linewidth=2.5, linestyle='--', zorder=5,
                         label=f'Quadratic fit  '
                               f'({c5[0]:.5f}θ²  {c5[1]:+.4f}θ  {c5[2]:+.3f})')

            # Annotate total energy over whole test
            total_energy_mJ = float(cycle_energies.sum())
            ax5.text(0.97, 0.05,
                     f'Total energy (all cycles): {total_energy_mJ:.3f} mJ',
                     transform=ax5.transAxes, ha='right', va='bottom',
                     fontsize=10, color=C_AVG,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                               edgecolor=C_AVG, alpha=0.8))

            ax5.set_xlabel("Peak Roll Angle in Cycle (°)", fontsize=12)
            ax5.set_ylabel("Energy Generated in Cycle (mJ)", fontsize=12)
            ax5.set_xlim(left=0)
            ax5.set_ylim(bottom=0)
            ax5.grid(True, linestyle='--', alpha=0.5)
            ax5.legend(fontsize=9)
            fig5.tight_layout()

            print(f"\n  Total energy across all cycles: {total_energy_mJ:.4f} mJ")

        else:
            print("\nNot enough zero-crossings to compute per-cycle stats.")
            print("  (Need at least 2 full roll cycles — try a longer recording.)")

    else:
        lines3,  labs3  = ax3.get_legend_handles_labels()
        lines3r, labs3r = ax3r.get_legend_handles_labels()
        ax3.legend(lines3 + lines3r, labs3 + labs3r, loc='upper right')
        print("\nNo power spikes detected above threshold. Generator remained idle.")

    fig1.tight_layout()
    fig1.subplots_adjust(top=0.92)
    plt.show()