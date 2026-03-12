"""
power_electronics.py
--------------------
Stage-by-stage power conditioning model for the WEC electrical chain.

Pipeline
--------
    Coil array (AC)
        │
        ▼
    Bridge rectifier        — full-wave, 4 diodes, drops 2*V_f per cycle
        │
        ▼
    Smoothing capacitor     — reduces ripple, models charge/discharge
        │
        ▼
    Boost converter         — steps up low-voltage (~0.1–2V) to usable rail
        │
        ▼
    INA219 sense + load     — current/power monitoring, then output

Each stage is modeled with realistic efficiency curves rather than
fixed constants, so the model degrades gracefully at low input voltages
(which is the hard operating regime for ocean WECs).

Key design question answered here
----------------------------------
At what EMF amplitude does the system cross the 1W output threshold?
And what does the power conditioning efficiency look like as a function
of input voltage — this determines whether you need more EMF or more
current from the coil design.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict
from config import WECConfig


# ---------------------------------------------------------------------------
# Configuration for power electronics
# ---------------------------------------------------------------------------
@dataclass
class PowerElectronicsConfig:
    """All power electronics parameters."""

    # Bridge rectifier (Schottky diodes recommended for low-voltage)
    # Schottky: V_f ~ 0.3V; standard Si: V_f ~ 0.7V
    diode_forward_voltage: float = 0.3      # V  (Schottky)
    n_diodes_series: int = 2                # per half-cycle in full bridge

    # Smoothing capacitor
    C_smooth: float = 470e-6               # F  (470 µF electrolytic)
    V_cap_initial: float = 0.0             # V  initial cap voltage

    # Boost converter (e.g. MT3608 or TPS61200 for low-input)
    V_out_target: float = 5.0              # V  target output rail
    boost_eta_max: float = 0.88            # peak efficiency
    boost_V_in_min: float = 0.3            # V  minimum input to start up
    boost_quiescent_mA: float = 1.0        # mA quiescent current draw

    # INA219 current/power monitor
    ina219_shunt_ohm: float = 0.1          # Ω  shunt resistor
    ina219_quiescent_mA: float = 0.5       # mA

    # Wiring / connector resistance
    R_wiring: float = 0.5                  # Ω  total wiring resistance

    # Coil wiring configuration
    coils_in_series: bool = True           # True=series(V boost), False=parallel(I boost)


# ---------------------------------------------------------------------------
# Output containers
# ---------------------------------------------------------------------------
@dataclass
class StageResult:
    """Power at each stage of the conditioning chain."""
    t: np.ndarray

    # Stage inputs / outputs
    V_ac_rms: float             # RMS AC voltage from coil array (V)
    I_ac_rms: float             # RMS AC current (A)
    P_ac: float                 # AC power into rectifier (W)

    V_rect: np.ndarray          # rectified voltage time series (V)
    V_rect_rms: float           # RMS rectified voltage (V)
    P_rect: float               # power after rectification (W)
    eta_rect: float             # rectifier efficiency

    V_smooth: np.ndarray        # smoothed (capacitor output) voltage (V)
    V_smooth_rms: float         # RMS smoothed voltage (V)
    P_smooth: float             # power after smoothing (W)
    eta_smooth: float           # smoothing stage efficiency

    V_boost_out: float          # boost converter output voltage (V)
    P_boost_out: float          # power after boost converter (W)
    eta_boost: float            # boost converter efficiency

    P_out_net: float            # net output power after quiescent loads (W)
    eta_total: float            # end-to-end efficiency

    # Loss breakdown
    P_loss_rectifier: float
    P_loss_capacitor: float
    P_loss_boost: float
    P_loss_quiescent: float
    P_loss_wiring: float


# ---------------------------------------------------------------------------
# Rectifier model
# ---------------------------------------------------------------------------
def rectifier_model(
    EMF: np.ndarray,
    R_coil: float,
    R_load: float,
    R_wiring: float,
    V_f: float,
    n_diodes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full-wave bridge rectifier with realistic diode drops.

    For each half-cycle, current flows through 2 diodes in series.
    The output voltage is:
        V_rect = |EMF * R_load / (R_coil + R_wiring + R_load)| - n_diodes * V_f

    Below the diode threshold the output is zero (no conduction).

    Returns
    -------
    V_rect, I_rect, P_rect : time series arrays
    """
    R_total = R_coil + R_wiring + R_load

    # Voltage divider to load
    V_load_ac = EMF * R_load / R_total

    # Rectify and subtract diode drops
    V_threshold = n_diodes * V_f
    V_rect = np.maximum(np.abs(V_load_ac) - V_threshold, 0.0)

    # Current through load after rectification
    I_rect = V_rect / R_load
    P_rect = V_rect * I_rect

    return V_rect, I_rect, P_rect


# ---------------------------------------------------------------------------
# Smoothing capacitor model
# ---------------------------------------------------------------------------
def capacitor_smooth(
    V_rect: np.ndarray,
    I_rect: np.ndarray,
    t: np.ndarray,
    C: float,
    R_load: float,
    V_cap_0: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate capacitor charge/discharge dynamics.

    The capacitor charges when V_rect > V_cap and discharges through R_load
    when the rectified voltage dips.

        dV_cap/dt = (I_charge - I_discharge) / C
        I_charge  = (V_rect - V_cap) / R_source   when V_rect > V_cap
        I_discharge = V_cap / R_load               always

    This is solved with a simple forward Euler step — fast and accurate
    enough for the slow dynamics of ocean wave frequencies.

    Returns
    -------
    V_smooth : smoothed voltage time series (V)
    I_out    : output current time series (A)
    """
    n       = len(t)
    V_cap   = np.zeros(n)
    V_cap[0] = V_cap_0
    R_src    = 1.0   # source impedance for charging (small, 1Ω default)

    for i in range(1, n):
        dt = t[i] - t[i-1]
        V_c = V_cap[i-1]
        V_r = V_rect[i]

        # Charging current (only when V_rect > V_cap)
        I_charge = max((V_r - V_c) / R_src, 0.0)

        # Discharge current through load
        I_discharge = V_c / R_load

        dVdt = (I_charge - I_discharge) / C
        V_cap[i] = max(V_c + dVdt * dt, 0.0)

    I_out = V_cap / R_load
    return V_cap, I_out


# ---------------------------------------------------------------------------
# Boost converter efficiency curve
# ---------------------------------------------------------------------------
def boost_efficiency(
    V_in: float,
    I_in: float,
    V_out: float,
    eta_max: float = 0.88,
    V_in_min: float = 0.3,
) -> float:
    """
    Realistic boost converter efficiency as a function of input conditions.

    Based on typical efficiency curves for ICs like TPS61200 / MT3608:
    - Efficiency peaks near rated current
    - Falls off sharply at very low input voltages (< 0.5V)
    - Falls off at very high step-up ratios

    Model:
        eta = eta_max * f_voltage(V_in) * f_current(I_in)

    where f_voltage accounts for startup threshold and
          f_current accounts for light-load efficiency loss.
    """
    if V_in < V_in_min:
        return 0.0

    # Voltage factor: ramps up from 0 at V_in_min to 1 at ~2*V_in_min
    f_voltage = np.clip((V_in - V_in_min) / V_in_min, 0.0, 1.0)
    # Smooth sigmoid-like rolloff
    f_voltage = f_voltage**0.5

    # Conversion ratio penalty: efficiency drops at high step-up
    ratio = V_out / max(V_in, 1e-3)
    f_ratio = np.clip(1.0 - 0.02 * max(ratio - 5.0, 0.0), 0.3, 1.0)

    # Current / load factor: light-load penalty below ~10mA
    P_in = V_in * max(I_in, 1e-6)
    f_load = np.clip(P_in / 0.01, 0.0, 1.0)**0.3   # 10mW reference

    eta = eta_max * f_voltage * f_ratio * f_load
    return float(np.clip(eta, 0.0, eta_max))


# ---------------------------------------------------------------------------
# Full pipeline: EMF time series → net output power
# ---------------------------------------------------------------------------
def compute_power_chain(
    t: np.ndarray,
    EMF: np.ndarray,
    cfg: WECConfig,
    pe_cfg: Optional[PowerElectronicsConfig] = None,
    R_load: Optional[float] = None,
) -> StageResult:
    """
    Run the complete power conditioning chain and return per-stage results.

    Parameters
    ----------
    t      : time array (s)
    EMF    : open-circuit EMF from electromagnetics (V)
    cfg    : WECConfig
    pe_cfg : PowerElectronicsConfig (uses defaults if None)
    R_load : load resistance (Ω); defaults to impedance-matched value

    Returns
    -------
    StageResult with power at every stage and efficiency breakdown
    """
    if pe_cfg is None:
        pe_cfg = PowerElectronicsConfig()

    # Coil array resistance
    if pe_cfg.coils_in_series:
        R_coil = cfg.coil.coil_resistance * cfg.coil.n_coils
    else:
        R_coil = cfg.coil.coil_resistance / cfg.coil.n_coils

    if R_load is None:
        R_load = R_coil   # impedance matching

    # --- Stage 1: AC from coil ---
    R_total_ac = R_coil + pe_cfg.R_wiring + R_load
    I_ac = EMF / R_total_ac
    V_ac = I_ac * R_load
    V_ac_rms = float(np.sqrt(np.mean(V_ac**2)))
    I_ac_rms = float(np.sqrt(np.mean(I_ac**2)))
    P_ac = V_ac_rms * I_ac_rms

    # Wiring loss
    P_loss_wiring = float(np.mean(I_ac**2)) * pe_cfg.R_wiring

    # --- Stage 2: Bridge rectifier ---
    V_rect, I_rect, P_rect_series = rectifier_model(
        EMF, R_coil, R_load, pe_cfg.R_wiring,
        pe_cfg.diode_forward_voltage, pe_cfg.n_diodes_series,
    )
    P_rect_avg = float(np.mean(P_rect_series))
    V_rect_rms = float(np.sqrt(np.mean(V_rect**2)))
    P_loss_rect = max(P_ac - P_rect_avg, 0.0)
    eta_rect = P_rect_avg / P_ac if P_ac > 0 else 0.0

    # --- Stage 3: Smoothing capacitor ---
    V_smooth, I_smooth = capacitor_smooth(
        V_rect, I_rect, t, pe_cfg.C_smooth, R_load, pe_cfg.V_cap_initial
    )
    P_smooth_avg = float(np.mean(V_smooth * I_smooth))
    V_smooth_rms = float(np.sqrt(np.mean(V_smooth**2)))
    P_loss_cap   = max(P_rect_avg - P_smooth_avg, 0.0)
    eta_smooth   = P_smooth_avg / P_rect_avg if P_rect_avg > 0 else 0.0

    # --- Stage 4: Boost converter ---
    # Use RMS values as operating point for efficiency calculation
    V_in_rms = V_smooth_rms
    I_in_est = P_smooth_avg / max(V_in_rms, 1e-6)
    eta_boost = boost_efficiency(
        V_in_rms, I_in_est,
        pe_cfg.V_out_target,
        pe_cfg.boost_eta_max,
        pe_cfg.boost_V_in_min,
    )
    P_boost_out = P_smooth_avg * eta_boost
    V_boost_out = pe_cfg.V_out_target if V_in_rms >= pe_cfg.boost_V_in_min else 0.0
    P_loss_boost = P_smooth_avg - P_boost_out

    # --- Stage 5: Quiescent loads ---
    P_quiescent = (
        (pe_cfg.boost_quiescent_mA + pe_cfg.ina219_quiescent_mA) * 1e-3
        * pe_cfg.V_out_target
    )
    P_out_net = max(P_boost_out - P_quiescent, 0.0)
    P_loss_quiescent = P_quiescent

    # --- End-to-end efficiency ---
    eta_total = P_out_net / P_ac if P_ac > 0 else 0.0

    return StageResult(
        t=t,
        V_ac_rms=V_ac_rms, I_ac_rms=I_ac_rms, P_ac=P_ac,
        V_rect=V_rect, V_rect_rms=V_rect_rms, P_rect=P_rect_avg, eta_rect=eta_rect,
        V_smooth=V_smooth, V_smooth_rms=V_smooth_rms, P_smooth=P_smooth_avg, eta_smooth=eta_smooth,
        V_boost_out=V_boost_out, P_boost_out=P_boost_out, eta_boost=eta_boost,
        P_out_net=P_out_net, eta_total=eta_total,
        P_loss_rectifier=P_loss_rect,
        P_loss_capacitor=P_loss_cap,
        P_loss_boost=P_loss_boost,
        P_loss_quiescent=P_loss_quiescent,
        P_loss_wiring=P_loss_wiring,
    )


# ---------------------------------------------------------------------------
# Sweep: output power vs. input EMF amplitude
# ---------------------------------------------------------------------------
def emf_power_sweep(
    EMF_amplitudes: np.ndarray,
    cfg: WECConfig,
    pe_cfg: Optional[PowerElectronicsConfig] = None,
    f_wave: float = 0.125,
    t_end: float = 60.0,
    dt: float = 0.02,
) -> Dict[str, np.ndarray]:
    """
    Compute net output power for a range of sinusoidal EMF amplitudes.
    Useful for finding the minimum EMF needed to cross the 1W threshold.

    Returns dict with arrays keyed by metric name.
    """
    if pe_cfg is None:
        pe_cfg = PowerElectronicsConfig()

    t = np.arange(0, t_end, dt)
    omega = 2 * np.pi * f_wave

    P_ac_arr     = np.zeros(len(EMF_amplitudes))
    P_rect_arr   = np.zeros(len(EMF_amplitudes))
    P_smooth_arr = np.zeros(len(EMF_amplitudes))
    P_boost_arr  = np.zeros(len(EMF_amplitudes))
    P_net_arr    = np.zeros(len(EMF_amplitudes))
    eta_arr      = np.zeros(len(EMF_amplitudes))

    for i, amp in enumerate(EMF_amplitudes):
        EMF = amp * np.sin(omega * t)
        sr  = compute_power_chain(t, EMF, cfg, pe_cfg)
        P_ac_arr[i]     = sr.P_ac
        P_rect_arr[i]   = sr.P_rect
        P_smooth_arr[i] = sr.P_smooth
        P_boost_arr[i]  = sr.P_boost_out
        P_net_arr[i]    = sr.P_out_net
        eta_arr[i]      = sr.eta_total

    return {
        "EMF_amplitude": EMF_amplitudes,
        "P_ac":     P_ac_arr,
        "P_rect":   P_rect_arr,
        "P_smooth": P_smooth_arr,
        "P_boost":  P_boost_arr,
        "P_net":    P_net_arr,
        "eta":      eta_arr,
    }


# ---------------------------------------------------------------------------
# Series vs parallel coil wiring comparison
# ---------------------------------------------------------------------------
def wiring_comparison(
    t: np.ndarray,
    EMF_per_coil: np.ndarray,
    cfg: WECConfig,
    pe_cfg: Optional[PowerElectronicsConfig] = None,
) -> Dict[str, StageResult]:
    """
    Compare series wiring (voltage boost) vs parallel wiring (current boost).

    In series:   V_total = N * V_per_coil,  R_total = N * R_coil
    In parallel: V_total = V_per_coil,      R_total = R_coil / N

    The optimal choice depends on the load resistance and EMF level.

    Parameters
    ----------
    EMF_per_coil : EMF generated by a single coil (V)
    """
    if pe_cfg is None:
        pe_cfg = PowerElectronicsConfig()

    N = cfg.coil.n_coils
    R_single = cfg.coil.coil_resistance

    # Series configuration
    pe_series         = PowerElectronicsConfig(**pe_cfg.__dict__)
    pe_series.coils_in_series = True
    EMF_series        = EMF_per_coil * N
    sr_series         = compute_power_chain(t, EMF_series, cfg, pe_series)

    # Parallel configuration
    pe_parallel              = PowerElectronicsConfig(**pe_cfg.__dict__)
    pe_parallel.coils_in_series = False
    EMF_parallel             = EMF_per_coil   # same voltage, more current
    sr_parallel              = compute_power_chain(t, EMF_parallel, cfg, pe_parallel)

    return {"series": sr_series, "parallel": sr_parallel}


# ---------------------------------------------------------------------------
# Print stage report
# ---------------------------------------------------------------------------
def print_stage_report(sr: StageResult, label: str = ""):
    tag = f"  [{label}] " if label else "  "
    print(f"\n{'='*55}")
    print(f"  Power Electronics Stage Report  {label}")
    print(f"{'='*55}")
    print(f"{tag}AC input:       V_rms={sr.V_ac_rms:.3f}V  I_rms={sr.I_ac_rms*1000:.2f}mA  P={sr.P_ac*1000:.3f}mW")
    print(f"{tag}After rectifier:V_rms={sr.V_rect_rms:.3f}V  P={sr.P_rect*1000:.3f}mW  η={sr.eta_rect*100:.1f}%")
    print(f"{tag}After cap:      V_rms={sr.V_smooth_rms:.3f}V  P={sr.P_smooth*1000:.3f}mW  η={sr.eta_smooth*100:.1f}%")
    print(f"{tag}After boost:    V_out={sr.V_boost_out:.2f}V  P={sr.P_boost_out*1000:.3f}mW  η={sr.eta_boost*100:.1f}%")
    print(f"{tag}Net output:     P_net={sr.P_out_net*1000:.3f}mW")
    print(f"{tag}End-to-end η:   {sr.eta_total*100:.1f}%")
    print(f"\n  Loss breakdown:")
    print(f"    Wiring       : {sr.P_loss_wiring*1000:.3f} mW")
    print(f"    Rectifier    : {sr.P_loss_rectifier*1000:.3f} mW")
    print(f"    Capacitor    : {sr.P_loss_capacitor*1000:.3f} mW")
    print(f"    Boost conv.  : {sr.P_loss_boost*1000:.3f} mW")
    print(f"    Quiescent    : {sr.P_loss_quiescent*1000:.3f} mW")
    total_loss = (sr.P_loss_wiring + sr.P_loss_rectifier + sr.P_loss_capacitor
                  + sr.P_loss_boost + sr.P_loss_quiescent)
    print(f"    TOTAL loss   : {total_loss*1000:.3f} mW")


# ---------------------------------------------------------------------------
# Quick validation
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    cfg    = WECConfig()
    pe_cfg = PowerElectronicsConfig()

    print(cfg.summary())
    print(f"\n  Coil R (series) = {cfg.coil.coil_resistance * cfg.coil.n_coils:.2f} Ω")
    print(f"  Diode V_f       = {pe_cfg.diode_forward_voltage:.2f} V (Schottky)")
    print(f"  Boost V_out     = {pe_cfg.V_out_target:.1f} V")

    # Test with a representative sinusoidal EMF
    t     = cfg.wave.t_eval
    omega = cfg.wave.omega_peak
    EMF_amp = 1.5    # V — representative for this system

    EMF = EMF_amp * np.sin(omega * t)
    sr  = compute_power_chain(t, EMF, cfg, pe_cfg)
    print_stage_report(sr, label=f"EMF_amp={EMF_amp}V")

    # EMF sweep — find 1W threshold
    print("\n--- EMF amplitude sweep ---")
    EMF_amps = np.linspace(0.1, 10.0, 80)
    sweep    = emf_power_sweep(EMF_amps, cfg, pe_cfg)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(sweep["EMF_amplitude"], sweep["P_ac"]*1000,     label="P_ac",     lw=2)
    axes[0].plot(sweep["EMF_amplitude"], sweep["P_rect"]*1000,   label="P_rect",   lw=2)
    axes[0].plot(sweep["EMF_amplitude"], sweep["P_smooth"]*1000, label="P_smooth", lw=2)
    axes[0].plot(sweep["EMF_amplitude"], sweep["P_boost"]*1000,  label="P_boost",  lw=2)
    axes[0].plot(sweep["EMF_amplitude"], sweep["P_net"]*1000,    label="P_net",    lw=2.5, color="black")
    axes[0].axhline(1000, color="red", ls="--", lw=1.5, label="1W target")
    axes[0].set_ylabel("Power (mW)"); axes[0].legend(fontsize=9)
    axes[0].set_title("Power Chain vs. EMF Amplitude", fontweight="bold")
    axes[0].grid(True, alpha=0.4)

    axes[1].plot(sweep["EMF_amplitude"], sweep["eta"]*100, color="steelblue", lw=2)
    axes[1].set_ylabel("End-to-End Efficiency (%)"); axes[1].set_xlabel("EMF Amplitude (V)")
    axes[1].set_title("Power Conditioning Efficiency", fontweight="bold")
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig("power_electronics_sweep.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Series vs parallel comparison
    print("\n--- Series vs Parallel wiring comparison ---")
    EMF_single = EMF_amp / cfg.coil.n_coils * np.sin(omega * t)
    comparison = wiring_comparison(t, EMF_single, cfg, pe_cfg)
    print_stage_report(comparison["series"],   label="Series wiring")
    print_stage_report(comparison["parallel"], label="Parallel wiring")