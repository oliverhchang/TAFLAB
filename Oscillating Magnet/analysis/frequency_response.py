"""
frequency_response.py
---------------------
Analytical frequency-domain analysis of the WEC system.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass
from typing import Optional
from physics.config import WECConfig


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------
@dataclass
class FreqResponseResult:
    f_Hz: np.ndarray          # frequency array (Hz)
    omega: np.ndarray         # frequency array (rad/s)
    H_mag: np.ndarray         # |X / alpha_0|  (m/rad)
    H_phase: np.ndarray       # phase (rad)
    P_normalized: np.ndarray  # normalized power ~ omega^2 * H_mag^2
    omega_n: float            # natural frequency (rad/s)
    f_n: float                # natural frequency (Hz)
    zeta: float               # total damping ratio
    zeta_load: float          # load-only damping ratio
    F0_coeff: np.ndarray      # forcing coefficient vs frequency


# ---------------------------------------------------------------------------
# Core transfer function
# ---------------------------------------------------------------------------
def compute_transfer_function(
    cfg: WECConfig,
    f_min: float = 0.01,
    f_max: float = 2.0,
    n_points: int = 2000,
) -> FreqResponseResult:
    """
    Compute the frequency response of magnet displacement to platform tilt.
    """

    m     = cfg.magnet.mass
    k     = cfg.spring.k_eff
    c_tot = cfg.damping.c_total
    c_L   = cfg.damping.c_load
    d     = cfg.platform.pivot_to_plate
    g     = 9.81

    # System parameters
    omega_n = np.sqrt(k / m)
    f_n     = omega_n / (2 * np.pi)
    zeta    = c_tot / (2 * m * omega_n)
    zeta_L  = c_L   / (2 * m * omega_n)

    # Frequency sweep
    f_arr   = np.linspace(f_min, f_max, n_points)
    omega   = 2 * np.pi * f_arr

    # Coupling / forcing coefficient (kappa for flat plate = d)
    # From Section 13.2 of derivation[cite: 604]: F0 = alpha_0 * (kappa*Omega^2 - g)
    kappa       = d
    F0_coeff    = kappa * omega**2 - g

    # Denominator: sqrt((omega_n^2 - Omega^2)^2 + (2*zeta*omega_n*Omega)^2)
    denom = np.sqrt(
        (omega_n**2 - omega**2)**2 + (2 * zeta * omega_n * omega)**2
    )

    # Magnitude |H| = |F0_coeff| / denom [cite: 608]
    H_mag = np.abs(F0_coeff) / denom

    # Phase calculation
    H_phase = np.arctan2(
        -2 * zeta * omega_n * omega,
        omega_n**2 - omega**2
    )
    # Correct phase sign for negative F0_coeff regions [cite: 634]
    H_phase[F0_coeff < 0] += np.pi

    # Normalized power: P ~ c_L * (omega * X)^2
    P_normalized = c_L * omega**2 * H_mag**2

    return FreqResponseResult(
        f_Hz=f_arr,
        omega=omega,
        H_mag=H_mag,
        H_phase=H_phase,
        P_normalized=P_normalized,
        omega_n=omega_n,
        f_n=f_n,
        zeta=zeta,
        zeta_load=zeta_L,
        F0_coeff=F0_coeff,
    )


# ---------------------------------------------------------------------------
# Find optimal spring constant to match a target frequency
# ---------------------------------------------------------------------------
def optimal_spring_constant(
    cfg: WECConfig,
    target_freq_Hz: float,
) -> float:
    """
    Compute the spring constant k_single needed so that omega_n = target_freq.

    omega_n = sqrt(k_eff / m)  =>  k_eff = m * omega_n^2
    k_single = k_eff / n_springs
    """
    m        = cfg.magnet.mass
    omega_t  = 2 * np.pi * target_freq_Hz
    k_eff    = m * omega_t**2
    k_single = k_eff / cfg.spring.n_springs

    return k_single


# ---------------------------------------------------------------------------
# Null-point analysis
# ---------------------------------------------------------------------------
def null_point_frequency(cfg: WECConfig) -> float:
    """
    Frequency at which forcing is zero: kappa * Omega^2 = g
    (Analogue of Section 14.1: 5*kappa/(7*R) = 1 condition)

    For flat plate: Omega_null = sqrt(g / kappa) = sqrt(g / d)
    """
    g     = 9.81
    kappa = cfg.platform.pivot_to_plate
    omega_null = np.sqrt(g / kappa)
    f_null     = omega_null / (2 * np.pi)
    return f_null


# ---------------------------------------------------------------------------
# Sweep over damping ratios
# ---------------------------------------------------------------------------
def damping_sweep(
    cfg: WECConfig,
    zeta_values: list,
    f_min: float = 0.01,
    f_max: float = 2.0,
) -> dict:
    """
    Compute frequency responses for multiple damping ratios.
    Useful for visualizing bandwidth vs. peak tradeoff.

    Returns dict keyed by zeta value.
    """
    results = {}
    for zeta_target in zeta_values:
        cfg_copy = WECConfig()
        # Back-calculate c_total from desired zeta
        omega_n = np.sqrt(cfg_copy.spring.k_eff / cfg_copy.magnet.mass)
        c_total_target = 2 * zeta_target * cfg_copy.magnet.mass * omega_n
        # Split evenly between internal and load
        cfg_copy.damping.c_internal = c_total_target / 2
        cfg_copy.damping.c_load     = c_total_target / 2
        results[zeta_target] = compute_transfer_function(cfg_copy, f_min, f_max)
    return results


# ---------------------------------------------------------------------------
# Spring constant sweep
# ---------------------------------------------------------------------------
def spring_sweep(
    cfg: WECConfig,
    k_values: np.ndarray,
    f_min: float = 0.01,
    f_max: float = 2.0,
) -> dict:
    """
    Compute frequency responses for a range of spring constants.
    Shows how tuning k shifts the resonance peak to match wave frequency.
    """
    results = {}
    for k in k_values:
        cfg_copy = WECConfig()
        cfg_copy.spring.k = k / cfg_copy.spring.n_springs  # per-spring
        results[k] = compute_transfer_function(cfg_copy, f_min, f_max)
    return results


# ---------------------------------------------------------------------------
# Compute average power for a given wave spectrum
# ---------------------------------------------------------------------------
def power_from_spectrum(
    fr: FreqResponseResult,
    S_alpha: np.ndarray,
    df: Optional[float] = None,
) -> float:
    """
    Compute average extracted power given a wave spectrum S_alpha(f).

    P_avg = integral[ c_L * omega^2 * |H(omega)|^2 * S_alpha(f) ] df

    Parameters
    ----------
    fr      : FreqResponseResult from compute_transfer_function
    S_alpha : platform tilt spectrum (rad^2/Hz) at same frequencies as fr.f_Hz
    df      : frequency resolution (Hz); computed from fr if not given

    Returns
    -------
    P_avg   : average power (W)
    """
    if df is None:
        df = fr.f_Hz[1] - fr.f_Hz[0]

    integrand = fr.P_normalized * S_alpha
    P_avg = np.trapz(integrand, fr.f_Hz)
    return P_avg


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_frequency_response(
    fr: FreqResponseResult,
    cfg: WECConfig,
    save_path: Optional[str] = None,
):
    """Full Bode-style plot: magnitude, phase, normalized power."""

    fig = plt.figure(figsize=(12, 10))
    gs  = gridspec.GridSpec(3, 1, hspace=0.45)

    # --- Magnitude ---
    ax1 = fig.add_subplot(gs[0])
    ax1.semilogy(fr.f_Hz, fr.H_mag * 1000, color="steelblue", lw=2,
                 label="|H(f)| = X / α₀")
    ax1.axvline(fr.f_n, color="darkorange", linestyle="--",
                label=f"ωₙ = {fr.f_n:.3f} Hz")
    ax1.axvline(cfg.wave.f_peak, color="green", linestyle=":",
                label=f"Wave peak = {cfg.wave.f_peak:.3f} Hz")
    f_null = null_point_frequency(cfg)
    ax1.axvline(f_null, color="red", linestyle="-.",
                label=f"Null point = {f_null:.3f} Hz")
    ax1.set_ylabel("|H(f)|  (mm / rad)", fontsize=11)
    ax1.set_title("WEC Frequency Response", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9); ax1.grid(True, which="both", alpha=0.4)
    ax1.set_xlim(fr.f_Hz[0], fr.f_Hz[-1])

    # --- Phase ---
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(fr.f_Hz, np.degrees(fr.H_phase), color="mediumpurple", lw=2)
    ax2.axvline(fr.f_n, color="darkorange", linestyle="--")
    ax2.axhline(-90, color="gray", linestyle=":", alpha=0.6)
    ax2.set_ylabel("Phase (deg)", fontsize=11)
    ax2.set_ylim(-200, 200)
    ax2.grid(True, alpha=0.4)

    # --- Normalized Power ---
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(fr.f_Hz, fr.P_normalized * 1000, color="crimson", lw=2,
             label="Normalized power")
    ax3.axvline(fr.f_n, color="darkorange", linestyle="--",
                label=f"ωₙ = {fr.f_n:.3f} Hz")
    ax3.axvline(cfg.wave.f_peak, color="green", linestyle=":",
                label=f"Wave = {cfg.wave.f_peak:.3f} Hz")
    ax3.set_xlabel("Frequency (Hz)", fontsize=11)
    ax3.set_ylabel("P_norm (mW / rad²)", fontsize=11)
    ax3.legend(fontsize=9); ax3.grid(True, alpha=0.4)

    # Annotations
    peak_idx = np.argmax(fr.H_mag)
    ax1.annotate(
        f"Peak: {fr.H_mag[peak_idx]*1000:.1f} mm/rad\n@ {fr.f_Hz[peak_idx]:.3f} Hz",
        xy=(fr.f_Hz[peak_idx], fr.H_mag[peak_idx]*1000),
        xytext=(fr.f_Hz[peak_idx] + 0.1, fr.H_mag[peak_idx]*1000 * 0.5),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=9,
    )

    plt.suptitle(
        f"ζ = {fr.zeta:.3f}  |  k_eff = {cfg.spring.k_eff:.2f} N/m  |  "
        f"m = {cfg.magnet.mass*1000:.1f} g",
        fontsize=10, y=0.98,
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


def plot_spring_sweep(
    sweep_results: dict,
    cfg: WECConfig,
    save_path: Optional[str] = None,
):
    """Plot displacement magnitude for multiple spring constants."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    cmap = plt.cm.viridis
    k_vals = list(sweep_results.keys())
    colors = cmap(np.linspace(0, 1, len(k_vals)))

    for k, color in zip(k_vals, colors):
        fr = sweep_results[k]
        f_n = fr.f_n
        axes[0].semilogy(fr.f_Hz, fr.H_mag * 1000, color=color, lw=1.5,
                         label=f"k={k:.1f} N/m → fₙ={f_n:.3f}Hz")
        axes[1].plot(fr.f_Hz, fr.P_normalized * 1000, color=color, lw=1.5)

    axes[0].axvline(cfg.wave.f_peak, color="red", linestyle="--", lw=2,
                    label=f"Target wave freq {cfg.wave.f_peak:.3f} Hz")
    axes[0].set_ylabel("|H(f)|  (mm/rad)", fontsize=11)
    axes[0].set_title("Spring Constant Sweep — Frequency Tuning", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=8, ncol=2); axes[0].grid(True, which="both", alpha=0.4)

    axes[1].axvline(cfg.wave.f_peak, color="red", linestyle="--", lw=2)
    axes[1].set_xlabel("Frequency (Hz)", fontsize=11)
    axes[1].set_ylabel("P_norm (mW/rad²)", fontsize=11)
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


def plot_damping_sweep(
    sweep_results: dict,
    cfg: WECConfig,
    save_path: Optional[str] = None,
):
    """Plot effect of damping ratio on bandwidth vs. peak tradeoff."""
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.plasma
    zeta_vals = list(sweep_results.keys())
    colors = cmap(np.linspace(0.1, 0.9, len(zeta_vals)))

    for zeta, color in zip(zeta_vals, colors):
        fr = sweep_results[zeta]
        ax.semilogy(fr.f_Hz, fr.H_mag * 1000, color=color, lw=2,
                    label=f"ζ = {zeta:.3f}")

    ax.axvline(cfg.wave.f_peak, color="black", linestyle="--", lw=1.5,
               label=f"Wave freq = {cfg.wave.f_peak:.3f} Hz")
    ax.set_xlabel("Frequency (Hz)", fontsize=11)
    ax.set_ylabel("|H(f)|  (mm/rad)", fontsize=11)
    ax.set_title("Damping Ratio Sweep — Bandwidth vs. Peak Tradeoff", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.4)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Print summary report
# ---------------------------------------------------------------------------
def print_resonance_report(fr: FreqResponseResult, cfg: WECConfig):
    g = 9.81
    kappa = cfg.platform.pivot_to_plate
    f_null = null_point_frequency(cfg)
    k_opt  = optimal_spring_constant(cfg, cfg.wave.f_peak)

    peak_idx = np.argmax(fr.H_mag)
    P_peak_idx = np.argmax(fr.P_normalized)

    print("=" * 55)
    print("  WEC Frequency Response Summary")
    print("=" * 55)
    print(f"  Natural frequency       : {fr.f_n:.4f} Hz")
    print(f"  Total damping ratio ζ   : {fr.zeta:.4f}")
    print(f"  Load damping ratio ζ_L  : {fr.zeta_load:.4f}")
    print(f"  Peak |H| = {fr.H_mag[peak_idx]*1000:.2f} mm/rad @ {fr.f_Hz[peak_idx]:.4f} Hz")
    print(f"  Peak P_norm @ {fr.f_Hz[P_peak_idx]:.4f} Hz")
    print(f"  Null-point frequency    : {f_null:.4f} Hz")
    print(f"  Wave peak frequency     : {cfg.wave.f_peak:.4f} Hz")
    print(f"  Optimal k (per spring)  : {k_opt:.4f} N/m  (to match wave freq)")
    print(f"  Optimal k_eff total     : {k_opt * cfg.spring.n_springs:.4f} N/m")
    print("=" * 55)


# ---------------------------------------------------------------------------
# Quick validation
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = WECConfig()

    # 1. Base frequency response
    print("\n--- Base Frequency Response ---")
    fr = compute_transfer_function(cfg)
    print_resonance_report(fr, cfg)
    plot_frequency_response(fr, cfg, save_path="freq_response.png")

    # 2. Spring sweep — find k to match 0.125 Hz (8s wave)
    print("\n--- Spring Constant Sweep ---")
    k_vals = np.array([1.0, 2.0, 5.0, 10.0, 20.0, 50.0])
    sweep = spring_sweep(cfg, k_vals)
    plot_spring_sweep(sweep, cfg, save_path="spring_sweep.png")

    # 3. Damping sweep
    print("\n--- Damping Ratio Sweep ---")
    zeta_vals = [0.05, 0.1, 0.2, 0.5, 1.0]
    damp_sweep = damping_sweep(cfg, zeta_vals)
    plot_damping_sweep(damp_sweep, cfg, save_path="damping_sweep.png")

    # 4. Optimal spring recommendation
    k_opt = optimal_spring_constant(cfg, cfg.wave.f_peak)
    print(f"\nTo match wave peak ({cfg.wave.f_peak:.4f} Hz):")
    print(f"  Set k_single = {k_opt:.4f} N/m per spring")
    print(f"  (with {cfg.spring.n_springs} springs in parallel)")