"""
wave_input.py
-------------
Ocean wave excitation modeling for the WEC system.

Provides two levels of fidelity:
    1. Deterministic sinusoid     — for validation and single-frequency tests
    2. Stochastic JONSWAP spectrum — realistic broadband ocean wave excitation

The output in both cases is a platform tilt time series alpha(t) [rad],
which feeds directly into dynamics.py as the prescribed excitation.

Physics
-------
Wave elevation eta(t) is modeled as a superposition of sinusoids:
    eta(t) = sum_i [ sqrt(2 * S(f_i) * df) * cos(2*pi*f_i*t + phi_i) ]

Platform tilt alpha(t) is related to wave slope:
    alpha(t) ≈ k_wave * eta(t)   [small angle, linear wave theory]

where k_wave = 2*pi*f^2 / g  (deep water dispersion relation)

For a buoy, the effective tilt depends on buoy geometry and mooring.
We use a simplified linear coupling:
    alpha(t) = alpha_scale * eta(t) / H_s
so that the RMS tilt matches the configured alpha_0.

JONSWAP Spectrum
----------------
S(f) = (alpha_PM * g^2) / (2*pi)^4 / f^5
       * exp(-5/4 * (f_p/f)^4)
       * gamma^exp(-0.5*((f/f_p - 1)/sigma)^2)

where:
    alpha_PM  = 0.0081 (Phillips constant)
    f_p       = peak frequency (Hz)
    gamma     = peak enhancement factor (3.3 typical)
    sigma     = 0.07 (f <= f_p), 0.09 (f > f_p)
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Callable
from scipy.interpolate import interp1d
from .config import WECConfig

# ---------------------------------------------------------------------------
# Output containers
# ---------------------------------------------------------------------------
@dataclass
class WaveSpectrum:
    """Frequency-domain wave description."""
    f_Hz: np.ndarray        # frequency array (Hz)
    S_eta: np.ndarray       # wave elevation spectrum (m²/Hz)
    S_alpha: np.ndarray     # platform tilt spectrum (rad²/Hz)
    H_s: float              # significant wave height (m)
    T_peak: float           # peak period (s)
    f_peak: float           # peak frequency (Hz)
    gamma: float            # JONSWAP enhancement factor
    m0: float               # zeroth moment (variance of eta)
    H_s_computed: float     # H_s from spectrum (= 4*sqrt(m0), check)


@dataclass
class WaveTimeSeries:
    """Time-domain wave realization."""
    t: np.ndarray           # time array (s)
    eta: np.ndarray         # wave elevation (m)
    alpha: np.ndarray       # platform tilt (rad)
    alpha_dot: np.ndarray   # platform angular velocity (rad/s)
    alpha_ddot: np.ndarray  # platform angular acceleration (rad/s²)
    spectrum: WaveSpectrum  # the spectrum used to generate this realization

    # Callable interpolants for use in dynamics.py solve_ivp
    alpha_func: Callable      = None
    alpha_ddot_func: Callable = None


# ---------------------------------------------------------------------------
# JONSWAP spectrum
# ---------------------------------------------------------------------------
def jonswap_spectrum(
    f_Hz: np.ndarray,
    H_s: float,
    T_peak: float,
    gamma: float = 3.3,
) -> np.ndarray:
    """
    Compute the JONSWAP wave elevation spectrum S_eta(f) [m²/Hz].

    Parameters
    ----------
    f_Hz   : frequency array (Hz), must start > 0
    H_s    : significant wave height (m)
    T_peak : peak wave period (s)
    gamma  : peak enhancement factor (1 = P-M, 3.3 = typical JONSWAP)

    Returns
    -------
    S_eta : one-sided spectrum (m²/Hz)
    """
    g     = 9.81
    f_p   = 1.0 / T_peak
    f     = np.maximum(f_Hz, 1e-6)   # avoid division by zero

    # --- Pierson-Moskowitz base ---
    # Normalised to match H_s via scaling after computation
    alpha_pm = 0.0081
    S_pm = (alpha_pm * g**2) / ((2*np.pi)**4 * f**5) * np.exp(-1.25 * (f_p/f)**4)

    # --- JONSWAP peak enhancement ---
    sigma = np.where(f <= f_p, 0.07, 0.09)
    r     = np.exp(-0.5 * ((f / f_p - 1.0) / sigma)**2)
    S_j   = S_pm * gamma**r

    # --- Rescale to match H_s ---
    # H_s = 4 * sqrt(m0),  m0 = integral(S) df
    df   = f[1] - f[0]
    m0   = np.trapz(S_j, f)
    H_s_raw = 4.0 * np.sqrt(m0)
    scale   = (H_s / H_s_raw)**2   # scale variance
    S_eta   = S_j * scale

    return S_eta


# ---------------------------------------------------------------------------
# Tilt spectrum from elevation spectrum
# ---------------------------------------------------------------------------
def elevation_to_tilt_spectrum(
    f_Hz: np.ndarray,
    S_eta: np.ndarray,
    buoy_length: float = 1.0,
    alpha_rms_target: Optional[float] = None,
) -> np.ndarray:
    """
    Convert wave elevation spectrum to platform tilt spectrum.

    Approach: linear wave slope coupling
        alpha = d(eta)/dx  =>  S_alpha(f) = k_wave(f)^2 * S_eta(f)

    Deep-water dispersion: k_wave = (2*pi*f)^2 / g

    Optionally rescale so RMS tilt matches alpha_rms_target.

    Parameters
    ----------
    f_Hz           : frequency array (Hz)
    S_eta          : wave elevation spectrum (m²/Hz)
    buoy_length    : characteristic buoy length (m) — affects coupling
    alpha_rms_target: if given, rescale spectrum to this RMS tilt (rad)

    Returns
    -------
    S_alpha : tilt spectrum (rad²/Hz)
    """
    g      = 9.81
    omega  = 2 * np.pi * f_Hz
    k_wave = omega**2 / g              # deep water wavenumber (rad/m)

    # Coupling: tilt ~ wave slope over buoy length
    # For a buoy of length L, tilt ≈ eta_slope ~ k*eta / (k*L) for long waves
    # Simple approximation: alpha ~ k_wave * eta
    coupling = k_wave

    S_alpha = coupling**2 * S_eta

    # Optionally rescale to match target RMS tilt
    if alpha_rms_target is not None:
        df    = f_Hz[1] - f_Hz[0]
        alpha_rms_current = np.sqrt(np.trapz(S_alpha, f_Hz))
        if alpha_rms_current > 0:
            scale   = (alpha_rms_target / alpha_rms_current)**2
            S_alpha = S_alpha * scale

    return S_alpha


# ---------------------------------------------------------------------------
# Generate stochastic time series from spectrum
# ---------------------------------------------------------------------------
def spectrum_to_timeseries(
    f_Hz: np.ndarray,
    S: np.ndarray,
    t: np.ndarray,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a time series from a one-sided power spectral density via
    random phase superposition (standard linear wave theory approach).

        x(t) = sum_i sqrt(2 * S(f_i) * df) * cos(2*pi*f_i*t + phi_i)

    Parameters
    ----------
    f_Hz : frequency array (Hz)
    S    : one-sided PSD (units²/Hz)
    t    : time array (s)
    seed : random seed for reproducibility

    Returns
    -------
    x    : time series (units matching S)
    """
    rng = np.random.default_rng(seed)
    df  = f_Hz[1] - f_Hz[0]

    # Random phases uniformly distributed in [0, 2*pi]
    phases = rng.uniform(0, 2*np.pi, size=len(f_Hz))

    # Amplitude for each component
    amplitudes = np.sqrt(2.0 * S * df)

    # Superposition
    # Broadcasting: t[:, None] * f[None, :] -> (n_t, n_f) matrix
    omega_arr = 2 * np.pi * f_Hz
    phases_t  = np.outer(t, omega_arr) + phases[None, :]   # (n_t, n_f)
    x         = np.sum(amplitudes[None, :] * np.cos(phases_t), axis=1)

    return x


# ---------------------------------------------------------------------------
# Numerical derivatives (central differences)
# ---------------------------------------------------------------------------
def _diff(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.gradient(y, t)


# ---------------------------------------------------------------------------
# Main: build WaveTimeSeries from config
# ---------------------------------------------------------------------------
def generate_wave_excitation(
    cfg: WECConfig,
    mode: str = "jonswap",
    seed: Optional[int] = 42,
    n_freq: int = 512,
    f_min: float = 0.02,
    f_max: float = 1.5,
) -> WaveTimeSeries:
    """
    Generate platform tilt time series alpha(t) for use in dynamics.py.

    Parameters
    ----------
    cfg    : WECConfig instance
    mode   : "jonswap" | "sinusoid"
    seed   : random seed (JONSWAP only)
    n_freq : number of frequency components (JONSWAP only)
    f_min  : minimum frequency for spectrum (Hz)
    f_max  : maximum frequency for spectrum (Hz)

    Returns
    -------
    WaveTimeSeries with alpha_func and alpha_ddot_func callables
    """

    t = cfg.wave.t_eval

    if mode == "sinusoid":
        return _sinusoid_timeseries(cfg, t)
    elif mode == "jonswap":
        return _jonswap_timeseries(cfg, t, seed, n_freq, f_min, f_max)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'jonswap' or 'sinusoid'.")


def _sinusoid_timeseries(cfg: WECConfig, t: np.ndarray) -> WaveTimeSeries:
    """Simple sinusoidal excitation — for validation."""
    Omega   = cfg.wave.omega_peak
    alpha_0 = cfg.platform.alpha_0

    alpha      = alpha_0 * np.sin(Omega * t)
    alpha_dot  = alpha_0 * Omega * np.cos(Omega * t)
    alpha_ddot = -alpha_0 * Omega**2 * np.sin(Omega * t)
    eta        = alpha / (Omega**2 / 9.81)  # back-compute elevation

    # Dummy spectrum for consistency
    f_arr  = np.array([cfg.wave.f_peak])
    S_eta  = np.array([0.0])
    S_alpha = np.array([0.0])
    spec = WaveSpectrum(
        f_Hz=f_arr, S_eta=S_eta, S_alpha=S_alpha,
        H_s=cfg.wave.significant_wave_height,
        T_peak=cfg.wave.T_peak, f_peak=cfg.wave.f_peak,
        gamma=cfg.wave.gamma_jonswap,
        m0=0.0, H_s_computed=0.0,
    )

    alpha_func      = interp1d(t, alpha,      fill_value="extrapolate")
    alpha_ddot_func = interp1d(t, alpha_ddot, fill_value="extrapolate")

    return WaveTimeSeries(
        t=t, eta=eta, alpha=alpha,
        alpha_dot=alpha_dot, alpha_ddot=alpha_ddot,
        spectrum=spec,
        alpha_func=alpha_func,
        alpha_ddot_func=alpha_ddot_func,
    )


def _jonswap_timeseries(
    cfg: WECConfig,
    t: np.ndarray,
    seed: Optional[int],
    n_freq: int,
    f_min: float,
    f_max: float,
) -> WaveTimeSeries:
    """JONSWAP stochastic time series."""

    H_s    = cfg.wave.significant_wave_height
    T_peak = cfg.wave.T_peak
    gamma  = cfg.wave.gamma_jonswap
    alpha_0 = cfg.platform.alpha_0

    # --- Build frequency array ---
    f_arr = np.linspace(f_min, f_max, n_freq)
    df    = f_arr[1] - f_arr[0]

    # --- Compute JONSWAP elevation spectrum ---
    S_eta = jonswap_spectrum(f_arr, H_s, T_peak, gamma)

    # --- Convert to tilt spectrum, rescale to match alpha_0 ---
    # alpha_rms ~ alpha_0 / sqrt(2) for sinusoid => use alpha_0 directly
    alpha_rms_target = alpha_0 / np.sqrt(2.0)
    S_alpha = elevation_to_tilt_spectrum(
        f_arr, S_eta, alpha_rms_target=alpha_rms_target
    )

    # --- Spectrum moments ---
    m0 = np.trapz(S_eta, f_arr)
    H_s_computed = 4.0 * np.sqrt(m0)

    spec = WaveSpectrum(
        f_Hz=f_arr, S_eta=S_eta, S_alpha=S_alpha,
        H_s=H_s, T_peak=T_peak, f_peak=1.0/T_peak,
        gamma=gamma, m0=m0, H_s_computed=H_s_computed,
    )

    # --- Generate stochastic alpha(t) ---
    alpha = spectrum_to_timeseries(f_arr, S_alpha, t, seed=seed)

    # --- Derivatives ---
    alpha_dot  = _diff(alpha, t)
    alpha_ddot = _diff(alpha_dot, t)

    # --- Elevation (for reference) ---
    eta = spectrum_to_timeseries(f_arr, S_eta, t, seed=seed)

    # --- Interpolants ---
    alpha_func      = interp1d(t, alpha,      fill_value="extrapolate", kind="cubic")
    alpha_ddot_func = interp1d(t, alpha_ddot, fill_value="extrapolate", kind="cubic")

    return WaveTimeSeries(
        t=t, eta=eta, alpha=alpha,
        alpha_dot=alpha_dot, alpha_ddot=alpha_ddot,
        spectrum=spec,
        alpha_func=alpha_func,
        alpha_ddot_func=alpha_ddot_func,
    )


# ---------------------------------------------------------------------------
# Spectrum statistics
# ---------------------------------------------------------------------------
def spectrum_stats(ws: WaveTimeSeries) -> dict:
    """Compute key statistics from a WaveTimeSeries."""
    spec = ws.spectrum
    df   = spec.f_Hz[1] - spec.f_Hz[0]

    m0 = np.trapz(spec.S_eta, spec.f_Hz)
    m1 = np.trapz(spec.f_Hz * spec.S_eta, spec.f_Hz)
    m2 = np.trapz(spec.f_Hz**2 * spec.S_eta, spec.f_Hz)

    T_mean    = m0 / m1 if m1 > 0 else 0.0
    T_e       = m0 / m1 if m1 > 0 else 0.0  # energy period

    alpha_rms = np.sqrt(np.mean(ws.alpha**2))
    alpha_max = np.max(np.abs(ws.alpha))

    return {
        "H_s_target":    spec.H_s,
        "H_s_spectrum":  spec.H_s_computed,
        "T_peak":        spec.T_peak,
        "T_mean":        T_mean,
        "alpha_rms_rad": alpha_rms,
        "alpha_rms_deg": np.degrees(alpha_rms),
        "alpha_max_deg": np.degrees(alpha_max),
        "m0_eta":        m0,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_wave_excitation(ws: WaveTimeSeries, save_path: Optional[str] = None):
    """Four-panel plot: spectrum, elevation, tilt time series, tilt PSD."""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    # --- Panel 1: Wave elevation spectrum ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(ws.spectrum.f_Hz, ws.spectrum.S_eta, color="steelblue", lw=2)
    ax1.axvline(ws.spectrum.f_peak, color="darkorange", linestyle="--",
                label=f"f_peak = {ws.spectrum.f_peak:.3f} Hz")
    ax1.set_xlabel("Frequency (Hz)"); ax1.set_ylabel("S_η (m²/Hz)")
    ax1.set_title("JONSWAP Wave Elevation Spectrum", fontweight="bold")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.4)

    # --- Panel 2: Tilt spectrum ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogy(ws.spectrum.f_Hz, ws.spectrum.S_alpha + 1e-20,
                 color="mediumpurple", lw=2)
    ax2.axvline(ws.spectrum.f_peak, color="darkorange", linestyle="--",
                label=f"f_peak = {ws.spectrum.f_peak:.3f} Hz")
    ax2.set_xlabel("Frequency (Hz)"); ax2.set_ylabel("S_α (rad²/Hz)")
    ax2.set_title("Platform Tilt Spectrum", fontweight="bold")
    ax2.legend(fontsize=9); ax2.grid(True, which="both", alpha=0.4)

    # --- Panel 3: Wave elevation time series ---
    ax3 = fig.add_subplot(gs[1, 0])
    t_plot = ws.t[:min(len(ws.t), 500)]
    ax3.plot(t_plot, ws.eta[:len(t_plot)], color="steelblue", lw=1.2)
    ax3.set_xlabel("Time (s)"); ax3.set_ylabel("η (m)")
    ax3.set_title("Wave Elevation η(t)", fontweight="bold")
    ax3.grid(True, alpha=0.4)

    # --- Panel 4: Platform tilt time series ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(t_plot, np.degrees(ws.alpha[:len(t_plot)]),
             color="crimson", lw=1.2)
    ax4.set_xlabel("Time (s)"); ax4.set_ylabel("α (deg)")
    ax4.set_title("Platform Tilt α(t)", fontweight="bold")
    ax4.grid(True, alpha=0.4)

    stats = spectrum_stats(ws)
    fig.suptitle(
        f"Wave Input  |  H_s = {stats['H_s_target']:.1f} m  |  "
        f"T_peak = {ws.spectrum.T_peak:.1f} s  |  "
        f"α_rms = {stats['alpha_rms_deg']:.2f}°",
        fontsize=11, y=1.01,
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Quick validation
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = WECConfig()

    print("--- Generating JONSWAP wave excitation ---")
    ws = generate_wave_excitation(cfg, mode="jonswap", seed=42)

    stats = spectrum_stats(ws)
    print("\nWave Statistics:")
    for k, v in stats.items():
        print(f"  {k:25s}: {v:.4f}")

    plot_wave_excitation(ws, save_path="wave_input.png")

    print("\n--- Validating sinusoid mode ---")
    ws_sin = generate_wave_excitation(cfg, mode="sinusoid")
    print(f"  Sinusoid alpha_0 = {np.degrees(np.max(ws_sin.alpha)):.2f} deg")
    print(f"  JONSWAP alpha_max = {np.degrees(np.max(np.abs(ws.alpha))):.2f} deg")