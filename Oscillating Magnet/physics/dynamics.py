"""
dynamics.py
-----------
Equations of motion for the sliding magnet on a spring-constrained flat plate
mounted on a pitching/rolling platform.

Derived from the Lagrangian framework in 3_95_Full_Derivation.pdf, adapted
for a flat plate + spring design (replaces curved bowl with linear spring).

Governing EOM (1D projection of 2D sliding motion):
    m*x'' + c_total*x' + k_eff*x = m * d * alpha''(t)

where:
    x       : magnet displacement on plate (m)
    alpha   : platform tilt angle (rad)  [prescribed excitation]
    d       : pivot-to-plate distance (m)  [coupling parameter kappa]
    m       : magnet mass (kg)
    c_total : total damping (internal + load) (N·s/m)
    k_eff   : effective spring constant (N/m)

Two solvers are provided:
    1. solve_linear()    — fast analytical solution for sinusoidal excitation
    2. solve_nonlinear() — full numerical ODE via scipy solve_ivp
                          includes nonlinear sin(alpha) gravity term
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from dataclasses import dataclass
from typing import Callable, Tuple
from .config import WECConfig


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------
@dataclass
class DynamicsResult:
    t: np.ndarray           # time array (s)
    x: np.ndarray           # magnet position on plate (m)
    x_dot: np.ndarray       # magnet velocity (m/s)
    x_ddot: np.ndarray      # magnet acceleration (m/s²)  [derived]
    alpha: np.ndarray       # platform tilt angle (rad)
    alpha_dot: np.ndarray   # platform angular velocity (rad/s)
    alpha_ddot: np.ndarray  # platform angular acceleration (rad/s²)

    @property
    def kinetic_energy(self) -> np.ndarray:
        """Translational KE of magnet (J)."""
        # Stored in result so we don't need cfg here
        return self._KE

    def set_energies(self, m: float, k: float):
        self._KE = 0.5 * m * self.x_dot**2
        self._PE = 0.5 * k * self.x**2

    @property
    def potential_energy(self) -> np.ndarray:
        return self._PE


# ---------------------------------------------------------------------------
# Helper: numerical derivatives with central differences
# ---------------------------------------------------------------------------
def _central_diff(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Second-order central difference derivative."""
    dy = np.gradient(y, t)
    return dy


# ---------------------------------------------------------------------------
# Analytical (linear) solver
# ---------------------------------------------------------------------------
def solve_linear(
    cfg: WECConfig,
    alpha_func: Callable[[np.ndarray], np.ndarray],
    alpha_ddot_func: Callable[[np.ndarray], np.ndarray],
    t: np.ndarray,
) -> DynamicsResult:
    """
    Analytical steady-state solution for sinusoidal excitation.

    Assumes:
        alpha(t) = alpha_0 * sin(Omega * t)
        Forcing:  F(t) = m * d * alpha_ddot = -m * d * alpha_0 * Omega^2 * sin(Omega*t)

    Steady-state response:
        x(t) = X * sin(Omega*t + phi)

    Where:
        X   = F0 / sqrt((omega_n^2 - Omega^2)^2 + (2*zeta*omega_n*Omega)^2)
        phi = atan2(-2*zeta*omega_n*Omega, omega_n^2 - Omega^2)
        F0  = d * alpha_0 * Omega^2   (forcing amplitude per unit mass)

    This is valid for steady-state only — use solve_nonlinear for transients.
    """

    m     = cfg.magnet.mass
    k     = cfg.spring.k_eff
    c     = cfg.damping.c_total
    d     = cfg.platform.pivot_to_plate
    alpha_0 = cfg.platform.alpha_0
    Omega = cfg.platform.Omega

    omega_n = np.sqrt(k / m)
    zeta    = c / (2 * m * omega_n)

    # Forcing amplitude (per unit mass)
    # F = m*d*alpha_ddot = -m*d*alpha_0*Omega^2 * sin(Omega*t)
    # The two forcing terms from the Lagrangian derivation:
    #   -g*alpha  (gravitational)  and  -kappa*alpha_ddot  (inertial)
    # For flat plate: kappa = d (pivot-to-plate distance)
    F0_gravity  = -cfg.wave.omega_peak**0    # placeholder, set per call below
    kappa       = d
    F0_inertial = kappa * alpha_0 * Omega**2
    F0_grav     = cfg.wave.omega_peak**0     # g * alpha_0
    # Net forcing amplitude (combined, same phase for pure sine)
    g = 9.81
    F0 = alpha_0 * (kappa * Omega**2 - g)   # from derivation Section 13.2

    # Frequency response magnitude and phase
    denom = np.sqrt((omega_n**2 - Omega**2)**2 + (2 * zeta * omega_n * Omega)**2)
    X_amp = abs(F0) / denom if denom > 0 else 0.0
    phi   = np.arctan2(-(2 * zeta * omega_n * Omega), (omega_n**2 - Omega**2))
    # Sign of F0 flips phase by pi if negative
    if F0 < 0:
        phi += np.pi

    x     = X_amp * np.sin(Omega * t + phi)
    x_dot = X_amp * Omega * np.cos(Omega * t + phi)

    # Recompute alpha arrays from function inputs
    alpha     = alpha_func(t)
    alpha_dot = _central_diff(alpha, t)
    alpha_ddot = _central_diff(alpha_dot, t)
    x_ddot    = _central_diff(x_dot, t)

    result = DynamicsResult(
        t=t, x=x, x_dot=x_dot, x_ddot=x_ddot,
        alpha=alpha, alpha_dot=alpha_dot, alpha_ddot=alpha_ddot,
    )
    result.set_energies(m, k)
    return result


# ---------------------------------------------------------------------------
# Nonlinear ODE solver
# ---------------------------------------------------------------------------
def solve_nonlinear(
    cfg: WECConfig,
    alpha_func: Callable[[float], float],
    alpha_ddot_func: Callable[[float], float],
    t_eval: np.ndarray,
    x0: float = 0.0,
    x_dot0: float = 0.0,
) -> DynamicsResult:
    """
    Full nonlinear numerical integration of the EOM using scipy solve_ivp.

    Nonlinear EOM (from derivation, adapted for flat plate):
        m*x'' + c*x' + k*x = m*d*alpha''(t) - m*g*sin(alpha(t))

    The sin(alpha) term becomes important for large tilt angles (> ~5 deg).

    State vector: [x, x_dot]

    Parameters
    ----------
    cfg          : WECConfig instance
    alpha_func   : callable(t) -> alpha(t) in radians
    alpha_ddot_func : callable(t) -> alpha_ddot(t) in rad/s²
    t_eval       : time points at which to evaluate solution
    x0, x_dot0   : initial conditions

    Returns
    -------
    DynamicsResult
    """

    m   = cfg.magnet.mass
    k   = cfg.spring.k_eff
    c   = cfg.damping.c_total
    d   = cfg.platform.pivot_to_plate
    g   = 9.81

    def ode(t, state):
        x, x_dot = state

        a     = alpha_func(t)
        a_ddot = alpha_ddot_func(t)

        # Nonlinear EOM:
        # m*x'' = -c*x' - k*x + m*d*alpha'' - m*g*sin(alpha)
        x_ddot = (
            -c / m * x_dot
            - k / m * x
            + d * a_ddot
            - g * np.sin(a)
        )
        return [x_dot, x_ddot]

    sol = solve_ivp(
        ode,
        t_span=(t_eval[0], t_eval[-1]),
        y0=[x0, x_dot0],
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-9,
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    t_out     = sol.t
    x_out     = sol.y[0]
    x_dot_out = sol.y[1]
    x_ddot_out = _central_diff(x_dot_out, t_out)

    alpha_arr     = np.array([alpha_func(ti) for ti in t_out])
    alpha_dot_arr = _central_diff(alpha_arr, t_out)
    alpha_ddot_arr = np.array([alpha_ddot_func(ti) for ti in t_out])

    result = DynamicsResult(
        t=t_out, x=x_out, x_dot=x_dot_out, x_ddot=x_ddot_out,
        alpha=alpha_arr, alpha_dot=alpha_dot_arr, alpha_ddot=alpha_ddot_arr,
    )
    result.set_energies(m, k)
    return result


# ---------------------------------------------------------------------------
# Convenience: build alpha functions from a sinusoid
# ---------------------------------------------------------------------------
def sinusoidal_excitation(alpha_0: float, Omega: float):
    """
    Returns (alpha_func, alpha_dot_func, alpha_ddot_func) for:
        alpha(t) = alpha_0 * sin(Omega * t)
    """
    alpha_func      = lambda t: alpha_0 * np.sin(Omega * t)
    alpha_dot_func  = lambda t: alpha_0 * Omega * np.cos(Omega * t)
    alpha_ddot_func = lambda t: -alpha_0 * Omega**2 * np.sin(Omega * t)
    return alpha_func, alpha_dot_func, alpha_ddot_func


# ---------------------------------------------------------------------------
# Convenience: build alpha functions from a time-series array (stochastic)
# ---------------------------------------------------------------------------
def array_excitation(t_arr: np.ndarray, alpha_arr: np.ndarray):
    """
    Returns (alpha_func, alpha_dot_func, alpha_ddot_func) by interpolating
    a pre-computed time series (e.g. from wave_input.py JONSWAP).
    """
    alpha_dot_arr  = _central_diff(alpha_arr, t_arr)
    alpha_ddot_arr = _central_diff(alpha_dot_arr, t_arr)

    alpha_func      = interp1d(t_arr, alpha_arr,      fill_value="extrapolate")
    alpha_dot_func  = interp1d(t_arr, alpha_dot_arr,  fill_value="extrapolate")
    alpha_ddot_func = interp1d(t_arr, alpha_ddot_arr, fill_value="extrapolate")

    return alpha_func, alpha_dot_func, alpha_ddot_func


# ---------------------------------------------------------------------------
# Power extraction from dynamics result
# ---------------------------------------------------------------------------
def compute_power(result: DynamicsResult, cfg: WECConfig) -> dict:
    """
    Compute power metrics from a solved DynamicsResult.

    P_instantaneous = c_L * x_dot^2   (load power)
    P_total_damping = c_total * x_dot^2

    Returns dict with time-series and scalar averages.
    """
    c_L   = cfg.damping.c_load
    c_tot = cfg.damping.c_total

    P_load  = c_L  * result.x_dot**2
    P_damp  = c_tot * result.x_dot**2

    # Skip first 20% of time (transient)
    n_skip = int(0.2 * len(result.t))

    return {
        "P_load_series":   P_load,
        "P_damp_series":   P_damp,
        "P_avg_load":      np.mean(P_load[n_skip:]),
        "P_peak_load":     np.max(P_load[n_skip:]),
        "P_avg_total_damp": np.mean(P_damp[n_skip:]),
        "efficiency_extraction": (
            np.mean(P_load[n_skip:]) / np.mean(P_damp[n_skip:])
            if np.mean(P_damp[n_skip:]) > 0 else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# Quick validation
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    cfg = WECConfig()
    print(cfg.summary())

    t = cfg.wave.t_eval
    Omega = cfg.wave.omega_peak
    alpha_0 = cfg.platform.alpha_0

    # Override platform Omega to match wave
    cfg.platform.Omega = Omega

    alpha_func, alpha_dot_func, alpha_ddot_func = sinusoidal_excitation(alpha_0, Omega)

    print("\n--- Running nonlinear solver ---")
    result_nl = solve_nonlinear(cfg, alpha_func, alpha_ddot_func, t)

    print("\n--- Running linear analytical solver ---")
    result_lin = solve_linear(cfg, alpha_func, alpha_ddot_func, t)

    power_nl = compute_power(result_nl, cfg)
    print(f"\nNonlinear: P_avg_load = {power_nl['P_avg_load']*1000:.2f} mW")
    print(f"Nonlinear: P_peak_load = {power_nl['P_peak_load']*1000:.2f} mW")

    power_lin = compute_power(result_lin, cfg)
    print(f"Linear:    P_avg_load = {power_lin['P_avg_load']*1000:.2f} mW")

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t, np.degrees(result_nl.alpha), label="α(t) platform tilt", color="steelblue")
    axes[0].set_ylabel("Platform Tilt (deg)")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(t, result_nl.x * 1000, label="x(t) nonlinear", color="darkorange")
    axes[1].plot(t, result_lin.x * 1000, label="x(t) linear", color="green", linestyle="--")
    axes[1].set_ylabel("Magnet Displacement (mm)")
    axes[1].legend(); axes[1].grid(True)

    axes[2].plot(t, power_nl["P_load_series"] * 1000, label="P_load nonlinear", color="crimson")
    axes[2].axhline(power_nl["P_avg_load"] * 1000, color="crimson", linestyle="--",
                    label=f"P_avg = {power_nl['P_avg_load']*1000:.2f} mW")
    axes[2].set_ylabel("Load Power (mW)")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(); axes[2].grid(True)

    plt.suptitle("WEC Dynamics Validation — Sinusoidal Excitation")
    plt.tight_layout()
    plt.savefig("dynamics_validation.png", dpi=150)
    plt.show()
    print("\nPlot saved to dynamics_validation.png")