"""
objective.py
------------
Defines the scalar objective function for WEC design Optimization.

The optimizer works with a flat "design vector" x of normalized parameters.
This module handles:
    1. Design vector <-> WECConfig translation
    2. Physics evaluation (fast frequency-domain or full time-domain)
    3. Constraint checking (physical feasibility bounds)
    4. Objective function: maximize P_avg subject to constraints

Design Vector
-------------
x = [k_eff, magnet_diameter, magnet_thickness, n_turns,
     coil_inner_r, coil_outer_r, air_gap, c_internal]

All values stored internally in SI units (m, N/m, etc.)

Evaluation Modes
----------------
"fast"  : frequency-domain P_avg estimate using transfer function
          ~1 ms per evaluation, good for global sweeps

"full"  : time-domain nonlinear ODE + EM model
          ~500 ms per evaluation, used for final verification

The fast mode uses:
    P_avg ≈ c_L * omega_n^2 * X_rms^2
          = c_L * omega_n^2 * (alpha_0 * |H(omega_wave)|)^2 / 2
"""

import numpy as np
import copy
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
from config import WECConfig, MagnetConfig, SpringConfig, CoilConfig, DampingConfig


# ---------------------------------------------------------------------------
# Design space bounds
# ---------------------------------------------------------------------------
# Each entry: (min, max, description, units)
DESIGN_BOUNDS = {
    "k_eff":            (0.5,   200.0,  "Effective spring constant",     "N/m"),
    "magnet_diameter":  (0.010, 0.060,  "Magnet diameter",               "m"),
    "magnet_thickness": (0.005, 0.040,  "Magnet thickness",              "m"),
    "n_turns":          (50,    2000,   "Turns per coil",                "—"),
    "coil_inner_r":     (0.003, 0.020,  "Coil inner radius",             "m"),
    "coil_outer_r":     (0.010, 0.040,  "Coil outer radius",             "m"),
    "air_gap":          (0.001, 0.010,  "Air gap magnet-to-coil",        "m"),
    "c_internal":       (0.005, 0.500,  "Internal damping coefficient",  "N·s/m"),
}

PARAM_NAMES = list(DESIGN_BOUNDS.keys())
N_PARAMS    = len(PARAM_NAMES)


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------
@dataclass
class ObjectiveResult:
    """Full result from one objective function evaluation."""
    design_vector: np.ndarray       # raw design vector (SI units)
    params: Dict[str, float]        # named parameters

    # Objectives
    P_avg_mW: float                 # primary: average extracted power (mW)
    P_peak_mW: float                # secondary: peak power (mW)

    # System state
    f_n_Hz: float                   # natural frequency (Hz)
    zeta_total: float               # total damping ratio
    zeta_load: float                # load damping ratio
    freq_ratio: float               # Omega_wave / omega_n

    # Constraints (all should be >= 0 when feasible)
    constraints: Dict[str, float]
    feasible: bool

    # Optional: full time-domain result for inspection
    em_result: Any = None
    dyn_result: Any = None

    def summary(self) -> str:
        lines = [
            f"  P_avg         = {self.P_avg_mW:.3f} mW",
            f"  P_peak        = {self.P_peak_mW:.3f} mW",
            f"  f_n           = {self.f_n_Hz:.4f} Hz",
            f"  freq ratio    = {self.freq_ratio:.3f}",
            f"  zeta_total    = {self.zeta_total:.4f}",
            f"  feasible      = {self.feasible}",
        ]
        if not self.feasible:
            violated = {k: v for k, v in self.constraints.items() if v < 0}
            lines.append(f"  violated      = {list(violated.keys())}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Design vector <-> WECConfig
# ---------------------------------------------------------------------------
def vector_to_config(x: np.ndarray, base_cfg: WECConfig) -> WECConfig:
    """
    Unpack a design vector into a WECConfig.
    Non-optimized parameters are inherited from base_cfg.
    """
    cfg = copy.deepcopy(base_cfg)

    k_eff, mag_d, mag_t, n_turns, r_in, r_out, gap, c_i = x

    # Spring
    cfg.spring.k        = k_eff / cfg.spring.n_springs
    cfg.spring.__dict__ # force update

    # Magnet
    cfg.magnet.diameter   = float(mag_d)
    cfg.magnet.thickness  = float(mag_t)
    cfg.magnet.__post_init__()

    # Coil
    cfg.coil.n_turns          = int(round(n_turns))
    cfg.coil.coil_inner_radius = float(r_in)
    cfg.coil.coil_outer_radius = float(r_out)
    cfg.coil.air_gap           = float(gap)

    # Damping
    cfg.damping.c_internal = float(c_i)
    # c_load will be set to impedance-matched value inside evaluator

    return cfg


def config_to_vector(cfg: WECConfig) -> np.ndarray:
    """Extract a design vector from a WECConfig."""
    return np.array([
        cfg.spring.k_eff,
        cfg.magnet.diameter,
        cfg.magnet.thickness,
        float(cfg.coil.n_turns),
        cfg.coil.coil_inner_radius,
        cfg.coil.coil_outer_radius,
        cfg.coil.air_gap,
        cfg.damping.c_internal,
    ])


def normalized_to_si(x_norm: np.ndarray) -> np.ndarray:
    """Scale normalized [0,1] design vector to SI units."""
    x_si = np.zeros(N_PARAMS)
    for i, key in enumerate(PARAM_NAMES):
        lo, hi, _, _ = DESIGN_BOUNDS[key]
        x_si[i] = lo + x_norm[i] * (hi - lo)
    return x_si


def si_to_normalized(x_si: np.ndarray) -> np.ndarray:
    """Scale SI design vector to normalized [0,1]."""
    x_norm = np.zeros(N_PARAMS)
    for i, key in enumerate(PARAM_NAMES):
        lo, hi, _, _ = DESIGN_BOUNDS[key]
        x_norm[i] = (x_si[i] - lo) / (hi - lo)
    return x_norm


# ---------------------------------------------------------------------------
# Constraint functions
# ---------------------------------------------------------------------------
def check_constraints(cfg: WECConfig, x: np.ndarray) -> Dict[str, float]:
    """
    Evaluate all physical feasibility constraints.
    Positive value = satisfied. Negative = violated.

    Constraints:
        1. coil_outer_r > coil_inner_r + 5mm  (minimum coil width)
        2. magnet_radius < coil_outer_r        (magnet fits in coil footprint)
        3. air_gap > 0.5mm                     (mechanical clearance)
        4. f_n in [0.05, 1.5] Hz              (tunable frequency range)
        5. zeta_total < 2.0                    (not overdamped)
        6. magnet_thickness < magnet_diameter  (aspect ratio)
        7. coil_outer_r < plate_half_width     (coil fits on plate)
        8. |x_max| < plate_half_width          (magnet stays on plate)
    """
    k_eff, mag_d, mag_t, n_turns, r_in, r_out, gap, c_i = x

    m       = cfg.magnet.mass
    k       = cfg.spring.k_eff
    omega_n = np.sqrt(k / m) if m > 0 else 1e-6
    f_n     = omega_n / (2 * np.pi)

    # Impedance-matched c_L (best case)
    R_coil = cfg.coil.coil_resistance * cfg.coil.n_coils
    # Estimate Gamma_rms heuristically (will be updated in full eval)
    Gamma_est = 0.05   # Wb/m conservative estimate
    c_L = Gamma_est**2 / (2 * R_coil) if R_coil > 0 else c_i
    c_total = c_i + c_L
    zeta    = c_total / (2 * m * omega_n) if (m > 0 and omega_n > 0) else 99.0

    plate_hw = 0.10   # half-width of plate (m) — 20cm plate

    # Max displacement estimate: X = alpha_0 * |H(omega_wave)| * d
    alpha_0 = cfg.platform.alpha_0
    Omega   = cfg.wave.omega_peak
    F0      = alpha_0 * abs(cfg.platform.pivot_to_plate * Omega**2 - 9.81)
    denom   = max(abs(omega_n**2 - Omega**2), 2 * zeta * omega_n * Omega, 1e-6)
    x_max_est = F0 / denom

    c = {
        "coil_width_min":    (r_out - r_in) - 0.005,
        "magnet_fits_coil":  r_out - mag_d/2,
        "gap_clearance":     gap - 0.0005,
        "f_n_min":           f_n - 0.05,
        "f_n_max":           1.5 - f_n,
        "not_overdamped":    2.0 - zeta,
        "magnet_aspect":     mag_d - mag_t,
        "coil_fits_plate":   plate_hw - r_out,
        "magnet_on_plate":   plate_hw - x_max_est,
    }
    return c


# ---------------------------------------------------------------------------
# Fast frequency-domain power estimate
# ---------------------------------------------------------------------------
def fast_power_estimate(cfg: WECConfig, Gamma_rms: float = 0.05) -> Tuple[float, float]:
    """
    Quick P_avg estimate using frequency-domain transfer function.

    P_avg ≈ c_L * (omega_wave * X_rms)^2
    X_rms = alpha_0/sqrt(2) * |H(omega_wave)|

    Returns (P_avg_W, f_n_Hz)
    """
    m       = cfg.magnet.mass
    k       = cfg.spring.k_eff
    omega_n = np.sqrt(k / m)
    f_n     = omega_n / (2 * np.pi)

    # Compute c_L from coil geometry
    R_coil  = cfg.coil.coil_resistance * cfg.coil.n_coils
    c_L     = Gamma_rms**2 / (2 * R_coil) if R_coil > 0 else 0.05
    c_total = cfg.damping.c_internal + c_L
    zeta    = c_total / (2 * m * omega_n)

    Omega   = cfg.wave.omega_peak
    kappa   = cfg.platform.pivot_to_plate
    g       = 9.81
    alpha_0 = cfg.platform.alpha_0

    # Forcing amplitude (Section 13.2 of derivation)
    F0 = alpha_0 * abs(kappa * Omega**2 - g)

    # Transfer function magnitude at excitation frequency
    denom = np.sqrt((omega_n**2 - Omega**2)**2 + (2*zeta*omega_n*Omega)**2)
    X_amp = F0 / denom if denom > 0 else 0.0

    # RMS velocity: x_dot_rms = X_amp * Omega / sqrt(2)
    x_dot_rms = X_amp * Omega / np.sqrt(2)

    # Average load power
    P_avg = c_L * x_dot_rms**2

    return P_avg, f_n


# ---------------------------------------------------------------------------
# Full time-domain power evaluation
# ---------------------------------------------------------------------------
def full_power_evaluation(
    cfg: WECConfig,
    wave_mode: str = "jonswap",
    seed: int = 42,
) -> Tuple[float, float, Any, Any]:
    """
    Full time-domain evaluation: wave -> dynamics -> EM -> P_avg.

    Returns (P_avg_W, P_peak_W, dyn_result, em_result)
    """
    from wave_input import generate_wave_excitation, array_excitation
    from dynamics import solve_nonlinear, array_excitation as dyn_array_excitation
    from electromagnetics import (
        build_coil_geometry, compute_flux_profile,
        build_flux_interpolants, solve_electromagnetics, compute_c_L,
    )

    # Wave
    ws = generate_wave_excitation(cfg, mode=wave_mode, seed=seed)
    alpha_f, _, alpha_ddot_f = dyn_array_excitation(ws.t, ws.alpha)

    # Flux profile
    geom  = build_coil_geometry(cfg)
    x_est = cfg.platform.alpha_0 * cfg.platform.pivot_to_plate * 2
    x_arr, Phi_arr, Gamma_arr = compute_flux_profile(
        cfg, geom, x_range=(-x_est, x_est), n_points=150
    )
    _, gamma_f = build_flux_interpolants(x_arr, Phi_arr, Gamma_arr)
    Gamma_rms  = float(np.sqrt(np.mean(Gamma_arr**2)))

    # Update c_load from geometry
    cfg.damping.c_load = compute_c_L(cfg, Gamma_rms)

    # Dynamics
    dyn = solve_nonlinear(cfg, alpha_f, alpha_ddot_f, cfg.wave.t_eval)

    # EM
    em = solve_electromagnetics(cfg, dyn.t, dyn.x, dyn.x_dot, gamma_func=gamma_f)

    return em.P_avg, em.P_peak, dyn, em


# ---------------------------------------------------------------------------
# Main objective function
# ---------------------------------------------------------------------------
def objective(
    x: np.ndarray,
    base_cfg: WECConfig,
    mode: str = "fast",
    penalty_weight: float = 10.0,
    Gamma_rms: float = 0.05,
) -> float:
    """
    Scalar objective function for optimizer.

    Returns NEGATIVE P_avg (minimizers minimize, we want to maximize power).
    Infeasible designs receive a large positive penalty.

    Parameters
    ----------
    x             : design vector in SI units (length N_PARAMS)
    base_cfg      : base WECConfig (non-optimized params)
    mode          : "fast" | "full"
    penalty_weight: penalty multiplier for constraint violations
    Gamma_rms     : flux gradient estimate (Wb/m) for fast mode

    Returns
    -------
    obj : float — negative P_avg (W) + constraint penalty
    """
    # Clip to bounds
    x_clipped = _clip_to_bounds(x)

    # Build config
    try:
        cfg = vector_to_config(x_clipped, base_cfg)
    except Exception:
        return 1.0   # bad config

    # Check constraints
    constraints = check_constraints(cfg, x_clipped)
    penalty = 0.0
    for name, val in constraints.items():
        if val < 0:
            penalty += penalty_weight * abs(val)

    if penalty > 0.5:
        return penalty   # skip expensive evaluation for clearly infeasible designs

    # Evaluate power
    try:
        if mode == "fast":
            P_avg, _ = fast_power_estimate(cfg, Gamma_rms=Gamma_rms)
        else:
            P_avg, _, _, _ = full_power_evaluation(cfg)
    except Exception:
        return 1.0 + penalty

    return -P_avg + penalty   # negative because we minimize


def objective_result(
    x: np.ndarray,
    base_cfg: WECConfig,
    mode: str = "fast",
    Gamma_rms: float = 0.05,
) -> ObjectiveResult:
    """
    Same as objective() but returns the full ObjectiveResult dataclass
    for inspection and plotting.
    """
    x_clipped = _clip_to_bounds(x)
    cfg = vector_to_config(x_clipped, base_cfg)
    constraints = check_constraints(cfg, x_clipped)
    feasible = all(v >= 0 for v in constraints.values())

    params = dict(zip(PARAM_NAMES, x_clipped))

    m       = cfg.magnet.mass
    k       = cfg.spring.k_eff
    omega_n = np.sqrt(k / m)
    f_n     = omega_n / (2 * np.pi)
    Omega   = cfg.wave.omega_peak
    R_coil  = cfg.coil.coil_resistance * cfg.coil.n_coils
    c_L     = Gamma_rms**2 / (2 * R_coil) if R_coil > 0 else 0.05
    c_total = cfg.damping.c_internal + c_L
    zeta    = c_total / (2 * m * omega_n)
    zeta_L  = c_L / (2 * m * omega_n)

    em_result = dyn_result = None

    if mode == "fast":
        P_avg, _ = fast_power_estimate(cfg, Gamma_rms=Gamma_rms)
        P_peak   = P_avg * 3.0   # rough estimate
    else:
        P_avg, P_peak, dyn_result, em_result = full_power_evaluation(cfg)

    return ObjectiveResult(
        design_vector=x_clipped,
        params=params,
        P_avg_mW=P_avg * 1000,
        P_peak_mW=P_peak * 1000,
        f_n_Hz=f_n,
        zeta_total=zeta,
        zeta_load=zeta_L,
        freq_ratio=Omega / omega_n,
        constraints=constraints,
        feasible=feasible,
        em_result=em_result,
        dyn_result=dyn_result,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clip_to_bounds(x: np.ndarray) -> np.ndarray:
    """Hard-clip design vector to physical bounds."""
    x_out = x.copy()
    for i, key in enumerate(PARAM_NAMES):
        lo, hi, _, _ = DESIGN_BOUNDS[key]
        x_out[i] = np.clip(x_out[i], lo, hi)
    return x_out


def default_design_vector(cfg: WECConfig) -> np.ndarray:
    """Extract design vector from the default config."""
    return config_to_vector(cfg)


def print_design_bounds():
    """Print the design space bounds table."""
    print(f"\n{'Parameter':<20} {'Min':>10} {'Max':>10}  {'Units':<12} Description")
    print("-" * 72)
    for key, (lo, hi, desc, unit) in DESIGN_BOUNDS.items():
        print(f"  {key:<18} {lo:>10.4g} {hi:>10.4g}  {unit:<12} {desc}")


# ---------------------------------------------------------------------------
# Quick validation
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = WECConfig()
    x0  = default_design_vector(cfg)

    print("Design space:")
    print_design_bounds()

    print(f"\nDefault design vector:")
    for i, (name, val) in enumerate(zip(PARAM_NAMES, x0)):
        lo, hi, _, unit = DESIGN_BOUNDS[name]
        norm = (val - lo) / (hi - lo)
        print(f"  {name:<20} = {val:.4g} {unit}  (normalized: {norm:.3f})")

    print("\nEvaluating default design (fast mode)...")
    result = objective_result(x0, cfg, mode="fast")
    print(result.summary())

    print("\nConstraints:")
    for k, v in result.constraints.items():
        status = "OK" if v >= 0 else "VIOLATED"
        print(f"  {k:<25} = {v:+.4f}  [{status}]")