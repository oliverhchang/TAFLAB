"""
electromagnetics.py
-------------------
Electromagnetic model for the WEC coil-magnet system.

Converts magnet velocity x_dot(t) → EMF(t) → Current(t) → Power(t)

Two flux linkage models are provided:
    1. Analytical dipole model   — fast, good for sweeps and Optimization
    2. Lookup table model        — for importing ANSYS Maxwell FEM results

Key physics
-----------
Faraday's law:
    EMF(t) = dΦ/dt = (dΦ/dx) * dx/dt = Gamma(x) * x_dot(t)

where Gamma(x) = dΦ/dx is the "flux gradient" [Wb/m], the key
geometric coupling parameter.

For the coil array, total flux linkage is the sum over all coils:
    Phi_total(x) = sum_i  N_i * Phi_i(x - x_i)

where x_i is the center position of coil i.

Electromagnetic back-force (load damping force on magnet):
    F_em(t) = -Gamma(x) * I(t) = -(Gamma^2 / R_total) * x_dot(t)

This is what c_L represents in dynamics.py:
    c_L = Gamma^2 / R_total   (derived from circuit equations)

Circuit model
-------------
    [Coil array] --- bridge rectifier --- smoothing cap --- boost converter --- load

    EMF = Gamma * x_dot
    I   = EMF / (R_coil + R_load)
    P_load = I^2 * R_load = EMF^2 * R_load / (R_coil + R_load)^2

    Optimal load resistance: R_load = R_coil  (impedance matching)
    Max power: P_max = EMF^2 / (4 * R_coil)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable, Tuple
from scipy.interpolate import interp1d
from .config import WECConfig


# ---------------------------------------------------------------------------
# Output containers
# ---------------------------------------------------------------------------
@dataclass
class EMResult:
    """Full electromagnetic solution at every time step."""
    t: np.ndarray           # time (s)
    x: np.ndarray           # magnet position (m)
    x_dot: np.ndarray       # magnet velocity (m/s)
    Phi: np.ndarray         # total flux linkage (Wb)
    Gamma: np.ndarray       # flux gradient dPhi/dx (Wb/m)
    EMF: np.ndarray         # open-circuit EMF (V)
    I_load: np.ndarray      # load current (A)
    V_load: np.ndarray      # voltage across load (V)
    P_load: np.ndarray      # instantaneous load power (W)
    P_avg: float            # average load power (W), excl. transient
    P_peak: float           # peak load power (W)
    c_L_eff: np.ndarray     # effective load damping coefficient (N·s/m)


@dataclass
class CoilGeometry:
    """Positions and parameters of each coil in the array."""
    x_centers: np.ndarray   # x-positions of coil centers on plate (m)
    y_centers: np.ndarray   # y-positions (for 2D layout)
    inner_radii: np.ndarray
    outer_radii: np.ndarray
    n_turns: np.ndarray
    resistances: np.ndarray


# ---------------------------------------------------------------------------
# Analytical dipole flux model
# ---------------------------------------------------------------------------
def dipole_flux_through_coil(
    x_magnet: float,
    y_magnet: float,
    z_gap: float,
    coil_inner_r: float,
    coil_outer_r: float,
    n_turns: int,
    magnet_radius: float,
    magnet_thickness: float,
    B_remanence: float,
) -> float:
    """
    Analytical estimate of flux linkage through a single coil from a
    cylindrical magnet using the on-axis dipole approximation.

    The magnet is modeled as a magnetic dipole with moment:
        m = M * Volume = (B_r / mu_0) * pi * r_m^2 * t_m

    Flux through a coaxial circular loop at axial distance z and
    radial offset rho from magnet axis:
        Phi = (mu_0 * m) / (2) * [ z / (z^2 + a^2)^(3/2) ]  (on-axis)

    For off-axis positions, we use a radial correction factor.
    This is the "near-field" approximation valid for z > r_m.

    For the coil (annular region), integrate over radius:
        Phi_coil = integral_{r_inner}^{r_outer} B_z(r, z, rho) * 2*pi*r dr

    Simplified here using the mean radius approximation.

    Parameters
    ----------
    x_magnet, y_magnet : magnet center position relative to coil center (m)
    z_gap              : axial distance from magnet face to coil plane (m)
    coil_inner_r       : inner radius of coil (m)
    coil_outer_r       : outer radius of coil (m)
    n_turns            : number of turns
    magnet_radius      : magnet cylinder radius (m)
    magnet_thickness   : magnet cylinder thickness (m)
    B_remanence        : remanent flux density (T)

    Returns
    -------
    Phi : flux linkage (Wb) = N * flux_per_turn
    """
    mu_0 = 4 * np.pi * 1e-7

    # Radial offset of magnet from coil center
    rho = np.sqrt(x_magnet**2 + y_magnet**2)

    # Magnetic dipole moment (A·m²)
    M_volume = B_remanence / mu_0              # magnetization (A/m)
    V_magnet = np.pi * magnet_radius**2 * magnet_thickness
    m_dipole = M_volume * V_magnet

    # Effective axial distance (from magnet face center to coil)
    z_eff = z_gap + magnet_thickness / 2.0
    z_eff = max(z_eff, 1e-4)   # prevent singularity

    # Mean coil radius
    r_coil = (coil_inner_r + coil_outer_r) / 2.0

    # On-axis B_z at coil plane, integrated over coil area
    # For a dipole: B_z = (mu_0 * m / 4*pi) * (2*z^2 - r^2) / (z^2 + r^2)^(5/2)
    # Integrate over annular coil:
    r_vals = np.linspace(coil_inner_r, coil_outer_r, 50)
    B_z = (mu_0 * m_dipole / (4 * np.pi)) * (
        (2 * z_eff**2 - r_vals**2) / (z_eff**2 + r_vals**2)**(5/2)
    )

    # Radial offset correction: Gaussian decay with offset
    # (heuristic, captures the flux drop as magnet moves off-center)
    sigma_r = magnet_radius + r_coil
    radial_factor = np.exp(-0.5 * (rho / sigma_r)**2)

    # Flux per turn (integrate B_z * 2*pi*r dr)
    phi_per_turn = np.trapz(B_z * 2 * np.pi * r_vals, r_vals) * radial_factor

    # Total flux linkage
    Phi = n_turns * phi_per_turn

    return Phi


# ---------------------------------------------------------------------------
# Build coil geometry from config
# ---------------------------------------------------------------------------
def build_coil_geometry(cfg: WECConfig) -> CoilGeometry:
    """
    Lay out coil centers on the flat plate in a hex-pack pattern.
    The 1/6/10 array from the proposal is approximated here.
    """
    n_coils  = cfg.coil.n_coils
    spacing  = cfg.coil.coil_spacing

    # Generate hex-pack positions
    positions = _hex_pack_positions(n_coils, spacing)
    x_c = positions[:, 0]
    y_c = positions[:, 1]

    r_in  = np.full(n_coils, cfg.coil.coil_inner_radius)
    r_out = np.full(n_coils, cfg.coil.coil_outer_radius)
    turns = np.full(n_coils, cfg.coil.n_turns)
    R_c   = np.full(n_coils, cfg.coil.coil_resistance)

    return CoilGeometry(
        x_centers=x_c, y_centers=y_c,
        inner_radii=r_in, outer_radii=r_out,
        n_turns=turns, resistances=R_c,
    )


def _hex_pack_positions(n: int, spacing: float) -> np.ndarray:
    """Generate n positions in a hexagonal packing pattern."""
    positions = []
    # Start with center, then rings
    positions.append([0.0, 0.0])

    ring = 1
    while len(positions) < n:
        # Each ring has 6*ring positions
        for i in range(6 * ring):
            if len(positions) >= n:
                break
            angle = np.pi / 3 * (i // ring) + np.pi / (3 * ring) * (i % ring)
            r     = spacing * ring
            positions.append([r * np.cos(angle), r * np.sin(angle)])
        ring += 1

    return np.array(positions[:n])


# ---------------------------------------------------------------------------
# Compute flux linkage vs position (1D sweep)
# ---------------------------------------------------------------------------
def compute_flux_profile(
    cfg: WECConfig,
    geometry: CoilGeometry,
    x_range: Tuple[float, float] = (-0.1, 0.1),
    n_points: int = 300,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute total flux linkage Phi(x) and gradient Gamma(x) = dPhi/dx
    as the magnet slides along the x-axis (y=0) over the coil array.

    Parameters
    ----------
    cfg      : WECConfig
    geometry : CoilGeometry from build_coil_geometry()
    x_range  : (x_min, x_max) sweep range (m)
    n_points : number of x positions

    Returns
    -------
    x_arr   : position array (m)
    Phi_arr : total flux linkage (Wb)
    Gamma_arr: flux gradient dPhi/dx (Wb/m)
    """
    x_arr   = np.linspace(x_range[0], x_range[1], n_points)
    Phi_arr = np.zeros(n_points)

    z_gap = cfg.coil.air_gap

    for i, x_m in enumerate(x_arr):
        phi_total = 0.0
        for j in range(len(geometry.x_centers)):
            # Magnet position relative to this coil's center
            dx = x_m - geometry.x_centers[j]
            dy = 0.0  # assuming 1D motion for now
            phi_j = dipole_flux_through_coil(
                x_magnet=dx,
                y_magnet=dy,
                z_gap=z_gap,
                coil_inner_r=geometry.inner_radii[j],
                coil_outer_r=geometry.outer_radii[j],
                n_turns=int(geometry.n_turns[j]),
                magnet_radius=cfg.magnet.radius,
                magnet_thickness=cfg.magnet.thickness,
                B_remanence=cfg.magnet.B_remanence,
            )
            phi_total += phi_j
        Phi_arr[i] = phi_total

    # Numerical gradient
    Gamma_arr = np.gradient(Phi_arr, x_arr)

    return x_arr, Phi_arr, Gamma_arr


# ---------------------------------------------------------------------------
# Build callable interpolants for Phi and Gamma
# ---------------------------------------------------------------------------
def build_flux_interpolants(
    x_arr: np.ndarray,
    Phi_arr: np.ndarray,
    Gamma_arr: np.ndarray,
) -> Tuple[Callable, Callable]:
    """
    Build fast interpolant functions for Phi(x) and Gamma(x).

    Returns
    -------
    phi_func   : callable(x) -> Phi (Wb)
    gamma_func : callable(x) -> Gamma = dPhi/dx (Wb/m)
    """
    phi_func   = interp1d(x_arr, Phi_arr,   kind="cubic", fill_value="extrapolate")
    gamma_func = interp1d(x_arr, Gamma_arr, kind="cubic", fill_value="extrapolate")
    return phi_func, gamma_func


# ---------------------------------------------------------------------------
# Load flux profile from ANSYS Maxwell lookup table
# ---------------------------------------------------------------------------
def load_ansys_flux_table(
    filepath: str,
    x_col: int = 0,
    phi_col: int = 1,
) -> Tuple[Callable, Callable]:
    """
    Load flux linkage vs position from an ANSYS Maxwell CSV export.

    Expected CSV format:
        position_mm, flux_linkage_Wb
        -50.0,       0.0001
        ...

    Returns phi_func, gamma_func callables.
    """
    data     = np.loadtxt(filepath, delimiter=",", skiprows=1)
    x_mm     = data[:, x_col]
    Phi_data = data[:, phi_col]

    x_m       = x_mm / 1000.0   # convert mm -> m
    Gamma_data = np.gradient(Phi_data, x_m)

    phi_func   = interp1d(x_m, Phi_data,   kind="cubic", fill_value="extrapolate")
    gamma_func = interp1d(x_m, Gamma_data, kind="cubic", fill_value="extrapolate")
    return phi_func, gamma_func


# ---------------------------------------------------------------------------
# Circuit model: EMF -> Current -> Power
# ---------------------------------------------------------------------------
def compute_circuit(
    EMF: np.ndarray,
    R_coil_total: float,
    R_load: float,
    rectifier_drop: float = 0.7,
    converter_efficiency: float = 0.85,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple circuit model: coil source + load resistance.

    AC circuit (before rectification):
        I_ac = EMF / (R_coil + R_load)
        V_load_ac = I_ac * R_load

    After bridge rectifier (4 diode drops total, 2 in series):
        V_rect = |V_load_ac| - 2 * V_diode

    After boost converter:
        P_out = V_rect * I_rect * eta_converter

    Parameters
    ----------
    EMF                 : open-circuit EMF time series (V)
    R_coil_total        : total coil array resistance (Ω)
    R_load              : load resistance (Ω)
    rectifier_drop      : voltage drop per diode junction (V)
    converter_efficiency: boost converter efficiency (0–1)

    Returns
    -------
    I_load, V_load, P_load : time series arrays
    """
    # AC stage
    R_total = R_coil_total + R_load
    I_ac    = EMF / R_total
    V_ac    = I_ac * R_load

    # Rectification — half-wave approximation for low-voltage signals
    # Full bridge: 2 diodes in series per half cycle
    V_rect = np.maximum(np.abs(V_ac) - 2 * rectifier_drop, 0.0)
    I_rect = V_rect / R_load

    # Boost converter
    P_out = V_rect * I_rect * converter_efficiency

    return I_rect, V_rect, P_out


# ---------------------------------------------------------------------------
# Main: solve full EM from dynamics result
# ---------------------------------------------------------------------------
def solve_electromagnetics(
    cfg: WECConfig,
    t: np.ndarray,
    x: np.ndarray,
    x_dot: np.ndarray,
    gamma_func: Optional[Callable] = None,
    R_load: Optional[float] = None,
    use_lookup: bool = False,
    ansys_filepath: Optional[str] = None,
) -> EMResult:
    """
    Compute full electromagnetic solution from magnet trajectory.

    Parameters
    ----------
    cfg          : WECConfig
    t, x, x_dot : dynamics result arrays
    gamma_func   : if provided, use this Gamma(x) interpolant
                   if None, compute analytically
    R_load       : load resistance (Ω); if None, use impedance-matched value
    use_lookup   : if True, load from ANSYS CSV (requires ansys_filepath)
    ansys_filepath: path to ANSYS Maxwell CSV export

    Returns
    -------
    EMResult
    """

    # --- Build flux interpolants if not provided ---
    if gamma_func is None:
        if use_lookup and ansys_filepath:
            phi_func, gamma_func = load_ansys_flux_table(ansys_filepath)
        else:
            geom = build_coil_geometry(cfg)
            x_range = (np.min(x) - 0.02, np.max(x) + 0.02)
            x_arr, Phi_arr, Gamma_arr = compute_flux_profile(
                cfg, geom, x_range=x_range
            )
            phi_func, gamma_func = build_flux_interpolants(x_arr, Phi_arr, Gamma_arr)

    # --- Evaluate Gamma(x) and Phi(x) at trajectory ---
    Gamma = gamma_func(x)
    try:
        Phi = phi_func(x)
    except Exception:
        Phi = np.zeros_like(x)

    # --- EMF = Gamma(x) * x_dot ---
    EMF = Gamma * x_dot

    # --- Circuit parameters ---
    # Total coil resistance (series or parallel depending on wiring)
    # Default: coils in series for voltage boost
    R_coil_total = cfg.coil.coil_resistance * cfg.coil.n_coils

    if R_load is None:
        R_load = R_coil_total   # impedance matching

    # --- Circuit solution ---
    I_load, V_load, P_load = compute_circuit(
        EMF, R_coil_total, R_load
    )

    # --- Effective load damping: c_L = Gamma^2 / R_total ---
    R_total = R_coil_total + R_load
    c_L_eff = Gamma**2 / R_total

    # --- Power statistics (skip first 20% transient) ---
    n_skip = int(0.2 * len(t))
    P_avg  = float(np.mean(P_load[n_skip:]))
    P_peak = float(np.max(P_load[n_skip:]))

    return EMResult(
        t=t, x=x, x_dot=x_dot,
        Phi=Phi, Gamma=Gamma,
        EMF=EMF, I_load=I_load, V_load=V_load,
        P_load=P_load, P_avg=P_avg, P_peak=P_peak,
        c_L_eff=c_L_eff,
    )


# ---------------------------------------------------------------------------
# Derived load damping coefficient from coil geometry
# ---------------------------------------------------------------------------
def compute_c_L(
    cfg: WECConfig,
    Gamma_rms: float,
    R_load: Optional[float] = None,
) -> float:
    """
    Compute the effective load damping coefficient c_L.

    c_L = Gamma_rms^2 / (R_coil + R_load)

    This links the electromagnetic model back to dynamics.py:
    use this value as cfg.damping.c_load.

    Parameters
    ----------
    Gamma_rms : RMS flux gradient over the magnet's expected travel (Wb/m)
    R_load    : load resistance; defaults to impedance-matched R_coil
    """
    R_coil = cfg.coil.coil_resistance * cfg.coil.n_coils
    if R_load is None:
        R_load = R_coil
    c_L = Gamma_rms**2 / (R_coil + R_load)
    return c_L


# ---------------------------------------------------------------------------
# Optimal load resistance
# ---------------------------------------------------------------------------
def optimal_load_resistance(cfg: WECConfig) -> float:
    """
    Optimal load resistance for maximum power transfer.
    For purely resistive circuit: R_load_opt = R_coil_total
    """
    return cfg.coil.coil_resistance * cfg.coil.n_coils


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_flux_profile(
    x_arr: np.ndarray,
    Phi_arr: np.ndarray,
    Gamma_arr: np.ndarray,
    cfg: WECConfig,
    save_path: Optional[str] = None,
):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(x_arr * 1000, Phi_arr * 1000, color="steelblue", lw=2)
    axes[0].set_ylabel("Flux Linkage Φ (mWb)", fontsize=11)
    axes[0].set_title("Flux Linkage and Gradient vs. Magnet Position",
                      fontsize=13, fontweight="bold")
    axes[0].grid(True, alpha=0.4)

    axes[1].plot(x_arr * 1000, Gamma_arr, color="crimson", lw=2)
    axes[1].axhline(0, color="black", lw=0.8, linestyle="--")
    axes[1].set_ylabel("Flux Gradient Γ = dΦ/dx (Wb/m)", fontsize=11)
    axes[1].set_xlabel("Magnet Position x (mm)", fontsize=11)
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


def plot_em_result(em: EMResult, save_path: Optional[str] = None):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)

    axes[0].plot(em.t, em.x * 1000, color="steelblue", lw=1.5)
    axes[0].set_ylabel("x (mm)"); axes[0].grid(True, alpha=0.4)
    axes[0].set_title("Electromagnetic Power Generation", fontsize=13, fontweight="bold")

    axes[1].plot(em.t, em.EMF, color="darkorange", lw=1.5, label="EMF")
    axes[1].plot(em.t, em.V_load, color="green", lw=1.2,
                 linestyle="--", label="V_load")
    axes[1].set_ylabel("Voltage (V)"); axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.4)

    axes[2].plot(em.t, em.I_load * 1000, color="mediumpurple", lw=1.5)
    axes[2].set_ylabel("Current (mA)"); axes[2].grid(True, alpha=0.4)

    axes[3].plot(em.t, em.P_load * 1000, color="crimson", lw=1.2,
                 label="P_load(t)", alpha=0.7)
    axes[3].axhline(em.P_avg * 1000, color="crimson", lw=2, linestyle="--",
                    label=f"P_avg = {em.P_avg*1000:.2f} mW")
    axes[3].axhline(em.P_peak * 1000, color="darkred", lw=1, linestyle=":",
                    label=f"P_peak = {em.P_peak*1000:.2f} mW")
    axes[3].set_ylabel("Power (mW)"); axes[3].set_xlabel("Time (s)")
    axes[3].legend(fontsize=9); axes[3].grid(True, alpha=0.4)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Quick validation
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from dynamics import solve_nonlinear, sinusoidal_excitation

    cfg = WECConfig()
    print(cfg.summary())

    # Build coil geometry
    print("\n--- Building coil geometry ---")
    geom = build_coil_geometry(cfg)
    print(f"  {len(geom.x_centers)} coils placed")
    print(f"  Coil resistance: {cfg.coil.coil_resistance:.2f} Ω each")
    print(f"  Total resistance (series): {cfg.coil.coil_resistance * cfg.coil.n_coils:.2f} Ω")

    # Compute flux profile
    print("\n--- Computing flux profile ---")
    x_arr, Phi_arr, Gamma_arr = compute_flux_profile(
        cfg, geom, x_range=(-0.08, 0.08), n_points=200
    )
    print(f"  Max |Phi| = {np.max(np.abs(Phi_arr))*1000:.4f} mWb")
    print(f"  Max |Gamma| = {np.max(np.abs(Gamma_arr)):.4f} Wb/m")
    print(f"  RMS Gamma = {np.sqrt(np.mean(Gamma_arr**2)):.4f} Wb/m")

    plot_flux_profile(x_arr, Phi_arr, Gamma_arr, cfg,
                      save_path="flux_profile.png")

    # Run dynamics then EM
    print("\n--- Running dynamics + EM pipeline ---")
    t     = cfg.wave.t_eval
    Omega = cfg.wave.omega_peak
    cfg.platform.Omega = Omega

    alpha_f, _, alpha_ddot_f = sinusoidal_excitation(cfg.platform.alpha_0, Omega)
    dyn = solve_nonlinear(cfg, alpha_f, alpha_ddot_f, t)

    # Effective c_L from geometry
    Gamma_rms = np.sqrt(np.mean(Gamma_arr**2))
    c_L = compute_c_L(cfg, Gamma_rms)
    print(f"  Computed c_L from geometry = {c_L:.4f} N·s/m")

    # EM solution
    phi_f, gamma_f = build_flux_interpolants(x_arr, Phi_arr, Gamma_arr)
    em = solve_electromagnetics(cfg, dyn.t, dyn.x, dyn.x_dot, gamma_func=gamma_f)

    print(f"\n  P_avg  = {em.P_avg*1000:.3f} mW")
    print(f"  P_peak = {em.P_peak*1000:.3f} mW")
    print(f"  EMF_rms = {np.sqrt(np.mean(em.EMF**2)):.4f} V")

    plot_em_result(em, save_path="em_result.png")

    # Optimal load resistance
    R_opt = optimal_load_resistance(cfg)
    print(f"\n  Optimal load resistance: {R_opt:.2f} Ω")