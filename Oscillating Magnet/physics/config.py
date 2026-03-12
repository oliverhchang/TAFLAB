"""
config.py
---------
All physical parameters for the WEC system in one place.
Edit values here; nothing else should contain magic numbers.
"""

from dataclasses import dataclass, field
import numpy as np


# ---------------------------------------------------------------------------
# Platform / Buoy
# ---------------------------------------------------------------------------
@dataclass
class PlatformConfig:
    """Describes the buoy / autonomous boat platform."""

    # Pivot-to-plate distance (m) — distance from boat rotation center
    # to the surface the magnet slides on
    pivot_to_plate: float = 0.15          # d in derivation (m)

    # Peak tilt amplitude under typical wave excitation (rad)
    alpha_0: float = np.radians(10.0)     # ~10 deg typical ocean wave tilt

    # Wave excitation frequency (rad/s) — overridden by WaveConfig
    # but useful for quick single-frequency tests
    Omega: float = 2 * np.pi * 0.5      # 0.5 Hz default ocean wave


# ---------------------------------------------------------------------------
# Magnet (sliding mass)
# ---------------------------------------------------------------------------
@dataclass
class MagnetConfig:
    """
    Cylindrical NdFeB magnet acting as the sliding proof mass.
    Default: largest commercially available ~40 x 20 mm cylinder.
    """

    # Geometry
    diameter: float = 0.040              # m  (40 mm)
    thickness: float = 0.020             # m  (20 mm)
    radius: float = field(init=False)

    # Material
    density: float = 7500.0             # kg/m³  (NdFeB typical)
    mass: float = field(init=False)

    # Magnetic properties (N42 NdFeB)
    B_remanence: float = 1.32           # T  (remanent flux density)
    grade: str = "N42"

    def __post_init__(self):
        self.radius = self.diameter / 2
        volume = np.pi * self.radius**2 * self.thickness
        self.mass = self.density * volume  # ~0.188 kg for default dims


# ---------------------------------------------------------------------------
# Spring (frequency tuning)
# ---------------------------------------------------------------------------
@dataclass
class SpringConfig:
    """
    Spring restoring force on the magnet.
    Natural frequency: omega_n = sqrt(k_eff / m_eff)
    Target: match dominant ocean wave frequency ~0.1–1 Hz
    """

    # Spring constant (N/m) — primary tuning parameter
    k: float = 5.0                      # N/m  (tune to match wave freq)

    # Number of springs (affects effective k)
    n_springs: int = 4

    @property
    def k_eff(self) -> float:
        """Effective spring constant from all springs in parallel."""
        return self.k * self.n_springs


# ---------------------------------------------------------------------------
# Coil Array
# ---------------------------------------------------------------------------
@dataclass
class CoilConfig:
    """
    Flat coil array beneath the sliding magnet.
    Geometry motivated by 1/6/10 array pattern from proposal.
    """

    # Individual coil geometry
    coil_inner_radius: float = 0.008     # m  (8 mm)
    coil_outer_radius: float = 0.020     # m  (20 mm)
    n_turns: int = 500                   # turns per coil
    wire_gauge_awg: int = 28             # AWG

    # Coil array layout
    n_coils: int = 10                    # total coils in array
    coil_spacing: float = 0.025          # center-to-center (m)

    # Air gap between magnet bottom face and coil top surface
    air_gap: float = 0.003               # m  (3 mm — from ANSYS sim)

    # Electrical properties (copper, AWG 28 ≈ 0.32 mm dia)
    wire_resistivity: float = 1.68e-8    # Ω·m
    wire_diameter: float = 0.00032       # m

    @property
    def mean_coil_radius(self) -> float:
        return (self.coil_inner_radius + self.coil_outer_radius) / 2

    @property
    def coil_resistance(self) -> float:
        """Approximate DC resistance per coil (Ω)."""
        mean_circumference = 2 * np.pi * self.mean_coil_radius
        total_length = mean_circumference * self.n_turns
        wire_area = np.pi * (self.wire_diameter / 2) ** 2
        return self.wire_resistivity * total_length / wire_area


# ---------------------------------------------------------------------------
# Damping
# ---------------------------------------------------------------------------
@dataclass
class DampingConfig:
    """
    Damping coefficients for the magnet sliding motion.
    c_i : internal (friction, eddy currents)
    c_L : load (electromagnetic energy extraction)

    Power extracted = c_L * x_dot^2
    Optimal load: c_L = c_i  (impedance matching)
    """

    c_internal: float = 0.05            # N·s/m  (friction + internal losses)
    c_load: float = 0.05                # N·s/m  (start at impedance-matched)

    @property
    def c_total(self) -> float:
        return self.c_internal + self.c_load


# ---------------------------------------------------------------------------
# Wave Input
# ---------------------------------------------------------------------------
@dataclass
class WaveConfig:
    """
    Ocean wave excitation parameters.
    Stochastic input modeled as JONSWAP spectrum.
    Deterministic sinusoid available for validation.
    """

    # Dominant wave period (s)
    T_peak: float = 8.0                 # s  (typical open ocean)

    @property
    def f_peak(self) -> float:          # Hz
        return 1.0 / self.T_peak

    @property
    def omega_peak(self) -> float:      # rad/s
        return 2 * np.pi * self.f_peak

    # JONSWAP parameters
    significant_wave_height: float = 1.5  # H_s (m)
    gamma_jonswap: float = 3.3            # peak enhancement factor

    # Simulation time
    t_start: float = 0.0
    t_end: float = 120.0                # s  (2 minutes)
    dt: float = 0.02                    # s  (50 Hz sample rate)

    @property
    def t_span(self):
        return (self.t_start, self.t_end)

    @property
    def t_eval(self):
        return np.arange(self.t_start, self.t_end, self.dt)


# ---------------------------------------------------------------------------
# Master Config — single object passed around the entire pipeline
# ---------------------------------------------------------------------------
@dataclass
class WECConfig:
    """
    Top-level configuration object.
    Pass this single object to every module.
    """

    platform: PlatformConfig = field(default_factory=PlatformConfig)
    magnet:   MagnetConfig   = field(default_factory=MagnetConfig)
    spring:   SpringConfig   = field(default_factory=SpringConfig)
    coil:     CoilConfig     = field(default_factory=CoilConfig)
    damping:  DampingConfig  = field(default_factory=DampingConfig)
    wave:     WaveConfig     = field(default_factory=WaveConfig)

    def __post_init__(self):
        # Resolve magnet mass after MagnetConfig post_init runs
        self.magnet.__post_init__()

    def summary(self) -> str:
        m = self.magnet.mass
        k = self.spring.k_eff
        omega_n = np.sqrt(k / m)
        f_n = omega_n / (2 * np.pi)
        zeta = self.damping.c_total / (2 * m * omega_n)

        lines = [
            "=" * 50,
            "WEC System Configuration Summary",
            "=" * 50,
            f"  Magnet mass        : {m*1000:.1f} g",
            f"  Effective spring k : {k:.2f} N/m",
            f"  Natural frequency  : {f_n:.4f} Hz  ({omega_n:.4f} rad/s)",
            f"  Wave peak freq     : {self.wave.f_peak:.4f} Hz",
            f"  Freq ratio Ω/ωn    : {self.wave.omega_peak/omega_n:.3f}",
            f"  Total damping ζ    : {zeta:.4f}",
            f"  Coil resistance    : {self.coil.coil_resistance:.2f} Ω",
            f"  Air gap            : {self.coil.air_gap*1000:.1f} mm",
            "=" * 50,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quick validation
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = WECConfig()
    print(cfg.summary())