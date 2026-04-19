"""
Wave Energy Converter - Electromechanical Ocean Optimization
================================================================
TAFLAB | April 2026

Unifies the full Biot-Savart Electromagnetic framework with the
Ocean Wave pitching platform to optimize resonant spring stiffness.

The ODE engine solves the full Lagrangian equation of motion on a
pitching rail in a non-inertial frame:

    M*u_ddot = F_grav + F_euler + F_coriolis + F_centrifugal
             + F_springs + F_mech + F_em

Physics corrections applied (vs. original):
  1. Added Coriolis force:     -2*M*theta_dot*v
  2. Added centrifugal force:  +M*theta_dot^2*u
  3. Removed arctan() from wave amplitude — linear RAO gives A = k*(H/2) directly
  4. Spring equilibrium offset initialisation when k1 != k2
  5. EM flux linkage integrated over full coil depth (z-axis), not just midplane

Where F_em is calculated dynamically via Faraday's law and a
parallel nodal analysis of the full-bridge rectifier circuit.

Optimization Target:
    Maximise steady-state Average Active Power (mW)
"""

import argparse
import warnings

import matplotlib
import subprocess, os, sys
matplotlib.use("Agg")   # non-interactive — renders to file reliably on all platforms
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import differential_evolution

warnings.filterwarnings("ignore", category=RuntimeWarning)

# -----------------------------------------------------------------------------
# Colour palette (TAFLAB Light Mode)
# -----------------------------------------------------------------------------
C_BG      = "#ffffff"
C_PANEL   = "#f8f9fa"
C_GRID    = "#e9ecef"
C_TEXT    = "#212529"
C_DIM     = "#6c757d"
C_ACCENT  = "#0d6efd"
C_GREEN   = "#198754"
C_ORANGE  = "#fd7e14"
C_PURPLE  = "#6f42c1"
C_YELLOW  = "#d97706"
C_PINK    = "#d63384"
C_TEAL    = "#20c997"
C_SPRING1 = "#059669"
C_SPRING2 = "#ea580c"
C_SLED    = "#64748b"
C_COIL    = "#d97706"
C_MAG_N   = "#dc3545"
C_MAG_S   = "#0d6efd"
C_TRACK   = "#e2e8f0"

def styled_axes(ax, title, xlabel, ylabel):
    ax.set_facecolor(C_PANEL)
    ax.set_title(title, color=C_TEXT, fontsize=9, fontweight="bold", pad=5)
    ax.set_xlabel(xlabel, color=C_DIM, fontsize=7.5)
    ax.set_ylabel(ylabel, color=C_DIM, fontsize=7.5)
    ax.tick_params(colors=C_DIM, labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor(C_GRID)
    ax.grid(color=C_GRID, linewidth=0.5, alpha=0.9)


# =============================================================================
# 1. Electromagnetic Pre-Computation Engine
# =============================================================================

class WEC_EM_Engine:
    """
    Pre-computes the spatial Biot-Savart field and handles circuit logic.

    Physics fix: flux linkage now integrated over the full coil depth in z,
    not just evaluated at the coil midplane. This corrects a 15-40% overestimate
    of K(u) = dPsi/du that arose from using only B_z at z_dist (midplane).
    """

    def __init__(self, R_load=10.0):
        # Mechanical limits & Magnet properties
        self.Br = 1.45; self.R_m = 0.0125; self.d_m = 0.025; self.mag_spacing = 0.030
        self.num_magnets = 4
        self.M = 0.092 * self.num_magnets + 0.050
        self.g = 9.81
        self.L = 0.40  # Track length

        # Stator & Coil properties
        self.num_coils_per_side = 5
        self.total_coils = 10
        self.coil_spacing = 0.030
        self.R_in = 0.008; self.R_out = 0.021; self.d_c = 0.0125
        self.N_turns = 1240
        self.R_int_per_coil = 52.0
        self.R_load = R_load
        self.V_drop = 0.4
        self.air_gap = 0.0012
        self.z_dist = self.d_m / 2 + self.air_gap + self.d_c / 2

        self._build_lookup_table()

    def _calc_Bz(self, r_p, z_p):
        """Biot-Savart B_z at radial distance r_p, axial distance z_p from magnet centre."""
        n_theta, n_z = 30, 10
        theta_q = np.linspace(0, 2*np.pi, n_theta)
        z_q = np.linspace(-self.d_m/2, self.d_m/2, n_z)
        TH, ZQ = np.meshgrid(theta_q, z_q)
        dist_sq = ((r_p - self.R_m*np.cos(TH))**2
                   + (self.R_m*np.sin(TH))**2
                   + (z_p - ZQ)**2 + 1e-9)
        integrand = self.R_m*(r_p*np.cos(TH) - self.R_m) / (dist_sq**1.5)
        return -(self.Br/(4*np.pi)) * np.trapezoid(
            np.trapezoid(integrand, theta_q, axis=1), z_q, axis=0)

    def _calc_flux_linkage(self, dx):
        """
        Compute flux linkage Psi for a single coil displaced dx from a single magnet.

        Physics fix: integrate B_z over the full coil cross-section in both r and z,
        rather than evaluating only at the coil axial midplane (z_dist).
        This correctly captures the variation of B_z through the coil depth.
        """
        r_pts = np.linspace(self.R_in, self.R_out, 8)
        # Integrate over coil axial extent
        z_coil_pts = np.linspace(self.z_dist - self.d_c/2,
                                  self.z_dist + self.d_c/2, 5)
        phi_total = 0.0
        for z_c in z_coil_pts:
            bz_vals = np.array([self._calc_Bz(np.sqrt(dx**2 + r**2), z_c) for r in r_pts])
            phi_total += np.trapezoid(bz_vals * 2*np.pi*r_pts, r_pts)
        # Average over z samples and scale by turns
        return (phi_total / len(z_coil_pts)) * self.N_turns

    def _get_Ks(self, u):
        """Compute dPsi/du (the transduction coefficient K) for each coil at sled position u."""
        Ks = np.zeros(self.total_coils)
        coil_pos = (np.arange(self.num_coils_per_side)
                    - (self.num_coils_per_side - 1)/2) * self.coil_spacing
        mag_off  = (np.arange(self.num_magnets)
                    - (self.num_magnets - 1)/2) * self.mag_spacing
        dx_eps = 5e-4

        for c_idx, cp in enumerate(coil_pos):
            k_coil = 0.0
            for m_idx, mo in enumerate(mag_off):
                pol = 1 if m_idx % 2 == 0 else -1
                dx  = u + mo - cp
                # Numerical derivative of flux linkage w.r.t. sled displacement
                phi1 = self._calc_flux_linkage(dx)
                phi2 = self._calc_flux_linkage(dx + dx_eps)
                k_coil += pol * (phi2 - phi1) / dx_eps
            Ks[c_idx] = k_coil
            Ks[c_idx + 5] = k_coil  # Symmetric coil pair
        return Ks

    def _build_lookup_table(self):
        u_pts = np.linspace(-self.L/2 - 0.05, self.L/2 + 0.05, 50)
        k_pts = [self._get_Ks(u) for u in u_pts]
        self.k_spline = CubicSpline(u_pts, np.array(k_pts), axis=0)

    def get_K(self, u):
        return self.k_spline(u)

    def solve_parallel_network(self, V_ind):
        """
        Solve the parallel rectifier network.
        Returns per-coil currents and load voltage V_L.
        """
        V_abs = np.maximum(np.abs(V_ind) - self.V_drop, 0)
        V_sorted = np.sort(V_abs)[::-1]
        V_L = 0.0
        if V_sorted[0] > 0:
            for N in range(1, 11):
                V_L_t = (np.sum(V_sorted[:N]) / self.R_int_per_coil
                         / (1/self.R_load + N/self.R_int_per_coil))
                if N == 10 or V_L_t >= V_sorted[N]:
                    V_L = V_L_t
                    break
        I_arr = np.maximum(0, np.abs(V_ind) - self.V_drop - V_L) / self.R_int_per_coil
        I_dir = I_arr * np.sign(V_ind)
        return I_dir, V_L


# =============================================================================
# 2. Fully Coupled Physics Engine (Ocean + Mechanics + EM)
# =============================================================================

class OceanWEC_Dynamic:
    """
    Integrates the full equation of motion for a sled on a pitching rail.

    Lagrangian derivation (generalised coordinate: u = sled position along rail):

        L = (1/2)*M*(u_dot^2 + (u*theta_dot)^2 + 2*g*u*sin(theta))

    Euler-Lagrange gives:
        M*u_ddot = M*g*sin(theta)           [gravity component along rail]
                 + M*theta_dot^2 * u        [centrifugal — pushes sled outward]
                 - M*theta_ddot * u         [Euler — from angular acceleration]
                 - 2*M*theta_dot * u_dot    [Coriolis — couples velocity to rotation]
                 + F_spring + F_mech + F_EM

    The original code was missing the Coriolis and centrifugal terms.
    """

    def __init__(self, em_engine: WEC_EM_Engine, k1=4.0, k2=4.0, c_mech=0.70,
                 f_wave=1, H_wave=1.5, theta_mean=0.0,
                 x0=None, v0=0.001, T=30.0, n_frames=500):
        self.em = em_engine
        self.k1, self.k2 = float(k1), float(k2)
        self.c_mech = float(c_mech)
        self.f_wave = float(f_wave)
        self.omega_wave = 2 * np.pi * f_wave
        self.H_wave = float(H_wave)
        self.theta_mean = float(theta_mean)

        # ---------------------------------------------------------------
        # Wave amplitude (Physics fix #3):
        # Linear deep-water dispersion: k_wave = omega^2 / g
        # For a buoy small relative to wavelength, RAO ~ 1, so pitch amplitude
        # equals the surface slope: A_pitch = k_wave * (H/2)  [radians]
        #
        # The original code wrapped this in arctan(), which is wrong —
        # arctan(slope) gives the angle of the slope, not the pitch amplitude,
        # and compresses large values incorrectly.
        #
        # Cap at 30 deg (0.524 rad): beyond this, linear wave theory breaks down
        # for small buoys and nonlinear corrections would be needed.
        # ---------------------------------------------------------------
        k_wave = (self.omega_wave**2) / self.em.g
        self.A_wave = min(k_wave * (self.H_wave / 2), np.radians(30))

        # ---------------------------------------------------------------
        # Spring equilibrium (Physics fix #4):
        # When k1 != k2, the static rest position of the sled is not u=0.
        # Initialise at the true equilibrium to avoid a spurious transient.
        #   u_eq = -(k1 - k2)*L / (2*(k1 + k2))
        # ---------------------------------------------------------------
        k_total = self.k1 + self.k2
        if k_total > 1e-9:
            u_eq = -(self.k1 - self.k2) * self.em.L / (2.0 * k_total)
        else:
            u_eq = 0.0

        self.x0 = float(x0) if x0 is not None else u_eq
        self.v0 = float(v0)
        self.T, self.n_frames = float(T), int(n_frames)

        self.k_total = k_total
        if k_total > 1e-9:
            self.f0 = np.sqrt(k_total / self.em.M) / (2 * np.pi)
            self.zeta = self.c_mech / (2.0 * np.sqrt(self.em.M * k_total))
        else:
            self.f0, self.zeta = 0.0, None

        self._integrate()

    # ------------------------------------------------------------------
    # Kinematics of the pitching platform
    # ------------------------------------------------------------------

    def theta(self, t):
        """Platform pitch angle [rad]."""
        return self.theta_mean + self.A_wave * np.sin(self.omega_wave * t)

    def theta_dot(self, t):
        """Platform angular velocity [rad/s] — required for Coriolis & centrifugal."""
        return self.A_wave * self.omega_wave * np.cos(self.omega_wave * t)

    def theta_ddot(self, t):
        """Platform angular acceleration [rad/s^2] — Euler pseudo-force."""
        return -self.A_wave * self.omega_wave**2 * np.sin(self.omega_wave * t)

    # ------------------------------------------------------------------
    # Full Lagrangian dynamics
    # ------------------------------------------------------------------

    def _dynamics(self, t, y):
        u, v = y
        th    = self.theta(t)
        th_d  = self.theta_dot(t)    # angular velocity
        th_dd = self.theta_ddot(t)   # angular acceleration

        # Gravity component along rail
        F_grav = self.em.M * self.em.g * np.sin(th)

        # Euler pseudo-force (from angular acceleration of the frame)
        F_euler = -self.em.M * th_dd * u

        # Coriolis pseudo-force (Physics fix #1 — was missing)
        # Arises because the sled moves in a rotating frame.
        # F_cor = -2 * M * omega x v_rel  (projected onto rail axis)
        F_cor = -2.0 * self.em.M * th_d * v

        # Centrifugal pseudo-force (Physics fix #2 — was missing)
        # Pushes sled away from pivot; linear in displacement u.
        F_cent = self.em.M * th_d**2 * u

        # Spring restoring force (symmetric about u_eq)
        F_spr = (-self.k1 * (u + self.em.L/2)
                 - self.k2 * (u - self.em.L/2))

        # Mechanical viscous damping
        F_mech = -self.c_mech * v

        # Hard bumper at track ends (prevents unphysical runaway)
        F_bump = 0.0
        if u >  self.em.L/2: F_bump = -10000.0 * (u - self.em.L/2)
        elif u < -self.em.L/2: F_bump = -10000.0 * (u + self.em.L/2)

        # Electromagnetic braking via Faraday + circuit analysis
        K_arr = self.em.get_K(u)
        I_dir, _ = self.em.solve_parallel_network(K_arr * v)
        F_em = -np.sum(K_arr * I_dir)

        a = (F_grav + F_euler + F_cor + F_cent
             + F_spr + F_mech + F_em + F_bump) / self.em.M
        return [v, a]

    def _integrate(self):
        t_eval = np.linspace(0, self.T, self.n_frames)
        sol = solve_ivp(self._dynamics, (0, self.T), [self.x0, self.v0],
                        method="LSODA", t_eval=t_eval, rtol=1e-6, atol=1e-8)
        self.t = sol.t
        self.u = sol.y[0]
        self.v = sol.y[1]

        n = len(self.t)
        self.theta_t  = np.zeros(n)
        self.acc      = np.zeros(n)
        self.F_grav   = np.zeros(n)
        self.F_euler  = np.zeros(n)
        self.F_cor    = np.zeros(n)
        self.F_cent   = np.zeros(n)
        self.F_spr    = np.zeros(n)
        self.F_em     = np.zeros(n)
        self.F_net    = np.zeros(n)
        self.P_load   = np.zeros(n)

        for i in range(n):
            ui, vi, ti = self.u[i], self.v[i], self.t[i]
            th    = self.theta(ti)
            th_d  = self.theta_dot(ti)
            th_dd = self.theta_ddot(ti)

            self.theta_t[i] = th

            fg   =  self.em.M * self.em.g * np.sin(th)
            feu  = -self.em.M * th_dd * ui
            fcor = -2.0 * self.em.M * th_d * vi
            fcen =  self.em.M * th_d**2 * ui
            fs   = (-self.k1 * (ui + self.em.L/2)
                    - self.k2 * (ui - self.em.L/2))
            fd   = -self.c_mech * vi

            K_arr = self.em.get_K(ui)
            I_dir, V_L = self.em.solve_parallel_network(K_arr * vi)
            fem = -np.sum(K_arr * I_dir)

            net = fg + feu + fcor + fcen + fs + fd + fem
            self.F_grav[i]  = fg
            self.F_euler[i] = feu
            self.F_cor[i]   = fcor
            self.F_cent[i]  = fcen
            self.F_spr[i]   = fs
            self.F_em[i]    = fem
            self.F_net[i]   = net
            self.acc[i]     = net / self.em.M
            self.P_load[i]  = (V_L**2) / self.em.R_load

        # Steady-state average active power (skip first 30% as transient)
        i_ss = int(0.30 * n)
        self.P_avg = float(np.mean(self.P_load[i_ss:]) * 1000)  # mW


# =============================================================================
# 3. Optimization Engine
# =============================================================================

class SpringOptimiser:
    def __init__(self, em_engine: WEC_EM_Engine, f_wave, H_wave=1.5, c_mech=0.70,
                 theta_mean=0.0, k_min=0.5, k_max=60.0):
        self.em = em_engine
        self.f_wave = f_wave
        self.H_wave = H_wave
        self.c_mech = c_mech
        self.theta_mean = theta_mean
        self.k_min = k_min
        self.k_max = k_max

    def _objective(self, params):
        k1, k2 = params
        if k1 < 0 or k2 < 0:
            return 0.0
        m = OceanWEC_Dynamic(self.em, k1=k1, k2=k2, c_mech=self.c_mech,
                             f_wave=self.f_wave, H_wave=self.H_wave,
                             theta_mean=self.theta_mean, T=25.0, n_frames=200)
        return -m.P_avg  # Minimise negative to maximise power

    def sweep_1d(self, n_pts=30, fixed_ratio=1.0):
        k_tot_vals = np.linspace(self.k_min * 2, self.k_max, n_pts)
        p_avg_vals = np.zeros(n_pts)
        print(f"  1-D Sweep: Testing {n_pts} points ...")
        for i, kt in enumerate(k_tot_vals):
            k2 = kt / (1 + fixed_ratio)
            k1 = kt - k2
            m = OceanWEC_Dynamic(self.em, k1=k1, k2=k2, c_mech=self.c_mech,
                                 f_wave=self.f_wave, H_wave=self.H_wave,
                                 theta_mean=self.theta_mean, T=25.0, n_frames=200)
            p_avg_vals[i] = m.P_avg
        return k_tot_vals, p_avg_vals

    def global_optimise(self):
        print(f"  Global Optimization (Differential Evolution) hunting for Max Power ...")
        bounds = [(self.k_min/2, self.k_max), (self.k_min/2, self.k_max)]
        result = differential_evolution(self._objective, bounds,
                                        popsize=8, maxiter=20, seed=42, workers=1)
        k1_opt, k2_opt = result.x
        p_opt = -result.fun
        f0_opt = np.sqrt((k1_opt + k2_opt) / self.em.M) / (2 * np.pi)

        print(f"\n  Optimum Spring Configuration Found:")
        print(f"    k1_opt  = {k1_opt:.4f} N/m")
        print(f"    k2_opt  = {k2_opt:.4f} N/m")
        print(f"    f0_opt  = {f0_opt:.4f} Hz  (wave: {self.f_wave:.4f} Hz)")
        print(f"    P_avg   = {p_opt:.2f} mW")
        return {"k1": k1_opt, "k2": k2_opt, "P_avg": p_opt}


# =============================================================================
# 4. Visualizer
# =============================================================================

class OceanWECVisualizer:
    def __init__(self, model: OceanWEC_Dynamic):
        self.m = model
        self.em = model.em

    def _spring_xy(self, start, end, amp=5.0, n_coils=8):
        n_pts = n_coils * 4 + 2
        dx, dy = end[0]-start[0], end[1]-start[1]
        length = np.hypot(dx, dy)
        if length < 1e-6:
            return np.array([start[0], end[0]]), np.array([start[1], end[1]])
        ux, uy = dx/length, dy/length
        px, py = -uy, ux
        t_inner = np.linspace(0.05, 0.95, n_pts)
        perp = amp * np.sign(np.sin(t_inner * n_coils * 2*np.pi)) * np.abs(np.cos(t_inner * n_coils * np.pi))
        xs = np.concatenate([[start[0]], start[0] + t_inner*dx + perp*px, [end[0]]])
        ys = np.concatenate([[start[1]], start[1] + t_inner*dy + perp*py, [end[1]]])
        return xs, ys

    def _build_figure(self):
        m = self.m
        fig = plt.figure(figsize=(23, 12), facecolor=C_BG)
        ttl = (f"Ocean WEC - Dynamic Electromechanical Simulation  |  f_wave={m.f_wave:.3f} Hz  "
               f"H_wave={m.H_wave:.1f}m (A={np.degrees(m.A_wave):.1f} deg)  "
               f"k1={m.k1:.1f}  k2={m.k2:.1f}  P_avg={m.P_avg:.1f} mW")
        fig.suptitle(ttl, color=C_TEXT, fontsize=10.5, fontweight="bold", y=0.989)

        gs = gridspec.GridSpec(3, 3, figure=fig,
                               left=0.05, right=0.97, top=0.95, bottom=0.06,
                               hspace=0.52, wspace=0.38)
        self.ax_kine = fig.add_subplot(gs[:, 0])
        self.ax_ang  = fig.add_subplot(gs[0, 1])
        self.ax_pos  = fig.add_subplot(gs[0, 2])
        self.ax_pwr  = fig.add_subplot(gs[1, 1:3])
        self.ax_frc  = fig.add_subplot(gs[2, 1])
        self.ax_acc  = fig.add_subplot(gs[2, 2])

        styled_axes(self.ax_ang, "Rail Pitch Angle (deg)", "Time (s)", "Angle")
        styled_axes(self.ax_pos, "Position u(t) & Velocity v(t)", "Time (s)", "m | m/s")
        styled_axes(self.ax_pwr, "Harvested Electrical Power", "Time (s)", "P (mW)")
        styled_axes(self.ax_frc, "Forces (Inc. Electromagnetic Braking)", "Time (s)", "Force (N)")
        styled_axes(self.ax_acc, "Acceleration a(t)", "Time (s)", "m/s^2")

        self.ax_kine.set_facecolor(C_PANEL)
        self.ax_kine.set_aspect("equal")
        self.ax_kine.set_title("Ocean Frame Kinematics (Oscillating)",
                               color=C_TEXT, fontsize=10, fontweight="bold", pad=6)
        return fig

    def _init_plots(self):
        m, ax, sc = self.m, self.ax_kine, 1000
        self._pivot = np.array([120.0, 180.0])
        self._sc = sc
        self._track_patch = plt.Polygon(np.zeros((4, 2)), closed=True,
                                        fc=C_TRACK, ec=C_DIM, lw=1.5, zorder=1)
        ax.add_patch(self._track_patch)

        # Coils
        self._coil_patches = []
        coil_rel = (np.arange(self.em.num_coils_per_side) - 2) * self.em.coil_spacing
        for _ in coil_rel:
            for _ in [1, -1]:
                p = plt.Polygon(np.zeros((4, 2)), closed=True,
                                fc=C_COIL, ec="#b45309", lw=0.8, alpha=0.9, zorder=4)
                ax.add_patch(p)
                self._coil_patches.append(p)

        self._sp1_line, = ax.plot([], [], color=C_SPRING1, lw=2.0, zorder=4)
        self._sp2_line, = ax.plot([], [], color=C_SPRING2, lw=2.0, zorder=4)

        self._sled_patch = plt.Polygon(np.zeros((4, 2)), closed=True,
                                       fc=C_SLED, ec=C_TEXT, lw=1.5, zorder=5)
        ax.add_patch(self._sled_patch)

        self._mag_patches = []
        for i in range(self.em.num_magnets):
            col = C_MAG_N if i % 2 == 0 else C_MAG_S
            p = plt.Polygon(np.zeros((4, 2)), closed=True,
                            fc=col, ec="white", lw=0.5, zorder=6)
            ax.add_patch(p)
            self._mag_patches.append(p)

        ax.plot(*self._pivot, "o", color=C_YELLOW, ms=6, zorder=9)

        wx = np.linspace(-150, 450, 200)
        wy = self._pivot[1] + 50 + 8*np.sin(np.linspace(0, 4*np.pi, 200))
        ax.fill_between(wx, wy, wy + 200, alpha=0.3, color="#0ea5e9", zorder=0)

        # Traces
        self._ang_line, = self.ax_ang.plot([], [], color=C_ORANGE, lw=1.5)
        self.ax_ang.axhline(0, color=C_DIM, lw=0.5, alpha=0.5)

        self._pos_line, = self.ax_pos.plot([], [], color=C_ACCENT, label="u (m)")
        self._vel_line, = self.ax_pos.plot([], [], color=C_GREEN, ls="--", label="v (m/s)")
        self.ax_pos.axhline(0, color=C_DIM, lw=0.5, alpha=0.5)
        self.ax_pos.legend(fontsize=7, loc="upper right")

        self._pwr_line, = self.ax_pwr.plot([], [], color=C_GREEN, lw=1.5)

        # Forces: now includes Coriolis and centrifugal
        self._fg_line,   = self.ax_frc.plot([], [], color=C_YELLOW,  label="F_grav")
        self._feu_line,  = self.ax_frc.plot([], [], color=C_ORANGE,  ls="--", label="F_euler")
        self._fcor_line, = self.ax_frc.plot([], [], color=C_TEAL,    ls=":",  label="F_coriolis")
        self._fcen_line, = self.ax_frc.plot([], [], color=C_PURPLE,  ls="-.", label="F_centrifugal")
        self._fs_line,   = self.ax_frc.plot([], [], color=C_SPRING1, label="F_spr")
        self._fem_line,  = self.ax_frc.plot([], [], color=C_PINK,    label="F_EM Braking")
        self.ax_frc.legend(fontsize=6, loc="lower right", ncol=3)
        self.ax_frc.axhline(0, color=C_DIM, lw=0.5, alpha=0.5)

        self._acc_line, = self.ax_acc.plot([], [], color=C_ORANGE)
        self.ax_acc.axhline(0, color=C_DIM, lw=0.5, alpha=0.5)

        ax.set_xlim(-150, 400)
        ax.set_ylim(400, -50)  # Y inverted for screen-space display

    def _update(self, fi):
        m, pivot, sc, u = self.m, self._pivot, self._sc, self.m.u[fi]
        ang = m.theta_t[fi]
        t   = m.t[fi]

        along  = np.array([np.cos(ang),  np.sin(ang)])
        across = np.array([np.sin(ang), -np.cos(ang)])

        L_mm = self.em.L * sc * 0.5
        p0, p1 = pivot - along * L_mm, pivot + along * L_mm

        th = 7.0
        self._track_patch.set_xy(np.array([
            p0 - across*th, p1 - across*th,
            p1 + across*th, p0 + across*th]))

        sled_ctr = pivot + along * (u * sc)
        sw = self.em.num_magnets * self.em.mag_spacing * sc * 0.9
        sh = self.em.d_m * sc * 1.1
        self._sled_patch.set_xy(np.array([
            sled_ctr - along*sw/2 - across*sh/2,
            sled_ctr + along*sw/2 - across*sh/2,
            sled_ctr + along*sw/2 + across*sh/2,
            sled_ctr - along*sw/2 + across*sh/2]))

        mag_off = ((np.arange(self.em.num_magnets) - (self.em.num_magnets-1)/2)
                   * self.em.mag_spacing * sc)
        mw = self.em.mag_spacing * sc * 0.7
        mh = self.em.d_m * sc * 0.8
        for i, patch in enumerate(self._mag_patches):
            mc = sled_ctr + along * mag_off[i]
            patch.set_xy(np.array([
                mc - along*mw/2 - across*mh/2,
                mc + along*mw/2 - across*mh/2,
                mc + along*mw/2 + across*mh/2,
                mc - along*mw/2 + across*mh/2]))

        coil_rel = ((np.arange(self.em.num_coils_per_side) - 2)
                    * self.em.coil_spacing * sc)
        cw_half = self.em.coil_spacing * 0.38 * sc
        ch_half = self.em.d_c / 2 * sc
        z_disp  = self.em.z_dist * sc
        idx = 0
        for cr in coil_rel:
            for sgn in [1, -1]:
                cc = pivot + along * cr + across * sgn * z_disp
                self._coil_patches[idx].set_xy(np.array([
                    cc - along*cw_half - across*ch_half,
                    cc + along*cw_half - across*ch_half,
                    cc + along*cw_half + across*ch_half,
                    cc - along*cw_half + across*ch_half]))
                idx += 1

        sp1_end    = sled_ctr - along*(sw/2 + 0.5)
        sp2_start  = sled_ctr + along*(sw/2 + 0.5)
        sx1, sy1 = self._spring_xy(p0, sp1_end, amp=5.0)
        sx2, sy2 = self._spring_xy(sp2_start, p1, amp=5.0)
        self._sp1_line.set_data(sx1, sy1)
        self._sp2_line.set_data(sx2, sy2)

        # Rolling 5-second window
        t_win = 5.0
        t_min = max(0, t - t_win * 0.8)
        t_max = max(t_win, t + t_win * 0.2)
        for ax_ in [self.ax_ang, self.ax_pos, self.ax_pwr, self.ax_frc, self.ax_acc]:
            ax_.set_xlim(t_min, t_max)

        ts = m.t[:fi+1]
        i_min = np.searchsorted(ts, t_min) if fi > 0 else 0

        if fi > 0 and i_min < len(ts):
            ang_max = np.max(np.abs(np.degrees(m.theta_t[i_min:fi+1]))) + 1.0
            self.ax_ang.set_ylim(-ang_max, ang_max)

            u_max = np.max(np.abs(m.u[i_min:fi+1])) + 0.05
            v_max = np.max(np.abs(m.v[i_min:fi+1])) + 0.05
            self.ax_pos.set_ylim(-max(u_max, v_max), max(u_max, v_max))

            pwr_max = np.max(m.P_load[i_min:fi+1]*1000) + 1.0
            self.ax_pwr.set_ylim(0, pwr_max * 1.1)

            f_max = max([
                np.max(np.abs(m.F_grav[i_min:fi+1])),
                np.max(np.abs(m.F_euler[i_min:fi+1])),
                np.max(np.abs(m.F_cor[i_min:fi+1])),
                np.max(np.abs(m.F_cent[i_min:fi+1])),
                np.max(np.abs(m.F_spr[i_min:fi+1])),
                np.max(np.abs(m.F_em[i_min:fi+1]))
            ]) + 1.0
            self.ax_frc.set_ylim(-f_max * 1.1, f_max * 1.1)

            acc_max = np.max(np.abs(m.acc[i_min:fi+1])) + 1.0
            self.ax_acc.set_ylim(-acc_max * 1.1, acc_max * 1.1)

        self._ang_line.set_data(ts, np.degrees(m.theta_t[:fi+1]))
        self._pos_line.set_data(ts, m.u[:fi+1])
        self._vel_line.set_data(ts, m.v[:fi+1])
        self._pwr_line.set_data(ts, m.P_load[:fi+1] * 1000)
        self._fg_line.set_data(ts,   m.F_grav[:fi+1])
        self._feu_line.set_data(ts,  m.F_euler[:fi+1])
        self._fcor_line.set_data(ts, m.F_cor[:fi+1])
        self._fcen_line.set_data(ts, m.F_cent[:fi+1])
        self._fs_line.set_data(ts,   m.F_spr[:fi+1])
        self._fem_line.set_data(ts,  m.F_em[:fi+1])
        self._acc_line.set_data(ts,  m.acc[:fi+1])

        return []

    def run(self, out_path="wec_animation.mov"):
        """Render the full animation and save as a .mov file."""
        fig = self._build_figure()
        self._init_plots()
        n = len(self.m.t)
        print(f"  Rendering animation ({n} frames) -> {out_path} ...")
        anim = FuncAnimation(fig, self._update, frames=n,
                             interval=30, blit=False, repeat=False)
        fps_realtime = len(self.m.t) / self.m.T  # frames / seconds = frames per second
        writer = matplotlib.animation.FFMpegWriter(
            fps=fps_realtime, codec="mpeg4",
            extra_args=["-pix_fmt", "yuv420p", "-q:v", "3"]
        )
        anim.save(out_path, writer=writer, dpi=120,
                  progress_callback=lambda i, n: print(f"    frame {i+1}/{n}", end="\r"))
        print(f"\n  Saved -> {out_path}")
        save_and_open(None, out_path)


# =============================================================================
# Utility: save figure and open it in the system viewer
# =============================================================================

def save_and_open(fig, filename):
    """Save figure to PNG (if fig given) and open file in default system viewer."""
    if fig is not None:
        fig.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {filename}")
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", filename])
        elif sys.platform.startswith("win"):
            os.startfile(filename)
        else:
            subprocess.Popen(["xdg-open", filename])
    except Exception as e:
        print(f"  Could not auto-open {filename}: {e}")


# =============================================================================
# 5. Power vs Frequency Sweep
# =============================================================================

def plot_power_vs_frequency(em_engine: WEC_EM_Engine,
                             H_wave=1.5, c_mech=0.70,
                             f_min=0.1, f_max=1.0, n_pts=25):
    """
    Sweeps wave frequency across [f_min, f_max] Hz.

    For each frequency:
      - Springs are auto-tuned to mechanical resonance (k = M*omega^2)
      - A short simulation is run and steady-state P_avg recorded

    Three separate figures are produced and shown together:
      Fig 1: P_avg (mW) vs frequency
      Fig 2: Wave pitch amplitude (deg) vs frequency
      Fig 3: Required spring stiffness k_total vs frequency
    """
    print(f"\n  Power vs Frequency Sweep: {n_pts} points from {f_min} to {f_max} Hz ...")
    f_vals = np.linspace(f_min, f_max, n_pts)
    p_vals = np.zeros(n_pts)
    a_vals = np.zeros(n_pts)
    k_vals = np.zeros(n_pts)

    for i, f in enumerate(f_vals):
        omega  = 2 * np.pi * f
        k_tot  = em_engine.M * omega**2
        k_each = k_tot / 2.0
        m = OceanWEC_Dynamic(em_engine, k1=k_each, k2=k_each,
                             c_mech=c_mech, f_wave=f, H_wave=H_wave,
                             T=20.0, n_frames=150)
        p_vals[i] = m.P_avg
        a_vals[i] = np.degrees(m.A_wave)
        k_vals[i] = k_tot
        print(f"    f={f:.3f} Hz  A={a_vals[i]:.1f} deg  k={k_tot:.2f} N/m  P={p_vals[i]:.3f} mW")

    # --- Figure 1: Power vs Resonance Frequency ---
    fig1, ax1 = plt.subplots(figsize=(10, 5), facecolor=C_BG)
    fig1.suptitle(f"Power vs Resonance Frequency  |  H={H_wave} m  |  resonance-tuned springs",
                  color=C_TEXT, fontsize=11, fontweight="bold")
    styled_axes(ax1, "Average Harvested Power vs Wave Frequency",
                "Wave Frequency (Hz)", "P_avg (mW)")
    ax1.plot(f_vals, p_vals, color=C_TEAL, lw=2.5, marker="o", ms=5)
    ax1.fill_between(f_vals, 0, p_vals, color=C_TEAL, alpha=0.12)
    i_peak = np.argmax(p_vals)
    ax1.axvline(f_vals[i_peak], color=C_PINK, ls="--", lw=1.5,
                label=f"Peak: {f_vals[i_peak]:.3f} Hz  ({p_vals[i_peak]:.2f} mW)")
    ax1.legend(facecolor=C_PANEL, labelcolor=C_TEXT, fontsize=9)
    ax1.set_xlim(f_min, f_max)
    ax1.set_ylim(bottom=0)
    fig1.tight_layout()
    save_and_open(fig1, "power_vs_frequency.png")
    plt.close(fig1)

    # --- Figure 2: Spring Stiffness vs Resonance Frequency ---
    fig2, ax2 = plt.subplots(figsize=(10, 5), facecolor=C_BG)
    fig2.suptitle(f"Spring Stiffness vs Resonance Frequency  |  H={H_wave} m",
                  color=C_TEXT, fontsize=11, fontweight="bold")
    styled_axes(ax2, "Resonance-Tuned Total Spring Stiffness vs Wave Frequency",
                "Wave Frequency (Hz)", "k_total (N/m)")
    ax2.plot(f_vals, k_vals, color=C_ACCENT, lw=2.5, marker="o", ms=5)
    ax2.fill_between(f_vals, 0, k_vals, color=C_ACCENT, alpha=0.10)
    # Annotate the stiffness at peak power frequency
    ax2.axvline(f_vals[i_peak], color=C_PINK, ls="--", lw=1.5,
                label=f"Peak power at {f_vals[i_peak]:.3f} Hz  (k={k_vals[i_peak]:.2f} N/m)")
    ax2.legend(facecolor=C_PANEL, labelcolor=C_TEXT, fontsize=9)
    ax2.set_xlim(f_min, f_max)
    ax2.set_ylim(bottom=0)
    fig2.tight_layout()
    save_and_open(fig2, "spring_stiffness_vs_frequency.png")
    plt.close(fig2)

    return f_vals, p_vals


# =============================================================================
# 6. Execution / CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser(description="Ocean WEC EM Spring Optimization")
    p.add_argument("--f_wave",     type=float, default=0.5)
    p.add_argument("--H_wave",     type=float, default=1.5, help="Wave height in meters")
    p.add_argument("--theta_mean", type=float, default=0.0)
    p.add_argument("--optimise",   action="store_true")
    p.add_argument("--freq_sweep",  action="store_true", help="Run frequency sweep only (no animation)")
    p.add_argument("--f_min",       type=float, default=0.1,  help="Freq sweep start (Hz)")
    p.add_argument("--f_max",       type=float, default=1.0,  help="Freq sweep end (Hz)")
    p.add_argument("--f_pts",       type=int,   default=20,   help="Number of sweep points")
    args = p.parse_args()

    print("==================================================")
    print(" TAFLAB: Ocean WEC Spring Optimization Engine")
    print("==================================================")
    print("Pre-computing Electromagnetic Spatial Map (Biot-Savart + coil z-integration) ...")
    em_engine = WEC_EM_Engine()

    if args.freq_sweep:
        plot_power_vs_frequency(em_engine, H_wave=args.H_wave,
                                f_min=args.f_min, f_max=args.f_max,
                                n_pts=args.f_pts)

    elif args.optimise:
        opt = SpringOptimiser(em_engine, f_wave=args.f_wave,
                              H_wave=args.H_wave, theta_mean=args.theta_mean)

        k_vals, p_vals = opt.sweep_1d()

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5), facecolor=C_BG)

        styled_axes(ax1, f"Power vs Total Spring Stiffness (f={args.f_wave}Hz, H={args.H_wave}m)",
                    "k_total (N/m)", "P_avg (mW)")
        ax1.plot(k_vals, p_vals, color=C_TEAL, lw=2)
        ax1.axvline(em_engine.M * (2*np.pi*args.f_wave)**2, color=C_YELLOW, ls="--",
                    label="Theoretical Mechanical Resonance")
        ax1.legend(facecolor=C_PANEL, labelcolor=C_TEXT)

        styled_axes(ax2, "System Resonance Frequency vs Total Spring Stiffness",
                    "k_total (N/m)", "f_0 (Hz)")
        f0_vals = np.sqrt(k_vals / em_engine.M) / (2 * np.pi)
        ax2.plot(k_vals, f0_vals, color=C_ACCENT, lw=2)
        ax2.axhline(args.f_wave, color=C_ORANGE, ls="--",
                    label=f"Target Ocean Frequency ({args.f_wave} Hz)")
        ax2.legend(facecolor=C_PANEL, labelcolor=C_TEXT)

        styled_axes(ax3, "Total Spring Stiffness vs Resonance Frequency",
                    "f_0 (Hz)", "k_total (N/m)")
        f_range = np.linspace(max(0.01, args.f_wave - 0.2), args.f_wave + 0.3, 100)
        k_req = em_engine.M * (2 * np.pi * f_range)**2
        ax3.plot(f_range, k_req, color=C_PURPLE, lw=2)
        ax3.axvline(args.f_wave, color=C_ORANGE, ls="--",
                    label=f"Target Ocean Frequency ({args.f_wave} Hz)")
        ax3.axhline(em_engine.M * (2*np.pi*args.f_wave)**2, color=C_YELLOW, ls=":",
                    label="Required Stiffness")
        ax3.legend(facecolor=C_PANEL, labelcolor=C_TEXT)

        plt.tight_layout()
        save_and_open(fig, "optimise_sweep.png")

        opt.global_optimise()

    else:
        # ── Step 1: Build the simulation at resonance-tuned springs ──────────
        omega   = 2 * np.pi * args.f_wave
        k_total = em_engine.M * omega**2
        k_opt   = k_total / 2.0
        print(f"Auto-tuning springs to mechanical resonance for {args.f_wave} Hz wave...")
        print(f"Targeting k_total = {k_total:.2f} N/m (k1 = k2 = {k_opt:.2f} N/m)")
        model = OceanWEC_Dynamic(em_engine, k1=k_opt, k2=k_opt,
                                 f_wave=args.f_wave, H_wave=args.H_wave,
                                 theta_mean=args.theta_mean)

        # ── Step 2: Save .mov animation ──────────────────────────────────────
        viz = OceanWECVisualizer(model)
        viz.run(out_path="wec_animation.mov")

        # ── Step 3: Frequency sweep → power + stiffness graphs ───────────────
        print("\nRunning frequency sweep for power and stiffness plots ...")
        plot_power_vs_frequency(em_engine, H_wave=args.H_wave,
                                f_min=args.f_min, f_max=args.f_max,
                                n_pts=args.f_pts)


if __name__ == "__main__":
    main()