"""
WECVisualizer  –  v3
────────────────────
Changes over v2:
  1. DOWNWARD MOTION  – the sled now moves DOWN (decreasing y) and to the
     left in the kinematic view, matching physical reality (gravity along
     the slope pulls the mass from top-right → bottom-left).
     The tilt axis is flipped: track runs from upper-right to lower-left;
     the "along" unit vector points down-slope (toward lower-left).

  2. HIGH-FIDELITY FIELD LINES  – the in-plane (Bx, Bz) overlay is replaced
     by a streamplot drawn on a much finer grid (120 × 80 by default) with
     linewidth and color mapped to field strength.  The Bz colour-map is also
     computed on the same fine grid.  Field-map refresh every 4 frames.

  3. MOV OUTPUT  – `viz.run(save_path="…/output.mov")` renders every frame
     to a PNG buffer and stitches them with ffmpeg into a QuickTime-compatible
     H.264 .mov.  Pillow is NOT used; this removes the 256-colour GIF
     limitation.  A progress bar is printed to stdout.
"""

import io
import os
import subprocess
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")          # off-screen; change to "TkAgg" for live window
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline


# ─── colour palette ──────────────────────────────────────────────────────────
# Kinematic panel stays dark; all other panels use a clean light theme.
C_BG     = "#ffffff"   # figure background  – white
C_PANEL  = "#f8f9fa"   # light panel fill
C_GRID   = "#dee2e6"   # subtle grid lines
C_TEXT   = "#212529"   # near-black text
C_DIM    = "#6c757d"   # secondary text / labels
C_ACCENT = "#1971c2"   # blue lines
C_WARM   = "#e03131"   # red bars
C_GREEN  = "#2f9e44"   # green traces
C_YELLOW = "#e67700"   # amber accents
C_MAG_N  = "#c92a2a"   # N-pole red
C_MAG_S  = "#1864ab"   # S-pole blue
C_COIL   = "#f08c00"   # coil amber
C_TRACK  = "#495057"   # track grey (dark, kinematic only)
C_FLUX   = "#6741d9"   # purple flux bars

# ── kinematic panel now uses the same light theme ────────────────────────────
CK_BG    = "#ffffff"
CK_PANEL = "#f8f9fa"
CK_GRID  = "#dee2e6"
CK_TEXT  = "#212529"
CK_DIM   = "#6c757d"


def styled_axes(ax, title="", xlabel="", ylabel=""):
    """Light-theme styling for non-kinematic panels."""
    ax.set_facecolor(C_PANEL)
    ax.tick_params(colors=C_DIM, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(C_GRID)
    ax.grid(True, color=C_GRID, linewidth=0.5, alpha=0.8)
    if title:
        ax.set_title(title, color=C_TEXT, fontsize=9, fontweight="bold", pad=6)
    if xlabel:
        ax.set_xlabel(xlabel, color=C_DIM, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=C_DIM, fontsize=8)


# ═════════════════════════════════════════════════════════════════════════════
class WECVisualizer:
    """Multi-panel animation for AnalyticalWEC  (v3)."""

    def __init__(self, wec, field_nx=120, field_nz=80):
        """
        Parameters
        ----------
        wec        : AnalyticalWEC instance (lookup table already generated)
        field_nx   : x-grid points for field map  (higher → sharper, slower)
        field_nz   : z-grid points for field map
        """
        self.wec      = wec
        self.field_nx = field_nx
        self.field_nz = field_nz
        self._pre_simulate()

    # ── ODE ──────────────────────────────────────────────────────────────────
    def _pre_simulate(self):
        w = self.wec
        print("  Pre-simulating trajectory …")
        y0    = [-w.track_length / 2, 0.001]
        t_end = 2.5
        t_eval = np.linspace(0, t_end, 300)

        def hit_end(t, y):
            return w.track_length / 2 - y[0]
        hit_end.terminal  = True
        hit_end.direction = -1

        sol = solve_ivp(
            w.system_dynamics, (0, t_end), y0,
            method="LSODA", t_eval=t_eval, events=hit_end, dense_output=True
        )
        if sol.status == 1:
            n   = len(sol.t) - 1
            u_s = np.pad(sol.y[0][:n], (0, len(t_eval) - n),
                         constant_values=w.track_length / 2)
            v_s = np.pad(sol.y[1][:n], (0, len(t_eval) - n),
                         constant_values=0.0)
        else:
            u_s, v_s = sol.y[0], sol.y[1]

        self.t_sim = t_eval
        self.u_sim = u_s
        self.v_sim = v_s

        print("  Pre-computing K / V / P / Φ arrays …")
        n_frames = len(self.t_sim)
        nc       = w.total_coils

        self.K_sim   = np.zeros((n_frames, nc))
        self.V_sim   = np.zeros((n_frames, nc))
        self.Phi_sim = np.zeros((n_frames, nc))
        self.P_sim   = np.zeros(n_frames)
        self.VL_sim  = np.zeros(n_frames)

        for i, (u, v) in enumerate(zip(self.u_sim, self.v_sim)):
            K        = w.get_K(u)
            V        = K * v
            I, VL    = w.solve_parallel_network(V)
            self.K_sim[i]   = K
            self.V_sim[i]   = V
            self.P_sim[i]   = VL ** 2 / w.R_load
            self.VL_sim[i]  = VL
            self.Phi_sim[i] = K * u

        # ── high-resolution field grid ────────────────────────────────────
        print(f"  Building {self.field_nx}×{self.field_nz} field grid …")
        z_pad       = w.z_dist * 3.5
        self.bz_x   = np.linspace(-w.track_length / 2 - 0.04,
                                   w.track_length / 2 + 0.04, self.field_nx)
        self.bz_z   = np.linspace(-z_pad, z_pad, self.field_nz)
        print("  Done.\n")

    # ── composite field  (Bz, Bx) on (bz_x, bz_z) grid ──────────────────────
    def _make_field_map(self, u_sled):
        w   = self.wec
        nx  = len(self.bz_x)
        nz  = len(self.bz_z)
        BZ  = np.zeros((nz, nx))
        BX  = np.zeros((nz, nx))
        mag_offsets = (np.arange(w.num_magnets) -
                       (w.num_magnets - 1) / 2) * w.mag_spacing

        for m_idx, m_off in enumerate(mag_offsets):
            polarity = 1 if m_idx % 2 == 0 else -1
            m_pos    = u_sled + m_off
            x_rel    = self.bz_x - m_pos          # shape (nx,)

            for iz, zp in enumerate(self.bz_z):
                for ix, xr in enumerate(x_rel):
                    r   = max(abs(xr), 1e-6)
                    bz  = w.calculate_Bz_vectorized(r, zp)
                    br  = w.calculate_Br_vectorized(r, zp)
                    sgn = np.sign(xr) if xr != 0 else 1.0
                    BZ[iz, ix] += polarity * bz
                    BX[iz, ix] += polarity * br * sgn

        return BZ, BX

    # ── figure ────────────────────────────────────────────────────────────────
    def _build_figure(self):
        fig = plt.figure(figsize=(20, 11), facecolor=C_BG)
        fig.suptitle(
            "WEC  ·  Magnetic Field, Flux & Kinematic Simulation  (v7)",
            color=C_TEXT, fontsize=13, fontweight="bold", y=0.985
        )
        gs = gridspec.GridSpec(
            3, 3, figure=fig,
            left=0.06, right=0.97, top=0.94, bottom=0.07,
            hspace=0.50, wspace=0.40
        )
        self.ax_kine  = fig.add_subplot(gs[:, 0])
        self.ax_field = fig.add_subplot(gs[0, 1:])
        self.ax_Phi   = fig.add_subplot(gs[1, 1])
        self.ax_V     = fig.add_subplot(gs[1, 2])
        self.ax_P     = fig.add_subplot(gs[2, 1])
        self.ax_vel   = fig.add_subplot(gs[2, 2])

        for ax, ttl, xl, yl in [
            (self.ax_field,
             "Bz Field Map  (z = 0 → magnet midline)  +  field-line streamplot",
             "Track position  x  (m)", "z  (m)  [z=0: magnet midline]"),
            (self.ax_Phi,  "Flux Linkage  Φ  per coil", "Coil index",  "Φ  (a.u.)"),
            (self.ax_V,    "Induced Voltage  V  per coil", "Coil index","V  (V)"),
            (self.ax_P,    "Load Power",   "Time (s)", "P  (mW)"),
            (self.ax_vel,  "Sled Velocity","Time (s)", "v  (m/s)"),
        ]:
            styled_axes(ax, ttl, xl, yl)

        self.ax_kine.set_facecolor(CK_PANEL)
        self.ax_kine.set_aspect("equal")
        for sp in self.ax_kine.spines.values():
            sp.set_edgecolor(CK_GRID)
        self.ax_kine.tick_params(colors=CK_DIM, labelsize=7)
        self.ax_kine.set_title(
            "Kinematic Side View  (30° tilt)  ↓ gravity down-slope",
            color=CK_TEXT, fontsize=9, fontweight="bold", pad=6
        )
        return fig

    # ── kinematic panel  (FIX 1: sled moves downward / to the left) ──────────
    def _init_kinematic(self):
        w   = self.wec
        ax  = self.ax_kine
        L   = w.track_length
        sc  = 1000          # m → mm

        # Track runs upper-right → lower-left.
        # along  = unit vector pointing DOWN-slope (increasing x to the left, y downward).
        # We do NOT invert the y-axis, so positive y = screen-down is correct.
        # along = (-cos θ, +sin θ) → goes left and downward in screen space.
        # across = perpendicular, pointing away from the slope surface (upward in screen).
        ang    = w.tilt_angle
        along  = np.array([-np.cos(ang),  np.sin(ang)])   # left + down
        across = np.array([-np.sin(ang), -np.cos(ang)])   # perpendicular "up" from surface

        self._along  = along
        self._across = across
        self._scale  = sc

        # p0 = START (upper-right), p1 = END (lower-left)
        # Place p0 in the upper-right region of the axes panel.
        p0 = np.array([L * sc * 0.95, L * sc * 0.95 * np.tan(ang)])
        p1 = p0 + along * L * sc
        self._p0 = p0

        TH = (w.d_m / 2 + w.air_gap + w.d_c) * sc

        # Track body
        corners = np.array([
            p0 - across * TH,
            p1 - across * TH,
            p1 + across * TH,
            p0 + across * TH,
        ])
        ax.add_patch(plt.Polygon(corners, closed=True,
                                 fc="#adb5bd", ec="#495057", lw=1.2, zorder=1))

        # Magnet channel
        mag_h = w.d_m / 2 * sc
        ax.add_patch(plt.Polygon(np.array([
            p0 - across * mag_h,
            p1 - across * mag_h,
            p1 + across * mag_h,
            p0 + across * mag_h,
        ]), closed=True, fc="#dee2e6", ec="#adb5bd", lw=0.8, zorder=2, alpha=0.9))

        # Coils at ±z_dist (mapped onto "across" direction)
        coil_rel    = (np.arange(w.num_coils_per_side) -
                       (w.num_coils_per_side - 1) / 2) * w.coil_spacing
        z_coil_disp = w.z_dist * sc
        cw_half     = w.coil_spacing * 0.38 * sc
        ch_half     = w.d_c / 2 * sc

        self._coil_patches_top = []
        self._coil_patches_bot = []
        for cr in coil_rel:
            for sgn, store in [(+1, self._coil_patches_top),
                               (-1, self._coil_patches_bot)]:
                ctr  = p0 + along * (L / 2 + cr) * sc + across * sgn * z_coil_disp
                rect = np.array([
                    ctr - along * cw_half - across * ch_half,
                    ctr + along * cw_half - across * ch_half,
                    ctr + along * cw_half + across * ch_half,
                    ctr - along * cw_half + across * ch_half,
                ])
                patch = plt.Polygon(rect, closed=True,
                                    fc=C_COIL, ec="#92400e", lw=0.8,
                                    alpha=0.90, zorder=4)
                ax.add_patch(patch)
                store.append(patch)

        # Gravity arrow – points in +y direction (screen downward, no inversion)
        mid = (p0 + p1) / 2
        ax.annotate(
            "", xy=mid + np.array([0, +32]), xytext=mid + np.array([0, 0]),
            arrowprops=dict(arrowstyle="-|>", color=C_YELLOW, lw=2.0,
                            mutation_scale=14),
            zorder=8
        )
        ax.text(mid[0] + 5, mid[1] + 22, "g", color=C_YELLOW,
                fontsize=10, fontweight="bold")

        # Tilt arc
        from matplotlib.patches import Arc
        ax.add_patch(Arc(p1, 55, 55, angle=0,
                         theta1=180 + np.degrees(ang),
                         theta2=180,
                         color=CK_DIM, lw=1))
        ax.text(p1[0] - 36, p1[1] + 8, "30°", color=CK_DIM, fontsize=7)

        ax.text(p0[0] + 4, p0[1] - 8, "START", color=CK_DIM, fontsize=7,
                ha="left", va="bottom")
        ax.text(p1[0] - 4, p1[1] + 8, "END", color=CK_DIM, fontsize=7,
                ha="right", va="top")

        # z-axis annotation
        z_arrow_base = p0 + along * L * sc * 0.15
        ax.annotate("", xy=z_arrow_base + across * z_coil_disp * 1.4,
                    xytext=z_arrow_base,
                    arrowprops=dict(arrowstyle="->", color="#a78bfa", lw=1.2))
        mid_z = z_arrow_base + across * z_coil_disp * 0.7
        ax.text(mid_z[0] - 8, mid_z[1], "z", color="#a78bfa",
                fontsize=8, fontweight="bold")
        tip_z = z_arrow_base + across * z_coil_disp * 1.5
        ax.text(tip_z[0], tip_z[1],
                f"+z_dist\n={w.z_dist*1000:.1f}mm",
                color="#a78bfa", fontsize=6, va="center")

        # Sled (animated)
        sled_w = w.num_magnets * w.mag_spacing * sc * 0.92
        sled_h = w.d_m * sc * 1.05
        self._sled_patch = plt.Polygon(np.zeros((4, 2)), closed=True,
                                        fc="#e9ecef", ec="#1971c2", lw=1.8,
                                        zorder=5)
        ax.add_patch(self._sled_patch)
        self._sled_w = sled_w
        self._sled_h = sled_h

        # Magnets (animated)
        self._mag_patches = []
        mag_offsets_sc = (np.arange(w.num_magnets) -
                          (w.num_magnets - 1) / 2) * w.mag_spacing * sc
        self._mag_offsets_sc = mag_offsets_sc
        for m_idx in range(w.num_magnets):
            col = C_MAG_N if m_idx % 2 == 0 else C_MAG_S
            mp  = plt.Polygon(np.zeros((4, 2)), closed=True,
                               fc=col, ec="white", lw=0.5, alpha=0.92, zorder=6)
            ax.add_patch(mp)
            lbl = ax.text(0, 0, "N" if m_idx % 2 == 0 else "S",
                          color="white", fontsize=7, fontweight="bold",
                          ha="center", va="center", zorder=7)
            self._mag_patches.append((mp, lbl))

        # Decorative field arcs
        self._field_arc_lines = []
        for _ in range(7):
            ln, = ax.plot([], [], color="#1971c2", lw=0.5, alpha=0.35, zorder=4)
            self._field_arc_lines.append(ln)

        # Velocity arrow
        self._vel_arrow = ax.annotate(
            "", xy=(0, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color="#2f9e44",
                            lw=2.2, mutation_scale=13),
            zorder=8
        )

        # Info label
        self._info_txt = ax.text(
            0.02, 0.97, "", transform=ax.transAxes,
            color="#212529", fontsize=7.5, va="top", fontfamily="monospace"
        )

        ax.legend(
            handles=[
                mpatches.Patch(fc=C_MAG_N, label="N pole"),
                mpatches.Patch(fc=C_MAG_S, label="S pole"),
                mpatches.Patch(fc=C_COIL,  label="Coil (±z_dist)"),
            ],
            loc="lower right", fontsize=6, framealpha=0.3,
            facecolor="#ffffff", edgecolor="#dee2e6", labelcolor="#495057"
        )

        # Axis limits: INVERT y so p0 (start) is at TOP, p1 (end) at BOTTOM.
        # p1 has larger y value (along has +sin component), so flipping puts it down.
        pad = 40
        all_x = [p0[0], p1[0]]
        all_y = [p0[1], p1[1]]
        ax.set_xlim(min(all_x) - TH - pad, max(all_x) + TH + pad)
        ax.set_ylim(max(all_y) + TH + pad, min(all_y) - TH - pad)   # ← inverted
        ax.set_xlabel("x (mm)", color=CK_DIM, fontsize=7)
        ax.set_ylabel("y (mm)", color=CK_DIM, fontsize=7)

    # ── field map (FIX 2: high-fidelity streamplot coloured by |B|) ──────────
    def _init_field(self):
        w  = self.wec
        ax = self.ax_field

        print("  Computing initial field map (this may take a moment) …")
        BZ0, BX0 = self._make_field_map(self.u_sim[0])
        vmax = np.percentile(np.abs(BZ0), 98) + 1e-9

        self._field_im = ax.imshow(
            BZ0,
            extent=[self.bz_x[0], self.bz_x[-1],
                    self.bz_z[0], self.bz_z[-1]],
            origin="lower", aspect="auto",
            cmap="RdBu_r", vmin=-vmax, vmax=vmax,
            interpolation="bilinear", zorder=1, alpha=0.85
        )
        cbar = plt.colorbar(self._field_im, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label("Bz  (T)", color=C_DIM, fontsize=8)
        cbar.ax.yaxis.set_tick_params(color=C_DIM, labelsize=7)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=C_DIM)

        # Reference lines
        ax.axhline( w.z_dist, color=C_COIL, lw=1.2, ls="--", alpha=0.9,
                    label=f"+z_dist = +{w.z_dist*1000:.1f} mm  (upper coils)")
        ax.axhline(-w.z_dist, color=C_COIL, lw=1.2, ls=":",  alpha=0.9,
                    label=f"−z_dist = −{w.z_dist*1000:.1f} mm  (lower coils)")
        ax.axhline(0, color=C_YELLOW, lw=1.0, ls="-", alpha=0.7,
                   label="z = 0  (magnet midline)")
        ax.legend(fontsize=6.5, loc="upper right",
                  facecolor=C_PANEL, edgecolor=C_GRID,
                  labelcolor=C_DIM, framealpha=0.9)

        # Static coil position markers
        coil_rel = (np.arange(w.num_coils_per_side) -
                    (w.num_coils_per_side - 1) / 2) * w.coil_spacing
        for cr in coil_rel:
            ax.axvline(cr, color=C_COIL, lw=0.8, alpha=0.5, ls="--")

        # ── Streamplot on a dedicated overlay axes ──────────────────────
        # We create a transparent twin axes (ax_stream) that exactly overlays
        # ax_field. On every refresh we call ax_stream.cla() which nukes ALL
        # stream artists (lines + arrow patches) atomically, then redraw fresh.
        # This is the only reliable way to avoid accumulating stale arrows.
        self.ax_stream = ax.inset_axes([0, 0, 1, 1])   # same bbox as ax_field
        self.ax_stream.set_xlim(self.bz_x[0], self.bz_x[-1])
        self.ax_stream.set_ylim(self.bz_z[0], self.bz_z[-1])
        self.ax_stream.set_facecolor("none")             # transparent background
        self.ax_stream.axis("off")                        # hide ticks/spines

        X2D, Z2D = np.meshgrid(self.bz_x, self.bz_z)
        Bmag = np.sqrt(BX0**2 + BZ0**2) + 1e-12
        self._stream_vmax = float(np.percentile(Bmag, 97))
        self._stream_X2D  = X2D
        self._stream_Z2D  = Z2D

        self.ax_stream.streamplot(
            X2D, Z2D, BX0, BZ0,
            color=Bmag,
            cmap="plasma",
            linewidth=1.4,
            density=1.8,
            arrowsize=0.9,
            arrowstyle="->",
            norm=plt.Normalize(0, self._stream_vmax)
        )

    # ── bars ──────────────────────────────────────────────────────────────────
    def _init_bars(self):
        w    = self.wec
        nc   = w.total_coils
        idxs = np.arange(nc)

        self._Phi_bars = self.ax_Phi.bar(
            idxs, np.zeros(nc),
            color=[C_FLUX] * nc, edgecolor=C_GRID, linewidth=0.5
        )
        self.ax_Phi.axhline(0, color=C_DIM, lw=0.5)
        self.ax_Phi.set_xticks(idxs)
        self.ax_Phi.set_xticklabels(
            [f"C{i+1}" for i in range(nc)],
            color=C_DIM, fontsize=6, rotation=45
        )
        self.ax_Phi.set_xlim(-0.5, nc - 0.5)

        self._V_bars = self.ax_V.bar(
            idxs, np.zeros(nc),
            color=[C_WARM] * nc, edgecolor=C_GRID, linewidth=0.5
        )
        self.ax_V.axhline(0, color=C_DIM, lw=0.5)
        self.ax_V.set_xticks(idxs)
        self.ax_V.set_xticklabels(
            [f"C{i+1}" for i in range(nc)],
            color=C_DIM, fontsize=6, rotation=45
        )
        self.ax_V.set_xlim(-0.5, nc - 0.5)

    # ── time traces ───────────────────────────────────────────────────────────
    def _init_traces(self):
        self._P_line,   = self.ax_P.plot([], [], color=C_GREEN,  lw=1.5)
        self._P_dot,    = self.ax_P.plot([], [], "o", color=C_GREEN, ms=5)
        self._vel_line, = self.ax_vel.plot([], [], color=C_ACCENT, lw=1.5)
        self._vel_dot,  = self.ax_vel.plot([], [], "o", color=C_ACCENT, ms=5)

        P_max = np.max(self.P_sim) * 1000 * 1.2 + 0.5
        v_max = np.max(np.abs(self.v_sim)) * 1.2 + 0.02
        t_max = self.t_sim[-1]

        self.ax_P.set_xlim(0, t_max);   self.ax_P.set_ylim(0, P_max)
        self.ax_vel.set_xlim(0, t_max); self.ax_vel.set_ylim(0, v_max)

        active = self.P_sim > 1e-7
        if active.any():
            self.ax_P.fill_between(
                self.t_sim, self.P_sim * 1000,
                where=active, alpha=0.15, color=C_GREEN
            )

    # ── per-frame update ──────────────────────────────────────────────────────
    def _update(self, fi):
        w     = self.wec
        u     = self.u_sim[fi]
        v     = self.v_sim[fi]
        t     = self.t_sim[fi]
        K_arr = self.K_sim[fi]
        V_arr = self.V_sim[fi]
        Phi   = self.Phi_sim[fi]
        P_mW  = self.P_sim[fi] * 1000
        sc    = self._scale
        along = self._along
        across= self._across
        p0    = self._p0

        # ── kinematic ──────────────────────────────────────────────────────
        # sled moves along "along" from p0;  u ∈ [−L/2, +L/2] → offset from midpoint
        sled_ctr = p0 + along * (w.track_length / 2 + u) * sc
        sw, sh   = self._sled_w, self._sled_h

        self._sled_patch.set_xy(np.array([
            sled_ctr - along * sw/2 - across * sh/2,
            sled_ctr + along * sw/2 - across * sh/2,
            sled_ctr + along * sw/2 + across * sh/2,
            sled_ctr - along * sw/2 + across * sh/2,
        ]))

        mw = w.mag_spacing * 0.70 * sc
        mh = w.d_m * 0.80 * sc
        for m_idx, (mp, lbl) in enumerate(self._mag_patches):
            mc = sled_ctr + along * self._mag_offsets_sc[m_idx]
            mp.set_xy(np.array([
                mc - along * mw/2 - across * mh/2,
                mc + along * mw/2 - across * mh/2,
                mc + along * mw/2 + across * mh/2,
                mc - along * mw/2 + across * mh/2,
            ]))
            lbl.set_position(mc)

        # Decorative field arcs
        m0 = sled_ctr + along * self._mag_offsets_sc[0]
        for ai, arc_ln in enumerate(self._field_arc_lines):
            r  = (ai + 1) * 3.5
            th = np.linspace(0, 2 * np.pi, 60)
            ax_pts = m0[0] + r * np.cos(th)
            ay_pts = m0[1] + r * 0.45 * np.sin(th)
            arc_ln.set_data(ax_pts, ay_pts)

        # Velocity arrow
        arrow_len = v * sc * 0.75
        self._vel_arrow.set_position(sled_ctr)
        self._vel_arrow.xy = sled_ctr + along * arrow_len

        self._info_txt.set_text(
            f"t = {t:.3f} s\n"
            f"u = {u*1000:+.1f} mm\n"
            f"v = {v:.3f} m/s\n"
            f"P = {P_mW:.2f} mW"
        )

        # ── field map + streamplot redraw every 4 frames ────────────────
        if fi % 4 == 0:
            BZ, BX = self._make_field_map(u)
            self._field_im.set_data(BZ)

            # cla() on the overlay axes is the only reliable way to clear ALL
            # stream artists (lines + arrow FancyArrowPatches) in one shot.
            self.ax_stream.cla()
            self.ax_stream.set_facecolor("none")
            self.ax_stream.axis("off")
            self.ax_stream.set_xlim(self.bz_x[0], self.bz_x[-1])
            self.ax_stream.set_ylim(self.bz_z[0], self.bz_z[-1])

            Bmag = np.sqrt(BX**2 + BZ**2) + 1e-12
            self.ax_stream.streamplot(
                self._stream_X2D, self._stream_Z2D, BX, BZ,
                color=Bmag,
                cmap="plasma",
                linewidth=1.4,
                density=1.8,
                arrowsize=0.9,
                arrowstyle="->",
                norm=plt.Normalize(0, self._stream_vmax)
            )

        # ── bars ─────────────────────────────────────────────────────────
        Phi_max = np.max(np.abs(self.Phi_sim)) * 1.1 + 1e-9
        V_max   = np.max(np.abs(self.V_sim))   * 1.1 + 0.01

        for j, bar in enumerate(self._Phi_bars):
            bar.set_height(Phi[j])
            bar.set_y(min(Phi[j], 0))
            bar.set_color(C_FLUX if Phi[j] >= 0 else C_WARM)

        for j, bar in enumerate(self._V_bars):
            bar.set_height(V_arr[j])
            bar.set_y(min(V_arr[j], 0))
            bar.set_color(C_GREEN if V_arr[j] >= 0 else C_WARM)

        self.ax_Phi.set_ylim(-Phi_max, Phi_max)
        self.ax_V.set_ylim(-V_max, V_max)

        # ── time traces ──────────────────────────────────────────────────
        self._P_line.set_data(self.t_sim[:fi+1], self.P_sim[:fi+1] * 1000)
        self._P_dot.set_data([t], [P_mW])
        self._vel_line.set_data(self.t_sim[:fi+1], self.v_sim[:fi+1])
        self._vel_dot.set_data([t], [v])

        return []

    # ── render & save to .mov (FIX 3) ────────────────────────────────────────
    def _save_mov(self, fig, save_path, fps=25, dpi=100):
        """
        Render each animation frame to a temporary directory of PNGs,
        then stitch with ffmpeg into a QuickTime H.264 .mov.
        Using temp files (not pipe) avoids frame-size and pipe-buffer issues.
        The vf scale filter forces even pixel dimensions required by libx264.
        """
        total   = len(self.t_sim)
        tmp_dir = tempfile.mkdtemp(prefix="wec_frames_")

        try:
            # ── 1. render all frames ──────────────────────────────────────
            for fi in range(total):
                self._update(fi)
                frame_path = os.path.join(tmp_dir, f"frame_{fi:05d}.png")
                fig.savefig(frame_path, format="png", dpi=dpi,
                            facecolor="white", bbox_inches="tight")

                if fi % 50 == 0 or fi == total - 1:
                    pct = (fi + 1) / total * 100
                    bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
                    print(f"\r  Rendering [{bar}] {pct:5.1f}%  frame {fi+1}/{total}",
                          end="", flush=True)

            print()

            # ── 2. stitch with ffmpeg ─────────────────────────────────────
            # vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" → forces even dimensions
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", os.path.join(tmp_dir, "frame_%05d.png"),
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-vcodec", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "18",
                "-preset", "medium",
                save_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print("  ffmpeg stderr:", result.stderr[-600:])
                raise RuntimeError(f"ffmpeg failed (rc={result.returncode})")

            print(f"  Saved → {save_path}")

        finally:
            # ── 3. clean up temp frames ───────────────────────────────────
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── public entry ──────────────────────────────────────────────────────────
    def run(self, interval_ms=40, save_path=None, fps=25, dpi=100):
        """
        Parameters
        ----------
        interval_ms : frame delay for live animation (ignored when saving)
        save_path   : if given, render to .mov instead of showing live
        fps         : frames per second in the output video
        dpi         : raster resolution (dots per inch)
        """
        fig = self._build_figure()
        self._init_kinematic()
        self._init_field()
        self._init_bars()
        self._init_traces()

        if save_path:
            self._save_mov(fig, save_path, fps=fps, dpi=dpi)
            plt.close(fig)
        else:
            anim = FuncAnimation(
                fig, self._update,
                frames=len(self.t_sim),
                interval=interval_ms,
                blit=False, repeat=False
            )
            plt.show()
            return anim


# ─── standalone demo / stub ───────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        import importlib
        wec_module = importlib.import_module("AnalyticalWEC")
        wec = wec_module.AnalyticalWEC(num_magnets=4, R_load=10.0)
    except ModuleNotFoundError:
        print("AnalyticalWEC not found – using inline stub.")

        class AnalyticalWEC:
            def __init__(self, num_magnets=4, R_load=None):
                self.mu0 = 4 * np.pi * 1e-7; self.g = 9.81
                self.num_magnets = num_magnets
                self.Br = 1.45; self.R_m = 0.0125; self.d_m = 0.025
                self.mag_spacing = 0.030
                self.M_single = 0.092
                self.M = self.M_single * num_magnets + 0.050
                self.num_coils_per_side = 5
                self.total_coils = self.num_coils_per_side * 2
                self.coil_spacing = 0.030
                self.R_in = 0.008; self.R_out = 0.021; self.d_c = 0.0125
                self.N_turns = 1240; self.R_int_per_coil = 52.0
                self.R_int = self.R_int_per_coil / self.total_coils
                self.R_load = R_load if R_load is not None else self.R_int
                self.V_drop = 0.6
                self.air_gap = 0.0012
                self.z_dist = self.d_m / 2 + self.air_gap + self.d_c / 2
                self.tilt_angle = np.deg2rad(30.0)
                self.track_length = 0.40
                self.C_m = 0.25; self.C_f = 0.45
                self.C_mech = self.C_m + self.C_f

            def calculate_Bz_vectorized(self, r_p, z_p):
                n_theta, n_z = 60, 20
                theta_q = np.linspace(0, 2 * np.pi, n_theta)
                z_q = np.linspace(-self.d_m / 2, self.d_m / 2, n_z)
                TH, ZQ = np.meshgrid(theta_q, z_q)
                eps = 1e-9
                dist_sq = ((r_p - self.R_m * np.cos(TH)) ** 2 +
                           (self.R_m * np.sin(TH)) ** 2 +
                           (z_p - ZQ) ** 2 + eps)
                integrand = self.R_m * (r_p * np.cos(TH) - self.R_m) / (dist_sq ** 1.5)
                integral  = np.trapezoid(
                    np.trapezoid(integrand, theta_q, axis=1), z_q, axis=0)
                return -(self.Br / (4 * np.pi)) * integral

            def calculate_Br_vectorized(self, r_p, z_p):
                dz = 1e-4
                Bz_p = self.calculate_Bz_vectorized(r_p, z_p + dz)
                Bz_m = self.calculate_Bz_vectorized(r_p, z_p - dz)
                dBz_dz = (Bz_p - Bz_m) / (2 * dz)
                r = max(r_p, 1e-6)
                return -(r / 2) * dBz_dz

            def get_individual_Ks(self, u):
                Ks = np.zeros(self.total_coils)
                coil_pos = (np.arange(self.num_coils_per_side) -
                            (self.num_coils_per_side - 1) / 2) * self.coil_spacing
                mag_off  = (np.arange(self.num_magnets) -
                            (self.num_magnets - 1) / 2) * self.mag_spacing
                r_pts    = np.linspace(self.R_in, self.R_out, 12)
                dx_eps   = 5e-4
                for c_idx, cp in enumerate(coil_pos):
                    k_coil = 0
                    for m_idx, mo in enumerate(mag_off):
                        pol = 1 if m_idx % 2 == 0 else -1
                        dx  = u + mo - cp
                        bz1 = np.array([self.calculate_Bz_vectorized(
                                np.sqrt(dx**2 + r**2 + 1e-9), self.z_dist)
                                for r in r_pts])
                        phi1 = np.trapezoid(bz1 * 2 * np.pi * r_pts, r_pts) * self.N_turns
                        dx2  = dx + dx_eps
                        bz2  = np.array([self.calculate_Bz_vectorized(
                                np.sqrt(dx2**2 + r**2 + 1e-9), self.z_dist)
                                for r in r_pts])
                        phi2 = np.trapezoid(bz2 * 2 * np.pi * r_pts, r_pts) * self.N_turns
                        k_coil += pol * (phi2 - phi1) / dx_eps
                    Ks[c_idx] = k_coil
                    Ks[c_idx + self.num_coils_per_side] = k_coil
                return Ks

            def generate_lookup_table(self):
                num_pts = 60
                u_pts   = np.linspace(-self.track_length / 2,
                                       self.track_length / 2, num_pts)
                k_pts   = [self.get_individual_Ks(u) for u in u_pts]
                self.u_table  = u_pts
                self.k_table  = np.array(k_pts)
                self.k_spline = CubicSpline(
                    self.u_table, self.k_table, axis=0, extrapolate=True)

            def get_K(self, u):
                return self.k_spline(u)

            def solve_parallel_network(self, V_induced_array):
                V_abs    = np.maximum(np.abs(V_induced_array) - self.V_drop, 0)
                V_sorted = np.sort(V_abs)[::-1]
                V_L = 0
                if V_sorted[0] > 0:
                    for N in range(1, len(V_abs) + 1):
                        V_L_t = (np.sum(V_sorted[:N]) / self.R_int_per_coil /
                                 (1 / self.R_load + N / self.R_int_per_coil))
                        if N == len(V_abs) or V_L_t >= V_sorted[N]:
                            V_L = V_L_t; break
                I_arr = np.maximum(0, V_abs - V_L) / self.R_int_per_coil
                I_dir = I_arr * np.sign(V_induced_array)
                return I_dir, V_L

            def system_dynamics(self, t, y):
                u, v = y
                if u > self.track_length / 2:
                    return [0.0, 0.0]
                F_gravity = self.M * self.g * np.sin(self.tilt_angle)
                K_array   = self.get_K(u)
                V_induced = K_array * v
                I_dir, _  = self.solve_parallel_network(V_induced)
                F_em      = -np.sum(K_array * I_dir)
                F_mech    = -self.C_mech * v
                dvdt      = (F_gravity + F_em + F_mech) / self.M
                return [v, dvdt]

        wec = AnalyticalWEC(num_magnets=4, R_load=10.0)

    print("Building lookup table …")
    wec.generate_lookup_table()

    print("Launching WECVisualizer v3 …")
    viz = WECVisualizer(wec, field_nx=60, field_nz=40)

    # ── save to .mov ──────────────────────────────────────────────────────
    import os, pathlib
    out_dir  = pathlib.Path.home() / "Downloads"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / "WEC_simulation_v7.mov")

    viz.run(save_path=out_path, fps=25, dpi=100)