"""
plots.py
--------
Unified visualization layer for the WEC analytical model.

All figures used in the paper live here. Each function takes the
result objects from the physics modules and produces a publication-
quality plot. No physics is computed here — only rendering.

Figure inventory
----------------
    Fig 1: System schematic overview (power flow block diagram)
    Fig 2: Wave input — JONSWAP spectrum + time series
    Fig 3: Frequency response — Bode + spring sweep
    Fig 4: Dynamics — platform tilt → magnet displacement
    Fig 5: Flux profile — Φ(x) and Γ(x)
    Fig 6: Electromagnetics — EMF, current, power time series
    Fig 7: Power chain — stage-by-stage waterfall
    Fig 8: Power flow Sankey diagram
    Fig 9: Efficiency sensitivity tornado chart
    Fig 10: Optimization convergence + Pareto front
    Fig 11: Summary dashboard (all key results, one figure)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.sankey import Sankey
from matplotlib.ticker import MultipleLocator
from typing import Optional, Dict, List
import warnings
warnings.filterwarnings("ignore")

# Paper-quality defaults
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        150,
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "grid.linestyle":    "--",
    "lines.linewidth":   2.0,
    "savefig.bbox":      "tight",
    "savefig.dpi":       300,
})

COLORS = {
    "wave":    "#2196F3",
    "mech":    "#FF9800",
    "em":      "#9C27B0",
    "elec":    "#F44336",
    "net":     "#4CAF50",
    "loss":    "#9E9E9E",
    "target":  "#F44336",
    "neutral": "#607D8B",
}


# ---------------------------------------------------------------------------
# Fig 2: Wave input
# ---------------------------------------------------------------------------
def plot_wave_input(ws, save_path: Optional[str] = None):
    """JONSWAP spectrum + elevation + tilt time series."""
    fig = plt.figure(figsize=(13, 8))
    gs  = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    # Spectrum
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.fill_between(ws.spectrum.f_Hz, ws.spectrum.S_eta,
                     alpha=0.3, color=COLORS["wave"])
    ax1.plot(ws.spectrum.f_Hz, ws.spectrum.S_eta,
             color=COLORS["wave"], lw=2)
    ax1.axvline(ws.spectrum.f_peak, color="darkorange", ls="--", lw=1.5,
                label=f"$f_p$ = {ws.spectrum.f_peak:.3f} Hz")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("$S_\\eta(f)$ (m²/Hz)")
    ax1.set_title("JONSWAP Wave Spectrum")
    ax1.legend(); ax1.set_xlim(0, 1.0)

    # Tilt spectrum
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogy(ws.spectrum.f_Hz, ws.spectrum.S_alpha + 1e-20,
                 color=COLORS["mech"], lw=2)
    ax2.axvline(ws.spectrum.f_peak, color="darkorange", ls="--", lw=1.5)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("$S_\\alpha(f)$ (rad²/Hz)")
    ax2.set_title("Platform Tilt Spectrum")
    ax2.set_xlim(0, 1.0)

    # Wave elevation
    ax3 = fig.add_subplot(gs[1, 0])
    n_plot = min(len(ws.t), 600)
    ax3.plot(ws.t[:n_plot], ws.eta[:n_plot], color=COLORS["wave"], lw=1.2)
    ax3.set_xlabel("Time (s)"); ax3.set_ylabel("$\\eta$ (m)")
    ax3.set_title("Wave Elevation $\\eta(t)$")

    # Platform tilt
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(ws.t[:n_plot], np.degrees(ws.alpha[:n_plot]),
             color=COLORS["mech"], lw=1.2)
    ax4.set_xlabel("Time (s)"); ax4.set_ylabel("$\\alpha$ (deg)")
    ax4.set_title("Platform Tilt $\\alpha(t)$")

    stats_str = (
        f"$H_s$ = {ws.spectrum.H_s:.1f} m  |  "
        f"$T_p$ = {ws.spectrum.T_peak:.1f} s  |  "
        f"$\\gamma$ = {ws.spectrum.gamma:.1f}  |  "
        f"$\\alpha_{{rms}}$ = {np.degrees(np.sqrt(np.mean(ws.alpha**2))):.2f}°"
    )
    fig.suptitle(f"Wave Excitation Input\n{stats_str}", fontsize=11)

    if save_path:
        plt.savefig(save_path)
        print(f"  Saved: {save_path}")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Fig 3: Frequency response
# ---------------------------------------------------------------------------
def plot_frequency_response(fr, cfg, save_path: Optional[str] = None):
    """Bode-style amplitude + phase + normalized power."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # Magnitude
    axes[0].semilogy(fr.f_Hz, fr.H_mag * 1000, color=COLORS["mech"], lw=2)
    axes[0].axvline(fr.f_n, color="darkorange", ls="--", lw=1.5,
                    label=f"$f_n$ = {fr.f_n:.3f} Hz")
    axes[0].axvline(cfg.wave.f_peak, color=COLORS["wave"], ls=":", lw=1.5,
                    label=f"$f_{{wave}}$ = {cfg.wave.f_peak:.3f} Hz")
    axes[0].set_ylabel("|H(f)| (mm/rad)")
    axes[0].set_title("Frequency Response: Magnet Displacement / Platform Tilt")
    axes[0].legend()

    # Phase
    axes[1].plot(fr.f_Hz, np.degrees(fr.H_phase), color=COLORS["em"], lw=2)
    axes[1].axvline(fr.f_n, color="darkorange", ls="--", lw=1.5)
    axes[1].axhline(-90, color=COLORS["loss"], ls=":", lw=1)
    axes[1].set_ylabel("Phase (deg)")
    axes[1].set_ylim(-210, 30)
    axes[1].set_yticks([-180, -90, 0])

    # Normalized power
    axes[2].plot(fr.f_Hz, fr.P_normalized * 1000, color=COLORS["net"], lw=2)
    axes[2].axvline(fr.f_n, color="darkorange", ls="--", lw=1.5,
                    label=f"$f_n$ = {fr.f_n:.3f} Hz")
    axes[2].axvline(cfg.wave.f_peak, color=COLORS["wave"], ls=":", lw=1.5)
    axes[2].set_ylabel("$P_{norm}$ (mW/rad²)")
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].legend()

    fig.suptitle(
        f"$k_{{eff}}$ = {cfg.spring.k_eff:.2f} N/m  |  "
        f"$m$ = {cfg.magnet.mass*1000:.1f} g  |  "
        f"$\\zeta$ = {fr.zeta:.3f}",
        fontsize=11,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"  Saved: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Fig 4: Dynamics
# ---------------------------------------------------------------------------
def plot_dynamics(dyn_result, cfg, save_path: Optional[str] = None):
    """Platform tilt, magnet displacement, velocity, and energy."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    t = dyn_result.t

    axes[0].plot(t, np.degrees(dyn_result.alpha), color=COLORS["wave"], lw=1.5)
    axes[0].set_ylabel("$\\alpha$ (deg)")
    axes[0].set_title("Dynamics: Platform → Magnet Response")

    axes[1].plot(t, dyn_result.x * 1000, color=COLORS["mech"], lw=1.5)
    axes[1].set_ylabel("$x$ (mm)")

    axes[2].plot(t, dyn_result.x_dot * 1000, color=COLORS["em"], lw=1.2)
    axes[2].set_ylabel("$\\dot{x}$ (mm/s)")

    KE = dyn_result._KE * 1000
    PE = dyn_result._PE * 1000
    axes[3].plot(t, KE, color=COLORS["wave"],  lw=1.5, label="KE")
    axes[3].plot(t, PE, color=COLORS["mech"],  lw=1.5, label="PE")
    axes[3].plot(t, KE + PE, color="black",    lw=1.2, ls="--", label="Total")
    axes[3].set_ylabel("Energy (mJ)")
    axes[3].set_xlabel("Time (s)")
    axes[3].legend(ncol=3)

    # Stats annotation
    n_skip = int(0.2 * len(t))
    x_rms  = np.sqrt(np.mean(dyn_result.x[n_skip:]**2)) * 1000
    v_rms  = np.sqrt(np.mean(dyn_result.x_dot[n_skip:]**2)) * 1000
    axes[1].annotate(f"$x_{{rms}}$ = {x_rms:.2f} mm",
                     xy=(0.02, 0.88), xycoords="axes fraction",
                     fontsize=9, color=COLORS["mech"])
    axes[2].annotate(f"$\\dot{{x}}_{{rms}}$ = {v_rms:.2f} mm/s",
                     xy=(0.02, 0.88), xycoords="axes fraction",
                     fontsize=9, color=COLORS["em"])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"  Saved: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Fig 5: Flux profile
# ---------------------------------------------------------------------------
def plot_flux_profile(x_arr, Phi_arr, Gamma_arr, cfg,
                      save_path: Optional[str] = None):
    """Flux linkage and gradient vs magnet position."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    axes[0].plot(x_arr * 1000, Phi_arr * 1000, color=COLORS["em"], lw=2)
    axes[0].fill_between(x_arr * 1000, 0, Phi_arr * 1000,
                         alpha=0.15, color=COLORS["em"])
    axes[0].set_ylabel("$\\Phi$ (mWb)")
    axes[0].set_title("Flux Linkage and Gradient vs. Magnet Position")
    axes[0].annotate(
        f"Peak $\\Phi$ = {np.max(np.abs(Phi_arr))*1000:.3f} mWb\n"
        f"from ANSYS: 171.1 mWb",
        xy=(0.02, 0.85), xycoords="axes fraction", fontsize=9,
    )

    axes[1].plot(x_arr * 1000, Gamma_arr, color=COLORS["mech"], lw=2)
    axes[1].axhline(0, color="black", lw=0.8, ls="--")
    # Mark coil positions
    for xc in np.linspace(x_arr[0]*1000, x_arr[-1]*1000, cfg.coil.n_coils + 2)[1:-1]:
        axes[1].axvline(xc, color=COLORS["loss"], lw=0.6, ls=":", alpha=0.7)
    axes[1].set_ylabel("$\\Gamma = d\\Phi/dx$ (Wb/m)")
    axes[1].set_xlabel("Magnet Position $x$ (mm)")
    axes[1].annotate(
        f"$\\Gamma_{{rms}}$ = {np.sqrt(np.mean(Gamma_arr**2)):.4f} Wb/m",
        xy=(0.02, 0.85), xycoords="axes fraction", fontsize=9,
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"  Saved: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Fig 6: Electromagnetics time series
# ---------------------------------------------------------------------------
def plot_em_timeseries(em_result, save_path: Optional[str] = None):
    """EMF, current, and power time series."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    t = em_result.t
    n_skip = int(0.2 * len(t))

    axes[0].plot(t, em_result.x * 1000, color=COLORS["mech"], lw=1.5)
    axes[0].set_ylabel("$x$ (mm)")
    axes[0].set_title("Electromagnetic Power Generation")

    axes[1].plot(t, em_result.EMF,   color=COLORS["em"],   lw=1.5, label="EMF (open circuit)")
    axes[1].plot(t, em_result.V_load, color=COLORS["elec"], lw=1.2, ls="--", label="$V_{load}$")
    axes[1].set_ylabel("Voltage (V)")
    axes[1].legend(ncol=2)

    axes[2].plot(t, em_result.I_load * 1000, color=COLORS["wave"], lw=1.5)
    axes[2].set_ylabel("$I_{load}$ (mA)")

    axes[3].plot(t, em_result.P_load * 1000, color=COLORS["elec"],
                 lw=1.0, alpha=0.6, label="$P(t)$")
    axes[3].axhline(em_result.P_avg * 1000, color=COLORS["net"], lw=2, ls="--",
                    label=f"$P_{{avg}}$ = {em_result.P_avg*1000:.2f} mW")
    axes[3].axhline(em_result.P_peak * 1000, color=COLORS["elec"], lw=1.5, ls=":",
                    label=f"$P_{{peak}}$ = {em_result.P_peak*1000:.2f} mW")
    axes[3].set_ylabel("$P_{load}$ (mW)")
    axes[3].set_xlabel("Time (s)")
    axes[3].legend(ncol=3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"  Saved: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Fig 7: Power chain waterfall
# ---------------------------------------------------------------------------
def plot_power_chain_waterfall(
    stage_result,
    save_path: Optional[str] = None,
):
    """Waterfall bar chart showing power at each conditioning stage."""
    stages = [
        ("AC Coil",    stage_result.P_ac     * 1000, COLORS["em"]),
        ("Rectifier",  stage_result.P_rect   * 1000, COLORS["mech"]),
        ("Capacitor",  stage_result.P_smooth * 1000, COLORS["wave"]),
        ("Boost Conv", stage_result.P_boost_out * 1000, COLORS["elec"]),
        ("Net Output", stage_result.P_out_net * 1000, COLORS["net"]),
    ]

    labels = [s[0] for s in stages]
    values = [s[1] for s in stages]
    colors = [s[2] for s in stages]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # Left: waterfall bars
    ax = axes[0]
    bars = ax.bar(labels, values, color=colors, edgecolor="white",
                  linewidth=1.5, width=0.6)
    ax.axhline(1000, color=COLORS["target"], ls="--", lw=1.5, label="1 W target")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Average Power (mW)")
    ax.set_title("Power at Each Conditioning Stage")
    ax.legend()
    ax.set_ylim(0, max(values) * 1.25)

    # Right: loss breakdown pie
    ax2 = axes[1]
    losses = {
        "Wiring":     stage_result.P_loss_wiring     * 1000,
        "Rectifier":  stage_result.P_loss_rectifier  * 1000,
        "Capacitor":  stage_result.P_loss_capacitor  * 1000,
        "Boost Conv": stage_result.P_loss_boost      * 1000,
        "Quiescent":  stage_result.P_loss_quiescent  * 1000,
        "Net Output": stage_result.P_out_net         * 1000,
    }
    pie_colors = [COLORS["loss"]] * 5 + [COLORS["net"]]
    wedges, texts, autotexts = ax2.pie(
        list(losses.values()),
        labels=list(losses.keys()),
        colors=pie_colors,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.82,
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax2.set_title(f"Power Loss Breakdown\n(Total input = {stage_result.P_ac*1000:.1f} mW)")

    plt.suptitle(
        f"Power Conditioning Chain  |  "
        f"$V_{{AC,rms}}$ = {stage_result.V_ac_rms:.3f} V  |  "
        f"$\\eta_{{total}}$ = {stage_result.eta_total*100:.1f}%",
        fontsize=11,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"  Saved: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Fig 8: Sankey power flow
# ---------------------------------------------------------------------------
def plot_power_flow_sankey(pf_result, save_path: Optional[str] = None):
    """
    Simplified Sankey-style diagram of end-to-end power flow.
    Uses matplotlib horizontal bar representation as a readable alternative
    to the Sankey class (which is finicky at small scales).
    """
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, 1.5)
    ax.axis("off")

    P_ref = pf_result.P_wave_captured * 1000  # normalize to captured power

    def _bar(x, width, height, y, color, label, value_mW):
        rect = mpatches.FancyBboxPatch(
            (x, y - height/2), width, height,
            boxstyle="round,pad=0.02",
            facecolor=color, edgecolor="white", linewidth=1.5, alpha=0.85,
        )
        ax.add_patch(rect)
        ax.text(x + width/2, y, f"{label}\n{value_mW:.1f} mW",
                ha="center", va="center", fontsize=8, fontweight="bold",
                color="white" if height > 0.12 else "black")

    stages = [
        ("Wave\nCapture",  pf_result.P_wave_captured,    COLORS["wave"]),
        ("Mech\nInput",    pf_result.P_mech_input,       COLORS["mech"]),
        ("EM\nInduced",    pf_result.P_em_induced,       COLORS["em"]),
        ("AC\nCoil",       pf_result.stage.P_ac,         COLORS["elec"]),
        ("After\nRect",    pf_result.stage.P_rect,       COLORS["mech"]),
        ("After\nBoost",   pf_result.stage.P_boost_out,  COLORS["wave"]),
        ("Net\nOutput",    pf_result.stage.P_out_net,    COLORS["net"]),
    ]

    x_positions = np.linspace(0.2, 9.0, len(stages))
    max_P = max(s[1] for s in stages) * 1000

    for (x, (label, P, color)) in zip(x_positions, stages):
        h = max(P * 1000 / max_P * 1.0, 0.05)
        _bar(x - 0.35, 0.7, h, 0, color, label, P * 1000)

    # Arrows between stages
    for i in range(len(stages) - 1):
        x1 = x_positions[i] + 0.35
        x2 = x_positions[i+1] - 0.35
        ax.annotate("", xy=(x2, 0), xytext=(x1, 0),
                    arrowprops=dict(arrowstyle="->", color=COLORS["neutral"], lw=1.5))

    # Target line
    target_h = 1000 / max_P * 1.0
    ax.axhline(target_h / 2, color=COLORS["target"], ls="--", lw=1.5, alpha=0.7)
    ax.text(9.5, target_h / 2, "1 W\ntarget", color=COLORS["target"],
            fontsize=8, va="center")

    ax.set_title(
        f"End-to-End Power Flow  |  "
        f"$H_s$ = {pf_result.P_wave_density_W_per_m:.0f} W/m  |  "
        f"$\\eta_{{total}}$ = {pf_result.eta_total_chain*100:.2f}%",
        fontsize=12, fontweight="bold",
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"  Saved: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Fig 9: Efficiency sensitivity tornado chart
# ---------------------------------------------------------------------------
def plot_sensitivity_tornado(
    sensitivities: Dict[str, float],
    nominal_P_mW: float,
    save_path: Optional[str] = None,
):
    """Horizontal bar tornado chart of efficiency parameter sensitivities."""
    sorted_items = sorted(sensitivities.items(), key=lambda x: abs(x[1]), reverse=True)
    labels = [k.replace("_", " ") for k, _ in sorted_items]
    deltas = [v for _, v in sorted_items]

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.55)))
    y_pos = np.arange(len(labels))

    colors = [COLORS["net"] if d > 0 else COLORS["elec"] for d in deltas]
    bars   = ax.barh(y_pos, deltas, color=colors, edgecolor="white",
                     linewidth=1, height=0.6)

    ax.axvline(0, color="black", lw=1.2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("ΔP_net (mW)")
    ax.set_title(
        f"Efficiency Sensitivity  |  Nominal $P_{{net}}$ = {nominal_P_mW:.2f} mW\n"
        "(positive = beneficial change, negative = harmful change)",
        fontsize=11,
    )

    for bar, val in zip(bars, deltas):
        x_off = 0.1 if val >= 0 else -0.1
        ha    = "left" if val >= 0 else "right"
        ax.text(val + x_off, bar.get_y() + bar.get_height()/2,
                f"{val:+.2f} mW", ha=ha, va="center", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"  Saved: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Fig 11: Summary dashboard
# ---------------------------------------------------------------------------
def plot_summary_dashboard(
    ws,
    dyn_result,
    em_result,
    pf_result,
    cfg,
    save_path: Optional[str] = None,
):
    """Six-panel summary of the full pipeline for the paper."""
    fig = plt.figure(figsize=(16, 11))
    gs  = gridspec.GridSpec(3, 3, hspace=0.52, wspace=0.38)

    n_skip = int(0.2 * len(dyn_result.t))
    n_plot = min(len(ws.t), 500)

    # Panel 1: Wave spectrum
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.fill_between(ws.spectrum.f_Hz, ws.spectrum.S_eta, alpha=0.3, color=COLORS["wave"])
    ax1.plot(ws.spectrum.f_Hz, ws.spectrum.S_eta, color=COLORS["wave"])
    ax1.axvline(ws.spectrum.f_peak, color="darkorange", ls="--", lw=1.5)
    ax1.set_xlabel("f (Hz)"); ax1.set_ylabel("$S_\\eta$ (m²/Hz)")
    ax1.set_title("(a) Wave Spectrum"); ax1.set_xlim(0, 1.0)

    # Panel 2: Platform tilt
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(ws.t[:n_plot], np.degrees(ws.alpha[:n_plot]), color=COLORS["mech"], lw=1.0)
    ax2.set_xlabel("t (s)"); ax2.set_ylabel("$\\alpha$ (deg)")
    ax2.set_title("(b) Platform Tilt")

    # Panel 3: Magnet displacement
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(dyn_result.t[:n_plot], dyn_result.x[:n_plot]*1000,
             color=COLORS["em"], lw=1.0)
    x_rms = np.sqrt(np.mean(dyn_result.x[n_skip:]**2))*1000
    ax3.set_xlabel("t (s)"); ax3.set_ylabel("$x$ (mm)")
    ax3.set_title(f"(c) Magnet Displacement\n$x_{{rms}}$ = {x_rms:.2f} mm")

    # Panel 4: Flux gradient
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot([], [], color=COLORS["em"], lw=2,
             label="$\\Gamma(x)$ model")
    ax4.set_xlabel("x (mm)"); ax4.set_ylabel("$\\Gamma$ (Wb/m)")
    ax4.set_title("(d) Flux Gradient $\\Gamma(x)$")
    ax4.text(0.5, 0.5, "(run EM module\nto populate)",
             ha="center", va="center", transform=ax4.transAxes,
             fontsize=9, color=COLORS["loss"])

    # Panel 5: EMF and power
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(em_result.t[:n_plot], em_result.EMF[:n_plot],
             color=COLORS["em"], lw=1.0, alpha=0.8, label="EMF")
    ax5_r = ax5.twinx()
    ax5_r.plot(em_result.t[:n_plot], em_result.P_load[:n_plot]*1000,
               color=COLORS["elec"], lw=1.0, alpha=0.5)
    ax5.set_xlabel("t (s)"); ax5.set_ylabel("EMF (V)", color=COLORS["em"])
    ax5_r.set_ylabel("P (mW)", color=COLORS["elec"])
    ax5.set_title(f"(e) EMF & Power\n$P_{{avg}}$ = {em_result.P_avg*1000:.2f} mW")

    # Panel 6: Power chain waterfall
    ax6 = fig.add_subplot(gs[1, 2])
    sr = pf_result.stage
    stage_labels = ["AC", "Rect", "Cap", "Boost", "Net"]
    stage_vals   = [sr.P_ac, sr.P_rect, sr.P_smooth, sr.P_boost_out, sr.P_out_net]
    stage_colors = [COLORS["em"], COLORS["mech"], COLORS["wave"],
                    COLORS["elec"], COLORS["net"]]
    ax6.bar(stage_labels, [v*1000 for v in stage_vals],
            color=stage_colors, edgecolor="white", linewidth=1.2)
    ax6.axhline(1000, color=COLORS["target"], ls="--", lw=1.5)
    ax6.set_ylabel("P (mW)"); ax6.set_title("(f) Power Chain")

    # Panel 7: End-to-end efficiency summary (text)
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis("off")

    m   = cfg.magnet.mass
    k   = cfg.spring.k_eff
    f_n = np.sqrt(k/m) / (2*np.pi)

    summary_text = (
        f"System Summary  |  "
        f"$m$ = {m*1000:.1f} g  |  "
        f"$k_{{eff}}$ = {k:.2f} N/m  |  "
        f"$f_n$ = {f_n:.4f} Hz  |  "
        f"$f_{{wave}}$ = {cfg.wave.f_peak:.4f} Hz  |  "
        f"$f_n/f_{{wave}}$ = {f_n/cfg.wave.f_peak:.3f}\n"
        f"$P_{{wave}}$ = {pf_result.P_wave_density_W_per_m:.1f} W/m  |  "
        f"$P_{{captured}}$ = {pf_result.P_wave_captured*1000:.1f} mW  |  "
        f"$P_{{EM}}$ = {em_result.P_avg*1000:.2f} mW  |  "
        f"$P_{{net}}$ = {pf_result.P_net_output*1000:.2f} mW  |  "
        f"$\\eta_{{total}}$ = {pf_result.eta_total_chain*100:.2f}%  |  "
        f"Target: {'✓ MET' if pf_result.target_met else '✗ NOT MET (gap: ' + str(round(1000 - pf_result.P_net_output*1000, 1)) + ' mW)'}"
    )

    ax7.text(0.5, 0.6, summary_text, ha="center", va="center",
             transform=ax7.transAxes, fontsize=9.5,
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#f5f5f5",
                       edgecolor="#cccccc", linewidth=1.5))

    fig.suptitle(
        "WEC Analytical Model — Full Pipeline Summary",
        fontsize=14, fontweight="bold", y=1.01,
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"  Saved: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Quick validation — generate all plots with dummy/default data
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from config import WECConfig
    from wave_input import generate_wave_excitation
    from dynamics import solve_nonlinear, array_excitation
    from electromagnetics import (
        build_coil_geometry, compute_flux_profile,
        build_flux_interpolants, solve_electromagnetics,
    )
    from power_electronics import PowerElectronicsConfig, compute_power_chain
    from power_flow import compute_power_flow, efficiency_sensitivity
    from frequency_response import compute_transfer_function

    cfg    = WECConfig()
    pe_cfg = PowerElectronicsConfig()

    print("Generating all plots...")

    # Wave
    ws = generate_wave_excitation(cfg, mode="jonswap", seed=42)
    plot_wave_input(ws, save_path="fig_wave_input.png")

    # Frequency response
    fr = compute_transfer_function(cfg)
    plot_frequency_response(fr, cfg, save_path="fig_freq_response.png")

    # Dynamics
    alpha_f, _, alpha_ddot_f = array_excitation(ws.t, ws.alpha)
    dyn = solve_nonlinear(cfg, alpha_f, alpha_ddot_f, cfg.wave.t_eval)
    plot_dynamics(dyn, cfg, save_path="fig_dynamics.png")

    # EM
    geom = build_coil_geometry(cfg)
    x_arr, Phi_arr, Gamma_arr = compute_flux_profile(
        cfg, geom, x_range=(-0.06, 0.06), n_points=150
    )
    plot_flux_profile(x_arr, Phi_arr, Gamma_arr, cfg,
                      save_path="fig_flux_profile.png")
    _, gamma_f = build_flux_interpolants(x_arr, Phi_arr, Gamma_arr)
    em = solve_electromagnetics(cfg, dyn.t, dyn.x, dyn.x_dot, gamma_func=gamma_f)
    plot_em_timeseries(em, save_path="fig_em_timeseries.png")

    # Power flow
    pf = compute_power_flow(cfg, dyn, em, pe_cfg)
    plot_power_chain_waterfall(pf.stage, save_path="fig_power_chain.png")
    plot_power_flow_sankey(pf, save_path="fig_sankey.png")

    # Sensitivity
    sens = efficiency_sensitivity(cfg, dyn, em, pe_cfg)
    plot_sensitivity_tornado(sens, pf.P_net_output*1000,
                             save_path="fig_sensitivity.png")

    # Summary dashboard
    plot_summary_dashboard(ws, dyn, em, pf, cfg,
                           save_path="fig_summary_dashboard.png")

    print("\nAll figures saved.")