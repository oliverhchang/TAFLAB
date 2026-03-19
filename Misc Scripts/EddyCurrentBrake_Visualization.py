import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


# --- Physics Formulas ---
def b_field_cylinder(z_m, R_m=0.020, L_m=0.020, Br=1.26):
    """Calculate on-axis B-field of your 40x20mm cylinder."""
    term1 = (L_m + z_m) / np.sqrt(R_m ** 2 + (L_m + z_m) ** 2)
    term2 = z_m / np.sqrt(R_m ** 2 + z_m ** 2)
    return (Br / 2.0) * (term1 - term2)


def calculate_eddy_drag(z_m, velocity_mps, thickness_mm):
    """
    Approximates eddy current drag force.
    F_drag is proportional to thickness * velocity * (B_z)^2.
    """
    B_z = b_field_cylinder(z_m)
    thickness_m = thickness_mm / 1000.0

    # This is a lumped constant (K) representing the conductivity of 6061 Aluminum,
    # the swept area of the 40mm magnet, and geometric factors.
    # It is calibrated to visualize the drop-off curve accurately.
    K = 150000

    return K * thickness_m * velocity_mps * (B_z ** 2)


# --- Setup Initial Data ---
z_array_mm = np.linspace(10, 100, 500)
z_array_m = z_array_mm / 1000.0

init_velocity = 0.5  # m/s (Wave speed)
init_thickness = 6.35  # mm (1/4 inch aluminum)
init_clearance = 60.0  # mm

# Calculate initial curve
initial_drag = calculate_eddy_drag(z_array_m, init_velocity, init_thickness)

# --- Create the Plot ---
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(left=0.1, bottom=0.35)  # Make room for sliders

# Plot the main curve
line, = ax.plot(z_array_mm, initial_drag, lw=2, color='blue', label='Eddy Current Drag')

threshold_val = 0.1
ax.axhline(y=threshold_val, color='red', linestyle='--', lw=1.5, label='Negligible Drag Threshold (0.05 N)')

# Plot the interactive marker
marker_z_m = init_clearance / 1000.0
marker_drag = calculate_eddy_drag(marker_z_m, init_velocity, init_thickness)
marker, = ax.plot(init_clearance, marker_drag, 'ro', markersize=8, label='Current Clearance')

ax.set_title("Eddy Current Braking Force vs. Aluminum Clearance")
ax.set_xlabel("Clearance Distance to Aluminum Chassis (mm)")
ax.set_ylabel("Estimated Drag Force (Newtons)")
ax.set_xlim(10, 100)
ax.set_ylim(0, 2.0)  # Zoomed in to see the threshold crossing clearly
ax.grid(True, alpha=0.5)
ax.legend()

# --- Setup Sliders ---
axcolor = 'lightgoldenrodyellow'
ax_vel = plt.axes([0.15, 0.20, 0.65, 0.03], facecolor=axcolor)
ax_thick = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_clear = plt.axes([0.15, 0.10, 0.65, 0.03], facecolor=axcolor)

s_vel = Slider(ax_vel, 'Velocity (m/s)', 0.1, 2.0, valinit=init_velocity)
s_thick = Slider(ax_thick, 'Thickness (mm)', 1.0, 10.0, valinit=init_thickness)
s_clear = Slider(ax_clear, 'Clearance (mm)', 10.0, 100.0, valinit=init_clearance)


# --- Update Function for Interactivity ---
def update(val):
    v = s_vel.val
    t = s_thick.val
    z_clear = s_clear.val

    # Update the whole curve
    new_drag_curve = calculate_eddy_drag(z_array_m, v, t)
    line.set_ydata(new_drag_curve)

    # Update the marker position
    new_marker_drag = calculate_eddy_drag(z_clear / 1000.0, v, t)
    marker.set_data([z_clear], [new_marker_drag])  # Pass as sequences

    fig.canvas.draw_idle()


# Link sliders to the update function
s_vel.on_changed(update)
s_thick.on_changed(update)
s_clear.on_changed(update)

plt.show()