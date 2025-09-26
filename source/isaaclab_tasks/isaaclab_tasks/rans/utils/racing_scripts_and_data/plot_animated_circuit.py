import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.animation import FuncAnimation

# --- Data Loading and Preparation ---

file_path = "/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/rans/utils/racing_scripts_and_data/extracted_trajectories_RaceGates_bcn.csv"
try:
    df = pd.read_csv(file_path)
    print("File loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file was not found at {file_path}")
    print("Please update the 'file_path' variable to the correct location.")
    raise SystemExit

# --- Select Trajectory Data for Animation ---
start_trajectory_id = 20
num_course_segments = 11  # Adjust as needed
end_trajectory_id = start_trajectory_id + num_course_segments - 1

print(f"Combining trajectories from {start_trajectory_id} to {end_trajectory_id} to plot the path.")
full_course_data = df[df['trajectory'].between(start_trajectory_id, end_trajectory_id)].copy()
full_course_data.reset_index(drop=True, inplace=True)

# Extract car's position and signals
car_x = full_course_data['position_x'].to_numpy()
car_y = full_course_data['position_y'].to_numpy()
velocity_magnitude = np.sqrt(
    full_course_data['linear_velocity_x'].to_numpy()**2 +
    full_course_data['linear_velocity_y'].to_numpy()**2
)
# NEW: unaltered actions
actions_x = full_course_data['unaltered_actions_x'].to_numpy()
actions_y = full_course_data['unaltered_actions_y'].to_numpy()

# --- Extract ALL Gate Data ---

gate_data_source = df.iloc[0]
heading_columns = [col for col in df.columns if 'target_headings_' in col]
total_gates_on_track = len(heading_columns)
print(f"Detected a total of {total_gates_on_track} gates on the course.")

gate_x = np.array([gate_data_source[f'target_positions_{i}'] for i in range(0, 2 * total_gates_on_track, 2)])
gate_y = np.array([gate_data_source[f'target_positions_{i}'] for i in range(1, 2 * total_gates_on_track, 2)])
gate_headings = np.array([gate_data_source[f'target_headings_{i}'] for i in range(total_gates_on_track)])

# --- Helper: find one-lap end index using gate wrap detection ---

def find_one_lap_end_index(car_x, car_y, gate_x, gate_y, total_gates, start_near_tol=1.5, wrap_span=3):
    pts = np.stack([car_x, car_y], axis=1)
    gates = np.stack([gate_x, gate_y], axis=1)

    diff = pts[:, None, :] - gates[None, :, :]
    dist2 = np.einsum('tgi,tgi->tg', diff, diff)
    nearest_gate = np.argmin(dist2, axis=1)

    boundaries = []
    for i in range(1, len(nearest_gate)):
        if (nearest_gate[i-1] >= total_gates - wrap_span) and (nearest_gate[i] <= wrap_span):
            boundaries.append(i)

    if not boundaries:
        d0 = np.sqrt((car_x - gate_x[0])**2 + (car_y - gate_y[0])**2)
        ignore = min(100, len(d0)//10)
        idx = np.where(d0[ignore:] < start_near_tol)[0]
        if len(idx) == 0:
            print("No lap boundary or start/finish proximity found; using full data.")
            return len(car_x) - 1
        return ignore + idx[0]

    start_near = np.sqrt((car_x[0] - gate_x[0])**2 + (car_y[0] - gate_y[0])**2) < start_near_tol
    if start_near and len(boundaries) >= 2:
        end_idx = boundaries[1]
    else:
        end_idx = boundaries[0]
    return int(end_idx)

# Trim to exactly one lap
end_frame_inclusive = find_one_lap_end_index(
    car_x, car_y, gate_x, gate_y, total_gates_on_track,
    start_near_tol=1.5, wrap_span=3
)
car_x = car_x[:end_frame_inclusive + 1]
car_y = car_y[:end_frame_inclusive + 1]
velocity_magnitude = velocity_magnitude[:end_frame_inclusive + 1]
actions_x = actions_x[:end_frame_inclusive + 1]
actions_y = actions_y[:end_frame_inclusive + 1]
print(f"Animating a single lap: {len(car_x)} frames.")

# Scales for bars
v_max = float(velocity_magnitude.max()) * 1.05  # headroom
a_max = float(max(np.max(np.abs(actions_x)), np.max(np.abs(actions_y)), 1e-6)) * 1.05  # symmetric range

# --- Animation Setup with 3 right-side bars (Speed, Action X, Action Y) ---

fig = plt.figure(figsize=(16, 8))
outer_gs = gridspec.GridSpec(1, 2, width_ratios=[12, 2], wspace=0.08)

# Main axis
ax = fig.add_subplot(outer_gs[0])

def plot_gates(ax, num_gates):
    gate_width = 0.5
    gate_height = 1.0
    for i in range(num_gates):
        center_x, center_y = gate_x[i], gate_y[i]
        heading = gate_headings[i]
        rect = patches.Rectangle(
            (-gate_width / 2, -gate_height / 2), gate_width, gate_height,
            linewidth=1.5, edgecolor='r', facecolor='none',
            label='Race Gate' if i == 0 else "", zorder=4
        )
        rect.set_transform(transforms.Affine2D().rotate(heading).translate(center_x, center_y) + ax.transData)
        ax.add_patch(rect)
        ax.text(center_x, center_y + 0.6, str(i), fontsize=10,
                ha='center', va='bottom', color='red', zorder=4, weight='bold')

plot_gates(ax, total_gates_on_track)
ax.plot(car_x, car_y, color='gray', linestyle='--', linewidth=0.7, zorder=1, label='Path (1 lap)')

robot_position, = ax.plot([], [], 'o', color='darkolivegreen', markersize=10, zorder=5, label='Robot')
trajectory_line, = ax.plot([], [], color='yellowgreen', linewidth=2, zorder=3, label='Traveled Path')
velocity_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_title(f'Catalunya Circuit')
# ax.grid(True, zorder=0)
ax.set_facecolor('whitesmoke')
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(car_x.min() - 2, car_x.max() + 2)
ax.set_ylim(car_y.min() - 2, car_y.max() + 2)
ax.legend(loc='upper right')

# Right stack of bars
right_gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer_gs[1], hspace=0.35)

# Speed bar (0 .. v_max)
bar_ax_speed = fig.add_subplot(right_gs[0])
bar_ax_speed.set_xlim(0, 1)
bar_ax_speed.set_ylim(0, v_max)
bar_ax_speed.set_xticks([])
bar_ax_speed.set_ylabel('Velocity (m/s)')
bar_ax_speed.set_title('Speed', pad=6)
bar_speed_outline = patches.Rectangle((0, 0), 1, v_max, fill=False, edgecolor='black', linewidth=1)
bar_ax_speed.add_patch(bar_speed_outline)
bar_speed = patches.Rectangle((0, 0), 1, 0, facecolor='red', edgecolor='none')
bar_ax_speed.add_patch(bar_speed)

# Action X bar (-a_max .. +a_max)
bar_ax_ax = fig.add_subplot(right_gs[1])
bar_ax_ax.set_xlim(0, 1)
bar_ax_ax.set_ylim(-a_max, a_max)
bar_ax_ax.set_xticks([])
# bar_ax_ax.set_ylabel('Throttle')
bar_ax_ax.set_title('Throttle', pad=6)
bar_ax_ax.axhline(0, linewidth=1, color='black')
bar_ax_ax_outline = patches.Rectangle((0, -a_max), 1, 2*a_max, fill=False, edgecolor='black', linewidth=1)
bar_ax_ax.add_patch(bar_ax_ax_outline)
bar_ax_x = patches.Rectangle((0, 0), 1, 0, facecolor='red', edgecolor='none')
bar_ax_ax.add_patch(bar_ax_x)

# Action Y bar (-a_max .. +a_max)
bar_ax_ay = fig.add_subplot(right_gs[2])
bar_ax_ay.set_xlim(0, 1)
bar_ax_ay.set_ylim(-a_max, a_max)
bar_ax_ay.set_xticks([])
# bar_ax_ay.set_ylabel('Steering')
bar_ax_ay.set_title('Steering', pad=6)
bar_ax_ay.axhline(0, linewidth=1, color='black')
bar_ax_ay_outline = patches.Rectangle((0, -a_max), 1, 2*a_max, fill=False, edgecolor='black', linewidth=1)
bar_ax_ay.add_patch(bar_ax_ay_outline)
bar_ay = patches.Rectangle((0, 0), 1, 0, facecolor='red', edgecolor='none')
bar_ax_ay.add_patch(bar_ay)

# --- Animation Functions ---

def init():
    robot_position.set_data([], [])
    trajectory_line.set_data([], [])
    velocity_text.set_text('')
    bar_speed.set_height(0)
    bar_ax_x.set_height(0); bar_ax_x.set_y(0)
    bar_ay.set_height(0);   bar_ay.set_y(0)
    return robot_position, trajectory_line, velocity_text, bar_speed, bar_ax_x, bar_ay

def update(frame):
    # Path & marker
    robot_position.set_data([car_x[frame]], [car_y[frame]])
    trajectory_line.set_data(car_x[:frame+1], car_y[:frame+1])

    # Text velocity (kept as-is)
    v = float(velocity_magnitude[frame])
    velocity_text.set_text(f'Velocity: {v:.2f} m/s')

    # Velocity bar (0..v_max)
    bar_speed.set_height(v)

    # Action X signed bar
    ax_val = float(actions_x[frame])
    if ax_val >= 0:
        bar_ax_x.set_y(0)
        bar_ax_x.set_height(ax_val)
    else:
        bar_ax_x.set_y(ax_val)
        bar_ax_x.set_height(-ax_val)

    # Action Y signed bar
    ay_val = float(actions_y[frame])
    if ay_val >= 0:
        bar_ay.set_y(0)
        bar_ay.set_height(ay_val)
    else:
        bar_ay.set_y(ay_val)
        bar_ay.set_height(-ay_val)

    return robot_position, trajectory_line, velocity_text, bar_speed, bar_ax_x, bar_ay

# Create and save the animation
animation_frames = len(car_x)
ani = FuncAnimation(fig, update, frames=animation_frames, init_func=init,
                    blit=True, interval=20)

output_filename = '/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/rans/utils/racing_scripts_and_data/robot_single_lap_with_speed_and_actions.mp4'
print("Saving animation...")
ani.save(output_filename, writer='ffmpeg', fps=30)
print(f"Animation saved successfully as '{output_filename}'")

# plt.show()
