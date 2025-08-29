import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import matplotlib.ticker as ticker
import numpy as np

def truncate_formatter(x, pos):
    return f"{int(x * 100) / 100:.2f}"

# Load the dataset
df = pd.read_csv("source/isaaclab_tasks/isaaclab_tasks/rans/utils/racing_scripts_and_data/extracted_trajectories_RaceGates.csv")
print("File loaded successfully.")

# Select the first trajectory
trajectory_0 = df[df['trajectory'] == 0].copy()

# --- Data Extraction ---
# Extract car's position
car_x = trajectory_0['position_x']
car_y = trajectory_0['position_y']
car_z = trajectory_0['position_z']

# Calculate velocity magnitude
velocity_magnitude = np.sqrt(
    trajectory_0['linear_velocity_x']**2 +
    trajectory_0['linear_velocity_y']**2
)
trajectory_0['velocity_magnitude'] = velocity_magnitude

# Extract gate data from the first row of the trajectory
gate_data = trajectory_0.iloc[0]
num_goals = int(gate_data['num_goals'])

# Extract gate center positions (assuming X, Y pairs)
gate_x = [gate_data[f'target_positions_{i}'] for i in range(0, 2 * num_goals, 2)]
gate_y = [gate_data[f'target_positions_{i}'] for i in range(1, 2 * num_goals, 2)]

gate_headings = [gate_data[f'target_headings_{i}'] for i in range(0, num_goals)]

# Function to plot gates
def plot_gates(ax, num_goals, gate_x, gate_y, gate_headings):
    gate_width = 0.5
    gate_height = 1.0
    for i in range(num_goals):
        center_x, center_y = gate_x[i], gate_y[i]
        heading = gate_headings[i]
        rect = patches.Rectangle(
            (-gate_width / 2, -gate_height / 2), gate_width, gate_height,
            linewidth=1, edgecolor='r', facecolor='none', label='Race Gate' if i == 0 else "", zorder=4
        )
        transform = transforms.Affine2D().rotate(heading).translate(center_x, center_y)
        rect.set_transform(transform + ax.transData)
        ax.add_patch(rect)
        ax.text(center_x, center_y + 0.8, str(i), fontsize=9, ha='center', va='bottom', color='red', zorder=4)

# --- Plot 1: Velocity Magnitude ---
fig_2d_vel, ax_2d_vel = plt.subplots(figsize=(15, 8))
plot_gates(ax_2d_vel, num_goals, gate_x, gate_y, gate_headings)

sc_vel = ax_2d_vel.scatter(car_x, car_y, c=trajectory_0['velocity_magnitude'], cmap='viridis', s=10, zorder=3)
cbar_vel = fig_2d_vel.colorbar(sc_vel, ax=ax_2d_vel, fraction=0.046, pad=0.04)
cbar_vel.set_label('Velocity Magnitude (m/s)')

ax_2d_vel.set_xlabel('X Position (m)')
ax_2d_vel.set_ylabel('Y Position (m)')
ax_2d_vel.set_title('2D Race Track and Car Trajectory (Trajectory 0) - Velocity')
ax_2d_vel.legend(['Race Gates'])
ax_2d_vel.grid(True)
ax_2d_vel.set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.savefig('source/isaaclab_tasks/isaaclab_tasks/rans/utils/racing_scripts_and_data/racing_circuit_plot_velocity.svg')
print("Velocity plot saved successfully.")


# --- Plot 2: Throttle Value ---
fig_2d_thr, ax_2d_thr = plt.subplots(figsize=(15, 8))
plot_gates(ax_2d_thr, num_goals, gate_x, gate_y, gate_headings)

sc_thr = ax_2d_thr.scatter(car_x, car_y, c=trajectory_0['throttle_0'], cmap='viridis', s=10, zorder=3)
cbar_thr = fig_2d_thr.colorbar(sc_thr, ax=ax_2d_thr, fraction=0.046, pad=0.04)
cbar_thr.set_label('Throttle Value')
cbar_thr.ax.yaxis.set_major_formatter(ticker.FuncFormatter(truncate_formatter))

ax_2d_thr.set_xlabel('X Position (m)')
ax_2d_thr.set_ylabel('Y Position (m)')
ax_2d_thr.set_title('2D Race Track and Car Trajectory (Trajectory 0) - Throttle')
ax_2d_thr.legend(['Race Gates'])
ax_2d_thr.grid(True)
ax_2d_thr.set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.savefig('source/isaaclab_tasks/isaaclab_tasks/rans/utils/racing_scripts_and_data/racing_circuit_plot_throttle.svg')
print("Throttle plot saved successfully.")