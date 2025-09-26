import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec  # Import gridspec
import numpy as np

csfont = {'fontname': 'serif'}

def truncate_formatter(x, pos):
    return f"{int(x * 100) / 100:.2f}"

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

# Define the file paths and corresponding trajectory numbers
file_paths = [
    "/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/rans/utils/racing_scripts_and_data/extracted_trajectories_RaceGates_bcn.csv",
    "/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/rans/utils/racing_scripts_and_data/extracted_trajectories_RaceGates_jpn.csv",
    "/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/rans/utils/racing_scripts_and_data/extracted_trajectories_RaceGates_itl.csv",
]
trajectory_numbers = [20, 6, 20]
trajectory_names = ["Catalunya Circuit", "Suzuka Circuit", "Monza Circuit"]


# --- MODIFIED LINES ---
# 1. Reduced the excessive figure size to something more standard.
fig = plt.figure(figsize=(24, 9))
# 2. Added `width_ratios` to make the first column 2x wider than the second.
gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1.7, 1])
# --- END MODIFIED LINES ---

# The first plot takes up the entire left column (both rows)
ax1 = fig.add_subplot(gs[:, 0])
# The second plot takes the top-right cell
ax2 = fig.add_subplot(gs[0, 1])
# The third plot takes the bottom-right cell, sharing axes with the one above
ax3 = fig.add_subplot(gs[1, 1], sharex=ax2, sharey=ax2)

# Create a list of the axes to iterate through
axs = [ax1, ax2, ax3]

# Keep a reference to the scatter plot object to create the shared colorbar
scatter_plot_ref = None

# Iterate through files, trajectories, and now the custom axes
for i, ax in enumerate(axs):
    file_path = file_paths[i]
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}  Skipping.")
        continue

    # Select the specific trajectory
    trajectory = df[df['trajectory'] == trajectory_numbers[i]].copy()

    # --- Data Extraction ---
    if trajectory.empty:
        print(f"Trajectory {trajectory_numbers[i]} not found in {file_path}. Skipping.")
        continue

    car_x = trajectory['position_x']
    car_y = trajectory['position_y']
    
    velocity_magnitude = np.sqrt(
        trajectory['linear_velocity_x']**2 +
        trajectory['linear_velocity_y']**2
    )
    trajectory['velocity_magnitude'] = velocity_magnitude
    
    gate_data = trajectory.iloc[0]
    num_goals = int(gate_data['num_goals'])
    
    gate_x = [gate_data[f'target_positions_{j}'] for j in range(0, 2 * num_goals, 2)]
    gate_y = [gate_data[f'target_positions_{j}'] for j in range(1, 2 * num_goals, 2)]
    gate_headings = [gate_data[f'target_headings_{j}'] for j in range(0, num_goals)]

    # --- Plotting on the respective subplot ---
    plot_gates(ax, num_goals, gate_x, gate_y, gate_headings)
    
    # Store the scatter plot reference
    scatter_plot_ref = ax.scatter(car_x, car_y, c=trajectory['velocity_magnitude'], cmap='viridis', s=15, zorder=3)
    
    ax.set_title(f'{trajectory_names[i]}', **csfont)
    ax.grid(False)
    ax.set_facecolor('whitesmoke')
    ax.set_aspect('equal', adjustable='box')

    # --- Custom label and legend handling for the new layout ---
    if ax == ax1: # Left plot
        ax.set_xlabel('X Position (m)', **csfont)
        ax.set_ylabel('Y Position (m)', fontsize=14, **csfont)
        ax.legend(loc='best')
    elif ax == ax2: # Top-right plot
        ax.tick_params(axis='x', labelbottom=False)
    elif ax == ax3: # Bottom-right plot
        ax.set_xlabel('X Position (m)', **csfont)

# Add a single colorbar for the entire figure
if scatter_plot_ref:
    # --- MODIFIED LINE ---
    # 3. Adjusted the colorbar position to fit the new layout.
    cbar_ax = fig.add_axes([0.9, 0.15, 0.015, 0.7]) 
    # --- END MODIFIED LINE ---
    cbar = fig.colorbar(scatter_plot_ref, cax=cbar_ax)
    cbar.set_label('Velocity (m/s)', **csfont)

# Use tight_layout to automatically adjust spacing and prevent overlap
plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.savefig('/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/rans/utils/racing_scripts_and_data/combined_layout_trajectories_velocity.svg')
print("Combined layout plot saved successfully.")