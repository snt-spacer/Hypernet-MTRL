import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import matplotlib.font_manager as fm
import numpy as np

csfont  = {'fontname':'serif'}


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

# Create a figure with three subplots, and share the x and y axes.
# Increased figure size for more space.
fig, axs = plt.subplots(1, 3, figsize=(28, 5), sharex=True, sharey=True)

# Keep a reference to the scatter plot object to create the shared colorbar
scatter_plot_ref = None

# Iterate through files and trajectories
for i, file_path in enumerate(file_paths):
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
    ax = axs[i]
    plot_gates(ax, num_goals, gate_x, gate_y, gate_headings)
    
    # Store the scatter plot reference
    scatter_plot_ref = ax.scatter(car_x, car_y, c=trajectory['velocity_magnitude'], cmap='viridis', s=10, zorder=3)
    
    # Set y-axis labels only for the leftmost subplot (ax[0])
    if i == 0:
        ax.set_ylabel('Y Position (m)', fontsize=14 , **csfont )
        ax.legend(loc='best') # Add legend to the first plot
    else:
        ax.tick_params(axis='y', labelleft=False) # Hide y-axis tick labels

    # RE-ENABLE X-AXIS LABELS AND TICKS FOR ALL SUBPLOTS
    ax.set_xlabel('X Position (m)', **csfont )
    ax.tick_params(axis='x', labelbottom=True)
    
    ax.set_title(f'{trajectory_names[i]}', **csfont )
    ax.grid(False)
    ax.set_facecolor('whitesmoke')
    ax.set_aspect('equal', adjustable='box')

# Add a single colorbar for the entire figure, using the reference from the last scatter plot
if scatter_plot_ref:
    fig.subplots_adjust(right=0.9)  # Make room for the colorbar, slightly less space
    cbar_ax = fig.add_axes([0.97, 0.15, 0.015, 0.65]) # Adjusted colorbar position and width
    cbar = fig.colorbar(scatter_plot_ref, cax=cbar_ax)
    cbar.set_label('Velocity (m/s)', **csfont )

# The manual adjustments for the colorbar make tight_layout unnecessary and cause the warning.
# We will remove it.
plt.tight_layout(rect=[0, 0, 0.98, 1]) 
plt.savefig('/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/rans/utils/racing_scripts_and_data/side_by_side_trajectories_velocity.svg')
print("Side-by-side plots saved successfully.")
