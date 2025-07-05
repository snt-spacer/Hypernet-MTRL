import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
import random # Import random module for hex color generation

print("starting")

# --- Configuration ---
CSV_PATH = "/workspace/isaaclab/logs/rsl_rl/multitask_memory_normW_hyperparams/2025-07-04_07-22-31_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-42/metrics/extracted_trajectories_GoToPosition.csv"
OUTPUT_DIR = "/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/rans/utils/plots/test_trajects/" # Ensure this directory exists
# COLORMAP_NAME is no longer used for individual trajectory colors, but kept if other parts need it.
# COLORMAP_NAME = 'viridis'
ALPHA_VALUE = 0.8 # Transparency for individual trajectory lines

# --- Load Data ---
df = pd.read_csv(CSV_PATH)
print("done reading csv")

# --- Data Preprocessing (Normalization) ---
# Normalize XY trajectory so that the target is at (0,0) for each trajectory
xy_df = df.copy()
xy_df['tx'] = xy_df.groupby('trajectory')['target_position_x'].transform('first')
xy_df['ty'] = xy_df.groupby('trajectory')['target_position_y'].transform('first')
xy_df['norm_position_x'] = xy_df['position_x'] - xy_df['tx']
xy_df['norm_position_y'] = xy_df['position_y'] - xy_df['ty']
xy_df = xy_df.drop(columns=['tx', 'ty'])

# --- Global Color Setup ---
# Get all unique trajectory names
trajectory_names = df['trajectory'].unique()
num_trajectories = len(trajectory_names)

# Create a dictionary to store random hex colors for each trajectory
# We'll use a fixed seed for reproducibility of the 'random' colors if desired
random.seed(42) # Optional: set a seed for reproducible random colors
trajectory_color_map_hex = {name: "#%06x" % random.randint(0, 0xFFFFFF) for name in trajectory_names}


# --- Plotting Functions ---

def plot_trajectories_with_colors(ax, data_df, x_col, y_col, title, xlabel, ylabel, show_legend=True, x_lim=None, y_lim=None, add_target_lines=False):
    """
    Plots individual trajectories with distinct random hex colors using a global trajectory_color_map_hex.
    """
    for trajectory_name, group in data_df.groupby('trajectory'):
        # Get the pre-assigned random hex color for this trajectory
        hex_color = trajectory_color_map_hex[trajectory_name]
        ax.plot(group[x_col], group[y_col], color=hex_color, alpha=ALPHA_VALUE, label=f'Trajectory {trajectory_name}')

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if add_target_lines:
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax.axvline(0, color='gray', linestyle='--', linewidth=1)
        ax.set_aspect('equal', adjustable='box') # Only for normalized XY plot

    if x_lim:
        ax.set_xlim(x_lim[0], x_lim[1])
    if y_lim:
        ax.set_ylim(y_lim[0], y_lim[1])

    if show_legend:
        # Check if the number of unique trajectories is small enough for a clear legend
        if num_trajectories <= 20: # Arbitrary threshold, adjust as needed
            ax.legend(title='Trajectories', bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            # For many trajectories, a legend can clutter the plot.
            pass # No legend for too many trajectories


# Helper to plot mean and sd for a given column (no change here, as it's not trajectory-specific colors)
def plot_mean_sd(data_df, y_col, fname, ylabel=None):
    grouped = data_df.groupby('step')[y_col]
    mean = grouped.mean()
    std = grouped.std()
    steps = mean.index
    plt.figure(figsize=(8,6))
    plt.plot(steps, mean, label='Mean', color='b')
    plt.fill_between(steps, mean-std, mean+std, color='b', alpha=0.2, label='±1 SD')
    plt.title(f'{y_col.replace("_", " ").title()} Mean ± SD Over Time')
    plt.xlabel('Step')
    plt.ylabel(ylabel if ylabel else y_col)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    plt.savefig(fname)
    plt.close()

# --- Plot 1: XY Trajectory (Target at 0,0) ---
fig1, ax1 = plt.subplots(figsize=(8,8))
plot_trajectories_with_colors(
    ax1, xy_df, 'norm_position_x', 'norm_position_y',
    'XY Trajectory (Target at 0,0)', 'Normalized Position X', 'Normalized Position Y',
    x_lim=(xy_df['norm_position_x'].min() - 1, xy_df['norm_position_x'].max() + 1),
    y_lim=(xy_df['norm_position_y'].min() - 1, xy_df['norm_position_y'].max() + 1),
    add_target_lines=True
)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}xy_trajectory_normalized_random_hex.png') # Changed filename
plt.close(fig1) # Close the figure to free memory
print("done plotting xy trajectory")

# --- Plot 2: Position distance over time ---
fig2, ax2 = plt.subplots(figsize=(8,6))
plot_trajectories_with_colors(
    ax2, df, 'step', 'position_distance',
    'Position Distance to Target Over Time', 'Step', 'Distance'
)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}position_distance_over_time_random_hex.png') # Changed filename
plt.close(fig2)
print("done plotting position distance")

# --- Plot 4: Linear velocities over time ---
fig4, axs4 = plt.subplots(2, 1, figsize=(10,8), sharex=True)
plot_trajectories_with_colors(
    axs4[0], df, 'step', 'linear_velocity_x',
    'Linear Velocity X Over Time', 'Step', 'Linear Velocity X', show_legend=False # Legend handled by only one subplot or manually
)
plot_trajectories_with_colors(
    axs4[1], df, 'step', 'linear_velocity_y',
    'Linear Velocity Y Over Time', 'Step', 'Linear Velocity Y', show_legend=False
)
# Add a single legend if desired, referring to both plots.
# Or, if too many trajectories, omit.
if num_trajectories <= 20:
    handles, labels = axs4[0].get_legend_handles_labels()
    fig4.legend(handles, labels, title='Trajectories', bbox_to_anchor=(1.05, 1), loc='upper left')
axs4[-1].set_xlabel('Step') # Set x-label only on the last subplot for shared x-axis
plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust rect for legend
plt.savefig(f'{OUTPUT_DIR}linear_velocities_over_time_random_hex.png') # Changed filename
plt.close(fig4)
print("done plotting linear velocities")

# --- Plot 5: Actions over time ---
action_cols = [col for col in df.columns if col.startswith('actions_')]
fig5, axs5 = plt.subplots(len(action_cols), 1, figsize=(10, 2*len(action_cols)), sharex=True)
# Ensure axs5 is always an array for consistent indexing, even if only one action
if len(action_cols) == 1:
    axs5 = [axs5]

for i, col in enumerate(action_cols):
    plot_trajectories_with_colors(
        axs5[i], df, 'step', col,
        f'{col} Over Time', 'Step', col, show_legend=False # Legend handled centrally or omitted
    )
if num_trajectories <= 20:
    handles, labels = axs5[0].get_legend_handles_labels()
    fig5.legend(handles, labels, title='Trajectories', bbox_to_anchor=(1.05, 1), loc='upper left')
axs5[-1].set_xlabel('Step')
plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust rect for legend
plt.savefig(f'{OUTPUT_DIR}actions_over_time_random_hex.png') # Changed filename
plt.close(fig5)
print("done plotting actions")

# --- Plot: Angular velocity z over time ---
fig6, ax6 = plt.subplots(figsize=(8,6))
plot_trajectories_with_colors(
    ax6, df, 'step', 'angular_velocity_z',
    'Angular Velocity Z Over Time', 'Step', 'Angular Velocity Z'
)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}angular_velocity_z_over_time_random_hex.png') # Changed filename
plt.close(fig6)
print("done plotting angular velocity z")

print("done plotting trajectory-specific plots")

# --- Mean and SD plots for time-series (No changes needed as these are aggregate plots) ---
print("\nstarting plotting mean and sd plots...")

# Position distance mean ± sd
dist_col = 'position_distance'
plot_mean_sd(df, dist_col, f'{OUTPUT_DIR}position_distance_over_time_mean_sd.png', ylabel='Distance')

# Combined mean±SD plot for linear velocities (x and y)
fig7, axs7 = plt.subplots(2, 1, figsize=(10,8), sharex=True)
for i, col in enumerate(['linear_velocity_x', 'linear_velocity_y']):
    grouped = df.groupby('step')[col]
    mean = grouped.mean()
    std = grouped.std()
    steps = mean.index
    axs7[i].plot(steps, mean, label='Mean', color='b')
    axs7[i].fill_between(steps, mean-std, mean+std, color='b', alpha=0.2, label='±1 SD')
    axs7[i].set_title(f'{col.replace("_", " ").title()} Mean ± SD Over Time')
    axs7[i].set_ylabel(col)
    axs7[i].legend()
axs7[-1].set_xlabel('Step')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}linear_velocities_over_time_mean_sd.png')
plt.close(fig7)

# Combined mean±SD plot for actions
fig8, axs8 = plt.subplots(len(action_cols), 1, figsize=(10, 8), sharex=True)
# Ensure axs8 is always an array
if len(action_cols) == 1:
    axs8 = [axs8]
for i, col in enumerate(action_cols):
    grouped = df.groupby('step')[col]
    mean = grouped.mean()
    std = grouped.std()
    steps = mean.index
    axs8[i].plot(steps, mean, label='Mean', color='b')
    axs8[i].fill_between(steps, mean-std, mean+std, color='b', alpha=0.2, label='±1 SD')
    axs8[i].set_title(f'{col} Mean ± SD Over Time')
    axs8[i].set_ylabel(col)
    axs8[i].legend()
axs8[-1].set_xlabel('Step')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}actions_over_time_mean_sd.png')
plt.close(fig8)

# Angular velocity z mean ± sd
plot_mean_sd(df, 'angular_velocity_z', f'{OUTPUT_DIR}angular_velocity_z_over_time_mean_sd.png', ylabel='Angular Velocity Z')

print('All plots saved to metrics/.')