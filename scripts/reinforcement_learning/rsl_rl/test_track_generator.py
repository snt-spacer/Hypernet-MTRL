import argparse
import sys
from isaaclab.app import AppLauncher
import cli_args  # isort: skip
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_tasks.rans.utils import PerEnvSeededRNG, TrackGenerator
import torch
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import numpy as np

# Function to plot gates
def plot_gates(ax, num_goals, gate_x, gate_y, gate_headings):
    gate_width = 1.5
    gate_height = 2.5
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

if __name__ == "__main__":
    device = "cuda:0"
    num_envs = 65536
    env_ids = torch.arange(num_envs, device=device)
    num_iterations = 100  # Number of times to run the generation for averaging
    
    rng = PerEnvSeededRNG(0, num_envs, device)
    track_gen = TrackGenerator(
        scale=50.0,
        rad=0.2,
        edgy=0.0,
        max_num_points=20,
        min_num_points=5,
        min_point_distance=0.05,
        rng=rng,
    )

    # Warm-up run (not included in timing)
    # print("Performing warm-up run...")
    # _ = track_gen.generate_tracks_points_non_fixed_points(env_ids)
    
    # # Timing loop
    # times = []
    # print(f"Running {num_iterations} iterations for timing...")
    
    # for i in range(num_iterations):
    #     start_time = time.time()
    #     points, tangents, num_goals = track_gen.generate_tracks_points_non_fixed_points(env_ids)
    #     end_time = time.time()
        
    #     iteration_time = end_time - start_time
    #     times.append(iteration_time)
    #     print(f"Iteration {i+1}/{num_iterations}: {iteration_time:.4f}s")
    
    # # Calculate and display statistics
    # avg_time = sum(times) / len(times)
    # min_time = min(times)
    # max_time = max(times)
    
    # print(f"\n--- Timing Results ---")
    # print(f"Average time: {avg_time:.4f}s")
    # print(f"Minimum time: {min_time:.4f}s")
    # print(f"Maximum time: {max_time:.4f}s")
    # print(f"Number of environments: {num_envs}")
    # print(f"Time per environment: {avg_time/num_envs*1000:.4f}ms")
    # print(f"Total number of points generated per iteration: {points.shape[0] * points.shape[1]}")
    
    # Plot 10 example tracks in 2 rows of 5
    print("\nGenerating plots for 10 example tracks...")
    
    # Create subplot layout - 2 rows, 5 columns
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    # fig.suptitle('Track Generation Examples with Gates and Bezier Curves', fontsize=16)
    
    # Generate and plot each track individually to get different tracks
    all_plot_points = []
    all_plot_tangents = []
    all_plot_num_goals = []
    all_plot_curves = []
    
    for i in range(10):
        # Calculate row and column for 2x5 grid
        row = i // 5
        col = i % 5
        ax = axes[row, col]
        
        # Generate a single track for each iteration
        single_env_id = torch.arange(10, device=device)
        plot_points, plot_tangents, plot_num_goals = track_gen.generate_tracks_points_non_fixed_points(single_env_id)
        
        # Generate Bezier curve for this track
        plot_curves = track_gen.get_bezier_curve_non_fixed_points(plot_points, plot_tangents, plot_num_goals)
        
        # Store for later use
        all_plot_points.append(plot_points[0])
        all_plot_tangents.append(plot_tangents[0])
        all_plot_num_goals.append(plot_num_goals[0])
        all_plot_curves.append(plot_curves[0])
        
        # Get data for this track
        track_points = plot_points[0].cpu().numpy()
        track_curve = plot_curves[0].cpu().numpy()
        num_actual_points = plot_num_goals[0].item()
        
        # Only use the actual points (not the padded ones)
        actual_points = track_points[:num_actual_points]
        
        # Plot the Bezier curve
        ax.plot(track_curve[:, 0], track_curve[:, 1], 'mediumseagreen', linewidth=5, alpha=0.7, label='Bezier Curve')

        # Plot the control points (gates) as rectangles
        plot_gates(ax, num_actual_points, actual_points[:, 0], actual_points[:, 1], plot_tangents[0][:num_actual_points].cpu().numpy())
        
        # Connect gates with lines to show the intended path
        # Close the loop by adding the first point at the end
        gate_loop = np.vstack([actual_points, actual_points[0:1]])
        ax.plot(gate_loop[:, 0], gate_loop[:, 1], 'slateblue', alpha=0.5, linewidth=1, label='Gate Connections')
        
        # Set equal aspect ratio and add grid
        ax.set_aspect('equal')
        ax.grid(False)
        ax.set_facecolor('whitesmoke')#whitesmoke
        ax.set_title(f'Track {i+1}')
        
        # Add legend only to the first plot to avoid clutter
        if i == 0:
            ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.grid(False)
    plt.savefig('source/track_examples.svg', bbox_inches='tight')
    plt.show()
    
    # # Additional plot: Show curve segments for one track in detail
    # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # # Use the first track for detailed visualization
    # detailed_track_idx = 0
    # track_points = plot_points[detailed_track_idx].cpu().numpy()
    # track_curve = plot_curves[detailed_track_idx].cpu().numpy()
    # num_actual_points = plot_num_goals[detailed_track_idx].item()
    # actual_points = track_points[:num_actual_points]
    
    # # Plot the full Bezier curve
    # ax.plot(track_curve[:, 0], track_curve[:, 1], 'b-', linewidth=3, label='Complete Bezier Curve')
    
    # # Plot individual segments in different colors
    # points_per_segment = track_gen._num_points_per_segment
    # colors = plt.cm.rainbow(np.linspace(0, 1, num_actual_points))
    
    # for i in range(num_actual_points):
    #     start_idx = i * points_per_segment
    #     end_idx = (i + 1) * points_per_segment
    #     segment = track_curve[start_idx:end_idx]
    #     ax.plot(segment[:, 0], segment[:, 1], color=colors[i], linewidth=2, alpha=0.8, 
    #            label=f'Segment {i}' if i < 5 else '')  # Only label first 5 to avoid clutter
    
    # # Plot gates as rectangles
    # plot_gates(ax, num_actual_points, actual_points[:, 0], actual_points[:, 1], tangents[:num_actual_points])
    
    # # Draw tangent vectors at gates
    # scale = 5.0  # Scale factor for tangent visualization
    # for j, (point, angle) in enumerate(zip(actual_points, tangents[:num_actual_points])):
    #     dx = scale * np.cos(angle)
    #     dy = scale * np.sin(angle)
    #     ax.arrow(point[0], point[1], dx, dy, head_width=1, head_length=1, 
    #             fc='green', ec='green', alpha=0.7)
    
    # ax.set_aspect('equal')
    # ax.grid(True, alpha=0.3)
    # ax.set_title(f'Detailed Track Visualization (Track {detailed_track_idx+1})\n'
    #             f'Gates: {num_actual_points}, Segments: {num_actual_points}, Points per segment: {points_per_segment}')
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # plt.tight_layout()
    # plt.savefig('source/detailed_track.png', dpi=300, bbox_inches='tight')
    # # plt.show()
    
    # print("Plots saved as 'source/track_examples.png' and 'detailed_track.png'")