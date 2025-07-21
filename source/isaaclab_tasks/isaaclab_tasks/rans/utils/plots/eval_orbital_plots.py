import torch
from tasks import TaskPlotsFactory
from robots import RobotPlotsFactory

import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import yaml

def main():
    
    # Base directory where evaluation results are saved by the bash script
    base_metrics_dir = "source/evaluation_metrics/orbital_exp"
    save_plots_folder_path = "source/plots/orbital_thruster_patterns"  # Updated save path

    # Thruster pattern names for better labeling
    thruster_pattern_names = [
        "All Thrusters",        # pattern_0: [True, True, True, True, True, True, True, True]
        "6 Thrusters (No Front)", # pattern_1: [False, False, True, True, True, True, True, True]
        "6 Thrusters (Mixed)",    # pattern_2: [False, True, True, True, False, True, True, True]
        "3 Thrusters" , # pattern_3: [False, True, False, True, False, True, False, True]
        "4 Front Thrusters",      # pattern_4: [True, True, False, False, False, False, False, False]
        "1 Thruster Only",        # pattern_5: [True, False, False, False, False, False, False, False]
    ]

    # Custom run names for the plots - customize these as needed
    # This will map run0, run1, etc. to your preferred labels
    # The script will automatically generate fallback names based on experiment and pattern
    custom_run_names = {
        # Example mappings - adjust based on your actual experiments
        "run0": "Memory PPO - All Thrusters",
        "run1": "Memory PPO - 6T (No Front)", 
        "run2": "Memory PPO - 6T (Mixed)",
        "run3": "Memory PPO - 3T",
        "run4": "Memory PPO - 4 Front Thrusters",
        "run5": "Memory PPO - 1 Thruster Only",
        "run6": "Standard PPO - All Thrusters",
        "run7": "Standard PPO - 6T (No Front)",
        "run8": "Standard PPO - 6T (Mixed)",
        "run9": "Standard PPO - 3T",
        "run10": "Standard PPO - 4 Front Thrusters",
        "run11": "Standard PPO - 1 Thruster Only",
        # Add more mappings as needed...
    }

    plot_cfg = {
        "title": "",
        "box_colors": [
            "#FF3D50", "#FFB3BA",  # Red variants for experiment 1
            "#FFA034", "#FFD4A3",  # Orange variants
            "#2FA1FF", "#A3D4FF",  # Blue variants
            "#A734FF", "#D4A3FF",  # Purple variants
            "#FFFF3D", "#FFFFBA",  # Yellow variants
            "#4DFF3D", "#B3FFB3",  # Green variants
            "#FF3DBB", "#FFB3E6",  # Pink variants
            "#623652", "#9B8AA3",  # Dark variants
            "#3DFFB3", "#BAFFF3",  # Light green variants
            "#FF573D", "#FFB3A3",  # Coral variants
            "#3DFFB3", "#B3FFDA",  # Mint variants
        ],
        "runs_names": [],  # Will be populated dynamically with custom labels
        "zoom_in": False,
    }
    
    # Automatically discover all experiment patterns
    print(f"Scanning {base_metrics_dir} for experiment patterns...")
    
    if not os.path.exists(base_metrics_dir):
        print(f"Error: Directory {base_metrics_dir} does not exist!")
        return
    
    # Find all timestamped experiment directories
    experiment_dirs = [d for d in os.listdir(base_metrics_dir) 
                       if os.path.isdir(os.path.join(base_metrics_dir, d)) and d != "logs"]
    
    if not experiment_dirs:
        print(f"No experiment directories found in {base_metrics_dir}")
        return
    
    print(f"Found {len(experiment_dirs)} experiment directories:")
    for exp_dir in sorted(experiment_dirs):
        print(f"  - {exp_dir}")
        
        # Check what pattern folders exist in this experiment
        exp_path = os.path.join(base_metrics_dir, exp_dir)
        pattern_folders = [d for d in os.listdir(exp_path) 
                          if os.path.isdir(os.path.join(exp_path, d)) and d.startswith("pattern_")]
        print(f"    Patterns: {sorted(pattern_folders)}")
    
    # Store experiment directories for processing
    base_experiments = sorted(experiment_dirs)
    
    
    if not os.path.exists(save_plots_folder_path):
        os.makedirs(save_plots_folder_path)

    dfs = {}
    # trajectories_dfs = {}
    labels = {}
    list_of_tasks = []
    
    print(f"\nProcessing all experiments...")
    
    # Process each timestamped experiment directory
    run_counter = 0
    for experiment_dir in base_experiments:
        experiment_path = os.path.join(base_metrics_dir, experiment_dir)
        print(f"\nProcessing experiment: {experiment_dir}")
        
        # Find all pattern folders in this experiment
        pattern_folders = [d for d in os.listdir(experiment_path) 
                          if os.path.isdir(os.path.join(experiment_path, d)) and d.startswith("pattern_")]
        
        # Sort pattern folders to ensure consistent ordering
        pattern_folders.sort()
        
        for pattern_folder in pattern_folders:
            pattern_path = os.path.join(experiment_path, pattern_folder)
            pattern_idx = int(pattern_folder.split("_")[1])
            
            print(f"  Processing {pattern_folder}")
            
            # Extract task and robot info from experiment directory name
            # Format: 2025-07-17_18-34-33_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-42
            exp_parts = experiment_dir.split("_")
            task_name = "GoToPosition"  # Default, can be extracted from folder name if needed
            robot_name = "ModularFreeflyer"  # Default, can be extracted from folder name if needed
            
            # Try to extract task and robot from folder name
            for i, part in enumerate(exp_parts):
                if part in ["GoToPosition", "GoToPose", "Rendezvous"]:
                    task_name = part
                    if i + 1 < len(exp_parts) and exp_parts[i + 1] in ["ModularFreeflyer"]:
                        robot_name = exp_parts[i + 1]
                    break
            
            if pattern_idx < len(thruster_pattern_names):
                run_name = thruster_pattern_names[pattern_idx]
            else:
                run_name = f"Pattern {pattern_idx}"
            
            # Create a unique group key for this experiment + pattern combination
            group_key = f"{task_name}_{experiment_dir}_pattern-{pattern_idx}"
            
            if group_key not in dfs:
                dfs[group_key] = []
                # trajectories_dfs[group_key] = []
                labels[group_key] = []
            
            if task_name not in list_of_tasks:
                list_of_tasks.append(task_name)
            
            try:
                # Look for metrics files directly in the pattern folder
                metrics_files = glob.glob(os.path.join(pattern_path, "*_metrics.csv"))
                if metrics_files:
                    df = pd.read_csv(metrics_files[0])
                    dfs[group_key].append(df)
                    labels[group_key].append(f"{experiment_dir}_{os.path.basename(metrics_files[0])}")
                    print(f"    Loaded pattern {pattern_idx} ({run_name}): {os.path.basename(metrics_files[0])}")
                else:
                    print(f"    Warning: No metrics CSV found in {pattern_path}")
                    continue

                # trajectories_files = glob.glob(os.path.join(pattern_path, f"extracted_trajectories_{task_name}.csv"))
                # if trajectories_files:
                #     trajectories_dfs[group_key].append(pd.read_csv(trajectories_files[0]))
                #     print(f"    Loaded trajectories: {os.path.basename(trajectories_files[0])}")
                # else:
                #     print(f"    Warning: No trajectories file found in {pattern_path}")
                #     trajectories_dfs[group_key].append(pd.DataFrame())

                # Look for env_info.yaml in the pattern folder or parent experiment folder
                env_info_files = glob.glob(os.path.join(pattern_path, "env_info.yaml"))
                if not env_info_files:
                    env_info_files = glob.glob(os.path.join(experiment_path, "env_info.yaml"))
                
                if env_info_files:
                    with open(env_info_files[0], 'r') as f:
                        env_info = yaml.safe_load(f)
                else:
                    env_info = {"task_name": task_name, "robot_name": robot_name, "num_envs": 4}
                        
            except Exception as e:
                print(f"    Error reading files in {pattern_path}: {e}")
                continue
            
            run_counter += 1
    
    # Create combined plots for all experiments
    if dfs:
        print(f"\nCreating combined plots for all experiments...")
        
        # --- MODIFICATION START ---
        # Create final plot labels and rename dictionary keys to match them.
        final_plot_labels = []
        new_dfs = {}
        # new_trajectories_dfs = {}
        new_labels = {}
        
        sorted_keys = sorted(dfs.keys())
        
        print("\nMapping runs to custom plot labels:")
        for i, key in enumerate(sorted_keys):
            run_id = f"run{i}"
            
            # Generate an automatic fallback name from the original key
            parts = key.split('_')
            auto_name_fallback = key
            if len(parts) >= 3 and parts[-1].startswith('pattern-'):
                pattern_idx = int(parts[-1].split('-')[1])
                pattern_name = thruster_pattern_names[pattern_idx] if pattern_idx < len(thruster_pattern_names) else f"Pattern {pattern_idx}"
                
                # Extract meaningful experiment identifier from timestamped folder name
                # Look for algorithm type (ppo, ppo-memory) and other identifying features
                exp_identifier = "Unknown"
                for part in parts[1:-1]:  # Skip task name and pattern part
                    if "ppo-memory" in part:
                        exp_identifier = "Memory PPO"
                        break
                    elif "ppo" in part and "memory" not in part:
                        exp_identifier = "Standard PPO"
                        break
                    elif part.startswith("2025-"):
                        # If we find a timestamp, we can use it as identifier
                        date_part = part
                        exp_identifier = f"Exp {date_part}"
                        break
                
                auto_name_fallback = f"{exp_identifier} - {pattern_name}"
            
            # Get the final label from custom_run_names, or use the fallback
            final_label = custom_run_names.get(run_id, auto_name_fallback)
            final_plot_labels.append(final_label)
            
            # Create the new dictionary key, e.g., "GoToPosition_Memory PPO - All Thrusters"
            # This ensures the key's suffix matches the label used in the plot.
            task_part = parts[0]
            new_key = f"{task_part}_{final_label}"
            
            new_dfs[new_key] = dfs[key]
            # new_trajectories_dfs[new_key] = trajectories_dfs[key]
            new_labels[new_key] = labels[key]
            
            print(f"  '{key}' -> Plotting as: '{final_label}'")

        # Update data structures to use the new keys
        dfs = new_dfs
        # trajectories_dfs = new_trajectories_dfs
        labels = new_labels
        
        # Assign the custom labels to the plot configuration
        plot_cfg["runs_names"] = final_plot_labels
        
        # Group data by task using the new keys
        task_dfs = {}
        # task_trajectories_dfs = {}
        
        for task in list_of_tasks:
            task_dfs[task] = {}
            # task_trajectories_dfs[task] = {}
            for key in dfs:
                if key.startswith(f"{task}_"):
                    task_dfs[task][key] = dfs[key]
                    # task_trajectories_dfs[task][key] = trajectories_dfs[key]
        # --- MODIFICATION END ---
        
        # Create plots for each task
        for task_name in task_dfs:
            if task_dfs[task_name]:
                print(f"\nCreating plots for task: {task_name}")
                
                task_plots_factory = TaskPlotsFactory.create(
                    task_name, 
                    dfs=task_dfs[task_name],
                    trajectories_dfs={},#task_trajectories_dfs[task_name],
                    labels=labels,
                    env_info=env_info,
                    folder_path=save_plots_folder_path,
                    plot_cfg=plot_cfg,
                )
                task_plots_factory.plot()
                
                robot_plots_factory = RobotPlotsFactory.create(
                    robot_name, 
                    dfs=dfs, # Robot plots may need all data across tasks
                    trajectories_dfs={},#trajectories_dfs,
                    labels=labels,
                    env_info=env_info,
                    folder_path=save_plots_folder_path,
                    plot_cfg=plot_cfg,
                )
                robot_plots_factory.plot()
    
    print(f"\nAll plots completed!")
    print(f"Results saved in: {save_plots_folder_path}")

if __name__ == "__main__":
    main()