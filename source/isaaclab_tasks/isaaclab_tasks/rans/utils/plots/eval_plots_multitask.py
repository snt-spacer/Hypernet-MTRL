import torch
from tasks import TaskPlotsFactory
from robots import RobotPlotsFactory

import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import yaml

def main():
    list_of_folders = [
        # 32
        [
            "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x32/2025-06-11_09-23-20_rsl-rl_ppo_GoToPosition_FloatingPlatform_r-0_seed-1",
            "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x32/2025-06-11_09-44-40_rsl-rl_ppo_GoToPosition_FloatingPlatform_r-0_seed-2",
        ],
        [
            "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x32/2025-06-11_10-06-00_rsl-rl_ppo_GoToPose_FloatingPlatform_r-0_seed-1",
            "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x32/2025-06-11_10-27-12_rsl-rl_ppo_GoToPose_FloatingPlatform_r-0_seed-2",
        ],
        [
            "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x32/2025-06-11_10-48-29_rsl-rl_ppo_TrackVelocities_FloatingPlatform_r-0_seed-1",
            "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x32/2025-06-11_11-10-33_rsl-rl_ppo_TrackVelocities_FloatingPlatform_r-0_seed-2",
        ],
        [
            "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x32/2025-06-11_11-32-22_rsl-rl_ppo_GoThroughPoses_FloatingPlatform_r-0_seed-1",
            "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x32/2025-06-11_11-59-24_rsl-rl_ppo_GoThroughPoses_FloatingPlatform_r-0_seed-2",
        ],
        [
            "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x32/2025-06-11_12-26-10_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-1",
            "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x32/2025-06-11_13-16-26_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-2",
        ],
        # 64
        [
            "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64/2025-06-11_14-16-44_rsl-rl_ppo_GoToPosition_FloatingPlatform_r-0_seed-1",
            "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64/2025-06-11_14-32-02_rsl-rl_ppo_GoToPosition_FloatingPlatform_r-0_seed-2",
        ],
        [
            "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64/2025-06-11_14-47-23_rsl-rl_ppo_GoToPose_FloatingPlatform_r-0_seed-1",
            "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64/2025-06-11_15-02-44_rsl-rl_ppo_GoToPose_FloatingPlatform_r-0_seed-2",
        ],
        [
            "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64/2025-06-11_15-18-14_rsl-rl_ppo_TrackVelocities_FloatingPlatform_r-0_seed-1",
            "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64/2025-06-11_15-34-23_rsl-rl_ppo_TrackVelocities_FloatingPlatform_r-0_seed-2",
        ],
        [
            "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64/2025-06-11_15-50-25_rsl-rl_ppo_GoThroughPoses_FloatingPlatform_r-0_seed-1",
            "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64/2025-06-11_16-10-22_rsl-rl_ppo_GoThroughPoses_FloatingPlatform_r-0_seed-2",
        ],
        [
            "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64/2025-06-11_16-30-37_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-1",
            "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64/2025-06-11_17-09-28_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-2",
        ],
          # 128  
        [
            "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x128/2025-06-11_14-19-20_rsl-rl_ppo_GoToPosition_FloatingPlatform_r-0_seed-1",
            "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x128/2025-06-11_14-50-50_rsl-rl_ppo_GoToPosition_FloatingPlatform_r-0_seed-2",
        ],
        [
            "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x128/2025-06-11_15-22-26_rsl-rl_ppo_GoToPose_FloatingPlatform_r-0_seed-1",
            "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x128/2025-06-11_15-53-51_rsl-rl_ppo_GoToPose_FloatingPlatform_r-0_seed-2",
        ],
        [
            "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x128/2025-06-11_16-25-11_rsl-rl_ppo_TrackVelocities_FloatingPlatform_r-0_seed-1",
            "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x128/2025-06-11_16-57-36_rsl-rl_ppo_TrackVelocities_FloatingPlatform_r-0_seed-2",
        ],
        [
            "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x128/2025-06-11_17-30-19_rsl-rl_ppo_GoThroughPoses_FloatingPlatform_r-0_seed-1",
            "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x128/2025-06-11_18-10-32_rsl-rl_ppo_GoThroughPoses_FloatingPlatform_r-0_seed-2",
        ],
        [
            "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x128/2025-06-11_18-50-46_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-1",
            "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x128/2025-06-11_20-04-55_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-2",
        ],
    ]

    plot_cfg = {
        "title": "",
        "box_colors": [
            # '#FFB3BA',  # Light Pink
            # '#FFDFBA',  # Light Orange
            # # '#FFFFBA',  # Light Yellow
            # # '#BAFFC9',  # Light Green
            # '#BAE1FF',  # Light Blue
            # '#D0BAFF',   # Light Purple

            "#FF3D50", 
            "#FF3D50", 
            "#FF3D50", 
            "#FF3D50",
            "#FFA034",

            "#2FA1FF", 
            "#2FA1FF",  
            "#2FA1FF",  
            "#2FA1FF", 
            "#A734FF",

            "#FFFF3D", 
            "#FFFF3D", 
            "#FFFF3D", 
            "#FFFF3D",
            "#4DFF3D", 
        ],
        "runs_names": [
            "Detumble 32",
            "Docking 32",
            "Track Velocities 32",
            "Rendezvous 32",
            "MTRL 32",
            "Detumble 64",
            "Docking 64",
            "Track Velocities 64",
            "Rendezvous 64",
            "MTRL 64",
            "Detumble 128",
            "Docking 128",
            "Track Velocities 128",
            "Rendezvous 128",
            "MTRL 128",
        ],
        "zoom_in": False,
    }

    save_plots_folder_path = "/workspace/isaaclab/source/plots/mtlrl_deep_nets" # Specify the folder path where you want to save the plots
    if not os.path.exists(save_plots_folder_path):
        os.makedirs(save_plots_folder_path)

    dfs = {}
    labels = {}
    list_of_tasks = []
    for group_idx, group in enumerate(list_of_folders):
        tasks_names = group[0].split("/")[-1].split("_")[4].split("-")
        for task_idx, task_name in enumerate(tasks_names):
            if task_name not in list_of_tasks:
                list_of_tasks.append(task_name)
            robot_name = group[0].split("/")[-1].split("_")[5]
            run_name = plot_cfg["runs_names"][group_idx]
            gorup_key = f"{task_name}_group-{group_idx}_{run_name}"
            dfs[gorup_key] = []
            labels[gorup_key] = []
            for folder_path in group:
                try:
                    experiment_name = None # Initialize to None in case no match is found
                    try:
                        experiment_name = next(
                            f for f in glob.glob(os.path.join(folder_path, "metrics", "*_metrics.csv"))
                            if task_name in f.split("/")[-1]
                        )
                    except StopIteration:
                        print(f"Error reading file in {experiment_name} in {folder_path}")
                        exit(0)

                    # experiment_name = glob.glob(os.path.join(folder_path, "metrics", "*_metrics.csv"))[task_idx]
                    metrics_file_path = os.path.join(folder_path, "metrics", experiment_name)
                    df = pd.read_csv(metrics_file_path)
                    dfs[gorup_key].append(df)
                    labels[gorup_key].append(experiment_name.split("/")[-1])

                    env_info_file_path = os.path.join(folder_path, "metrics", "env_info.yaml")
                    with open(env_info_file_path, 'r') as f:
                        env_info = yaml.safe_load(f)
                except Exception as e:
                    print(f"Error reading file in {experiment_name}: {e}")
                    exit(0)
    
    task_dfs = {}
    # 
    for task in list_of_tasks:
        # Select all DataFrames whose key starts with the task name
        task_dfs[task] = [dfs[key] for key in dfs if key.startswith(task)]

    for task_name in task_dfs:
        # Collect the per-group DataFrames and labels
        task_group_keys = [key for key in dfs if key.startswith(task_name)]
        
        task_group_dfs = {
            key: dfs[key] for key in task_group_keys
        }
        task_group_labels = {
            key: labels[key] for key in task_group_keys
        }

        print("-"*30)
        print(f"Taksk name: {task_name}, task_group_dfs: {task_group_dfs.keys()}, task_group_labels: {task_group_labels.keys()}")
        task_plots_factory = TaskPlotsFactory.create(
            task_name,
            dfs=task_group_dfs,
            labels=task_group_labels,
            env_info=env_info,
            folder_path=save_plots_folder_path,
            plot_cfg=plot_cfg,
        )
        task_plots_factory.plot()
    
        robot_plots_factory = RobotPlotsFactory.create(
            robot_name, 
            dfs=dfs,
            labels=labels,
            env_info=env_info,
            folder_path=folder_path,
            plot_cfg=plot_cfg,
        )
        robot_plots_factory.plot()

if __name__ == "__main__":
    main()
    