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
        [
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-27_12-08-03_rsl-rl_ppo_GoToPosition_FloatingPlatform_r-0_seed-1",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-27_12-30-15_rsl-rl_ppo_GoToPosition_FloatingPlatform_r-0_seed-2",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-27_12-52-10_rsl-rl_ppo_GoToPosition_FloatingPlatform_r-0_seed-3",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-27_13-14-26_rsl-rl_ppo_GoToPosition_FloatingPlatform_r-0_seed-4",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-27_13-36-31_rsl-rl_ppo_GoToPosition_FloatingPlatform_r-0_seed-5",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-27_13-58-32_rsl-rl_ppo_GoToPosition_FloatingPlatform_r-0_seed-6",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-27_14-20-47_rsl-rl_ppo_GoToPosition_FloatingPlatform_r-0_seed-7",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-27_14-42-56_rsl-rl_ppo_GoToPosition_FloatingPlatform_r-0_seed-8",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-27_15-04-58_rsl-rl_ppo_GoToPosition_FloatingPlatform_r-0_seed-9",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-27_15-27-00_rsl-rl_ppo_GoToPosition_FloatingPlatform_r-0_seed-10",
        ],
        [
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-27_15-49-05_rsl-rl_ppo_GoToPose_FloatingPlatform_r-0_seed-1",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-27_16-11-05_rsl-rl_ppo_GoToPose_FloatingPlatform_r-0_seed-2",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-27_16-33-14_rsl-rl_ppo_GoToPose_FloatingPlatform_r-0_seed-3",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-27_16-55-20_rsl-rl_ppo_GoToPose_FloatingPlatform_r-0_seed-4",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-27_17-17-23_rsl-rl_ppo_GoToPose_FloatingPlatform_r-0_seed-5",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-27_17-39-11_rsl-rl_ppo_GoToPose_FloatingPlatform_r-0_seed-6",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-27_18-01-08_rsl-rl_ppo_GoToPose_FloatingPlatform_r-0_seed-7",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-27_18-22-57_rsl-rl_ppo_GoToPose_FloatingPlatform_r-0_seed-8",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-27_18-44-47_rsl-rl_ppo_GoToPose_FloatingPlatform_r-0_seed-9",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-27_19-06-25_rsl-rl_ppo_GoToPose_FloatingPlatform_r-0_seed-10",
        ],
        [
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-27_19-28-25_rsl-rl_ppo_TrackVelocities_FloatingPlatform_r-0_seed-1",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-27_19-51-51_rsl-rl_ppo_TrackVelocities_FloatingPlatform_r-0_seed-2",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-27_20-15-34_rsl-rl_ppo_TrackVelocities_FloatingPlatform_r-0_seed-3",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-27_20-39-08_rsl-rl_ppo_TrackVelocities_FloatingPlatform_r-0_seed-4",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-27_21-02-41_rsl-rl_ppo_TrackVelocities_FloatingPlatform_r-0_seed-5",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-27_21-26-18_rsl-rl_ppo_TrackVelocities_FloatingPlatform_r-0_seed-6",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-27_21-49-42_rsl-rl_ppo_TrackVelocities_FloatingPlatform_r-0_seed-7",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-27_22-13-20_rsl-rl_ppo_TrackVelocities_FloatingPlatform_r-0_seed-8",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-27_22-37-05_rsl-rl_ppo_TrackVelocities_FloatingPlatform_r-0_seed-9",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-27_23-00-56_rsl-rl_ppo_TrackVelocities_FloatingPlatform_r-0_seed-10",
        ],
        [
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-27_23-24-42_rsl-rl_ppo_GoThroughPoses_FloatingPlatform_r-0_seed-1",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-27_23-50-03_rsl-rl_ppo_GoThroughPoses_FloatingPlatform_r-0_seed-2",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-28_00-15-53_rsl-rl_ppo_GoThroughPoses_FloatingPlatform_r-0_seed-3",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-28_00-41-41_rsl-rl_ppo_GoThroughPoses_FloatingPlatform_r-0_seed-4",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-28_01-07-34_rsl-rl_ppo_GoThroughPoses_FloatingPlatform_r-0_seed-5",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-28_01-33-19_rsl-rl_ppo_GoThroughPoses_FloatingPlatform_r-0_seed-6",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-28_01-58-53_rsl-rl_ppo_GoThroughPoses_FloatingPlatform_r-0_seed-7",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-28_02-24-30_rsl-rl_ppo_GoThroughPoses_FloatingPlatform_r-0_seed-8",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-28_02-50-32_rsl-rl_ppo_GoThroughPoses_FloatingPlatform_r-0_seed-9",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-28_03-16-29_rsl-rl_ppo_GoThroughPoses_FloatingPlatform_r-0_seed-10",
        ],
        [
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-28_03-42-31_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-1",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-28_04-18-05_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-2",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-28_04-54-38_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-3",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-28_05-30-44_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-4",
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
            "#FFA034", 
            "#2FA5FF", 
            "#7434FF",
            "#4FFF75",
        ],
        "runs_names": [
            "Detumble",
            "Docking",
            "Track Velocities",
            "Rendezvous",
            "MTRL",
        ],
        "zoom_in": False,
    }

    save_plots_folder_path = "/workspace/isaaclab/source/plots/mtlrl_256" # Specify the folder path where you want to save the plots
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
    