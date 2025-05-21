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
            "/workspace/isaaclab/logs/rsl_rl/Single/2025-05-21_06-18-20_rsl-rl_ppo_GoToPosition_Turtlebot2_r-0_seed-42",
        ],
        [
            "/workspace/isaaclab/logs/rsl_rl/Single/2025-05-21_06-25-59_rsl-rl_ppo_GoToPose_Turtlebot2_r-0_seed-42",
        ],
        [
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-20_15-34-23_rsl-rl_ppo_GoToPosition_Turtlebot2_r-0_seed-42",
        ],
        [
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-20_17-42-15_rsl-rl_ppo_GoToPosition_Turtlebot2_r-0_seed-42",
        ],
        [
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-20_17-45-42_rsl-rl_ppo_GoToPosition_Turtlebot2_r-0_seed-42",
        ],
        [
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-20_17-39-10_rsl-rl_ppo_GoToPosition-GoToPose_Turtlebot2_r-0_seed-42",
        ],
        [
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-20_17-48-31_rsl-rl_ppo_GoToPosition-GoToPose_Turtlebot2_r-0_seed-42",
        ],
    ]

    runs_names = [
        "Single-GoToPosition",
        "Single-GoToPose",
        "Multitask-Single-Obs",
        "Multitask-Multi-newObs",
        "Multitask-Multi-newObs-Tid",
        "Multitask(Pose,Posi)-Multi-newObs",
        "Multitask(Pose,Posi)-Multi-newObs-Tid",
    ]

    plot_cfg = {
        "title": "",
        "box_colors": [
            # "Grey",
            "Brown",
            # "Orange",
            # "Purple",
            # "Pink",
            "Green",
            "Blue",
        ],
        "runs_names": [
            "Single-GoToPose",
            "Single-GoToPosition",
            "Multitask-Single-Obs",
            "Multitask-Multi-newObs",
            "Multitask-Multi-newObs-Tid",
            "Multitask(Pose,Posi)-Multi-newObs",
            "Multitask(Pose,Posi)-Multi-newObs-Tid",
        ],
        "zoom_in": True,
    }

    save_plots_folder_path = "/workspace/isaaclab/source/plots/new_obs_buffer" # Specify the folder path where you want to save the plots
    if not os.path.exists(save_plots_folder_path):
        os.makedirs(save_plots_folder_path)

    dfs = {}
    labels = {}
    for group_idx, group in enumerate(list_of_folders):
        tasks_names = group[0].split("/")[-1].split("_")[4].split("-")
        for task_idx, task_name in enumerate(tasks_names):
            robot_name = group[0].split("/")[-1].split("_")[5]
            gorup_key = f"{task_name}_group-{group_idx}_{runs_names[group_idx]}"
            dfs[gorup_key] = []
            labels[gorup_key] = []
            for folder_path in group:
                try:
                    experiment_name = glob.glob(os.path.join(folder_path, "metrics", "*_metrics.csv"))[task_idx]
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
    # breakpoint()
    for task in tasks_names:
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
    