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
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-22_13-24-37_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities_Turtlebot2_r-0_seed-1",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-22_13-35-01_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities_Turtlebot2_r-0_seed-2",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-22_13-45-23_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities_Turtlebot2_r-0_seed-3",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-22_13-55-42_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities_Turtlebot2_r-0_seed-4",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-22_14-06-08_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities_Turtlebot2_r-0_seed-5",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-22_14-16-29_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities_Turtlebot2_r-0_seed-6",
        ],
        [
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-22_16-26-02_rsl-rl_ppo_GoToPosition_Turtlebot2_r-0_seed-1",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-22_16-33-36_rsl-rl_ppo_GoToPosition_Turtlebot2_r-0_seed-2",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-22_16-41-17_rsl-rl_ppo_GoToPosition_Turtlebot2_r-0_seed-3",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-22_16-48-49_rsl-rl_ppo_GoToPosition_Turtlebot2_r-0_seed-4",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-22_16-56-25_rsl-rl_ppo_GoToPosition_Turtlebot2_r-0_seed-5",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-22_17-03-57_rsl-rl_ppo_GoToPosition_Turtlebot2_r-0_seed-6",
        ],
        [
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-22_18-07-40_rsl-rl_ppo_GoToPose_Turtlebot2_r-0_seed-1",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-22_18-15-24_rsl-rl_ppo_GoToPose_Turtlebot2_r-0_seed-2",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-22_18-23-03_rsl-rl_ppo_GoToPose_Turtlebot2_r-0_seed-3",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-22_18-30-48_rsl-rl_ppo_GoToPose_Turtlebot2_r-0_seed-4",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-22_18-38-26_rsl-rl_ppo_GoToPose_Turtlebot2_r-0_seed-5",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-22_18-46-07_rsl-rl_ppo_GoToPose_Turtlebot2_r-0_seed-6",
        ],
        [
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-23_06-20-58_rsl-rl_ppo_TrackVelocities_Turtlebot2_r-0_seed-1",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-23_06-28-34_rsl-rl_ppo_TrackVelocities_Turtlebot2_r-0_seed-2",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-23_06-36-14_rsl-rl_ppo_TrackVelocities_Turtlebot2_r-0_seed-3",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-23_06-43-56_rsl-rl_ppo_TrackVelocities_Turtlebot2_r-0_seed-4",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-23_06-51-40_rsl-rl_ppo_TrackVelocities_Turtlebot2_r-0_seed-5",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-23_06-59-26_rsl-rl_ppo_TrackVelocities_Turtlebot2_r-0_seed-6",
        ],


        [
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-23_16-54-58_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities_Turtlebot2_r-0_seed-1",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-23_17-14-06_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities_Turtlebot2_r-0_seed-2",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-23_17-33-08_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities_Turtlebot2_r-0_seed-3",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-23_17-52-07_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities_Turtlebot2_r-0_seed-4",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-23_18-11-06_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities_Turtlebot2_r-0_seed-5",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-23_18-30-08_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities_Turtlebot2_r-0_seed-6",
        ],
        [
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-23_12-32-12_rsl-rl_ppo_GoToPosition_Turtlebot2_r-0_seed-1",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-23_12-46-39_rsl-rl_ppo_GoToPosition_Turtlebot2_r-0_seed-2",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-23_13-01-04_rsl-rl_ppo_GoToPosition_Turtlebot2_r-0_seed-3",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-23_13-15-33_rsl-rl_ppo_GoToPosition_Turtlebot2_r-0_seed-4",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-23_13-29-46_rsl-rl_ppo_GoToPosition_Turtlebot2_r-0_seed-5",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-23_13-44-18_rsl-rl_ppo_GoToPosition_Turtlebot2_r-0_seed-6",
        ],
        [
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-23_13-58-47_rsl-rl_ppo_GoToPose_Turtlebot2_r-0_seed-1",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-23_14-13-30_rsl-rl_ppo_GoToPose_Turtlebot2_r-0_seed-2",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-23_14-28-20_rsl-rl_ppo_GoToPose_Turtlebot2_r-0_seed-3",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-23_14-43-07_rsl-rl_ppo_GoToPose_Turtlebot2_r-0_seed-4",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-23_14-57-39_rsl-rl_ppo_GoToPose_Turtlebot2_r-0_seed-5",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-23_15-12-26_rsl-rl_ppo_GoToPose_Turtlebot2_r-0_seed-6",
        ],
        [
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-23_15-27-11_rsl-rl_ppo_TrackVelocities_Turtlebot2_r-0_seed-1",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-23_15-41-51_rsl-rl_ppo_TrackVelocities_Turtlebot2_r-0_seed-2",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-23_15-56-28_rsl-rl_ppo_TrackVelocities_Turtlebot2_r-0_seed-3",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-23_16-10-58_rsl-rl_ppo_TrackVelocities_Turtlebot2_r-0_seed-4",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-23_16-25-34_rsl-rl_ppo_TrackVelocities_Turtlebot2_r-0_seed-5",
            "/workspace/isaaclab/logs/rsl_rl/multitask/2025-05-23_16-40-15_rsl-rl_ppo_TrackVelocities_Turtlebot2_r-0_seed-6",
        ],
    ]

    plot_cfg = {
        "title": "",
        "box_colors": [
            '#FFB3BA',  # Light Pink
            '#FFDFBA',  # Light Orange
            # '#FFFFBA',  # Light Yellow
            # '#BAFFC9',  # Light Green
            '#BAE1FF',  # Light Blue
            '#D0BAFF',   # Light Purple

            "#FF3D50", 
            "#FFA034", 
            "#2FA5FF", 
            "#7434FF",  
        ],
        "runs_names": [
            "MTRL (Pose, Position, Vel) 32",
            "MTRL (Position) 32",
            "MTRL (Pose) 32",
            "MTRL (Vel) 32",
            "MTRL (Pose, Position, Vel) 128",
            "MTRL (Position) 128",
            "MTRL (Pose) 128",
            "MTRL (Vel) 128",
        ],
        "zoom_in": False,
    }

    save_plots_folder_path = "/workspace/isaaclab/source/plots/multitask_nn-128" # Specify the folder path where you want to save the plots
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
                        print(f"Error reading file in {experiment_name}: {e}")
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
    