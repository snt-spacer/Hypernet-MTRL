import torch
from tasks import TaskPlotsFactory
from robots import RobotPlotsFactory

import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import yaml

def main():
    # list_of_folders = [
    #     [
    #         "/workspace/isaaclab/logs/rsl_rl/jetbot_direct/2025-04-11_08-23-54_rsl-rl_GoToPosition_Jetbot_r-1_seed-4",
    #     ],
    #     [
    #         "/workspace/isaaclab/logs/rsl_rl/jetbot_direct/2025-04-11_08-03-24_rsl-rl_GoToPosition_Jetbot_r-1_seed-2",
    #     ],
    #     [
    #         "/workspace/isaaclab/logs/rsl_rl/jetbot_direct/2025-04-11_08-21-51_rsl-rl_GoToPosition_Jetbot_r-5_seed-3",

    #     ],
    # ]
    # list_of_folders = [
    #     [
    #         "/workspace/isaaclab/logs/rsl_rl/jetbot_direct/2025-04-11_08-44-23_rsl-rl_GoToPose_Jetbot_r-1_seed-1",
    #         "/workspace/isaaclab/logs/rsl_rl/jetbot_direct/2025-04-11_08-54-43_rsl-rl_GoToPose_Jetbot_r-1_seed-2",
    #     ],
    #     [
    #         "/workspace/isaaclab/logs/rsl_rl/jetbot_direct/2025-04-11_09-05-03_rsl-rl_GoToPose_Jetbot_r-1_seed-3",
    #     ],
    #     [
    #         "/workspace/isaaclab/logs/rsl_rl/jetbot_direct/2025-04-11_09-15-26_rsl-rl_GoToPose_Jetbot_r-1_seed-4",

    #     ],
    #     [
    #         "/workspace/isaaclab/logs/rsl_rl/jetbot_direct/2025-04-11_09-29-49_rsl-rl_GoToPose_Jetbot_r-1_seed-5",
    #     ]
    # ]
    # list_of_folders = [
    #     [
    #         "/workspace/isaaclab/logs/rsl_rl/jetbot_direct/2025-04-11_09-40-50_rsl-rl_GoThroughPositions_Jetbot_r-1_seed-1",
    #     ],
    #     [
    #         "/workspace/isaaclab/logs/rsl_rl/jetbot_direct/2025-04-11_09-52-44_rsl-rl_GoThroughPositions_Jetbot_r-1_seed-2",
    #     ]
    # ]
    list_of_folders = [
        [
            "/workspace/isaaclab/logs/rsl_rl/jetbot_direct/2025-04-10_17-46-10_rsl-rl_TrackVelocities_Jetbot_r-3_seed-3",
        ],
        [
            "/workspace/isaaclab/logs/rsl_rl/jetbot_direct/2025-04-10_18-02-49_rsl-rl_TrackVelocities_Jetbot_r-1_seed-5",
        ]
    ]

    save_plots_folder_path = "/workspace/isaaclab/source/plots/test_plots" # Specify the folder path where you want to save the plots
    if not os.path.exists(save_plots_folder_path):
        os.makedirs(save_plots_folder_path)

    dfs = {}
    labels = {}
    for group_idx, group in enumerate(list_of_folders):
        task_name = group[0].split("/")[-1].split("_")[3]
        robot_name = group[0].split("/")[-1].split("_")[4]
        gorup_key = f"{task_name}_group-{group_idx}"
        dfs[gorup_key] = []
        labels[gorup_key] = []
        for folder_path in group:
            try:
                experiment_name = glob.glob(os.path.join(folder_path, "metrics", "*_metrics.csv"))[0]
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

    task_plots_factory = TaskPlotsFactory.create(
        task_name, 
        dfs=dfs,
        labels=labels,
        env_info=env_info,
        folder_path=save_plots_folder_path,
    )
    task_plots_factory.plot()
    
    robot_plots_factory = RobotPlotsFactory.create(
        robot_name, 
        dfs=dfs,
        labels=labels,
        env_info=env_info,
        folder_path=folder_path,
    )
    robot_plots_factory.plot()

if __name__ == "__main__":
    main()
    