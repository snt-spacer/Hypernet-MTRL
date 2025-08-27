import torch
from tasks import TaskPlotsFactory
from robots import RobotPlotsFactory

import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import yaml

def main():
    # Define your CSV file groups directly, now each group can contain multiple sets of files (for different seeds)
    list_of_grouped_csv_data = [
        # {
        #     "group_name": "Expert_GoToPosition", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "GoToPosition",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_13-40-53_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-1/metrics/2025-07-25_13-40-53_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_13-40-53_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_13-40-53_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_14-05-17_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-2/metrics/2025-07-25_14-05-17_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_14-05-17_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_14-05-17_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_14-29-44_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-3/metrics/2025-07-25_14-29-44_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_14-29-44_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_14-29-44_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_14-54-10_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-4/metrics/2025-07-25_14-54-10_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_14-54-10_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_14-54-10_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_15-18-28_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-5/metrics/2025-07-25_15-18-28_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_15-18-28_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_15-18-28_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_15-42-48_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-6/metrics/2025-07-25_15-42-48_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-6_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_15-42-48_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_15-42-48_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_16-07-01_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-7/metrics/2025-07-25_16-07-01_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-7_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_16-07-01_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_16-07-01_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_16-31-37_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-8/metrics/2025-07-25_16-31-37_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-8_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_16-31-37_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_16-31-37_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_16-55-54_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-9/metrics/2025-07-25_16-55-54_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-9_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_16-55-54_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_16-55-54_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_17-19-36_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-10/metrics/2025-07-25_17-19-36_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-10_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_17-19-36_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_17-19-36_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml",
        #         },
        #     ]
        # },
        # {
        #     "group_name": "baselineGobsTid32x32_GoToPosition", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "GoToPosition",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_10-21-16_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-07-25_10-21-16_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_10-21-16_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_10-21-16_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_10-48-38_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-07-25_10-48-38_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_10-48-38_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_10-48-38_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_11-15-42_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-07-25_11-15-42_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_11-15-42_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_11-15-42_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_11-42-24_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-07-25_11-42-24_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_11-42-24_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_11-42-24_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_07-17-03_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-07-25_07-17-03_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_07-17-03_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_07-17-03_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_07-44-01_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-07-25_07-44-01_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-6_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_07-44-01_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_07-44-01_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_08-11-02_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-07-25_08-11-02_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-7_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_08-11-02_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_08-11-02_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_08-38-05_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-07-25_08-38-05_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-8_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_08-38-05_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_08-38-05_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_09-05-05_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-07-25_09-05-05_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-9_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_09-05-05_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_09-05-05_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_09-32-09_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-07-25_09-32-09_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-10_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_09-32-09_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_09-32-09_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml",
        #         },
        #     ]
        # },
        # {
        #     "group_name": "hypernetGobsTid32x32_GoToPosition", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "GoToPosition",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_15-43-21_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-07-23_15-43-21_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_15-43-21_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_15-43-21_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_16-13-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-07-23_16-13-54_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_16-13-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_16-13-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_16-44-19_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-07-23_16-44-19_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_16-44-19_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_16-44-19_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_17-14-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-07-23_17-14-46_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_17-14-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_17-14-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_17-45-18_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-07-23_17-45-18_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_17-45-18_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_17-45-18_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_18-15-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-07-23_18-15-51_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-6_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_18-15-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_18-15-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_18-47-07_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-07-23_18-47-07_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-7_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_18-47-07_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_18-47-07_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_19-18-15_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-07-23_19-18-15_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-8_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_19-18-15_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_19-18-15_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_19-48-44_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-07-23_19-48-44_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-9_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_19-48-44_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_19-48-44_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_20-19-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-07-23_20-19-26_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-10_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_20-19-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_20-19-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml",
        #         },
        #     ]
        # },
        #         {
        #     "group_name": "hypernetGobsTid64x64_GoToPosition", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "GoToPosition",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_09-04-44_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-07-31_09-04-44_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_09-04-44_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_09-04-44_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_10-05-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-07-31_10-05-53_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_10-05-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_10-05-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_11-07-10_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-07-31_11-07-10_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_11-07-10_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_11-07-10_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_12-09-17_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-07-31_12-09-17_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_12-09-17_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_12-09-17_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_13-10-37_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-07-31_13-10-37_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_13-10-37_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_13-10-37_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_14-11-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-07-31_14-11-51_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-6_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_14-11-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_14-11-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_15-12-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-07-31_15-12-54_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-7_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_15-12-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_15-12-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_16-15-55_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-07-31_16-15-55_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-8_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_16-15-55_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_16-15-55_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_17-17-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-07-31_17-17-48_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-9_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_17-17-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_17-17-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_18-19-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-07-31_18-19-22_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-10_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_18-19-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_18-19-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml",
        #         },
        #     ]
        # },
        # {
        #     "group_name": "hypernetGobsTid64x64x64_GoToPosition", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "GoToPosition",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_14-58-33_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-01_14-58-33_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_14-58-33_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_14-58-33_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_16-00-03_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-01_16-00-03_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_16-00-03_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_16-00-03_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_17-01-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-01_17-01-42_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_17-01-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_17-01-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_18-03-09_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-01_18-03-09_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_18-03-09_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_18-03-09_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_19-04-49_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-01_19-04-49_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_19-04-49_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_19-04-49_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_20-06-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-08-01_20-06-22_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-6_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_20-06-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_20-06-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_21-07-41_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-08-01_21-07-41_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-7_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_21-07-41_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_21-07-41_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_22-09-18_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-01_22-09-18_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-8_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_22-09-18_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_22-09-18_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_23-10-59_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-08-01_23-10-59_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-9_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_23-10-59_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_23-10-59_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_00-12-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-08-02_00-12-26_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-10_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_00-12-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_00-12-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml",
        #         },
        #     ]
        # },
        # {
        #     "group_name": "Hypernet w Gobs Tid 512x128x64x32_GoToPosition", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "GoToPosition",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_11-14-08_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-02_11-14-08_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_11-14-08_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_11-14-08_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_12-17-10_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-02_12-17-10_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_12-17-10_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_12-17-10_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_13-20-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-02_13-20-22_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_13-20-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_13-20-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_14-22-20_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-02_14-22-20_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_14-22-20_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_14-22-20_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_15-24-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-02_15-24-51_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_15-24-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_15-24-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_16-28-23_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-08-02_16-28-23_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-6_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_16-28-23_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_16-28-23_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_17-32-04_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-08-02_17-32-04_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-7_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_17-32-04_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_17-32-04_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_18-35-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-02_18-35-26_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-8_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_18-35-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_18-35-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_20-42-11_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-08-02_20-42-11_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-10_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_20-42-11_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_20-42-11_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml",
        #         },
        #     ]
        # },
        # {
        #     "group_name": "HypernetGobsTid32x32x32x32x32x32_GoToPosition", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "GoToPosition",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_23-44-25_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-02_23-44-25_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_23-44-25_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_23-44-25_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_00-46-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-03_00-46-48_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_00-46-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_00-46-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_01-48-56_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-03_01-48-56_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_01-48-56_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_01-48-56_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_02-51-56_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-03_02-51-56_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_02-51-56_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_02-51-56_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_03-54-07_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-03_03-54-07_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_03-54-07_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_03-54-07_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_04-56-09_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-08-03_04-56-09_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-6_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_04-56-09_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_04-56-09_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_07-01-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-03_07-01-02_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-8_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_07-01-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_07-01-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_08-03-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-08-03_08-03-02_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-9_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_08-03-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_08-03-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_09-05-14_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-08-03_09-05-14_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-10_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_09-05-14_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_09-05-14_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml",
        #         },
        #     ]
        # },
        # {
        #     "group_name": "Hypernet w Gobs Tid 64, 64, 64, 128, 64, 64, 64_GoToPosition", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "GoToPosition",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_13-50-19_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-03_13-50-19_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_13-50-19_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_13-50-19_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_14-52-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-03_14-52-53_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_14-52-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_14-52-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_16-59-17_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-03_16-59-17_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_16-59-17_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_16-59-17_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_18-01-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-08-03_18-01-48_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-6_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_18-01-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_18-01-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_19-04-40_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-08-03_19-04-40_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-7_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_19-04-40_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_19-04-40_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_20-07-35_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-03_20-07-35_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-8_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_20-07-35_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_20-07-35_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_22-15-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-08-03_22-15-26_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-10_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_22-15-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_22-15-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml",
        #         },
        #     ]
        # },
        # {
        #     "group_name": "Hypernet Critic w Gobs Tid 64x64x64_GoToPosition", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "GoToPosition",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_10-48-13_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-04_10-48-13_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_10-48-13_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_10-48-13_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_13-07-24_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-04_13-07-24_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_13-07-24_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_13-07-24_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_14-16-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-04_14-16-26_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_14-16-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_14-16-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_16-35-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-08-04_16-35-53_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-6_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_16-35-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_16-35-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_18-55-16_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-04_18-55-16_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-8_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_18-55-16_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_18-55-16_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml",
        #         },
        #     ]
        # },
        # {
        #     "group_name": "Hypernet Embeddings w Gobs Tid 64x64x64_GoToPosition", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "GoToPosition",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_16-34-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-05_16-34-42_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_16-34-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_16-34-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_17-36-32_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-05_17-36-32_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_17-36-32_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_17-36-32_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_18-38-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-05_18-38-42_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_18-38-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_18-38-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_19-41-15_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-05_19-41-15_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_19-41-15_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_19-41-15_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_20-42-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-05_20-42-54_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_20-42-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_20-42-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_21-44-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-08-05_21-44-42_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-6_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_21-44-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_21-44-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_22-46-39_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-08-05_22-46-39_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-7_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_22-46-39_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_22-46-39_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_23-47-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-05_23-47-47_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-8_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_23-47-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_23-47-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-06_00-49-40_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-08-06_00-49-40_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-9_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-06_00-49-40_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-06_00-49-40_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-06_01-51-29_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-08-06_01-51-29_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-10_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-06_01-51-29_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-06_01-51-29_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml",
        #         },
        #     ]
        # },
        # {
        #     "group_name": "New rendezvous reward Hypernet w Gobs Tid 64x64x64_GoToPosition", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "GoToPosition",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-06_11-49-57_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-06_11-49-57_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-06_11-49-57_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-06_11-49-57_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml",
        #         },
        #     ]
        # },
        # {
        #     "group_name": "Expert_GoToPose", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "GoToPose",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_17-43-52_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-1/metrics/2025-07-25_17-43-52_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_17-43-52_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_17-43-52_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_18-08-31_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-2/metrics/2025-07-25_18-08-31_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_18-08-31_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_18-08-31_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_18-32-31_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-3/metrics/2025-07-25_18-32-31_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_18-32-31_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_18-32-31_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_18-56-59_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-4/metrics/2025-07-25_18-56-59_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_18-56-59_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_18-56-59_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_19-21-06_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-5/metrics/2025-07-25_19-21-06_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_19-21-06_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_19-21-06_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_19-45-24_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-6/metrics/2025-07-25_19-45-24_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-6_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_19-45-24_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_19-45-24_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_20-10-29_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-7/metrics/2025-07-25_20-10-29_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-7_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_20-10-29_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_20-10-29_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_20-34-43_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-8/metrics/2025-07-25_20-34-43_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-8_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_20-34-43_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_20-34-43_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_20-59-12_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-9/metrics/2025-07-25_20-59-12_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-9_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_20-59-12_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_20-59-12_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_21-24-22_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-10/metrics/2025-07-25_21-24-22_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-10_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_21-24-22_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_21-24-22_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml",
        #         },
        #     ]
        # },
        # {
        #     "group_name": "baselineGobsTid32x32_GoToPose", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "GoToPose",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_10-21-16_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-07-25_10-21-16_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_10-21-16_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_10-21-16_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_10-48-38_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-07-25_10-48-38_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_10-48-38_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_10-48-38_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_11-15-42_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-07-25_11-15-42_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_11-15-42_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_11-15-42_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_11-42-24_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-07-25_11-42-24_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_11-42-24_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_11-42-24_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_07-17-03_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-07-25_07-17-03_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_07-17-03_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_07-17-03_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_07-44-01_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-07-25_07-44-01_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-6_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_07-44-01_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_07-44-01_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_08-11-02_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-07-25_08-11-02_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-7_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_08-11-02_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_08-11-02_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_08-38-05_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-07-25_08-38-05_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-8_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_08-38-05_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_08-38-05_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_09-05-05_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-07-25_09-05-05_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-9_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_09-05-05_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_09-05-05_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_09-32-09_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-07-25_09-32-09_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-10_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_09-32-09_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_09-32-09_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml",
        #         },
        #     ]
        # },
        # {
        #     "group_name": "hypernetGobsTid32x32_GoToPose", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "GoToPose",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_15-43-21_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-07-23_15-43-21_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_15-43-21_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_15-43-21_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_16-13-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-07-23_16-13-54_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_16-13-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_16-13-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_16-44-19_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-07-23_16-44-19_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_16-44-19_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_16-44-19_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_17-14-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-07-23_17-14-46_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_17-14-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_17-14-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_17-45-18_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-07-23_17-45-18_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_17-45-18_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_17-45-18_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_18-15-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-07-23_18-15-51_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-6_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_18-15-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_18-15-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_18-47-07_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-07-23_18-47-07_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-7_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_18-47-07_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_18-47-07_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_19-18-15_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-07-23_19-18-15_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-8_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_19-18-15_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_19-18-15_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_19-48-44_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-07-23_19-48-44_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-9_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_19-48-44_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_19-48-44_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_20-19-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-07-23_20-19-26_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-10_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_20-19-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_20-19-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml",
        #         },
        #     ]
        # },
        #         {
        #     "group_name": "hypernetGobsTid64x64_GoToPose", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "GoToPose",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_09-04-44_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-07-31_09-04-44_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_09-04-44_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_09-04-44_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_10-05-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-07-31_10-05-53_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_10-05-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_10-05-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_11-07-10_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-07-31_11-07-10_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_11-07-10_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_11-07-10_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_12-09-17_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-07-31_12-09-17_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_12-09-17_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_12-09-17_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_13-10-37_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-07-31_13-10-37_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_13-10-37_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_13-10-37_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_14-11-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-07-31_14-11-51_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-6_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_14-11-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_14-11-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_15-12-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-07-31_15-12-54_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-7_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_15-12-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_15-12-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_16-15-55_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-07-31_16-15-55_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-8_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_16-15-55_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_16-15-55_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_17-17-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-07-31_17-17-48_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-9_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_17-17-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_17-17-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_18-19-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-07-31_18-19-22_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-10_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_18-19-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_18-19-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml",
        #         },
        #     ]
        # },
        # {
        #     "group_name": "hypernetGobsTid64x64x64_GoToPose", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "GoToPose",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_14-58-33_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-01_14-58-33_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_14-58-33_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_14-58-33_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_16-00-03_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-01_16-00-03_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_16-00-03_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_16-00-03_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_17-01-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-01_17-01-42_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_17-01-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_17-01-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_18-03-09_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-01_18-03-09_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_18-03-09_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_18-03-09_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_19-04-49_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-01_19-04-49_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_19-04-49_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_19-04-49_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_20-06-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-08-01_20-06-22_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-6_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_20-06-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_20-06-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_21-07-41_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-08-01_21-07-41_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-7_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_21-07-41_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_21-07-41_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_22-09-18_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-01_22-09-18_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-8_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_22-09-18_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_22-09-18_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_23-10-59_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-08-01_23-10-59_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-9_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_23-10-59_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_23-10-59_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_00-12-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-08-02_00-12-26_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-10_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_00-12-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_00-12-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml",
        #         },
        #     ]
        # },
        # {
        #     "group_name": "Hypernet w Gobs Tid 512x128x64x32_GoToPose", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "GoToPose",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_11-14-08_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-02_11-14-08_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_11-14-08_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_11-14-08_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_12-17-10_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-02_12-17-10_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_12-17-10_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_12-17-10_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_13-20-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-02_13-20-22_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_13-20-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_13-20-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_14-22-20_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-02_14-22-20_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_14-22-20_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_14-22-20_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_15-24-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-02_15-24-51_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_15-24-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_15-24-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_16-28-23_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-08-02_16-28-23_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-6_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_16-28-23_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_16-28-23_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_17-32-04_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-08-02_17-32-04_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-7_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_17-32-04_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_17-32-04_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_18-35-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-02_18-35-26_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-8_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_18-35-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_18-35-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_20-42-11_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-08-02_20-42-11_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-10_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_20-42-11_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_20-42-11_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml",
        #         },
        #     ]
        # },
        # {
        #     "group_name": "HypernetGobsTid32x32x32x32x32x32_GoToPose", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "GoToPose",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_23-44-25_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-02_23-44-25_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_23-44-25_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_23-44-25_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_00-46-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-03_00-46-48_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_00-46-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_00-46-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_01-48-56_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-03_01-48-56_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_01-48-56_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_01-48-56_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_02-51-56_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-03_02-51-56_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_02-51-56_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_02-51-56_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_03-54-07_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-03_03-54-07_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_03-54-07_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_03-54-07_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_04-56-09_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-08-03_04-56-09_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-6_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_04-56-09_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_04-56-09_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_07-01-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-03_07-01-02_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-8_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_07-01-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_07-01-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_08-03-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-08-03_08-03-02_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-9_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_08-03-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_08-03-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_09-05-14_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-08-03_09-05-14_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-10_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_09-05-14_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_09-05-14_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml",
        #         },
        #     ]
        # },
        # {
        #     "group_name": "Hypernet w Gobs Tid 64, 64, 64, 128, 64, 64, 64_GoToPose", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "GoToPose",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_13-50-19_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-03_13-50-19_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_13-50-19_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_13-50-19_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_14-52-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-03_14-52-53_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_14-52-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_14-52-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_16-59-17_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-03_16-59-17_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_16-59-17_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_16-59-17_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_18-01-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-08-03_18-01-48_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-6_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_18-01-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_18-01-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_19-04-40_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-08-03_19-04-40_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-7_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_19-04-40_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_19-04-40_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_20-07-35_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-03_20-07-35_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-8_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_20-07-35_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_20-07-35_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_22-15-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-08-03_22-15-26_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-10_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_22-15-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_22-15-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml",
        #         },
        #     ]
        # },
        # {
        #     "group_name": "Hypernet Critic w Gobs Tid 64x64x64_GoToPose", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "GoToPose",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_10-48-13_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-04_10-48-13_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_10-48-13_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_10-48-13_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_13-07-24_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-04_13-07-24_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_13-07-24_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_13-07-24_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_14-16-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-04_14-16-26_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_14-16-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_14-16-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_16-35-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-08-04_16-35-53_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-6_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_16-35-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_16-35-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_18-55-16_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-04_18-55-16_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-8_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_18-55-16_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_18-55-16_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml",
        #         },
        #     ]
        # },
        # {
        #     "group_name": "Hypernet Embeddings w Gobs Tid 64x64x64_GoToPose", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "GoToPose",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_16-34-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-05_16-34-42_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_16-34-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_16-34-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_17-36-32_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-05_17-36-32_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_17-36-32_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_17-36-32_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_18-38-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-05_18-38-42_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_18-38-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_18-38-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_19-41-15_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-05_19-41-15_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_19-41-15_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_19-41-15_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_20-42-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-05_20-42-54_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_20-42-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_20-42-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_21-44-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-08-05_21-44-42_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-6_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_21-44-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_21-44-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_22-46-39_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-08-05_22-46-39_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-7_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_22-46-39_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_22-46-39_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_23-47-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-05_23-47-47_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-8_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_23-47-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_23-47-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-06_00-49-40_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-08-06_00-49-40_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-9_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-06_00-49-40_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-06_00-49-40_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-06_01-51-29_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-08-06_01-51-29_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-10_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-06_01-51-29_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-06_01-51-29_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml",
        #         },
        #     ]
        # },
        # {
        #     "group_name": "New rendezvous reward Hypernet w Gobs Tid 64x64x64_GoToPose", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "GoToPose",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-06_11-49-57_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-06_11-49-57_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-06_11-49-57_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_GoToPose.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-06_11-49-57_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml",
        #         },
        #     ]
        # },
        # {
        #     "group_name": "Expert_TrackVelocities", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "TrackVelocities",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_21-49-01_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-1/metrics/2025-07-25_21-49-01_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_21-49-01_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_21-49-01_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_22-17-14_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-2/metrics/2025-07-25_22-17-14_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_22-17-14_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_22-17-14_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_22-44-58_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-3/metrics/2025-07-25_22-44-58_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_22-44-58_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_22-44-58_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_23-13-10_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-4/metrics/2025-07-25_23-13-10_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_23-13-10_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_23-13-10_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_23-41-48_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-5/metrics/2025-07-25_23-41-48_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_23-41-48_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_23-41-48_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_00-10-01_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-6/metrics/2025-07-26_00-10-01_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-6_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_00-10-01_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_00-10-01_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_00-38-23_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-7/metrics/2025-07-26_00-38-23_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-7_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_00-38-23_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_00-38-23_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_01-06-25_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-8/metrics/2025-07-26_01-06-25_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-8_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_01-06-25_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_01-06-25_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_01-34-14_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-9/metrics/2025-07-26_01-34-14_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-9_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_01-34-14_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_01-34-14_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_02-02-39_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-10/metrics/2025-07-26_02-02-39_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-10_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_02-02-39_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_02-02-39_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml",
        #         },
        #     ]
        # },
        # {
        #     "group_name": "baselineGobsTid32x32_TrackVelocities", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "TrackVelocities",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_10-21-16_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-07-25_10-21-16_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_10-21-16_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_10-21-16_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_10-48-38_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-07-25_10-48-38_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_10-48-38_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_10-48-38_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_11-15-42_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-07-25_11-15-42_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_11-15-42_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_11-15-42_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_11-42-24_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-07-25_11-42-24_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_11-42-24_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_11-42-24_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_07-17-03_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-07-25_07-17-03_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_07-17-03_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_07-17-03_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_07-44-01_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-07-25_07-44-01_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-6_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_07-44-01_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_07-44-01_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_08-11-02_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-07-25_08-11-02_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-7_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_08-11-02_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_08-11-02_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_08-38-05_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-07-25_08-38-05_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-8_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_08-38-05_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_08-38-05_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_09-05-05_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-07-25_09-05-05_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-9_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_09-05-05_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_09-05-05_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_09-32-09_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-07-25_09-32-09_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-10_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_09-32-09_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_09-32-09_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml",
        #         },
        #     ]
        # },
        # {
        #     "group_name": "hypernetGobsTid32x32_TrackVelocities", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "TrackVelocities",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_15-43-21_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-07-23_15-43-21_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_15-43-21_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_15-43-21_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_16-13-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-07-23_16-13-54_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_16-13-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_16-13-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_16-44-19_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-07-23_16-44-19_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_16-44-19_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_16-44-19_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_17-14-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-07-23_17-14-46_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_17-14-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_17-14-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_17-45-18_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-07-23_17-45-18_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_17-45-18_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_17-45-18_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_18-15-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-07-23_18-15-51_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-6_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_18-15-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_18-15-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_18-47-07_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-07-23_18-47-07_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-7_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_18-47-07_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_18-47-07_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_19-18-15_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-07-23_19-18-15_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-8_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_19-18-15_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_19-18-15_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_19-48-44_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-07-23_19-48-44_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-9_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_19-48-44_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_19-48-44_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_20-19-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-07-23_20-19-26_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-10_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_20-19-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_20-19-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml",
        #         },
        #     ]
        # },
        #         {
        #     "group_name": "hypernetGobsTid64x64_TrackVelocities", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "TrackVelocities",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_09-04-44_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-07-31_09-04-44_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_09-04-44_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_09-04-44_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_10-05-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-07-31_10-05-53_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_10-05-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_10-05-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_11-07-10_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-07-31_11-07-10_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_11-07-10_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_11-07-10_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_12-09-17_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-07-31_12-09-17_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_12-09-17_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_12-09-17_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_13-10-37_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-07-31_13-10-37_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_13-10-37_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_13-10-37_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_14-11-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-07-31_14-11-51_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-6_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_14-11-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_14-11-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_15-12-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-07-31_15-12-54_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-7_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_15-12-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_15-12-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_16-15-55_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-07-31_16-15-55_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-8_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_16-15-55_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_16-15-55_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_17-17-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-07-31_17-17-48_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-9_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_17-17-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_17-17-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_18-19-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-07-31_18-19-22_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-10_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_18-19-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_18-19-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml",
        #         },
        #     ]
        # },
        # {
        #     "group_name": "hypernetGobsTid64x64x64_TrackVelocities", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "TrackVelocities",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_14-58-33_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-01_14-58-33_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_14-58-33_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_14-58-33_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_16-00-03_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-01_16-00-03_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_16-00-03_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_16-00-03_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_17-01-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-01_17-01-42_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_17-01-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_17-01-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_18-03-09_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-01_18-03-09_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_18-03-09_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_18-03-09_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_19-04-49_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-01_19-04-49_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_19-04-49_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_19-04-49_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_20-06-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-08-01_20-06-22_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-6_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_20-06-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_20-06-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_21-07-41_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-08-01_21-07-41_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-7_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_21-07-41_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_21-07-41_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_22-09-18_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-01_22-09-18_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-8_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_22-09-18_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_22-09-18_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_23-10-59_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-08-01_23-10-59_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-9_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_23-10-59_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_23-10-59_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_00-12-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-08-02_00-12-26_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-10_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_00-12-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_00-12-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml",
        #         },
        #     ]
        # },
        # {
        #     "group_name": "Hypernet w Gobs Tid 512x128x64x32_TrackVelocities", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "TrackVelocities",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_11-14-08_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-02_11-14-08_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_11-14-08_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_11-14-08_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_12-17-10_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-02_12-17-10_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_12-17-10_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_12-17-10_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_13-20-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-02_13-20-22_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_13-20-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_13-20-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_14-22-20_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-02_14-22-20_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_14-22-20_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_14-22-20_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_15-24-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-02_15-24-51_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_15-24-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_15-24-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_16-28-23_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-08-02_16-28-23_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-6_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_16-28-23_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_16-28-23_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_17-32-04_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-08-02_17-32-04_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-7_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_17-32-04_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_17-32-04_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_18-35-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-02_18-35-26_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-8_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_18-35-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_18-35-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_20-42-11_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-08-02_20-42-11_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-10_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_20-42-11_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_20-42-11_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml",
        #         },
        #     ]
        # },
        # {
        #     "group_name": "HypernetGobsTid32x32x32x32x32x32_TrackVelocities", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "TrackVelocities",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_23-44-25_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-02_23-44-25_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_23-44-25_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_23-44-25_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_00-46-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-03_00-46-48_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_00-46-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_00-46-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_01-48-56_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-03_01-48-56_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_01-48-56_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_01-48-56_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_02-51-56_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-03_02-51-56_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_02-51-56_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_02-51-56_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_03-54-07_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-03_03-54-07_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_03-54-07_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_03-54-07_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_04-56-09_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-08-03_04-56-09_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-6_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_04-56-09_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_04-56-09_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_07-01-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-03_07-01-02_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-8_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_07-01-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_07-01-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_08-03-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-08-03_08-03-02_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-9_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_08-03-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_08-03-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_09-05-14_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-08-03_09-05-14_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-10_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_09-05-14_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_09-05-14_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml",
        #         },
        #     ]
        # },
        # {
        #     "group_name": "Hypernet w Gobs Tid 64, 64, 64, 128, 64, 64, 64_TrackVelocities", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "TrackVelocities",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_13-50-19_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-03_13-50-19_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_13-50-19_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_13-50-19_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_14-52-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-03_14-52-53_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_14-52-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_14-52-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_16-59-17_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-03_16-59-17_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_16-59-17_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_16-59-17_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_18-01-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-08-03_18-01-48_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-6_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_18-01-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_18-01-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_19-04-40_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-08-03_19-04-40_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-7_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_19-04-40_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_19-04-40_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_20-07-35_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-03_20-07-35_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-8_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_20-07-35_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_20-07-35_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_22-15-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-08-03_22-15-26_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-10_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_22-15-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_22-15-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml",
        #         },
        #     ]
        # },
        # {
        #     "group_name": "Hypernet Critic w Gobs Tid 64x64x64_TrackVelocities", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "TrackVelocities",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_10-48-13_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-04_10-48-13_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_10-48-13_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_10-48-13_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_13-07-24_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-04_13-07-24_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_13-07-24_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_13-07-24_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_14-16-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-04_14-16-26_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_14-16-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_14-16-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_16-35-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-08-04_16-35-53_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-6_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_16-35-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_16-35-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_18-55-16_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-04_18-55-16_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-8_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_18-55-16_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_18-55-16_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml",
        #         },
        #     ]
        # },
        # {
        #     "group_name": "Hypernet Embeddings w Gobs Tid 64x64x64_TrackVelocities", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "TrackVelocities",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_16-34-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-05_16-34-42_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_16-34-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_16-34-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_17-36-32_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-05_17-36-32_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_17-36-32_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_17-36-32_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_18-38-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-05_18-38-42_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_18-38-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_18-38-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_19-41-15_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-05_19-41-15_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_19-41-15_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_19-41-15_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_20-42-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-05_20-42-54_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_20-42-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_20-42-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_21-44-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-08-05_21-44-42_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-6_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_21-44-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_21-44-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_22-46-39_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-08-05_22-46-39_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-7_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_22-46-39_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_22-46-39_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_23-47-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-05_23-47-47_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-8_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_23-47-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_23-47-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-06_00-49-40_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-08-06_00-49-40_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-9_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-06_00-49-40_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-06_00-49-40_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-06_01-51-29_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-08-06_01-51-29_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-10_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-06_01-51-29_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-06_01-51-29_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml",
        #         },
        #     ]
        # },
        # {
        #     "group_name": "New rendezvous reward Hypernet w Gobs Tid 64x64x64_TrackVelocities", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "TrackVelocities",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-06_11-49-57_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-06_11-49-57_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-06_11-49-57_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_TrackVelocities.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-06_11-49-57_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml",
        #         },
        #     ]
        # },
        # {
        #     "group_name": "Expert_Rendezvous", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "Rendezvous",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_02-31-28_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-07-26_02-31-28_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_02-31-28_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_02-31-28_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_02-57-35_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-07-26_02-57-35_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_02-57-35_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_02-57-35_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_03-24-24_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-07-26_03-24-24_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_03-24-24_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_03-24-24_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_03-50-36_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-07-26_03-50-36_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_03-50-36_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_03-50-36_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_04-17-17_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-07-26_04-17-17_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_04-17-17_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_04-17-17_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_04-44-17_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-07-26_04-44-17_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-6_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_04-44-17_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_04-44-17_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_05-11-15_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-07-26_05-11-15_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-7_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_05-11-15_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_05-11-15_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_05-38-04_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-07-26_05-38-04_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-8_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_05-38-04_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_05-38-04_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_06-04-38_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-07-26_06-04-38_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-9_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_06-04-38_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_06-04-38_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_06-31-49_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-07-26_06-31-49_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-10_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_06-31-49_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_06-31-49_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml",
        #         },
        #     ]
        # },
        # {
        #     "group_name": "baselineGobsTid32x32_Rendezvous", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "Rendezvous",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_10-21-16_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-07-25_10-21-16_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_10-21-16_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_10-21-16_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_10-48-38_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-07-25_10-48-38_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_10-48-38_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_10-48-38_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_11-15-42_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-07-25_11-15-42_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_11-15-42_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_11-15-42_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_11-42-24_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-07-25_11-42-24_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_11-42-24_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_11-42-24_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_07-17-03_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-07-25_07-17-03_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_07-17-03_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_07-17-03_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_07-44-01_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-07-25_07-44-01_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-6_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_07-44-01_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_07-44-01_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_08-11-02_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-07-25_08-11-02_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-7_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_08-11-02_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_08-11-02_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_08-38-05_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-07-25_08-38-05_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-8_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_08-38-05_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_08-38-05_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_09-05-05_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-07-25_09-05-05_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-9_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_09-05-05_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_09-05-05_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_09-32-09_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-07-25_09-32-09_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-10_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_09-32-09_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_09-32-09_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml",
        #         },
        #     ]
        # },
        # # {
        # #     "group_name": "hypernetGobsTid32x32_Rendezvous", # This is the name that will appear in plot_cfg["runs_names"]
        # #     "task_name": "Rendezvous",
        # #     "robot_name": "ModularFreeflyer",
        # #     "runs": [
        # #         {
        # #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_15-43-21_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-07-23_15-43-21_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-1_metrics.csv",
        # #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_15-43-21_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_Rendezvous.csv",
        # #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_15-43-21_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml",
        # #         },
        # #         {
        # #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_16-13-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-07-23_16-13-54_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-2_metrics.csv",
        # #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_16-13-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_Rendezvous.csv",
        # #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_16-13-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml",
        # #         },
        # #         {
        # #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_16-44-19_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-07-23_16-44-19_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-3_metrics.csv",
        # #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_16-44-19_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_Rendezvous.csv",
        # #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_16-44-19_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml",
        # #         },
        # #         {
        # #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_17-14-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-07-23_17-14-46_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-4_metrics.csv",
        # #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_17-14-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_Rendezvous.csv",
        # #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_17-14-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml",
        # #         },
        # #         {
        # #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_17-45-18_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-07-23_17-45-18_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-5_metrics.csv",
        # #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_17-45-18_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_Rendezvous.csv",
        # #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_17-45-18_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml",
        # #         },
        # #         {
        # #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_18-15-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-07-23_18-15-51_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-6_metrics.csv",
        # #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_18-15-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_Rendezvous.csv",
        # #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_18-15-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml",
        # #         },
        # #         {
        # #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_18-47-07_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-07-23_18-47-07_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-7_metrics.csv",
        # #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_18-47-07_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_Rendezvous.csv",
        # #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_18-47-07_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml",
        # #         },
        # #         {
        # #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_19-18-15_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-07-23_19-18-15_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-8_metrics.csv",
        # #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_19-18-15_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_Rendezvous.csv",
        # #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_19-18-15_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml",
        # #         },
        # #         {
        # #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_19-48-44_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-07-23_19-48-44_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-9_metrics.csv",
        # #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_19-48-44_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_Rendezvous.csv",
        # #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_19-48-44_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml",
        # #         },
        # #         {
        # #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_20-19-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-07-23_20-19-26_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-10_metrics.csv",
        # #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_20-19-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_Rendezvous.csv",
        # #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_20-19-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml",
        # #         },
        # #     ]
        # # },
        # {
        #     "group_name": "hypernetGobsTid64x64_Rendezvous", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "Rendezvous",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_09-04-44_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-07-31_09-04-44_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_09-04-44_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_09-04-44_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_10-05-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-07-31_10-05-53_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_10-05-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_10-05-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_11-07-10_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-07-31_11-07-10_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_11-07-10_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_11-07-10_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_12-09-17_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-07-31_12-09-17_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_12-09-17_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_12-09-17_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_13-10-37_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-07-31_13-10-37_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_13-10-37_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_13-10-37_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_14-11-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-07-31_14-11-51_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-6_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_14-11-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_14-11-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_15-12-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-07-31_15-12-54_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-7_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_15-12-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_15-12-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_16-15-55_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-07-31_16-15-55_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-8_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_16-15-55_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_16-15-55_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_17-17-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-07-31_17-17-48_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-9_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_17-17-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_17-17-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_18-19-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-07-31_18-19-22_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-10_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_18-19-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_18-19-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml",
        #         },
        #     ]
        # },
        # {
        #     "group_name": "hypernetGobsTid64x64x64_Rendezvous", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "Rendezvous",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_14-58-33_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-01_14-58-33_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_14-58-33_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_14-58-33_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_16-00-03_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-01_16-00-03_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_16-00-03_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_16-00-03_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_17-01-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-01_17-01-42_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_17-01-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_17-01-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_18-03-09_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-01_18-03-09_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_18-03-09_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_18-03-09_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_19-04-49_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-01_19-04-49_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_19-04-49_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_19-04-49_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_20-06-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-08-01_20-06-22_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-6_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_20-06-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_20-06-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_21-07-41_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-08-01_21-07-41_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-7_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_21-07-41_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_21-07-41_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_22-09-18_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-01_22-09-18_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-8_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_22-09-18_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_22-09-18_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_23-10-59_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-08-01_23-10-59_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-9_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_23-10-59_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_23-10-59_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_00-12-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-08-02_00-12-26_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-10_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_00-12-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_00-12-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml",
        #         },
        #     ]
        # },
        # {
        #     "group_name": "Hypernet w Gobs Tid 512x128x64x32_Rendezvous", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "Rendezvous",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_11-14-08_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-02_11-14-08_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_11-14-08_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_11-14-08_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_12-17-10_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-02_12-17-10_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_12-17-10_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_12-17-10_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_13-20-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-02_13-20-22_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_13-20-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_13-20-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_14-22-20_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-02_14-22-20_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_14-22-20_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_14-22-20_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_15-24-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-02_15-24-51_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_15-24-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_15-24-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_16-28-23_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-08-02_16-28-23_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-6_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_16-28-23_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_16-28-23_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_17-32-04_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-08-02_17-32-04_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-7_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_17-32-04_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_17-32-04_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_18-35-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-02_18-35-26_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-8_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_18-35-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_18-35-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_20-42-11_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-08-02_20-42-11_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-10_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_20-42-11_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_20-42-11_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml",
        #         },
        #     ]
        # },
        # {
        #     "group_name": "HypernetGobsTid32x32x32x32x32x32_Rendezvous", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "Rendezvous",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_23-44-25_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-02_23-44-25_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_23-44-25_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_23-44-25_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_00-46-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-03_00-46-48_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_00-46-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_00-46-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_01-48-56_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-03_01-48-56_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_01-48-56_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_01-48-56_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_02-51-56_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-03_02-51-56_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_02-51-56_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_02-51-56_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_03-54-07_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-03_03-54-07_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_03-54-07_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_03-54-07_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_04-56-09_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-08-03_04-56-09_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-6_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_04-56-09_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_04-56-09_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_07-01-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-03_07-01-02_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-8_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_07-01-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_07-01-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_08-03-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-08-03_08-03-02_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-9_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_08-03-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_08-03-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_09-05-14_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-08-03_09-05-14_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-10_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_09-05-14_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_09-05-14_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml",
        #         },
        #     ]
        # },
        # {
        #     "group_name": "Hypernet w Gobs Tid 64, 64, 64, 128, 64, 64, 64_Rendezvous", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "Rendezvous",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_13-50-19_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-03_13-50-19_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_13-50-19_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_13-50-19_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_14-52-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-03_14-52-53_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_14-52-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_14-52-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_16-59-17_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-03_16-59-17_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_16-59-17_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_16-59-17_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_18-01-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-08-03_18-01-48_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-6_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_18-01-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_18-01-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_19-04-40_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-08-03_19-04-40_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-7_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_19-04-40_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_19-04-40_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_20-07-35_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-03_20-07-35_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-8_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_20-07-35_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_20-07-35_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_22-15-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-08-03_22-15-26_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-10_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_22-15-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_22-15-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml",
        #         },
        #     ]
        # },
        # {
        #     "group_name": "Hypernet Critic w Gobs Tid 64x64x64_Rendezvous", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "Rendezvous",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_10-48-13_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-04_10-48-13_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_10-48-13_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_10-48-13_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_13-07-24_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-04_13-07-24_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_13-07-24_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_13-07-24_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_14-16-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-04_14-16-26_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_14-16-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_14-16-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_16-35-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-08-04_16-35-53_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-6_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_16-35-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_16-35-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_18-55-16_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-04_18-55-16_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-8_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_18-55-16_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_18-55-16_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml",
        #         },
        #     ]
        # },
        # {
        #     "group_name": "Hypernet Embeddings w Gobs Tid 64x64x64_Rendezvous", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "Rendezvous",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_16-34-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-05_16-34-42_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_16-34-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_16-34-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_17-36-32_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-05_17-36-32_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_17-36-32_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_17-36-32_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_18-38-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-05_18-38-42_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_18-38-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_18-38-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_19-41-15_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-05_19-41-15_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_19-41-15_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_19-41-15_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_20-42-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-05_20-42-54_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_20-42-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_20-42-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_21-44-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-08-05_21-44-42_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-6_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_21-44-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_21-44-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_22-46-39_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-08-05_22-46-39_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-7_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_22-46-39_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_22-46-39_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_23-47-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-05_23-47-47_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-8_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_23-47-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_23-47-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-06_00-49-40_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-08-06_00-49-40_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-9_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-06_00-49-40_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-06_00-49-40_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-06_01-51-29_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-08-06_01-51-29_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-10_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-06_01-51-29_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-06_01-51-29_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml",
        #         },
        #     ]
        # },
        # {
        #     "group_name": "New rendezvous reward Hypernet w Gobs Tid 64x64x64_Rendezvous", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "Rendezvous",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-06_11-49-57_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-06_11-49-57_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-06_11-49-57_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_Rendezvous.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-06_11-49-57_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml",
        #         },
        #     ]
        # },


        # Experts
        {
            "group_name": "Expert 32x32_GoToPosition",
            "task_name": "GoToPosition",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_17-01-14_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-1/metrics/2025-08-12_17-01-14_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_17-01-14_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_17-01-14_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_17-27-40_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-2/metrics/2025-08-12_17-27-40_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_17-27-40_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_17-27-40_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_17-54-17_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-3/metrics/2025-08-12_17-54-17_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_17-54-17_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_17-54-17_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_18-20-41_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-4/metrics/2025-08-12_18-20-41_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_18-20-41_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_18-20-41_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_18-47-16_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-5/metrics/2025-08-12_18-47-16_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_18-47-16_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_18-47-16_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_19-13-45_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-6/metrics/2025-08-12_19-13-45_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-6_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_19-13-45_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_19-13-45_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_19-40-14_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-7/metrics/2025-08-12_19-40-14_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-7_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_19-40-14_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_19-40-14_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_20-07-03_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-8/metrics/2025-08-12_20-07-03_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-8_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_20-07-03_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_20-07-03_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_20-33-34_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-9/metrics/2025-08-12_20-33-34_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-9_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_20-33-34_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_20-33-34_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_21-00-10_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-10/metrics/2025-08-12_21-00-10_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-10_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_21-00-10_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_21-00-10_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml"
                }
            ]
        },

        {
            "group_name": "Expert 32x32_GoToPose",
            "task_name": "GoToPose",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_21-26-57_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-1/metrics/2025-08-12_21-26-57_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_21-26-57_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_21-26-57_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_21-53-23_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-2/metrics/2025-08-12_21-53-23_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_21-53-23_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_21-53-23_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_22-20-17_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-3/metrics/2025-08-12_22-20-17_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_22-20-17_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_22-20-17_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_22-46-57_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-4/metrics/2025-08-12_22-46-57_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_22-46-57_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_22-46-57_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_23-13-35_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-5/metrics/2025-08-12_23-13-35_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_23-13-35_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_23-13-35_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_23-40-28_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-6/metrics/2025-08-12_23-40-28_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-6_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_23-40-28_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_23-40-28_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_00-07-14_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-7/metrics/2025-08-13_00-07-14_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-7_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_00-07-14_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_00-07-14_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_00-33-48_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-8/metrics/2025-08-13_00-33-48_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-8_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_00-33-48_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_00-33-48_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_01-00-23_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-9/metrics/2025-08-13_01-00-23_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-9_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_01-00-23_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_01-00-23_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_01-26-52_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-10/metrics/2025-08-13_01-26-52_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-10_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_01-26-52_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_01-26-52_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml"
                }
            ]
        },

        {
            "group_name": "Expert 32x32_TrackVelocities",
            "task_name": "TrackVelocities",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_01-53-37_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-1/metrics/2025-08-13_01-53-37_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_01-53-37_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_01-53-37_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_02-21-30_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-2/metrics/2025-08-13_02-21-30_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_02-21-30_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_02-21-30_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_02-49-20_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-3/metrics/2025-08-13_02-49-20_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_02-49-20_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_02-49-20_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_03-17-26_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-4/metrics/2025-08-13_03-17-26_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_03-17-26_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_03-17-26_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_03-45-10_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-5/metrics/2025-08-13_03-45-10_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_03-45-10_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_03-45-10_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_04-13-04_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-6/metrics/2025-08-13_04-13-04_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-6_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_04-13-04_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_04-13-04_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_04-40-48_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-7/metrics/2025-08-13_04-40-48_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-7_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_04-40-48_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_04-40-48_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_05-08-34_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-8/metrics/2025-08-13_05-08-34_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-8_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_05-08-34_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_05-08-34_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_05-36-32_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-9/metrics/2025-08-13_05-36-32_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-9_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_05-36-32_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_05-36-32_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_06-04-04_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-10/metrics/2025-08-13_06-04-04_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-10_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_06-04-04_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_06-04-04_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml"
                }
            ]
        },

        {
            "group_name": "Expert 32x32_Rendezvous",
            "task_name": "Rendezvous",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_12-11-23_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-12_12-11-23_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_12-11-23_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_12-11-23_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_12-40-28_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-12_12-40-28_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_12-40-28_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_12-40-28_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_13-09-12_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-12_13-09-12_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_13-09-12_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_13-09-12_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_13-37-58_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-12_13-37-58_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_13-37-58_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_13-37-58_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_14-06-53_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-12_14-06-53_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_14-06-53_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_14-06-53_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_14-35-52_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-08-12_14-35-52_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-6_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_14-35-52_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_14-35-52_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_15-04-48_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-08-12_15-04-48_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-7_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_15-04-48_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_15-04-48_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_15-33-44_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-12_15-33-44_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-8_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_15-33-44_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_15-33-44_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_16-02-57_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-08-12_16-02-57_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-9_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_16-02-57_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_16-02-57_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_16-32-07_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-08-12_16-32-07_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-10_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_16-32-07_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_16-32-07_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml"
                }
            ]
        },

        # Baseline 32x32 w Gobs Tinf
        {
            "group_name": "Baseline w Gobs Tinf 32x32_GoToPosition",
            "task_name": "GoToPosition",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_11-30-10_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-10_11-30-10_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_11-30-10_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_11-30-10_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_12-24-10_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-10_12-24-10_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_12-24-10_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_12-24-10_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_13-18-14_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-10_13-18-14_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_13-18-14_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_13-18-14_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_14-11-59_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-10_14-11-59_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_14-11-59_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_14-11-59_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_15-06-14_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-10_15-06-14_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_15-06-14_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_15-06-14_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_16-01-00_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-08-10_16-01-00_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-6_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_16-01-00_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_16-01-00_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_16-55-10_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-08-10_16-55-10_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-7_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_16-55-10_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_16-55-10_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_17-50-55_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-10_17-50-55_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-8_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_17-50-55_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_17-50-55_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_18-44-46_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-08-10_18-44-46_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-9_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_18-44-46_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_18-44-46_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_19-39-06_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-08-10_19-39-06_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-10_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_19-39-06_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_19-39-06_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml",
                },
            ]
        },
        {
            "group_name": "Baseline w Gobs Tinf 32x32_GoToPose",
            "task_name": "GoToPose",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_11-30-10_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-10_11-30-10_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_11-30-10_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_11-30-10_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_12-24-10_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-10_12-24-10_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_12-24-10_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_12-24-10_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_13-18-14_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-10_13-18-14_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_13-18-14_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_13-18-14_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_14-11-59_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-10_14-11-59_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_14-11-59_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_14-11-59_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_15-06-14_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-10_15-06-14_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_15-06-14_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_15-06-14_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_16-01-00_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-08-10_16-01-00_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-6_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_16-01-00_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_16-01-00_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_16-55-10_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-08-10_16-55-10_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-7_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_16-55-10_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_16-55-10_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_17-50-55_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-10_17-50-55_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-8_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_17-50-55_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_17-50-55_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_18-44-46_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-08-10_18-44-46_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-9_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_18-44-46_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_18-44-46_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_19-39-06_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-08-10_19-39-06_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-10_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_19-39-06_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_19-39-06_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml",
                },
            ]
        },
        {
            "group_name": "Baseline w Gobs Tinf 32x32_TrackVelocities",
            "task_name": "TrackVelocities",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_11-30-10_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-10_11-30-10_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_11-30-10_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_11-30-10_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_12-24-10_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-10_12-24-10_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_12-24-10_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_12-24-10_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_13-18-14_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-10_13-18-14_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_13-18-14_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_13-18-14_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_14-11-59_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-10_14-11-59_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_14-11-59_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_14-11-59_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_15-06-14_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-10_15-06-14_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_15-06-14_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_15-06-14_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_16-01-00_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-08-10_16-01-00_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-6_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_16-01-00_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_16-01-00_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_16-55-10_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-08-10_16-55-10_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-7_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_16-55-10_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_16-55-10_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_17-50-55_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-10_17-50-55_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-8_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_17-50-55_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_17-50-55_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_18-44-46_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-08-10_18-44-46_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-9_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_18-44-46_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_18-44-46_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_19-39-06_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-08-10_19-39-06_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-10_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_19-39-06_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_19-39-06_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml",
                },
            ]
        },
        {
            "group_name": "Baseline w Gobs Tinf 32x32_Rendezvous",
            "task_name": "Rendezvous",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_11-30-10_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-10_11-30-10_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_11-30-10_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_11-30-10_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_12-24-10_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-10_12-24-10_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_12-24-10_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_12-24-10_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_13-18-14_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-10_13-18-14_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_13-18-14_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_13-18-14_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_14-11-59_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-10_14-11-59_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_14-11-59_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_14-11-59_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_15-06-14_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-10_15-06-14_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_15-06-14_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_15-06-14_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_16-01-00_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-08-10_16-01-00_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-6_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_16-01-00_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_16-01-00_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_16-55-10_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-08-10_16-55-10_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-7_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_16-55-10_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_16-55-10_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_17-50-55_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-10_17-50-55_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-8_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_17-50-55_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_17-50-55_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_18-44-46_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-08-10_18-44-46_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-9_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_18-44-46_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_18-44-46_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_19-39-06_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-08-10_19-39-06_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-10_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_19-39-06_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_19-39-06_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml",
                },
            ]
        },

        # Baseline 32x32 w/o Gobs Tinf
        {
            "group_name": "Baseline w/o Gobs Tinf 32x32_GoToPosition",
            "task_name": "GoToPosition",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_13-10-17_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-14_13-10-17_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_13-10-17_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_13-10-17_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_14-04-11_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-14_14-04-11_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_14-04-11_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_14-04-11_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_15-00-56_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-14_15-00-56_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_15-00-56_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_15-00-56_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_15-55-17_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-14_15-55-17_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_15-55-17_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_15-55-17_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_16-49-14_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-14_16-49-14_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_16-49-14_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_16-49-14_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_17-42-55_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-08-14_17-42-55_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-6_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_17-42-55_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_17-42-55_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_18-36-07_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-08-14_18-36-07_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-7_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_18-36-07_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_18-36-07_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_19-31-16_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-14_19-31-16_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-8_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_19-31-16_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_19-31-16_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_20-25-00_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-08-14_20-25-00_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-9_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_20-25-00_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_20-25-00_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_21-18-32_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-08-14_21-18-32_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-10_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_21-18-32_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_21-18-32_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml"
                }
            ]
        },
        {
            "group_name": "Baseline w/o Gobs Tinf 32x32_GoToPose",
            "task_name": "GoToPose",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_13-10-17_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-14_13-10-17_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_13-10-17_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_13-10-17_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_14-04-11_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-14_14-04-11_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_14-04-11_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_14-04-11_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_15-00-56_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-14_15-00-56_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_15-00-56_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_15-00-56_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_15-55-17_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-14_15-55-17_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_15-55-17_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_15-55-17_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_16-49-14_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-14_16-49-14_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_16-49-14_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_16-49-14_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_17-42-55_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-08-14_17-42-55_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-6_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_17-42-55_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_17-42-55_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_18-36-07_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-08-14_18-36-07_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-7_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_18-36-07_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_18-36-07_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_19-31-16_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-14_19-31-16_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-8_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_19-31-16_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_19-31-16_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_20-25-00_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-08-14_20-25-00_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-9_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_20-25-00_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_20-25-00_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_21-18-32_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-08-14_21-18-32_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-10_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_21-18-32_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_21-18-32_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml"
                }
            ]
        },
        {
            "group_name": "Baseline w/o Gobs Tinf 32x32_TrackVelocities",
            "task_name": "TrackVelocities",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_13-10-17_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-14_13-10-17_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_13-10-17_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_13-10-17_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_14-04-11_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-14_14-04-11_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_14-04-11_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_14-04-11_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_15-00-56_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-14_15-00-56_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_15-00-56_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_15-00-56_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_15-55-17_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-14_15-55-17_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_15-55-17_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_15-55-17_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_16-49-14_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-14_16-49-14_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_16-49-14_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_16-49-14_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_17-42-55_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-08-14_17-42-55_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-6_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_17-42-55_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_17-42-55_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_18-36-07_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-08-14_18-36-07_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-7_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_18-36-07_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_18-36-07_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_19-31-16_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-14_19-31-16_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-8_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_19-31-16_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_19-31-16_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_20-25-00_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-08-14_20-25-00_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-9_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_20-25-00_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_20-25-00_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_21-18-32_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-08-14_21-18-32_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-10_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_21-18-32_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_21-18-32_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml"
                }
            ]
        },
        {
            "group_name": "Baseline w/o Gobs Tinf 32x32_Rendezvous",
            "task_name": "Rendezvous",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_13-10-17_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-14_13-10-17_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_13-10-17_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_13-10-17_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_14-04-11_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-14_14-04-11_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_14-04-11_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_14-04-11_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_15-00-56_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-14_15-00-56_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_15-00-56_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_15-00-56_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_15-55-17_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-14_15-55-17_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_15-55-17_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_15-55-17_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_16-49-14_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-14_16-49-14_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_16-49-14_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_16-49-14_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_17-42-55_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-08-14_17-42-55_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-6_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_17-42-55_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_17-42-55_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_18-36-07_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-08-14_18-36-07_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-7_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_18-36-07_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_18-36-07_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_19-31-16_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-14_19-31-16_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-8_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_19-31-16_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_19-31-16_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_20-25-00_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-08-14_20-25-00_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-9_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_20-25-00_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_20-25-00_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_21-18-32_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-08-14_21-18-32_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-10_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_21-18-32_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_21-18-32_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml"
                }
            ]
        },
        
        # Hypernet Init 64x64x64 w Gobs Tinf
        {
            "group_name": "Hypernet Init w Gobs Tinf 64x64x64_GoToPosition",
            "task_name": "GoToPosition",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_18-34-24_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-09_18-34-24_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_18-34-24_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_18-34-24_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_19-36-10_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-09_19-36-10_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_19-36-10_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_19-36-10_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_20-38-03_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-09_20-38-03_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_20-38-03_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_20-38-03_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_21-39-56_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-09_21-39-56_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_21-39-56_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_21-39-56_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_23-43-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-08-09_23-43-47_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-6_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_23-43-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_23-43-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_00-45-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-08-10_00-45-42_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-7_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_00-45-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_00-45-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_01-47-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-10_01-47-47_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-8_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_01-47-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_01-47-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_02-49-37_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-08-10_02-49-37_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-9_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_02-49-37_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_02-49-37_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_03-52-03_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-08-10_03-52-03_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-10_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_03-52-03_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_03-52-03_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml",
                },
            ]
        },
        {
            "group_name": "Hypernet Init w Gobs Tinf 64x64x64_GoToPose",
            "task_name": "GoToPose",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_18-34-24_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-09_18-34-24_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_18-34-24_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_18-34-24_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_19-36-10_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-09_19-36-10_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_19-36-10_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_19-36-10_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_20-38-03_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-09_20-38-03_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_20-38-03_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_20-38-03_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_21-39-56_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-09_21-39-56_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_21-39-56_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_21-39-56_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_23-43-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-08-09_23-43-47_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-6_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_23-43-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_23-43-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_00-45-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-08-10_00-45-42_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-7_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_00-45-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_00-45-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_01-47-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-10_01-47-47_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-8_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_01-47-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_01-47-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_02-49-37_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-08-10_02-49-37_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-9_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_02-49-37_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_02-49-37_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_03-52-03_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-08-10_03-52-03_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-10_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_03-52-03_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_03-52-03_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml",
                },
            ]
        },
        {
            "group_name": "Hypernet Init w Gobs Tinf 64x64x64_TrackVelocities",
            "task_name": "TrackVelocities",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_18-34-24_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-09_18-34-24_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_18-34-24_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_18-34-24_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_19-36-10_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-09_19-36-10_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_19-36-10_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_19-36-10_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_20-38-03_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-09_20-38-03_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_20-38-03_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_20-38-03_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_21-39-56_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-09_21-39-56_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_21-39-56_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_21-39-56_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_23-43-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-08-09_23-43-47_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-6_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_23-43-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_23-43-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_00-45-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-08-10_00-45-42_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-7_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_00-45-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_00-45-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_01-47-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-10_01-47-47_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-8_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_01-47-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_01-47-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_02-49-37_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-08-10_02-49-37_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-9_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_02-49-37_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_02-49-37_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_03-52-03_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-08-10_03-52-03_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-10_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_03-52-03_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_03-52-03_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml",
                },
            ]
        },
        {
            "group_name": "Hypernet Init w Gobs Tinf 64x64x64_Rendezvous",
            "task_name": "Rendezvous",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_18-34-24_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-09_18-34-24_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_18-34-24_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_18-34-24_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_19-36-10_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-09_19-36-10_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_19-36-10_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_19-36-10_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_20-38-03_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-09_20-38-03_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_20-38-03_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_20-38-03_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_21-39-56_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-09_21-39-56_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_21-39-56_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_21-39-56_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_23-43-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-08-09_23-43-47_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-6_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_23-43-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_23-43-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_00-45-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-08-10_00-45-42_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-7_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_00-45-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_00-45-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_01-47-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-10_01-47-47_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-8_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_01-47-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_01-47-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_02-49-37_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-08-10_02-49-37_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-9_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_02-49-37_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_02-49-37_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_03-52-03_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-08-10_03-52-03_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-10_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_03-52-03_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_03-52-03_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml",
                },
            ]
        },

        # Hypernet Critic Init 64x64x64 w Gobs Tinf
        {
            "group_name": "Hypernet Critic Init w Gobs Tinf 64x64x64_GoToPosition",
            "task_name": "GoToPosition",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_01-33-49_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-11_01-33-49_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_01-33-49_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_01-33-49_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_06-36-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-08-11_06-36-22_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-9_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_06-36-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_06-36-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml",
                }
            ]
        },
        {
            "group_name": "Hypernet Critic Init w Gobs Tinf 64x64x64_GoToPose",
            "task_name": "GoToPose",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_01-33-49_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-11_01-33-49_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_01-33-49_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_01-33-49_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_06-36-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-08-11_06-36-22_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-9_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_06-36-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_06-36-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml",
                }
            ]
        },
        {
            "group_name": "Hypernet Critic Init w Gobs Tinf 64x64x64_TrackVelocities",
            "task_name": "TrackVelocities",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_01-33-49_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-11_01-33-49_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_01-33-49_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_01-33-49_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_06-36-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-08-11_06-36-22_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-9_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_06-36-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_06-36-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml",
                }
            ]
        },
        {
            "group_name": "Hypernet Critic Init w Gobs Tinf 64x64x64_Rendezvous",
            "task_name": "Rendezvous",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_01-33-49_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-11_01-33-49_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_01-33-49_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_01-33-49_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_06-36-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-08-11_06-36-22_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-9_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_06-36-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_06-36-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml",
                }
            ]
        },

        # Hypernet Critic Emb 32x32 w Gobs Tinf
        {
            "group_name": "Hypernet Critic Emb 32x32 w Gobs Tinf_GoToPosition",
            "task_name": "GoToPosition",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_10-01-14_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-11_10-01-14_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_10-01-14_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_10-01-14_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_11-09-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-11_11-09-48_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_11-09-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_11-09-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_12-18-36_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-11_12-18-36_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_12-18-36_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_12-18-36_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_13-27-58_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-11_13-27-58_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_13-27-58_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_13-27-58_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_14-37-12_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-11_14-37-12_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_14-37-12_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_14-37-12_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_15-46-33_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-08-11_15-46-33_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-6_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_15-46-33_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_15-46-33_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_16-56-16_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-08-11_16-56-16_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-7_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_16-56-16_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_16-56-16_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_18-05-29_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-11_18-05-29_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-8_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_18-05-29_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_18-05-29_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_19-14-49_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-08-11_19-14-49_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-9_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_19-14-49_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_19-14-49_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_20-23-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-08-11_20-23-46_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-10_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_20-23-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_20-23-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml",
                },
            ]
        },
        {
            "group_name": "Hypernet Critic Emb 32x32 w Gobs Tinf_GoToPose",
            "task_name": "GoToPose",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_10-01-14_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-11_10-01-14_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_10-01-14_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_10-01-14_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_11-09-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-11_11-09-48_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_11-09-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_11-09-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_12-18-36_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-11_12-18-36_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_12-18-36_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_12-18-36_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_13-27-58_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-11_13-27-58_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_13-27-58_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_13-27-58_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_14-37-12_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-11_14-37-12_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_14-37-12_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_14-37-12_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_15-46-33_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-08-11_15-46-33_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-6_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_15-46-33_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_15-46-33_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_16-56-16_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-08-11_16-56-16_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-7_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_16-56-16_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_16-56-16_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_18-05-29_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-11_18-05-29_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-8_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_18-05-29_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_18-05-29_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_19-14-49_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-08-11_19-14-49_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-9_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_19-14-49_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_19-14-49_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_20-23-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-08-11_20-23-46_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-10_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_20-23-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_20-23-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml",
                },
            ]
        },
        {
            "group_name": "Hypernet Critic Emb 32x32 w Gobs Tinf_TrackVelocities",
            "task_name": "TrackVelocities",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_10-01-14_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-11_10-01-14_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_10-01-14_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_10-01-14_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_11-09-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-11_11-09-48_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_11-09-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_11-09-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_12-18-36_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-11_12-18-36_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_12-18-36_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_12-18-36_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_13-27-58_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-11_13-27-58_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_13-27-58_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_13-27-58_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_14-37-12_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-11_14-37-12_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_14-37-12_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_14-37-12_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_15-46-33_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-08-11_15-46-33_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-6_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_15-46-33_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_15-46-33_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_16-56-16_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-08-11_16-56-16_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-7_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_16-56-16_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_16-56-16_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_18-05-29_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-11_18-05-29_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-8_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_18-05-29_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_18-05-29_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_19-14-49_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-08-11_19-14-49_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-9_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_19-14-49_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_19-14-49_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_20-23-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-08-11_20-23-46_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-10_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_20-23-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_20-23-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml",
                },
            ]
        },
        {
            "group_name": "Hypernet Critic Emb 32x32 w Gobs Tinf_Rendezvous",
            "task_name": "Rendezvous",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_10-01-14_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-11_10-01-14_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_10-01-14_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_10-01-14_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_11-09-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-11_11-09-48_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_11-09-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_11-09-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_12-18-36_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-11_12-18-36_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_12-18-36_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_12-18-36_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_13-27-58_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-11_13-27-58_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_13-27-58_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_13-27-58_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_14-37-12_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-11_14-37-12_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_14-37-12_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_14-37-12_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_15-46-33_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-08-11_15-46-33_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-6_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_15-46-33_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_15-46-33_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_16-56-16_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-08-11_16-56-16_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-7_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_16-56-16_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_16-56-16_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_18-05-29_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-11_18-05-29_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-8_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_18-05-29_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_18-05-29_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_19-14-49_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-08-11_19-14-49_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-9_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_19-14-49_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_19-14-49_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml",
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_20-23-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-08-11_20-23-46_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-10_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_20-23-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_20-23-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml",
                },
            ]
        },

        # Hypernet Critic Emb 32x32 w/o Gobs Tinf
        {
            "group_name": "Hypernet Critic Emb 32x32 w/o Gobs Tinf_GoToPosition",
            "task_name": "GoToPosition",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_09-00-04_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-13_09-00-04_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_09-00-04_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_09-00-04_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_10-11-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-13_10-11-51_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_10-11-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_10-11-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_11-23-25_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-13_11-23-25_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_11-23-25_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_11-23-25_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_12-32-30_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-13_12-32-30_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_12-32-30_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_12-32-30_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_13-40-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-13_13-40-54_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_13-40-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_13-40-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_14-49-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-08-13_14-49-26_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-6_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_14-49-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_14-49-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_15-57-56_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-08-13_15-57-56_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-7_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_15-57-56_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_15-57-56_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_17-06-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-13_17-06-02_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-8_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_17-06-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_17-06-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_18-13-50_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-08-13_18-13-50_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-9_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_18-13-50_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_18-13-50_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_19-21-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-08-13_19-21-53_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-10_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_19-21-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_19-21-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml"
                }
            ]
        },
        {
            "group_name": "Hypernet Critic Emb 32x32 w/o Gobs Tinf_GoToPose",
            "task_name": "GoToPose",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_09-00-04_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-13_09-00-04_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_09-00-04_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_09-00-04_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_10-11-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-13_10-11-51_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_10-11-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_10-11-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_11-23-25_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-13_11-23-25_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_11-23-25_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_11-23-25_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_12-32-30_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-13_12-32-30_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_12-32-30_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_12-32-30_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_13-40-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-13_13-40-54_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_13-40-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_13-40-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_14-49-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-08-13_14-49-26_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-6_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_14-49-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_14-49-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_15-57-56_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-08-13_15-57-56_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-7_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_15-57-56_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_15-57-56_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_17-06-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-13_17-06-02_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-8_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_17-06-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_17-06-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_18-13-50_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-08-13_18-13-50_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-9_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_18-13-50_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_18-13-50_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_19-21-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-08-13_19-21-53_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-10_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_19-21-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_19-21-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml"
                }
            ]
        },
        {
            "group_name": "Hypernet Critic Emb 32x32 w/o Gobs Tinf_TrackVelocities",
            "task_name": "TrackVelocities",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_09-00-04_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-13_09-00-04_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_09-00-04_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_09-00-04_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_10-11-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-13_10-11-51_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_10-11-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_10-11-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_11-23-25_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-13_11-23-25_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_11-23-25_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_11-23-25_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_12-32-30_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-13_12-32-30_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_12-32-30_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_12-32-30_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_13-40-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-13_13-40-54_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_13-40-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_13-40-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_14-49-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-08-13_14-49-26_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-6_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_14-49-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_14-49-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_15-57-56_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-08-13_15-57-56_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-7_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_15-57-56_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_15-57-56_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_17-06-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-13_17-06-02_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-8_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_17-06-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_17-06-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_18-13-50_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-08-13_18-13-50_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-9_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_18-13-50_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_18-13-50_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_19-21-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-08-13_19-21-53_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-10_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_19-21-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_19-21-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml"
                }
            ]
        },
        {
            "group_name": "Hypernet Critic Emb 32x32 w/o Gobs Tinf_Rendezvous",
            "task_name": "Rendezvous",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_09-00-04_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-13_09-00-04_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_09-00-04_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_09-00-04_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_10-11-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-13_10-11-51_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_10-11-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_10-11-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_11-23-25_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-13_11-23-25_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_11-23-25_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_11-23-25_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_12-32-30_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-13_12-32-30_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_12-32-30_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_12-32-30_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_13-40-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-13_13-40-54_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_13-40-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_13-40-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_14-49-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-08-13_14-49-26_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-6_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_14-49-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_14-49-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_15-57-56_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-08-13_15-57-56_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-7_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_15-57-56_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_15-57-56_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_17-06-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-13_17-06-02_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-8_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_17-06-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_17-06-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_18-13-50_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-08-13_18-13-50_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-9_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_18-13-50_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_18-13-50_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_19-21-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-08-13_19-21-53_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-10_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_19-21-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_19-21-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml"
                }
            ]
        },


        # Hypernet Critic Emb 64x64x64 w/o Gobs Tinf
        {
            "group_name": "Hypernet Critic Emb 64x64x64 w/o Gobs Tinf_GoToPosition",
            "task_name": "GoToPosition",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_21-50-52_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-13_21-50-52_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_21-50-52_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_21-50-52_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_23-00-04_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-13_23-00-04_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_23-00-04_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_23-00-04_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_00-09-18_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-14_00-09-18_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_00-09-18_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_00-09-18_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_01-18-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-14_01-18-02_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_01-18-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_01-18-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_02-26-08_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-14_02-26-08_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_02-26-08_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_02-26-08_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_03-34-58_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-08-14_03-34-58_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-6_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_03-34-58_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_03-34-58_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_05-52-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-14_05-52-47_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-8_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_05-52-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_05-52-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_07-03-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-08-14_07-03-02_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-9_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_07-03-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_07-03-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_08-12-01_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-08-14_08-12-01_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-10_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_08-12-01_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_08-12-01_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml"
                }
            ]
        },
        {
            "group_name": "Hypernet Critic Emb 64x64x64 w/o Gobs Tinf_GoToPose",
            "task_name": "GoToPose",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_21-50-52_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-13_21-50-52_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_21-50-52_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_21-50-52_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_23-00-04_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-13_23-00-04_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_23-00-04_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_23-00-04_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_00-09-18_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-14_00-09-18_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_00-09-18_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_00-09-18_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_01-18-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-14_01-18-02_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_01-18-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_01-18-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_02-26-08_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-14_02-26-08_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_02-26-08_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_02-26-08_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_03-34-58_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-08-14_03-34-58_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-6_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_03-34-58_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_03-34-58_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_05-52-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-14_05-52-47_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-8_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_05-52-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_05-52-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_07-03-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-08-14_07-03-02_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-9_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_07-03-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_07-03-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_08-12-01_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-08-14_08-12-01_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-10_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_08-12-01_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_08-12-01_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml"
                }
            ]
        },
        {
            "group_name": "Hypernet Critic Emb 64x64x64 w/o Gobs Tinf_TrackVelocities",
            "task_name": "TrackVelocities",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_21-50-52_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-13_21-50-52_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_21-50-52_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_21-50-52_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_23-00-04_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-13_23-00-04_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_23-00-04_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_23-00-04_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_00-09-18_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-14_00-09-18_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_00-09-18_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_00-09-18_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_01-18-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-14_01-18-02_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_01-18-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_01-18-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_02-26-08_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-14_02-26-08_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_02-26-08_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_02-26-08_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_03-34-58_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-08-14_03-34-58_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-6_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_03-34-58_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_03-34-58_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_05-52-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-14_05-52-47_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-8_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_05-52-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_05-52-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_07-03-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-08-14_07-03-02_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-9_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_07-03-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_07-03-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_08-12-01_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-08-14_08-12-01_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-10_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_08-12-01_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_08-12-01_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml"
                }
            ]
        },
        {
            "group_name": "Hypernet Critic Emb 64x64x64 w/o Gobs Tinf_Rendezvous",
            "task_name": "Rendezvous",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_21-50-52_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-13_21-50-52_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_21-50-52_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_21-50-52_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_23-00-04_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-13_23-00-04_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_23-00-04_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_23-00-04_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_00-09-18_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-14_00-09-18_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_00-09-18_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_00-09-18_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_01-18-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-14_01-18-02_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_01-18-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_01-18-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_02-26-08_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-14_02-26-08_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_02-26-08_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_02-26-08_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_03-34-58_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-08-14_03-34-58_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-6_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_03-34-58_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_03-34-58_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_05-52-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-14_05-52-47_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-8_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_05-52-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_05-52-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_07-03-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-08-14_07-03-02_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-9_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_07-03-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_07-03-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_08-12-01_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-08-14_08-12-01_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-10_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_08-12-01_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_08-12-01_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml"
                }
            ]
        },

        # Hypernet Emb 32x32 w/o Gobs Tinf
        {
            "group_name": "Hypernet Emb 32x32 w/o Gobs Tinf GoToPosition",
            "task_name": "GoToPosition",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_09-08-45_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-15_09-08-45_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_09-08-45_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_09-08-45_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_10-12-15_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-15_10-12-15_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_10-12-15_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_10-12-15_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_11-16-09_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-15_11-16-09_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_11-16-09_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_11-16-09_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_13-21-15_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-15_13-21-15_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_13-21-15_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_13-21-15_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_15-25-37_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-08-15_15-25-37_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-7_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_15-25-37_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_15-25-37_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_16-28-01_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-15_16-28-01_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-8_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_16-28-01_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_16-28-01_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_17-29-41_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-08-15_17-29-41_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-9_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_17-29-41_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_17-29-41_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml"
                }
            ]
        },
        {
            "group_name": "Hypernet Emb 32x32 w/o Gobs Tinf GoToPose",
            "task_name": "GoToPose",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_09-08-45_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-15_09-08-45_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_09-08-45_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_09-08-45_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_10-12-15_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-15_10-12-15_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_10-12-15_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_10-12-15_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_11-16-09_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-15_11-16-09_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_11-16-09_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_11-16-09_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_13-21-15_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-15_13-21-15_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_13-21-15_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_13-21-15_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_15-25-37_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-08-15_15-25-37_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-7_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_15-25-37_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_15-25-37_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_16-28-01_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-15_16-28-01_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-8_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_16-28-01_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_16-28-01_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_17-29-41_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-08-15_17-29-41_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-9_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_17-29-41_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_17-29-41_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml"
                }
            ]
        },
        {
            "group_name": "Hypernet Emb 32x32 w/o Gobs Tinf TrackVelocites",
            "task_name": "TrackVelocities",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_09-08-45_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-15_09-08-45_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_09-08-45_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_09-08-45_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_10-12-15_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-15_10-12-15_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_10-12-15_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_10-12-15_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_11-16-09_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-15_11-16-09_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_11-16-09_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_11-16-09_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_13-21-15_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-15_13-21-15_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_13-21-15_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_13-21-15_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_15-25-37_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-08-15_15-25-37_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-7_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_15-25-37_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_15-25-37_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_16-28-01_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-15_16-28-01_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-8_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_16-28-01_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_16-28-01_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_17-29-41_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-08-15_17-29-41_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-9_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_17-29-41_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_17-29-41_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml"
                }
            ]
        },
        {
            "group_name": "Hypernet Emb 32x32 w/o Gobs Tinf Rendezvous",
            "task_name": "Rendezvous",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_09-08-45_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-15_09-08-45_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_09-08-45_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_09-08-45_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_10-12-15_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-15_10-12-15_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_10-12-15_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_10-12-15_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_11-16-09_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-15_11-16-09_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_11-16-09_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_11-16-09_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_13-21-15_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-15_13-21-15_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_13-21-15_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_13-21-15_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_15-25-37_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-08-15_15-25-37_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-7_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_15-25-37_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_15-25-37_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_16-28-01_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-15_16-28-01_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-8_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_16-28-01_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_16-28-01_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_17-29-41_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-08-15_17-29-41_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-9_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_17-29-41_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_17-29-41_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml"
                }
            ]
        },

        # Reward with action rate at target Hypernet last layer
        # Hypernet Critic Emb 64, 64, 64 w/o Gobs Tinf
        {
            "group_name": "Hypernet Critic L Emb 64, 64, 64 w/o Gobs Tinf GoToPosition",
            "task_name": "GoToPosition",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_08-26-39_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-21_08-26-39_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_08-26-39_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_08-26-39_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_09-25-57_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-21_09-25-57_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_09-25-57_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_09-25-57_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_10-25-14_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-21_10-25-14_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_10-25-14_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_10-25-14_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_11-24-07_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-21_11-24-07_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_11-24-07_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_11-24-07_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_12-22-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-08-21_12-22-53_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-6_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_12-22-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_12-22-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_13-21-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-08-21_13-21-54_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-7_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_13-21-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_13-21-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_14-20-58_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-21_14-20-58_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-8_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_14-20-58_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_14-20-58_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_15-19-37_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-08-21_15-19-37_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-9_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_15-19-37_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_15-19-37_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_16-17-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-08-21_16-17-46_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-10_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_16-17-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_16-17-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml"
                }
            ]
        },
        {
            "group_name": "Hypernet Critic L Emb 64, 64, 64 w/o Gobs Tinf GoToPose",
            "task_name": "GoToPose",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_08-26-39_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-21_08-26-39_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_08-26-39_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_08-26-39_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                # {
                #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_09-25-57_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-21_09-25-57_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-3_metrics.csv",
                #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_09-25-57_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPose.csv",
                #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_09-25-57_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                # },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_10-25-14_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-21_10-25-14_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_10-25-14_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_10-25-14_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_11-24-07_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-21_11-24-07_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_11-24-07_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_11-24-07_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_12-22-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-08-21_12-22-53_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-6_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_12-22-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_12-22-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_13-21-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-08-21_13-21-54_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-7_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_13-21-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_13-21-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_14-20-58_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-21_14-20-58_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-8_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_14-20-58_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_14-20-58_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_15-19-37_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-08-21_15-19-37_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-9_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_15-19-37_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_15-19-37_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml"
                },
                # {
                #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_16-17-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-08-21_16-17-46_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-10_metrics.csv",
                #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_16-17-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_GoToPose.csv",
                #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_16-17-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml"
                # }
            ]
        },
        {
            "group_name": "Hypernet Critic L Emb 64, 64, 64 w/o Gobs Tinf TrackVelocities",
            "task_name": "TrackVelocities",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_08-26-39_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-21_08-26-39_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_08-26-39_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_08-26-39_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_09-25-57_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-21_09-25-57_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_09-25-57_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_09-25-57_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_10-25-14_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-21_10-25-14_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_10-25-14_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_10-25-14_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_11-24-07_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-21_11-24-07_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_11-24-07_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_11-24-07_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_12-22-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-08-21_12-22-53_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-6_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_12-22-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_12-22-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_13-21-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-08-21_13-21-54_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-7_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_13-21-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_13-21-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_14-20-58_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-21_14-20-58_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-8_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_14-20-58_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_14-20-58_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_15-19-37_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-08-21_15-19-37_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-9_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_15-19-37_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_15-19-37_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_16-17-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-08-21_16-17-46_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-10_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_16-17-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_16-17-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml"
                }
            ]
        },
        {
            "group_name": "Hypernet Critic L Emb 64, 64, 64 w/o Gobs Tinf Rendezvous",
            "task_name": "Rendezvous",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_08-26-39_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-21_08-26-39_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_08-26-39_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_08-26-39_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_09-25-57_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-21_09-25-57_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_09-25-57_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_09-25-57_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_10-25-14_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-21_10-25-14_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_10-25-14_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_10-25-14_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_11-24-07_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-21_11-24-07_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_11-24-07_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_11-24-07_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_12-22-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/2025-08-21_12-22-53_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-6_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_12-22-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_12-22-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_13-21-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/2025-08-21_13-21-54_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-7_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_13-21-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_13-21-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_14-20-58_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/2025-08-21_14-20-58_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-8_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_14-20-58_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_14-20-58_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_15-19-37_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/2025-08-21_15-19-37_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-9_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_15-19-37_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_15-19-37_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_16-17-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/2025-08-21_16-17-46_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-10_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_16-17-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_16-17-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/metrics/env_info.yaml"
                }
            ]
        },

        # Hypernet Critic L Emb 32x32 w/o Gobs Tinf
        {
            "group_name": "Hypernet Critic L Emb 32x32 w/o Gobs Tinf GoToPosition",
            "task_name": "GoToPosition",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_07-23-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-22_07-23-51_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_07-23-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_07-23-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_08-22-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-22_08-22-47_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_08-22-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_08-22-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_09-20-23_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-22_09-20-23_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_09-20-23_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_09-20-23_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_10-18-16_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-22_10-18-16_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_10-18-16_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_10-18-16_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_11-17-01_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-22_11-17-01_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_11-17-01_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_11-17-01_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                }
            ]
        },
        {
            "group_name": "Hypernet Critic L Emb 32x32 w/o Gobs Tinf GoToPose",
            "task_name": "GoToPose",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_07-23-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-22_07-23-51_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_07-23-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_07-23-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_08-22-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-22_08-22-47_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_08-22-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_08-22-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_09-20-23_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-22_09-20-23_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_09-20-23_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_09-20-23_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_10-18-16_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-22_10-18-16_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_10-18-16_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_10-18-16_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_11-17-01_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-22_11-17-01_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_11-17-01_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_11-17-01_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                }
            ]
        },
        {
            "group_name": "Hypernet Critic L Emb 32x32 w/o Gobs Tinf TrackVelocities",
            "task_name": "TrackVelocities",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_07-23-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-22_07-23-51_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_07-23-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_07-23-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_08-22-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-22_08-22-47_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_08-22-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_08-22-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_09-20-23_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-22_09-20-23_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_09-20-23_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_09-20-23_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_10-18-16_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-22_10-18-16_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_10-18-16_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_10-18-16_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_11-17-01_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-22_11-17-01_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_11-17-01_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_11-17-01_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                }
            ]
        },
        {
            "group_name": "Hypernet Critic L Emb 32x32 w/o Gobs Tinf Rendezvous",
            "task_name": "Rendezvous",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_07-23-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-22_07-23-51_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_07-23-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_07-23-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_08-22-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-22_08-22-47_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_08-22-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_08-22-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_09-20-23_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-22_09-20-23_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_09-20-23_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_09-20-23_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_10-18-16_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-22_10-18-16_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_10-18-16_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_10-18-16_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_11-17-01_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-22_11-17-01_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_11-17-01_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_11-17-01_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                }
            ]
        },

        # Baseline L 32x32 w Tinf in Gobs
        {
            "group_name": "Baseline L 32x32 w Tinf in Gobs GoToPosition",
            "task_name": "GoToPosition",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_13-56-13_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-22_13-56-13_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_13-56-13_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_13-56-13_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_14-50-46_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-22_14-50-46_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_14-50-46_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_14-50-46_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_15-46-04_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-22_15-46-04_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_15-46-04_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_15-46-04_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_16-40-36_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-22_16-40-36_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_16-40-36_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_16-40-36_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_17-35-32_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-22_17-35-32_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_17-35-32_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_17-35-32_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                }
            ]
        },
        {
            "group_name": "Baseline L 32x32 w Tinf in Gobs GoToPose",
            "task_name": "GoToPose",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_13-56-13_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-22_13-56-13_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_13-56-13_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_13-56-13_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_14-50-46_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-22_14-50-46_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_14-50-46_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_14-50-46_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_15-46-04_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-22_15-46-04_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_15-46-04_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_15-46-04_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_16-40-36_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-22_16-40-36_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_16-40-36_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_16-40-36_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_17-35-32_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-22_17-35-32_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_17-35-32_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_17-35-32_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                }
            ]
        },
        {
            "group_name": "Baseline L 32x32 w Tinf in Gobs TrackVelocities",
            "task_name": "TrackVelocities",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_13-56-13_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-22_13-56-13_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_13-56-13_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_13-56-13_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_14-50-46_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-22_14-50-46_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_14-50-46_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_14-50-46_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_15-46-04_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-22_15-46-04_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_15-46-04_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_15-46-04_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_16-40-36_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-22_16-40-36_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_16-40-36_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_16-40-36_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_17-35-32_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-22_17-35-32_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_17-35-32_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_17-35-32_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                }
            ]
        },
        {
            "group_name": "Baseline L 32x32 w Tinf in Gobs Rendezvous",
            "task_name": "Rendezvous",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_13-56-13_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-22_13-56-13_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_13-56-13_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_13-56-13_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_14-50-46_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-22_14-50-46_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_14-50-46_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_14-50-46_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_15-46-04_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-22_15-46-04_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_15-46-04_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_15-46-04_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_16-40-36_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-22_16-40-36_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_16-40-36_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_16-40-36_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_17-35-32_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-22_17-35-32_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_17-35-32_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_17-35-32_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                }
            ]
        },

        # Hypernet Critic L Emb 64x64 w/o Tinf in Gobs (tanh, 256C)
        {
            "group_name": "Hypernet Critic L Emb 64x64 w/o Tinf in Gobs (tanh, 256C) GoToPosition",
            "task_name": "GoToPosition",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_06-53-13_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-25_06-53-13_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_06-53-13_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_06-53-13_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_07-51-23_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-25_07-51-23_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_07-51-23_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_07-51-23_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_08-50-04_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-25_08-50-04_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_08-50-04_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_08-50-04_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_09-48-52_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-25_09-48-52_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_09-48-52_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_09-48-52_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_10-47-20_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-25_10-47-20_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_10-47-20_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_10-47-20_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                }
            ]
        },
        {
            "group_name": "Hypernet Critic L Emb 64x64 w/o Tinf in Gobs (tanh, 256C) GoToPose",
            "task_name": "GoToPose",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_06-53-13_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-25_06-53-13_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_06-53-13_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_06-53-13_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_07-51-23_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-25_07-51-23_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_07-51-23_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_07-51-23_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_08-50-04_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-25_08-50-04_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_08-50-04_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_08-50-04_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_09-48-52_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-25_09-48-52_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_09-48-52_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_09-48-52_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_10-47-20_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-25_10-47-20_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_10-47-20_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_10-47-20_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                }
            ]
        },
        {
            "group_name": "Hypernet Critic L Emb 64x64 w/o Tinf in Gobs (tanh, 256C) TrackVelocities",
            "task_name": "TrackVelocities",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_06-53-13_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-25_06-53-13_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_06-53-13_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_06-53-13_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_07-51-23_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-25_07-51-23_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_07-51-23_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_07-51-23_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_08-50-04_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-25_08-50-04_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_08-50-04_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_08-50-04_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_09-48-52_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-25_09-48-52_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_09-48-52_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_09-48-52_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_10-47-20_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-25_10-47-20_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_10-47-20_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_10-47-20_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                }
            ]
        },
        {
            "group_name": "Hypernet Critic L Emb 64x64 w/o Tinf in Gobs (tanh, 256C) Rendezvous",
            "task_name": "Rendezvous",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_06-53-13_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-25_06-53-13_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_06-53-13_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_06-53-13_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_07-51-23_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-25_07-51-23_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_07-51-23_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_07-51-23_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_08-50-04_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-25_08-50-04_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_08-50-04_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_08-50-04_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_09-48-52_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-25_09-48-52_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_09-48-52_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_09-48-52_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_10-47-20_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-25_10-47-20_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_10-47-20_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_10-47-20_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                }
            ]
        },

        # Experts L
        {
            "group_name": "Expert L GoToPosition (C 256)",
            "task_name": "GoToPosition",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_14-50-34_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-1/metrics/2025-08-26_14-50-34_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_14-50-34_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_14-50-34_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_15-17-15_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-2/metrics/2025-08-26_15-17-15_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_15-17-15_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_15-17-15_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_15-43-52_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-3/metrics/2025-08-26_15-43-52_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_15-43-52_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_15-43-52_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_16-10-15_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-4/metrics/2025-08-26_16-10-15_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_16-10-15_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_16-10-15_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_16-36-46_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-5/metrics/2025-08-26_16-36-46_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_16-36-46_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_16-36-46_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                }
            ]
        },
        {
            "group_name": "Expert L GoToPose (C 256)",
            "task_name": "GoToPose",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_17-03-26_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-1/metrics/2025-08-26_17-03-26_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_17-03-26_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_17-03-26_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_17-30-04_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-2/metrics/2025-08-26_17-30-04_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_17-30-04_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_17-30-04_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_17-56-58_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-3/metrics/2025-08-26_17-56-58_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_17-56-58_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_17-56-58_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_18-23-36_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-4/metrics/2025-08-26_18-23-36_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_18-23-36_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_18-23-36_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_18-50-24_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-5/metrics/2025-08-26_18-50-24_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_18-50-24_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_18-50-24_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                }
            ]
        },
        {
            "group_name": "Expert L TrackVelocities (C 256)",
            "task_name": "TrackVelocities",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_19-17-29_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-1/metrics/2025-08-26_19-17-29_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_19-17-29_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_19-17-29_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_19-45-45_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-2/metrics/2025-08-26_19-45-45_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_19-45-45_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_19-45-45_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_20-13-52_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-3/metrics/2025-08-26_20-13-52_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_20-13-52_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_20-13-52_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_20-41-46_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-4/metrics/2025-08-26_20-41-46_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_20-41-46_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_20-41-46_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_21-09-28_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-5/metrics/2025-08-26_21-09-28_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_21-09-28_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_21-09-28_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                }
            ]
        },
        {
            "group_name": "Expert L Rendezvous (C 256)",
            "task_name": "Rendezvous",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_12-25-35_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-26_12-25-35_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_12-25-35_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_12-25-35_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_12-54-36_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-26_12-54-36_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_12-54-36_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_12-54-36_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_13-23-22_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-26_13-23-22_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_13-23-22_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_13-23-22_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_13-52-31_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-26_13-52-31_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_13-52-31_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_13-52-31_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_14-21-47_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-26_14-21-47_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_14-21-47_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_14-21-47_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                }
            ]
        },

        # Baseline L 32x32 w Tinf in Gobs (tanh C 256) 
        {
            "group_name": "Baseline L 32x32 w Tinf in Gobs (tanh C 256) GoToPosition",
            "task_name": "GoToPosition",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_09-00-16_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-27_09-00-16_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_09-00-16_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_09-00-16_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_09-55-55_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-27_09-55-55_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_09-55-55_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_09-55-55_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_10-51-46_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-27_10-51-46_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_10-51-46_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_10-51-46_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_11-48-03_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-27_11-48-03_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_11-48-03_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_11-48-03_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_12-43-08_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-27_12-43-08_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_12-43-08_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_12-43-08_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                }
            ]
        },
        {
            "group_name": "Baseline L 32x32 w Tinf in Gobs (tanh C 256) GoToPose",
            "task_name": "GoToPose",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_09-00-16_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-27_09-00-16_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_09-00-16_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_09-00-16_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_09-55-55_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-27_09-55-55_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_09-55-55_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_09-55-55_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_10-51-46_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-27_10-51-46_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_10-51-46_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_10-51-46_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_11-48-03_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-27_11-48-03_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_11-48-03_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_11-48-03_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_12-43-08_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-27_12-43-08_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_12-43-08_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_12-43-08_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                }
            ]
        },
        {
            "group_name": "Baseline L 32x32 w Tinf in Gobs (tanh C 256) TrackVelocities",
            "task_name": "TrackVelocities",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_09-00-16_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-27_09-00-16_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_09-00-16_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_09-00-16_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_09-55-55_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-27_09-55-55_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_09-55-55_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_09-55-55_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_10-51-46_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-27_10-51-46_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_10-51-46_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_10-51-46_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_11-48-03_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-27_11-48-03_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_11-48-03_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_11-48-03_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_12-43-08_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-27_12-43-08_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_12-43-08_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_12-43-08_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                }
            ]
        },
        {
            "group_name": "Baseline L 32x32 w Tinf in Gobs (tanh C 256) Rendezvous",
            "task_name": "Rendezvous",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_09-00-16_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-27_09-00-16_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_09-00-16_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_09-00-16_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_09-55-55_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-27_09-55-55_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_09-55-55_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_09-55-55_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_10-51-46_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-27_10-51-46_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_10-51-46_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_10-51-46_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_11-48-03_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-27_11-48-03_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_11-48-03_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_11-48-03_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_12-43-08_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-27_12-43-08_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_12-43-08_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_12-43-08_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                }
            ]
        },

        
    ]

    plot_cfg = {
        "title": "",
        "box_colors": [
            "#FF3D50",
            "#FFA034",
            "#2FA1FF",
            "#A734FF",
            "#FFFF3D",
            "#4DFF3D",
            "#FF3DBB",
            "#623652",
            "#00CED1",  # Dark Turquoise
            "#1E90FF",  # Dodger Blue
            "#4682B4",  # Steel Blue
            "#32CD32",  # Lime Green
            "#008080",  # Teal
            "#20B2AA",  # Light Sea Green
            "#8A2BE2",  # Blue Violet
            "#9932CC",  # Dark Orchid
            "#BA55D3",  # Medium Orchid
            "#FF8C00",  # Dark Orange
            "#D2691E",  # Chocolate
            "#B8860B",  # Dark Goldenrod
            "#FF69B4",  # Hot Pink
            "#DB7093",  # Pale Violet Red
            "#C71585",  # Medium Violet Red
            "#DC143C",  # Crimson
            "#B22222",  # Firebrick
            "#808080",  # Grey
            "#D3D3D3",  # Light Grey
        ],
        "runs_names": [], # This will be filled with group names
        "zoom_in": False,
    }

    save_plots_folder_path = "/workspace/isaaclab/source/plots/orbital_control" # Specify the folder path where you want to save the plots
    if not os.path.exists(save_plots_folder_path):
        os.makedirs(save_plots_folder_path)

    dfs = {}
    trajectories_dfs = {}
    labels = {} # This will now store labels for individual seeds within a group
    env_infos = {} # Store env_info for each group (assuming it's consistent within a group)
    list_of_tasks = []
    robot_names_per_task = {} # To store robot name for each task

    for group_data in list_of_grouped_csv_data:
        group_name = group_data["group_name"]
        task_name = group_data["task_name"]
        robot_name = group_data["robot_name"]
        plot_cfg["runs_names"].append(group_name)

        # Add task and robot to their respective lists/dicts
        if task_name not in list_of_tasks:
            list_of_tasks.append(task_name)
        robot_names_per_task[task_name] = robot_name # Assuming robot_name is consistent per task

        dfs[group_name] = []
        trajectories_dfs[group_name] = []
        labels[group_name] = []

        # Loop through individual runs (seeds) within this group
        for run_info in group_data["runs"]:
            try:
                # Load metrics CSV
                metrics_file_path = run_info["metrics_csv"]
                df = pd.read_csv(metrics_file_path)
                dfs[group_name].append(df)
                labels[group_name].append(f"{group_name} - {os.path.basename(os.path.dirname(os.path.dirname(metrics_file_path)))}") # Label with group name and folder name
                # Load trajectories CSV
                trajectories_file_path = run_info["trajectories_csv"]
                if os.path.exists(trajectories_file_path):
                    # trajectories_df = pd.read_csv(trajectories_file_path)
                    # trajectories_dfs[group_name].append(trajectories_df)
                    pass
                else:
                    print(f"Warning: No trajectories file found at {trajectories_file_path} for group {group_name}")
                    trajectories_dfs[group_name].append(pd.DataFrame()) # Empty DataFrame as fallback

                # Load env info (assuming it's consistent for all seeds within a group)
                # env_info_file_path = run_info["env_info_yaml"]
                # with open(env_info_file_path, 'r') as f:
                #     env_infos[group_name] = yaml.safe_load(f)

            except Exception as e:
                print(f"Error reading file for run in group {group_name}: {e}")
                exit(0)

    # Group by task for plotting
    task_dfs = {}
    task_trajectories_dfs = {}
    task_labels = {}
    task_env_infos = {}

    for task in set(list_of_tasks): # Use set to get unique task names
        task_dfs[task] = {}
        task_trajectories_dfs[task] = {}
        task_labels[task] = {}
        task_env_infos[task] = {}

        for group_name, group_dfs in dfs.items():
            # Check if this group belongs to the current task
            # This assumes that group_name itself somehow implies the task,
            # or you might need a more explicit mapping if not.
            # For this example, we'll use the task_name stored in `list_of_grouped_csv_data`
            # and access it via the `group_data` structure.
            found_group_task_name = None
            for g_data in list_of_grouped_csv_data:
                if g_data["group_name"] == group_name:
                    found_group_task_name = g_data["task_name"]
                    break

            if found_group_task_name == task:
                task_dfs[task][group_name] = group_dfs
                # task_trajectories_dfs[task][group_name] = trajectories_dfs[group_name]
                task_labels[task][group_name] = labels[group_name]
                task_env_infos[task][group_name] = env_infos.get(group_name, {}) # Get env_info for this group

    # Create plots for each task
    for task_name in task_dfs:
        # Create a combined plot_cfg for TaskPlotsFactory if needed
        # For now, we pass the general plot_cfg. The TaskPlotsFactory should handle grouping internally.

        task_plots_factory = TaskPlotsFactory.create(
            task_name,
            dfs=task_dfs[task_name],
            trajectories_dfs=task_trajectories_dfs[task_name],
            labels=task_labels[task_name], # Pass the labels for each group
            env_info=task_env_infos[task_name], # Pass environment info per group (if multiple exist)
            folder_path=save_plots_folder_path,
            plot_cfg=plot_cfg,
        )
        task_plots_factory.plot()

        current_robot_name = robot_names_per_task.get(task_name)
        if current_robot_name:
            robot_plots_factory = RobotPlotsFactory.create(
                current_robot_name,
                dfs=task_dfs[task_name],
                trajectories_dfs=task_trajectories_dfs[task_name],
                labels=task_labels[task_name],
                env_info=task_env_infos[task_name],
                folder_path=save_plots_folder_path,
                plot_cfg=plot_cfg,
            )
            robot_plots_factory.plot()
        else:
            print(f"Warning: Could not determine robot name for task {task_name}. Skipping robot plots.")


if __name__ == "__main__":
    main()