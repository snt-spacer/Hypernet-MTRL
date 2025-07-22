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
        #     "group_name": "HyperNet Pattern 0", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "GoToPosition",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_06-00-25_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-42/pattern_0/2025-07-21_06-00-25_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-42_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_06-00-25_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-42/pattern_0/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_07-54-39_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-37/pattern_0/2025-07-21_07-54-39_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-37_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_07-54-39_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-37/pattern_0/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_09-43-20_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-8888/pattern_0/2025-07-21_09-43-20_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-8888_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_09-43-20_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-8888/pattern_0/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_11-38-10_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-653/pattern_0/2025-07-21_11-38-10_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-653_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_11-38-10_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-653/pattern_0/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "",
        #         },
        #         # Add more seeds for this group here
        #     ]
        # },
        # {
        #     "group_name": "Vanilla Pattern 0", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "GoToPosition",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_20-47-43_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-42/pattern_0/2025-07-21_20-47-43_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-42_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_20-47-43_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-42/pattern_0/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "",
        #         },
        #         # Add more seeds for this group here
        #     ]
        # },
        # {
        #     "group_name": "HyperNet Pattern 1", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "GoToPosition",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_06-00-25_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-42/pattern_1/2025-07-21_06-00-25_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-42_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_06-00-25_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-42/pattern_1/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_07-54-39_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-37/pattern_1/2025-07-21_07-54-39_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-37_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_07-54-39_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-37/pattern_0/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_09-43-20_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-8888/pattern_1/2025-07-21_09-43-20_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-8888_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_09-43-20_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-8888/pattern_1/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_11-38-10_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-653/pattern_1/2025-07-21_11-38-10_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-653_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_11-38-10_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-653/pattern_1/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "",
        #         },
        #         # Add more seeds for this group here
        #     ]
        # },
        # {
        #     "group_name": "Vanilla Pattern 1", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "GoToPosition",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_20-47-43_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-42/pattern_1/2025-07-21_20-47-43_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-42_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_20-47-43_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-42/pattern_1/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "",
        #         },
        #         # Add more seeds for this group here
        #     ]
        # },
        # {
        #     "group_name": "HyperNet Pattern 2", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "GoToPosition",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_06-00-25_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-42/pattern_2/2025-07-21_06-00-25_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-42_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_06-00-25_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-42/pattern_2/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_07-54-39_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-37/pattern_2/2025-07-21_07-54-39_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-37_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_07-54-39_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-37/pattern_0/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_09-43-20_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-8888/pattern_2/2025-07-21_09-43-20_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-8888_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_09-43-20_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-8888/pattern_2/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_11-38-10_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-653/pattern_2/2025-07-21_11-38-10_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-653_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_11-38-10_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-653/pattern_2/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "",
        #         },
        #         # Add more seeds for this group here
        #     ]
        # },
        # {
        #     "group_name": "Vanilla Pattern 2", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "GoToPosition",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_20-47-43_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-42/pattern_2/2025-07-21_20-47-43_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-42_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_20-47-43_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-42/pattern_2/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "",
        #         },
        #         # Add more seeds for this group here
        #     ]
        # },
        # {
        #     "group_name": "HyperNet Pattern 3", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "GoToPosition",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_06-00-25_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-42/pattern_3/2025-07-21_06-00-25_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-42_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_06-00-25_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-42/pattern_3/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_07-54-39_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-37/pattern_3/2025-07-21_07-54-39_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-37_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_07-54-39_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-37/pattern_0/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_09-43-20_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-8888/pattern_3/2025-07-21_09-43-20_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-8888_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_09-43-20_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-8888/pattern_3/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_11-38-10_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-653/pattern_3/2025-07-21_11-38-10_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-653_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_11-38-10_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-653/pattern_3/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "",
        #         },
        #         # Add more seeds for this group here
        #     ]
        # },
        # {
        #     "group_name": "Vanilla Pattern 3", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "GoToPosition",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_20-47-43_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-42/pattern_3/2025-07-21_20-47-43_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-42_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_20-47-43_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-42/pattern_3/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "",
        #         },
        #         # Add more seeds for this group here
        #     ]
        # },
        # {
        #     "group_name": "HyperNet Pattern 4", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "GoToPosition",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_06-00-25_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-42/pattern_4/2025-07-21_06-00-25_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-42_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_06-00-25_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-42/pattern_4/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_07-54-39_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-37/pattern_4/2025-07-21_07-54-39_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-37_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_07-54-39_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-37/pattern_4/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_09-43-20_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-8888/pattern_4/2025-07-21_09-43-20_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-8888_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_09-43-20_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-8888/pattern_4/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_11-38-10_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-653/pattern_4/2025-07-21_11-38-10_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-653_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_11-38-10_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-653/pattern_4/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "",
        #         },
        #         # Add more seeds for this group here
        #     ]
        # },
        # {
        #     "group_name": "Vanilla Pattern 4", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "GoToPosition",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_20-47-43_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-42/pattern_4/2025-07-21_20-47-43_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-42_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_20-47-43_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-42/pattern_4/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "",
        #         },
        #         # Add more seeds for this group here
        #     ]
        # },
        # {
        #     "group_name": "HyperNet Pattern 5", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "GoToPosition",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_06-00-25_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-42/pattern_5/2025-07-21_06-00-25_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-42_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_06-00-25_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-42/pattern_5/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_07-54-39_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-37/pattern_5/2025-07-21_07-54-39_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-37_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_07-54-39_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-37/pattern_5/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_09-43-20_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-8888/pattern_5/2025-07-21_09-43-20_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-8888_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_09-43-20_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-8888/pattern_5/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "",
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_11-38-10_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-653/pattern_5/2025-07-21_11-38-10_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-653_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_11-38-10_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-653/pattern_5/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "",
        #         },
        #         # Add more seeds for this group here
        #     ]
        # },
        # {
        #     "group_name": "Vanilla Pattern 5", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "GoToPosition",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_20-47-43_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-42/pattern_5/2025-07-21_20-47-43_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-42_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-21_20-47-43_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-42/pattern_5/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "",
        #         },
        #         # Add more seeds for this group here
        #     ]
        # },
        # Add more experiment groups as needed


        # {
        #     "group_name": "HyperNet Pattern 0", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "GoToPosition",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-17_18-34-33_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-42/pattern_0/2025-07-17_18-34-33_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-42_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-17_18-34-33_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-42/pattern_0/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "",
        #         },
        #         # Add more seeds for this group here
        #     ]
        # },
        # {
        #     "group_name": "Vanilla Pattern 0", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "GoToPosition",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-18_07-33-48_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-42/pattern_0/2025-07-18_07-33-48_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-42_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-18_07-33-48_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-42/pattern_0/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "",
        #         },
        #         # Add more seeds for this group here
        #     ]
        # },
        # {
        #     "group_name": "HyperNet Pattern 1", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "GoToPosition",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-17_18-34-33_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-42/pattern_1/2025-07-17_18-34-33_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-42_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-17_18-34-33_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-42/pattern_1/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "",
        #         },
        #         # Add more seeds for this group here
        #     ]
        # },
        # {
        #     "group_name": "Vanilla Pattern 1", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "GoToPosition",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-18_07-33-48_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-42/pattern_1/2025-07-18_07-33-48_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-42_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-18_07-33-48_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-42/pattern_1/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "",
        #         },
        #         # Add more seeds for this group here
        #     ]
        # },
        # {
        #     "group_name": "HyperNet Pattern 2", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "GoToPosition",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-17_18-34-33_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-42/pattern_2/2025-07-17_18-34-33_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-42_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-17_18-34-33_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-42/pattern_2/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "",
        #         },
        #         # Add more seeds for this group here
        #     ]
        # },
        # {
        #     "group_name": "Vanilla Pattern 2", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "GoToPosition",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-18_07-33-48_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-42/pattern_2/2025-07-18_07-33-48_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-42_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-18_07-33-48_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-42/pattern_2/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "",
        #         },
        #         # Add more seeds for this group here
        #     ]
        # },
        # {
        #     "group_name": "HyperNet Pattern 3", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "GoToPosition",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-17_18-34-33_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-42/pattern_3/2025-07-17_18-34-33_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-42_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-17_18-34-33_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-42/pattern_3/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "",
        #         },
        #         # Add more seeds for this group here
        #     ]
        # },
        # {
        #     "group_name": "Vanilla Pattern 3", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "GoToPosition",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-18_07-33-48_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-42/pattern_3/2025-07-18_07-33-48_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-42_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-18_07-33-48_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-42/pattern_3/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "",
        #         },
        #         # Add more seeds for this group here
        #     ]
        # },
        # {
        #     "group_name": "HyperNet Pattern 4", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "GoToPosition",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-17_18-34-33_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-42/pattern_4/2025-07-17_18-34-33_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-42_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-17_18-34-33_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-42/pattern_4/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "",
        #         },
        #         # Add more seeds for this group here
        #     ]
        # },
        # {
        #     "group_name": "Vanilla Pattern 4", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "GoToPosition",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-18_07-33-48_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-42/pattern_4/2025-07-18_07-33-48_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-42_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-18_07-33-48_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-42/pattern_4/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "",
        #         },
        #         # Add more seeds for this group here
        #     ]
        # },
        # {
        #     "group_name": "HyperNet Pattern 5", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "GoToPosition",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-17_18-34-33_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-42/pattern_5/2025-07-17_18-34-33_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-42_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-17_18-34-33_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-42/pattern_5/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "",
        #         },
        #         # Add more seeds for this group here
        #     ]
        # },
        # {
        #     "group_name": "Vanilla Pattern 5", # This is the name that will appear in plot_cfg["runs_names"]
        #     "task_name": "GoToPosition",
        #     "robot_name": "ModularFreeflyer",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-18_07-33-48_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-42/pattern_5/2025-07-18_07-33-48_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-42_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/source/evaluation_metrics/orbital_exp/2025-07-18_07-33-48_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-42/pattern_5/extracted_trajectories_GoToPosition.csv",
        #             "env_info_yaml": "",
        #         },
        #         # Add more seeds for this group here
        #     ]
        # },
    ]

    plot_cfg = {
        "title": "",
        "box_colors": [
            # "#FF3D50",
            # "#FFA034",
            # "#2FA1FF",
            # "#A734FF",
            # "#FFFF3D",
            # "#4DFF3D",
            # "#FF3DBB",
            # "#623652",
            # "#00CED1",  # Dark Turquoise
            # "#1E90FF",  # Dodger Blue
            # "#4682B4",  # Steel Blue
            # "#32CD32",  # Lime Green
            # "#008080",  # Teal
            # "#20B2AA",  # Light Sea Green
            # "#8A2BE2",  # Blue Violet
            # "#9932CC",  # Dark Orchid
            # "#BA55D3",  # Medium Orchid
            # "#FF8C00",  # Dark Orange
            # "#D2691E",  # Chocolate
            # "#B8860B",  # Dark Goldenrod
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

    save_plots_folder_path = "/workspace/isaaclab/source/plots/orbital_thruster_patterns_v2" # Specify the folder path where you want to save the plots
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