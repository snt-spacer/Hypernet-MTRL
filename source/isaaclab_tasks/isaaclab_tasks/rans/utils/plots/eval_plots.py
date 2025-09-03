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
        # [
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-13_12-34-20_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-42",
        # ],
        # [
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-13_12-23-18_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-42",
        # ],
        # [
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-13_12-10-01_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-42",
        # ],
        # [
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-13_11-56-31_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-42",
        # ],
        # [
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-13_11-13-22_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-42",
        # ],
        # [
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-13_10-59-34_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-42",
        # ],
        # [
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-15_14-42-48_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-42",
        # ],
        # [
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-15_15-36-48_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-42",
        # ],
        # [
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-15_10-26-40_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-42",
        # ],
        # [
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-15_11-18-58_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-42",
        # ],
        # [
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-15_12-04-29_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-42",
        # ],
        # [
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-15_12-49-51_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-42",
        # ],
        # [
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-15_13-37-19_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-42",
        # ],
        # [
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-15_16-54-39_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-42",
        # ],

        # Hypernetwork general obs w/o track info
        # [
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-28_11-29-34_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-42", 
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-28_12-11-16_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-37", 
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-29_10-52-34_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-456", 
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-30_12-34-08_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-789", 
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-30_13-16-15_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-1337", 
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-30_13-58-38_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-2048", 
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-30_14-40-44_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-3141", 
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-30_15-22-36_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-5678",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-30_16-05-09_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-9999",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-30_16-47-37_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-1111", 
        # ],

        # # Hypernetwork general obs w track info
        # [
        #     # "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-30_20-41-40_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-42",
        #     # "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-30_21-29-31_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-37",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-30_22-17-27_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-456", #G
        #     # "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-30_23-05-48_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-789",
        #     # "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-30_23-53-53_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-1337",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-31_00-41-57_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-2048", #G
        #     # "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-31_01-30-11_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-3141",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-31_02-17-57_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-5678", #G
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-31_03-05-57_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-9999",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-31_03-53-55_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-1111", #G
        # ],

        # # Hypernet Critic 32x32 w Tinfo in Gobs
        # [
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-04_10-07-04_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-42",
        #     # "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-04_11-07-35_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-37",
        #     # "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-04_12-08-08_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-456",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-04_13-08-34_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-789",
        #     # "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-04_14-09-09_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-1337",
        #     # "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-04_15-09-42_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-2048",
        #     # "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-04_16-10-07_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-3141",
        #     # "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-04_17-26-42_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-5678", 
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-04_18-27-27_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-9999",
        #     # "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-04_19-27-54_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-1111",
        # ],

        # # Baseline w/ track info in general obs
        # [
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-31_08-32-00_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-42",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-31_09-07-09_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-37",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-31_09-42-18_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-456",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-31_10-17-05_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-789",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-31_10-52-15_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-1337",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-31_11-27-09_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-2048",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-31_12-02-13_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-3141",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-31_12-37-16_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-5678",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-31_13-12-31_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-9999",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-31_13-48-01_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-1111",
        # ],

        # # Baseline w/o track info 
        # [
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-31_16-10-49_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-42",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-31_16-45-52_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-37",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-31_17-20-36_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-456",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-31_17-55-38_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-789",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-31_18-30-49_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-1337",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-31_19-06-02_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-2048",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-31_19-41-05_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-3141",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-31_20-16-21_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-5678",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-31_20-51-40_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-9999",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-31_21-26-32_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-1111",
        # ],

        # # Hypernet 64x64x64 w Tinfo in Gobs
        # [
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-01_14-07-00_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-42",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-01_14-55-56_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-37",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-01_15-44-38_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-456",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-01_16-32-57_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-789",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-01_17-21-33_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-1337",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-01_18-10-18_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-2048",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-01_18-59-03_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-3141",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-01_19-47-38_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-5678",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-01_20-36-24_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-9999",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-01_21-24-55_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-1111",
        # ],

        # # Hypernet 32x32 w/o hypernet init w Tinfo in Gobs
        # [
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-02_10-45-16_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-42",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-02_11-33-13_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-37",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-02_12-21-07_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-456",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-02_13-09-07_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-789",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-02_13-56-53_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-1337",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-02_14-45-20_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-2048",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-02_15-33-49_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-3141",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-02_16-22-20_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-5678",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-02_17-10-48_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-9999",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-02_17-58-54_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-1111",
        # ],

        # # Hypernet 512x256x128x64x32 w Tinfo in Gobs
        # [
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-02_23-23-48_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-42",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-03_00-14-14_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-37",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-03_01-05-00_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-456",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-03_01-55-19_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-789",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-03_02-45-59_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-1337",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-03_03-36-49_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-2048",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-03_04-27-50_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-3141",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-03_07-00-42_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-1111",
        # ],

        # # Hypernet 32 w Tinfo in Gobs
        # [
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-03_13-57-54_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-42",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-03_14-45-36_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-37",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-03_15-33-14_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-456",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-03_16-20-57_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-789",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-03_17-08-46_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-1337",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-03_17-56-40_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-2048",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-03_18-44-20_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-3141",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-03_19-32-18_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-5678",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-03_20-19-54_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-9999",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-03_21-07-48_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-1111",
        # ],

        # MAX GATES 35
        # Hypernet Critic 32x32 Tinfo in Gobs
        # [
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-05_15-58-12_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-42",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-05_17-48-14_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-37",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-05_19-34-38_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-456", #NG
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-05_21-24-11_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-789",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-05_23-11-04_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-1337",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-06_00-57-08_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-2048",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-06_02-41-15_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-3141",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-06_04-31-22_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-5678", #NG
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-06_06-11-58_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-9999", #NG
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-06_08-04-24_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-1111", #NG
        # ],

        # MAX GATES 40
        # Hypernet 32x32 w Tinfo in Gobs
        # [
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-08_08-42-03_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-42",
        #     # "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-08_12-38-13_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-37",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-08_20-13-24_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-789",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-08_23-57-09_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-1337",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-09_03-51-29_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-2048",
        #     # "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-09_07-32-42_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-3141",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-09_11-20-15_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-5678",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-09_15-14-05_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-9999",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-09_19-10-52_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-1111",
        # ],

        # Baseline 32x32 Gobs w Tinf
        # [
        #     # "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-12_11-23-04_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-42",
        #     # "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-13_08-16-46_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-37",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-13_15-13-54_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-456",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-13_22-12-09_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-789",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-14_05-03-03_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-1337",
        #     # "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-14_12-12-19_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-2048",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-14_19-12-24_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-3141",
        #     # "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-15_02-08-00_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-5678",
        #     # "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-15_08-57-27_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-9999",
        #     # "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-15_15-48-19_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-1111",
        # ],

        # Hypernetwork 128x256x128x64x32 w/o Tinf in Gobs
        # [
        #     # "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-16_18-57-30_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-42",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-17_09-01-10_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-37",
        #     # "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-17_14-21-32_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-456",
        #     # "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-16_18-59-39_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-789",
        #     # "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-17_00-22-35_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-1337",
        #     # "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-16_19-07-27_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-2048",
        #     # "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-17_00-34-54_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-3141",
        #     # "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-16_19-13-31_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-5678",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-17_02-36-33_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-9999",
        #     # "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-17_09-22-19_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-1111",
        # ],

        # Hypernet 32x32 w/o Tinfo in Gobs
        # [
        #     # "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-17_22-19-27_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-42", #NG
        #     # "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-18_03-26-03_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-37", #SLOW
        #     # "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-18_11-48-12_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-456", #SLOW
        #     # "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-18_08-02-45_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-789", #SLOW
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-17_22-17-55_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-1337", #VG
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-18_03-33-00_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-2048", #VG no touch
        #     # "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-17_22-15-00_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-314", #NG
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-18_03-33-06_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-5678", #G
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-17_22-08-59_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-9999", #G
        #     # "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-18_04-47-00_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-1111", #SLOW
        # ],

        # Hypernet Critic L 32x32 w/o Tinfo in Gobs
        # [
        #     # "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-22_07-51-39_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-42",
        #     # # "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-22_16-43-30_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-37", #SLOW
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-22_23-24-43_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-456", #vG
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-22_07-58-45_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-789", # G
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-22_16-27-44_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-1337", #VG
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-22_08-11-21_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-3141", #G
        # ],

        ######################################################################################
        # VANILLA 2.0 Throttle Scale 120
        ######################################################################################
        # Hypernet Critic L 32x32 w/o Tinfo in Gobs (C 256 tanh)
        [
            "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-29_16-32-48_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-1",
            "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-30_17-09-52_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-2",
            "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-31_15-28-42_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-4",
            "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-08-31_22-09-24_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-5",
        ],
        # Baseline L 32x32 w Tinfo in Gobs  (C 256 tanh)
        [
            "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-09-01_12-23-52_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-1",
            "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-09-01_12-35-48_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-2",
            # "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-09-01_19-31-02_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-3",
            "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-09-01_12-45-50_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-4",
            "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-09-01_21-55-27_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-5",
        ],

    ]

    plot_cfg = {
        "title": "",
        "box_colors": [
            "#FF3D50",  # Red
            "#FFA034",  # Orange
            "#2FA1FF",  # Blue
            "#A734FF",  # Purple
            "#FFFF3D",  # Yellow
            "#4DFF3D",  # Green
            "#FF3DBB",  # Pink
            "#623652",  # Dark purple
            "#00CED1",  # Dark turquoise
            "#FF6347",  # Tomato
            "#32CD32",  # Lime green
            "#FF1493",  # Deep pink
            "#00BFFF",  # Deep sky blue
            "#FFD700",  # Gold
            "#8A2BE2",  # Blue violet
            "#FF4500",  # Orange red
            "#20B2AA",  # Light sea green
            "#DC143C",  # Crimson
            "#9370DB",  # Medium purple
            "#FF8C00",  # Dark orange
            "#40E0D0",  # Turquoise
            "#FF69B4",  # Hot pink
            "#7FFF00",  # Chartreuse
            "#FF00FF",  # Magenta
            "#1E90FF",  # Dodger blue
            "#FF7F50",  # Coral
            "#98FB98",  # Pale green
            "#DDA0DD",  # Plum
            "#F0E68C",  # Khaki
            "#87CEEB",  # Sky blue
        ],
        "runs_names": [
            "Hypernet Critic L Th 120 32x32 w-o Tinfo in Gobs",
            "Baseline Th 32x32 w Tinfo in Gobs",
            
            # "Hypernet 32x32 w Tinfo in Gobs",
            # "Baseline 32x32 w Tinfo in Gobs",
            # # "Hypernet 128x256x128x64x32 w-o Tinfo in Gobs",
            # # "Hypernet 32x32 w-o Tinfo in Gobs",
            # "Hypernet Critic L 32x32 w-o Tinfo in Gobs",
            


            # "Hypernetwork 32x32 w/o Tinf in Gobs",
            # "Hypernetwork 32x32 w Tinf in Gobs",
            # "Hypernet Critic 32x32 w Tinfo in Gobs",
            # "Baseline Gobs w/ Tinf",
            # "Baseline Gobs w/o Tinf",
            # "Hypernet 64x64x64 w Tinfo in Gobs",
            # "Hypernet 32x32 w/o hypernet init w Tinfo in Gobs",
            # "Hypernet 512x256x128x64x32 w Tinfo in Gobs",
            # "Hypernet 32 w Tinfo in Gobs",

            # "Expert TrackID 2 32,32",
            # "Expert 37",
            # "Expert 42",
            # "Memory 42",
            # "Memory -1",
            # "Baseline -1",
            # "Baseline 42",
            # "Expert TrackID 2 32,32",
            # "Baseline 32,32",
            # "Hypernetwork 32,32",
            # "Hypernetwork 64,64",
            # "Hypernetwork 64,32,64",
            # "Hypernetwork 32",
            # "Hypernetwork 512,128,64,32",
            # "Expert TrackID 1 32,32",
        ],
        "zoom_in": False,
    }

    save_plots_folder_path = "/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/rans/utils/racing_scripts_and_data/plots" # Specify the folder path where you want to save the plots
    if not os.path.exists(save_plots_folder_path):
        os.makedirs(save_plots_folder_path)

    dfs = {}
    trajectories_dfs = {}
    labels = {}
    list_of_tasks = []
    
    for group_idx, group in enumerate(list_of_folders):
        task_name = group[0].split("/")[-1].split("_")[4]
        robot_name = group[0].split("/")[-1].split("_")[5]
        # Extract seed from folder path (e.g., "seed-42" from the folder name)
        seed_info = group[0].split("/")[-1].split("_")[-1]  # Gets "seed-42"
        # Use the run name from plot_cfg if available, otherwise use group index
        run_name = plot_cfg["runs_names"][group_idx] if group_idx < len(plot_cfg["runs_names"]) else f"group-{group_idx}"
        group_key = f"{task_name}_group-{group_idx}_{run_name}"  # Creates "RaceGates_group-0_seed-42"
        dfs[group_key] = []
        trajectories_dfs[group_key] = []
        labels[group_key] = []
        list_of_tasks.append(task_name)
        
        for folder_path in group:
            try:
                # Load metrics CSV
                experiment_name = glob.glob(os.path.join(folder_path, "metrics", "*_metrics.csv"))[0]
                metrics_file_path = os.path.join(folder_path, "metrics", experiment_name)
                df = pd.read_csv(metrics_file_path)
                dfs[group_key].append(df)
                labels[group_key].append(experiment_name.split("/")[-1])

                # Load trajectories CSV
                trajectories_pattern = os.path.join(folder_path, "metrics", f"extracted_trajectories_{task_name}.csv")
                trajectories_files = glob.glob(trajectories_pattern)
                if trajectories_files:
                    trajectories_file_path = trajectories_files[0]
                    trajectories_df = pd.read_csv(trajectories_file_path)
                    trajectories_dfs[group_key].append(trajectories_df)
                else:
                    print(f"Warning: No trajectories file found for {folder_path}")
                    trajectories_dfs[group_key].append(pd.DataFrame())  # Empty DataFrame as fallback

                # Load env info
                env_info_file_path = os.path.join(folder_path, "metrics", "env_info.yaml")
                with open(env_info_file_path, 'r') as f:
                    env_info = yaml.safe_load(f)
                    
            except Exception as e:
                print(f"Error reading file in {folder_path}: {e}")
                exit(0)

    # Group by task
    task_dfs = {}
    task_trajectories_dfs = {}
    
    for task in list_of_tasks:
        task_dfs[task] = {}
        task_trajectories_dfs[task] = {}
        for key in dfs:
            if task in key:
                task_dfs[task][key] = dfs[key]
                task_trajectories_dfs[task][key] = trajectories_dfs[key]

    # Create plots for each task
    for task_name in task_dfs:
        task_plots_factory = TaskPlotsFactory.create(
            task_name, 
            dfs=task_dfs[task_name],
            trajectories_dfs=task_trajectories_dfs[task_name],
            labels=labels,
            env_info=env_info,
            folder_path=save_plots_folder_path,
            plot_cfg=plot_cfg,
        )
        task_plots_factory.plot()
        
        robot_plots_factory = RobotPlotsFactory.create(
            robot_name, 
            dfs=dfs,
            trajectories_dfs=trajectories_dfs,
            labels=labels,
            env_info=env_info,
            folder_path=folder_path,
            plot_cfg=plot_cfg,
        )
        robot_plots_factory.plot()

if __name__ == "__main__":
    main()
