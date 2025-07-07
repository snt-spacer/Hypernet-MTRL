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
        # [
        #     "/workspace/isaaclab/multitask_baseline-models-dart/logs/rsl_rl/2025-07-01_05-03-13_rsl-rl_ppo_GoToPosition_FloatingPlatform_r-0_seed-1",
        #     "/workspace/isaaclab/multitask_baseline-models-dart/logs/rsl_rl/2025-07-01_05-09-51_rsl-rl_ppo_GoToPosition_FloatingPlatform_r-0_seed-2",
        #     "/workspace/isaaclab/multitask_baseline-models-dart/logs/rsl_rl/2025-07-01_05-16-31_rsl-rl_ppo_GoToPosition_FloatingPlatform_r-0_seed-3",
        #     "/workspace/isaaclab/multitask_baseline-models-dart/logs/rsl_rl/2025-07-01_05-23-12_rsl-rl_ppo_GoToPosition_FloatingPlatform_r-0_seed-4",
        #     "/workspace/isaaclab/multitask_baseline-models-dart/logs/rsl_rl/2025-07-01_05-29-50_rsl-rl_ppo_GoToPosition_FloatingPlatform_r-0_seed-5",
        #     "/workspace/isaaclab/multitask_baseline-models-dart/logs/rsl_rl/2025-07-01_05-36-30_rsl-rl_ppo_GoToPosition_FloatingPlatform_r-0_seed-6",
        #     "/workspace/isaaclab/multitask_baseline-models-dart/logs/rsl_rl/2025-07-01_05-43-09_rsl-rl_ppo_GoToPosition_FloatingPlatform_r-0_seed-7",
        #     "/workspace/isaaclab/multitask_baseline-models-dart/logs/rsl_rl/2025-07-01_05-49-50_rsl-rl_ppo_GoToPosition_FloatingPlatform_r-0_seed-8",
        #     "/workspace/isaaclab/multitask_baseline-models-dart/logs/rsl_rl/2025-07-01_05-56-28_rsl-rl_ppo_GoToPosition_FloatingPlatform_r-0_seed-9",
        #     "/workspace/isaaclab/multitask_baseline-models-dart/logs/rsl_rl/2025-07-01_06-03-09_rsl-rl_ppo_GoToPosition_FloatingPlatform_r-0_seed-10",
        # ],
        # [
        #     "/workspace/isaaclab/multitask_baseline-models-dart/logs/rsl_rl/2025-07-01_06-09-48_rsl-rl_ppo_GoToPose_FloatingPlatform_r-0_seed-1",
        #     "/workspace/isaaclab/multitask_baseline-models-dart/logs/rsl_rl/2025-07-01_06-16-32_rsl-rl_ppo_GoToPose_FloatingPlatform_r-0_seed-2",
        #     "/workspace/isaaclab/multitask_baseline-models-dart/logs/rsl_rl/2025-07-01_06-23-14_rsl-rl_ppo_GoToPose_FloatingPlatform_r-0_seed-3",
        #     "/workspace/isaaclab/multitask_baseline-models-dart/logs/rsl_rl/2025-07-01_06-29-57_rsl-rl_ppo_GoToPose_FloatingPlatform_r-0_seed-4",
        #     "/workspace/isaaclab/multitask_baseline-models-dart/logs/rsl_rl/2025-07-01_06-36-38_rsl-rl_ppo_GoToPose_FloatingPlatform_r-0_seed-5",
        #     "/workspace/isaaclab/multitask_baseline-models-dart/logs/rsl_rl/2025-07-01_06-43-18_rsl-rl_ppo_GoToPose_FloatingPlatform_r-0_seed-6",
        #     "/workspace/isaaclab/multitask_baseline-models-dart/logs/rsl_rl/2025-07-01_06-50-01_rsl-rl_ppo_GoToPose_FloatingPlatform_r-0_seed-7",
        #     "/workspace/isaaclab/multitask_baseline-models-dart/logs/rsl_rl/2025-07-01_06-56-47_rsl-rl_ppo_GoToPose_FloatingPlatform_r-0_seed-8",
        #     "/workspace/isaaclab/multitask_baseline-models-dart/logs/rsl_rl/2025-07-01_07-03-27_rsl-rl_ppo_GoToPose_FloatingPlatform_r-0_seed-9",
        #     "/workspace/isaaclab/multitask_baseline-models-dart/logs/rsl_rl/2025-07-01_07-10-09_rsl-rl_ppo_GoToPose_FloatingPlatform_r-0_seed-10",
        # ],
        # [
        #     "/workspace/isaaclab/multitask_baseline-models-dart/logs/rsl_rl/2025-07-01_07-16-52_rsl-rl_ppo_TrackVelocities_FloatingPlatform_r-0_seed-1",
        #     "/workspace/isaaclab/multitask_baseline-models-dart/logs/rsl_rl/2025-07-01_07-24-23_rsl-rl_ppo_TrackVelocities_FloatingPlatform_r-0_seed-2",
        #     "/workspace/isaaclab/multitask_baseline-models-dart/logs/rsl_rl/2025-07-01_07-31-58_rsl-rl_ppo_TrackVelocities_FloatingPlatform_r-0_seed-3",
        #     "/workspace/isaaclab/multitask_baseline-models-dart/logs/rsl_rl/2025-07-01_07-39-33_rsl-rl_ppo_TrackVelocities_FloatingPlatform_r-0_seed-4",
        #     "/workspace/isaaclab/multitask_baseline-models-dart/logs/rsl_rl/2025-07-01_07-47-06_rsl-rl_ppo_TrackVelocities_FloatingPlatform_r-0_seed-5",
        #     "/workspace/isaaclab/multitask_baseline-models-dart/logs/rsl_rl/2025-07-01_07-54-38_rsl-rl_ppo_TrackVelocities_FloatingPlatform_r-0_seed-6",
        #     "/workspace/isaaclab/multitask_baseline-models-dart/logs/rsl_rl/2025-07-01_08-02-13_rsl-rl_ppo_TrackVelocities_FloatingPlatform_r-0_seed-7",
        #     "/workspace/isaaclab/multitask_baseline-models-dart/logs/rsl_rl/2025-07-01_08-10-00_rsl-rl_ppo_TrackVelocities_FloatingPlatform_r-0_seed-8",
        #     "/workspace/isaaclab/multitask_baseline-models-dart/logs/rsl_rl/2025-07-01_08-17-33_rsl-rl_ppo_TrackVelocities_FloatingPlatform_r-0_seed-9",
        #     "/workspace/isaaclab/multitask_baseline-models-dart/logs/rsl_rl/2025-07-01_08-25-05_rsl-rl_ppo_TrackVelocities_FloatingPlatform_r-0_seed-10",
        # ],
        # [
        #     "/workspace/isaaclab/multitask_baseline-models-dart/logs/rsl_rl/2025-07-01_08-32-39_rsl-rl_ppo_GoThroughPoses_FloatingPlatform_r-0_seed-1",
        #     "/workspace/isaaclab/multitask_baseline-models-dart/logs/rsl_rl/2025-07-01_08-40-52_rsl-rl_ppo_GoThroughPoses_FloatingPlatform_r-0_seed-2",
        #     "/workspace/isaaclab/multitask_baseline-models-dart/logs/rsl_rl/2025-07-01_08-49-01_rsl-rl_ppo_GoThroughPoses_FloatingPlatform_r-0_seed-3",
        #     "/workspace/isaaclab/multitask_baseline-models-dart/logs/rsl_rl/2025-07-01_08-57-08_rsl-rl_ppo_GoThroughPoses_FloatingPlatform_r-0_seed-4",
        #     "/workspace/isaaclab/multitask_baseline-models-dart/logs/rsl_rl/2025-07-01_09-05-17_rsl-rl_ppo_GoThroughPoses_FloatingPlatform_r-0_seed-5",
        #     "/workspace/isaaclab/multitask_baseline-models-dart/logs/rsl_rl/2025-07-01_09-13-30_rsl-rl_ppo_GoThroughPoses_FloatingPlatform_r-0_seed-6",
        #     "/workspace/isaaclab/multitask_baseline-models-dart/logs/rsl_rl/2025-07-01_09-21-39_rsl-rl_ppo_GoThroughPoses_FloatingPlatform_r-0_seed-7",
        #     "/workspace/isaaclab/multitask_baseline-models-dart/logs/rsl_rl/2025-07-01_09-29-50_rsl-rl_ppo_GoThroughPoses_FloatingPlatform_r-0_seed-8",
        #     "/workspace/isaaclab/multitask_baseline-models-dart/logs/rsl_rl/2025-07-01_09-37-58_rsl-rl_ppo_GoThroughPoses_FloatingPlatform_r-0_seed-9",
        #     "/workspace/isaaclab/multitask_baseline-models-dart/logs/rsl_rl/2025-07-01_09-46-06_rsl-rl_ppo_GoThroughPoses_FloatingPlatform_r-0_seed-10",
        # ],
        # [
        #     "/workspace/isaaclab/multitask_baseline-models-dart/logs/rsl_rl/2025-07-01_09-54-17_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-1",
        #     "/workspace/isaaclab/multitask_baseline-models-dart/logs/rsl_rl/2025-07-01_10-19-13_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-2",
        #     "/workspace/isaaclab/multitask_baseline-models-dart/logs/rsl_rl/2025-07-01_10-44-17_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-3",
        #     "/workspace/isaaclab/multitask_baseline-models-dart/logs/rsl_rl/2025-07-01_11-09-10_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-4",
        #     "/workspace/isaaclab/multitask_baseline-models-dart/logs/rsl_rl/2025-07-01_11-34-12_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-5",
        #     "/workspace/isaaclab/multitask_baseline-models-dart/logs/rsl_rl/2025-07-01_11-59-13_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-6",
        #     "/workspace/isaaclab/multitask_baseline-models-dart/logs/rsl_rl/2025-07-01_12-24-11_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-7",
        #     "/workspace/isaaclab/multitask_baseline-models-dart/logs/rsl_rl/2025-07-01_12-49-09_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-8",
        #     "/workspace/isaaclab/multitask_baseline-models-dart/logs/rsl_rl/2025-07-01_13-14-05_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-9",
        #     "/workspace/isaaclab/multitask_baseline-models-dart/logs/rsl_rl/2025-07-01_13-39-04_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-10",
        # ],
        # # 64
        # [
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-15_13-22-50_rsl-rl_ppo_GoToPosition_FloatingPlatform_r-0_seed-1",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-15_13-49-37_rsl-rl_ppo_GoToPosition_FloatingPlatform_r-0_seed-2",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-15_14-16-31_rsl-rl_ppo_GoToPosition_FloatingPlatform_r-0_seed-3",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-15_14-43-22_rsl-rl_ppo_GoToPosition_FloatingPlatform_r-0_seed-4",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-15_15-10-29_rsl-rl_ppo_GoToPosition_FloatingPlatform_r-0_seed-5",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-15_15-37-28_rsl-rl_ppo_GoToPosition_FloatingPlatform_r-0_seed-6",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-15_16-04-34_rsl-rl_ppo_GoToPosition_FloatingPlatform_r-0_seed-7",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-15_16-31-28_rsl-rl_ppo_GoToPosition_FloatingPlatform_r-0_seed-8",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-15_16-58-09_rsl-rl_ppo_GoToPosition_FloatingPlatform_r-0_seed-9",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-15_17-25-25_rsl-rl_ppo_GoToPosition_FloatingPlatform_r-0_seed-10",
        # ],
        # [
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-15_17-52-20_rsl-rl_ppo_GoToPose_FloatingPlatform_r-0_seed-1",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-15_18-19-28_rsl-rl_ppo_GoToPose_FloatingPlatform_r-0_seed-2",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-15_18-46-56_rsl-rl_ppo_GoToPose_FloatingPlatform_r-0_seed-3",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-15_19-13-37_rsl-rl_ppo_GoToPose_FloatingPlatform_r-0_seed-4",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-15_19-40-36_rsl-rl_ppo_GoToPose_FloatingPlatform_r-0_seed-5",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-15_20-07-46_rsl-rl_ppo_GoToPose_FloatingPlatform_r-0_seed-6",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-15_20-34-27_rsl-rl_ppo_GoToPose_FloatingPlatform_r-0_seed-7",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-15_21-01-18_rsl-rl_ppo_GoToPose_FloatingPlatform_r-0_seed-8",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-15_21-28-13_rsl-rl_ppo_GoToPose_FloatingPlatform_r-0_seed-9",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-15_21-55-23_rsl-rl_ppo_GoToPose_FloatingPlatform_r-0_seed-10",
        # ],
        # [
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-15_22-22-22_rsl-rl_ppo_TrackVelocities_FloatingPlatform_r-0_seed-1",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-15_22-51-35_rsl-rl_ppo_TrackVelocities_FloatingPlatform_r-0_seed-2",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-15_23-19-54_rsl-rl_ppo_TrackVelocities_FloatingPlatform_r-0_seed-3",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-15_23-48-12_rsl-rl_ppo_TrackVelocities_FloatingPlatform_r-0_seed-4",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-16_00-16-39_rsl-rl_ppo_TrackVelocities_FloatingPlatform_r-0_seed-5",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-16_00-44-56_rsl-rl_ppo_TrackVelocities_FloatingPlatform_r-0_seed-6",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-16_01-13-09_rsl-rl_ppo_TrackVelocities_FloatingPlatform_r-0_seed-7",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-16_01-41-38_rsl-rl_ppo_TrackVelocities_FloatingPlatform_r-0_seed-8",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-16_02-09-53_rsl-rl_ppo_TrackVelocities_FloatingPlatform_r-0_seed-9",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-16_02-37-47_rsl-rl_ppo_TrackVelocities_FloatingPlatform_r-0_seed-10",
        # ],
        # [
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-16_03-06-01_rsl-rl_ppo_GoThroughPoses_FloatingPlatform_r-0_seed-1",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-16_03-40-42_rsl-rl_ppo_GoThroughPoses_FloatingPlatform_r-0_seed-2",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-16_04-15-11_rsl-rl_ppo_GoThroughPoses_FloatingPlatform_r-0_seed-3",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-16_04-49-30_rsl-rl_ppo_GoThroughPoses_FloatingPlatform_r-0_seed-4",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-16_05-24-18_rsl-rl_ppo_GoThroughPoses_FloatingPlatform_r-0_seed-5",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-16_05-58-54_rsl-rl_ppo_GoThroughPoses_FloatingPlatform_r-0_seed-6",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-16_06-33-28_rsl-rl_ppo_GoThroughPoses_FloatingPlatform_r-0_seed-7",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-16_07-07-46_rsl-rl_ppo_GoThroughPoses_FloatingPlatform_r-0_seed-8",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-16_07-42-11_rsl-rl_ppo_GoThroughPoses_FloatingPlatform_r-0_seed-9",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-16_08-16-36_rsl-rl_ppo_GoThroughPoses_FloatingPlatform_r-0_seed-10",
        # ],
        # [
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-16_08-51-11_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-1",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-16_10-01-56_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-2",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-16_11-11-50_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-3",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-16_12-22-08_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-4",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-16_13-33-22_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-5",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-16_14-43-28_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-6",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-16_15-54-41_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-7",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-16_17-05-14_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-8",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-16_18-16-00_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-9",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-16_19-27-00_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-10",
        # ],
        # # 64x128
        # [
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64m128_10xseeds/2025-06-13_08-00-58_rsl-rl_ppo_GoToPosition_FloatingPlatform_r-0_seed-1",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64m128_10xseeds/2025-06-13_08-28-20_rsl-rl_ppo_GoToPosition_FloatingPlatform_r-0_seed-2",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64m128_10xseeds/2025-06-13_08-55-45_rsl-rl_ppo_GoToPosition_FloatingPlatform_r-0_seed-3",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64m128_10xseeds/2025-06-13_09-22-58_rsl-rl_ppo_GoToPosition_FloatingPlatform_r-0_seed-4",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64m128_10xseeds/2025-06-13_09-50-21_rsl-rl_ppo_GoToPosition_FloatingPlatform_r-0_seed-5",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64m128_10xseeds/2025-06-13_10-18-21_rsl-rl_ppo_GoToPosition_FloatingPlatform_r-0_seed-6",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64m128_10xseeds/2025-06-13_10-45-39_rsl-rl_ppo_GoToPosition_FloatingPlatform_r-0_seed-7",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64m128_10xseeds/2025-06-13_11-12-54_rsl-rl_ppo_GoToPosition_FloatingPlatform_r-0_seed-8",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64m128_10xseeds/2025-06-13_11-40-00_rsl-rl_ppo_GoToPosition_FloatingPlatform_r-0_seed-9",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64m128_10xseeds/2025-06-13_12-07-29_rsl-rl_ppo_GoToPosition_FloatingPlatform_r-0_seed-10",
        # ],
        # [
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64m128_10xseeds/2025-06-13_12-34-32_rsl-rl_ppo_GoToPose_FloatingPlatform_r-0_seed-1",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64m128_10xseeds/2025-06-13_13-02-04_rsl-rl_ppo_GoToPose_FloatingPlatform_r-0_seed-2",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64m128_10xseeds/2025-06-13_13-29-21_rsl-rl_ppo_GoToPose_FloatingPlatform_r-0_seed-3",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64m128_10xseeds/2025-06-13_13-56-35_rsl-rl_ppo_GoToPose_FloatingPlatform_r-0_seed-4",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64m128_10xseeds/2025-06-13_14-24-09_rsl-rl_ppo_GoToPose_FloatingPlatform_r-0_seed-5",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64m128_10xseeds/2025-06-13_14-51-54_rsl-rl_ppo_GoToPose_FloatingPlatform_r-0_seed-6",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64m128_10xseeds/2025-06-13_15-19-35_rsl-rl_ppo_GoToPose_FloatingPlatform_r-0_seed-7",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64m128_10xseeds/2025-06-13_15-47-26_rsl-rl_ppo_GoToPose_FloatingPlatform_r-0_seed-8",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64m128_10xseeds/2025-06-13_16-14-37_rsl-rl_ppo_GoToPose_FloatingPlatform_r-0_seed-9",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64m128_10xseeds/2025-06-13_16-42-14_rsl-rl_ppo_GoToPose_FloatingPlatform_r-0_seed-10",
        # ],
        # [
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64m128_10xseeds/2025-06-13_17-09-48_rsl-rl_ppo_TrackVelocities_FloatingPlatform_r-0_seed-1",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64m128_10xseeds/2025-06-13_17-38-26_rsl-rl_ppo_TrackVelocities_FloatingPlatform_r-0_seed-2",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64m128_10xseeds/2025-06-13_18-07-09_rsl-rl_ppo_TrackVelocities_FloatingPlatform_r-0_seed-3",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64m128_10xseeds/2025-06-13_18-35-33_rsl-rl_ppo_TrackVelocities_FloatingPlatform_r-0_seed-4",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64m128_10xseeds/2025-06-13_19-04-05_rsl-rl_ppo_TrackVelocities_FloatingPlatform_r-0_seed-5",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64m128_10xseeds/2025-06-13_19-32-58_rsl-rl_ppo_TrackVelocities_FloatingPlatform_r-0_seed-6",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64m128_10xseeds/2025-06-13_20-01-36_rsl-rl_ppo_TrackVelocities_FloatingPlatform_r-0_seed-7",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64m128_10xseeds/2025-06-13_20-30-15_rsl-rl_ppo_TrackVelocities_FloatingPlatform_r-0_seed-8",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64m128_10xseeds/2025-06-13_20-58-42_rsl-rl_ppo_TrackVelocities_FloatingPlatform_r-0_seed-9",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64m128_10xseeds/2025-06-13_21-27-39_rsl-rl_ppo_TrackVelocities_FloatingPlatform_r-0_seed-10",
        # ],
        # [
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64m128_10xseeds/2025-06-13_21-56-21_rsl-rl_ppo_GoThroughPoses_FloatingPlatform_r-0_seed-1",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64m128_10xseeds/2025-06-13_22-31-26_rsl-rl_ppo_GoThroughPoses_FloatingPlatform_r-0_seed-2",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64m128_10xseeds/2025-06-13_23-06-29_rsl-rl_ppo_GoThroughPoses_FloatingPlatform_r-0_seed-3",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64m128_10xseeds/2025-06-13_23-41-04_rsl-rl_ppo_GoThroughPoses_FloatingPlatform_r-0_seed-4",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64m128_10xseeds/2025-06-14_00-16-00_rsl-rl_ppo_GoThroughPoses_FloatingPlatform_r-0_seed-5",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64m128_10xseeds/2025-06-14_00-51-11_rsl-rl_ppo_GoThroughPoses_FloatingPlatform_r-0_seed-6",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64m128_10xseeds/2025-06-14_01-26-07_rsl-rl_ppo_GoThroughPoses_FloatingPlatform_r-0_seed-7",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64m128_10xseeds/2025-06-14_02-01-13_rsl-rl_ppo_GoThroughPoses_FloatingPlatform_r-0_seed-8",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64m128_10xseeds/2025-06-14_02-36-20_rsl-rl_ppo_GoThroughPoses_FloatingPlatform_r-0_seed-9",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64m128_10xseeds/2025-06-14_03-11-03_rsl-rl_ppo_GoThroughPoses_FloatingPlatform_r-0_seed-10",
        # ],
        # [
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64m128_10xseeds/2025-06-14_03-45-28_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-1",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64m128_10xseeds/2025-06-14_04-57-04_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-2",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64m128_10xseeds/2025-06-14_06-06-57_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-3",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64m128_10xseeds/2025-06-14_07-19-10_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-4",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64m128_10xseeds/2025-06-14_08-30-13_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-5",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64m128_10xseeds/2025-06-14_09-41-54_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-6",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64m128_10xseeds/2025-06-14_10-52-28_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-7",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64m128_10xseeds/2025-06-14_12-02-44_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-8",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64m128_10xseeds/2025-06-14_13-14-11_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-9",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64m128_10xseeds/2025-06-14_14-25-47_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-10",
        # ],

        #Memory
        # [
        #     # "/workspace/isaaclab/logs/rsl_rl/multitask_memory_new-eval/2025-06-29_19-31-21_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-1",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_memory_new-eval/2025-06-29_20-57-10_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-2",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_memory_new-eval/2025-06-29_22-18-33_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-3",
        #     # "/workspace/isaaclab/logs/rsl_rl/multitask_memory_new-eval/2025-06-29_23-40-03_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-4",
        #     # "/workspace/isaaclab/logs/rsl_rl/multitask_memory_new-eval/2025-06-30_01-01-29_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-5",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_memory_new-eval/2025-06-30_02-23-58_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-6",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_memory_new-eval/2025-06-30_03-45-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-7",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_memory_new-eval/2025-06-30_05-07-19_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-8",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_memory_new-eval/2025-06-30_06-28-35_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-9",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_memory_new-eval/2025-06-30_07-51-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-10",
        # ],

        # Memory normalized W
        # [
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_memory_normW/2025-07-01_07-31-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-1",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_memory_normW/2025-07-01_08-01-08_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-2",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_memory_normW/2025-07-01_08-30-17_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-3",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_memory_normW/2025-07-01_08-59-39_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-4",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_memory_normW/2025-07-01_09-28-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-5",
        #     # "/workspace/isaaclab/logs/rsl_rl/multitask_memory_normW/2025-07-01_09-57-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-6",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_memory_normW/2025-07-01_10-27-07_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-7",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_memory_normW/2025-07-01_10-56-05_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-8",
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_memory_normW/2025-07-01_11-25-14_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-9",
        # ],
        # Norm W hyperparams
        # [
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_memory_normW_hyperparams/2025-07-04_08-16-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-42",
        # ],
        # [
        #     "/workspace/isaaclab/logs/rsl_rl/multitask_memory_normW_hyperparams/2025-07-04_08-56-06_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-0",
        # ],
        # Rendezvous
        [
            "/workspace/isaaclab/logs/rsl_rl/multitask_memory_normW_hyperparams_rendezvous/2025-07-05_23-43-33_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_FloatingPlatform_r-0_seed-1",
        ],
        [
            "/workspace/isaaclab/logs/rsl_rl/multitask_memory_normW_hyperparams_rendezvous/2025-07-06_00-18-06_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_FloatingPlatform_r-0_seed-2",
        ],
        [
            "/workspace/isaaclab/logs/rsl_rl/multitask_memory_normW_hyperparams_rendezvous/2025-07-06_00-52-28_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_FloatingPlatform_r-0_seed-3",
        ],
        [
            "/workspace/isaaclab/logs/rsl_rl/multitask_memory_normW_hyperparams_rendezvous/2025-07-06_01-27-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_FloatingPlatform_r-0_seed-4",
        ],
        [
            "/workspace/isaaclab/logs/rsl_rl/multitask_memory_normW_hyperparams_rendezvous/2025-07-06_02-01-19_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_FloatingPlatform_r-0_seed-5",
        ],
        [
            "/workspace/isaaclab/logs/rsl_rl/multitask_memory_normW_hyperparams_rendezvous/2025-07-06_02-36-21_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_FloatingPlatform_r-0_seed-6",
        ],
        [
            "/workspace/isaaclab/logs/rsl_rl/multitask_memory_normW_hyperparams_rendezvous/2025-07-06_03-11-05_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_FloatingPlatform_r-0_seed-7",
        ],
        [
            "/workspace/isaaclab/logs/rsl_rl/multitask_memory_normW_hyperparams_rendezvous/2025-07-06_03-45-39_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_FloatingPlatform_r-0_seed-8",
        ],
        [
            "/workspace/isaaclab/logs/rsl_rl/multitask_memory_normW_hyperparams_rendezvous/2025-07-06_04-20-24_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_FloatingPlatform_r-0_seed-9",
        ],
        [
            "/workspace/isaaclab/logs/rsl_rl/multitask_memory_normW_hyperparams_rendezvous/2025-07-06_04-55-06_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_FloatingPlatform_r-0_seed-10",
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

            "#FF3DBB", 
            "#FF3DBB", 
            "#FF3DBB", 
            "#FF3DBB",
            "#623652", 
        ],
        "runs_names": [
            "Seed 0",
            "Seed 1",
            "Seed 2",
            "Seed 3",
            "Seed 4",
            "Seed 5",
            "Seed 6",
            "Seed 7",
            "Seed 8",
            "Seed 9",
            "Detumble 32x32",
            "Docking 32x32",
            "Track Velocities 32x32",
            "Rendezvous 32x32",
            "MTRL 32x32",
            # "Detumble 4x64",
            # "Docking 4x64",
            # "Track Velocities 4x64",
            # "Rendezvous 4x64",
            # "MTRL 4x64",
            # "Detumble 4x64m128",
            # "Docking 4x64m128",
            # "Track Velocities 4x64m128",
            # "Rendezvous 4x64m128",
            # "MTRL 4x64m128",
            "Memory 32x32",
            "Memory 32x32 normW",
            
        ],
        "zoom_in": False,
    }

    save_plots_folder_path = "/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/rans/utils/plots/test_trajects/plot_rendezvous" # Specify the folder path where you want to save the plots
    if not os.path.exists(save_plots_folder_path):
        os.makedirs(save_plots_folder_path)

    dfs = {}
    trajectories_dfs = {}
    labels = {}
    list_of_tasks = []
    for group_idx, group in enumerate(list_of_folders):
        tasks_names = group[0].split("/")[-1].split("_")[4].split("-")
        robot_name = group[0].split("/")[-1].split("_")[5]
        for task_idx, task_name in enumerate(tasks_names):
            if task_name not in list_of_tasks:
                list_of_tasks.append(task_name)
            run_name = plot_cfg["runs_names"][group_idx]
            gorup_key = f"{task_name}_group-{group_idx}_{run_name}"
            trajectories_key = f"{task_name}_trajectories_group-{group_idx}_{run_name}"
            dfs[gorup_key] = []
            labels[gorup_key] = []
            trajectories_dfs[trajectories_key] = []
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
                
                # Read the trajectories file if it exists
                try:
                    trajectories_path = os.path.join(folder_path, "metrics", f"extracted_trajectories_{task_name}.csv")
                    traj_df = pd.read_csv(trajectories_path)
                    trajectories_dfs[trajectories_key].append(traj_df)
                except Exception as e:
                    print(f"Error reading trajectories file in {trajectories_path}: {e}")
                
    
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
        task_group_trajectories_dfs = {
            key: trajectories_dfs[key] for key in trajectories_dfs if key.startswith(task_name)
        }

        print("-"*30)
        print(f"Taksk name: {task_name}, task_group_dfs: {task_group_dfs.keys()}, task_group_labels: {task_group_labels.keys()}")
        task_plots_factory = TaskPlotsFactory.create(
            task_name,
            dfs=task_group_dfs,
            trajectories_dfs=task_group_trajectories_dfs,
            labels=task_group_labels,
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
    