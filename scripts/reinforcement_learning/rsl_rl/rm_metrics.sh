#!/bin/bash
PATHS=(
/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_09-27-03_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1
/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_10-20-58_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2
/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_11-15-40_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3
/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_12-10-00_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4
/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_13-05-04_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5

/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_18-31-29_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1
/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_19-30-27_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2
/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_20-28-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3
/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_21-27-06_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4
/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_22-25-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5

/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_16-39-04_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-1
/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_17-08-03_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-2
/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_17-37-20_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-3
/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_18-06-26_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-4
/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_18-35-25_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-5
/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_19-03-53_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-1
/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_19-30-16_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-2
/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_19-56-54_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-3
/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_20-23-05_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-4
/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_20-49-18_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-5
/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_21-15-35_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-1
/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_21-42-12_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-2
/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_22-08-52_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-3
/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_22-35-35_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-4
/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_23-01-53_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-5
/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_23-27-59_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-1
/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_23-55-49_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-2
/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-09-01_00-23-46_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-3
/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-09-01_00-51-20_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-4
/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-09-01_01-18-58_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-5
)


# Remove metrics folder
for path in "${PATHS[@]}"
do
    rm -rf "$path/metrics"
done