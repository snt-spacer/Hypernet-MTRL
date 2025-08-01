
#!/bin/bash

# Common evaluation parameters
SCRIPT_PATH="./scripts/reinforcement_learning/rsl_rl/eval.py"
NUM_ENVS=4096

# Common arguments that apply to all evaluations (without task, as it will be extracted from checkpoint)
COMMON_ARGS="--task=Isaac-RANS-Single-v0 --headless --num_envs=${NUM_ENVS}"

# Array of checkpoint paths (relative to BASE_LOG_DIR)

# Experts
# CHECKPOINTS=(
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_13-40-53_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-1/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_14-05-17_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-2/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_14-29-44_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-3/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_14-54-10_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-4/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_15-18-28_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-5/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_15-42-48_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-6/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_16-07-01_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-7/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_16-31-37_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-8/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_16-55-54_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-9/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_17-19-36_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-10/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_17-43-52_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-1/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_18-08-31_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-2/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_18-32-31_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-3/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_18-56-59_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-4/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_19-21-06_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-5/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_19-45-24_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-6/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_20-10-29_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-7/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_20-34-43_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-8/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_20-59-12_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-9/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_21-24-22_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-10/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_21-49-01_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-1/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_22-17-14_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-2/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_22-44-58_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-3/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_23-13-10_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-4/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_23-41-48_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-5/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_00-10-01_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-6/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_00-38-23_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-7/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_01-06-25_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-8/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_01-34-14_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-9/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_02-02-39_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-10/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_02-31-28_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-1/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_02-57-35_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-2/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_03-24-24_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-3/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_03-50-36_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-4/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_04-17-17_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-5/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_04-44-17_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-6/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_05-11-15_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-7/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_05-38-04_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-8/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_06-04-38_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-9/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_06-31-49_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-10/model_3999.pt
# )



# Function to extract task from checkpoint path
extract_task_from_checkpoint() {
    local checkpoint_path="$1"
    local basename=$(basename "${checkpoint_path}")
    local dirname=$(dirname "${checkpoint_path}")
    local parent_dirname=$(basename "${dirname}")
    
    # Extract task from the parent directory name
    # More flexible pattern that looks for task names between underscores
    # Matches any of the known task names regardless of algorithm or robot
    if [[ "${parent_dirname}" =~ _(GoToPosition|GoToPose|TrackVelocities|Rendezvous)_ ]]; then
        local task_name="${BASH_REMATCH[1]}"
        echo "${task_name}"
    else
        echo "Orbital Not exist"
    fi
}

# Function to run evaluation for a single checkpoint
run_evaluation() {
    local checkpoint_path="$1"
    local task=$(extract_task_from_checkpoint "${checkpoint_path}")
    
    echo "Running evaluation for checkpoint: ${checkpoint_path}"
    echo "Extracted task: ${task}"
    ./isaaclab.sh -p ${SCRIPT_PATH} ${COMMON_ARGS} env.task_name=${task} env.robot_name=ModularFreeflyer --checkpoint=${checkpoint_path}

    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "✓ Evaluation completed successfully for: ${checkpoint_path}"
    else
        echo "✗ Evaluation failed for: ${checkpoint_path}"
    fi
    echo "----------------------------------------"
}

# Main execution: iterate through all checkpoints
echo "Starting control evaluation for ${#CHECKPOINTS[@]} checkpoints..."
echo "Number of environments: ${NUM_ENVS}"
echo "========================================"

for checkpoint in "${CHECKPOINTS[@]}"; do
    run_evaluation "${checkpoint}"
done

echo "All evaluations completed!"