

#!/bin/bash

# Common training parameters
SCRIPT_PATH="scripts/reinforcement_learning/rsl_rl/train_racing.py"
TASK="Isaac-RANS-Single-v0"
ROBOT_NAME="Leatherback"
TASK_NAME="RaceGates"
ALGORITHM="ppo"
TYPE_OF_TRAINING="padd"  # Options: "hyper" or "padd"
FIXED_TRACK_ID=0 # 0 is custom track, -1 for random
CUSTOM_TRACK_ID=2 # 0 BCN, 1 JPN, 2 Monza, 10 points, 4 points, 11 alphapilot
SAME_TRACK_FOR_ALL_ENVS="True"  # If True, all environments will use the same
SEED=1


# Common arguments that apply to all training runs
COMMON_ARGS="--task=${TASK} --headless env.robot_name=${ROBOT_NAME} env.task_name=${TASK_NAME} --algorithm=${ALGORITHM} --type_of_training=${TYPE_OF_TRAINING} --fixed_track_id=${FIXED_TRACK_ID} --same_track_for_all_envs=${SAME_TRACK_FOR_ALL_ENVS} --custom_track_id=${CUSTOM_TRACK_ID} --seed=${SEED}"


CHECKPOINTS=(
2025-09-01_12-23-52_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-1/model_2000.pt
2025-09-01_12-35-48_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-2/model_3700.pt
2025-09-01_12-45-50_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-4/model_3500.pt
2025-09-01_21-55-27_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-5/model_3200.pt
)
# Modify config

modify_cfg() {
    local checkpoint_path="$1"
    ./isaaclab.sh -p "scripts/reinforcement_learning/rsl_rl/racing_modify_cfg_finetune.py" --checkpoint_path="${checkpoint_path}"
}

# Function to run training for a single seed
run_training() {
    
    echo "Starting finetuning"
    echo "Command: ./isaaclab.sh -p ${SCRIPT_PATH} ${COMMON_ARGS}"
    
    ./isaaclab.sh -p ${SCRIPT_PATH} ${COMMON_ARGS}
    
    # Check if the command was successful
    # if [ $? -eq 0 ]; then
    #     echo "✓ Training completed successfully for seed: ${seed}"
    # else
    #     echo "✗ Training failed for seed: ${seed}"
    # fi
    echo "========================================"
}

# Main execution: iterate through all seeds
echo "Starting racing finetuning"
echo "Task: ${TASK}"
echo "Robot: ${ROBOT_NAME}"
echo "Task Name: ${TASK_NAME}"
echo "Algorithm: ${ALGORITHM}"
echo "Track ID: ${CUSTOM_TRACK_ID}"
echo "========================================"
for checkpoint in "${CHECKPOINTS[@]}"; do
    modify_cfg "${checkpoint}"
    run_training
done
run_training

echo "All training runs completed!"
echo "Seeds used: ${SEEDS[*]}"