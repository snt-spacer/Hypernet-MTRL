

#!/bin/bash

# Common training parameters
SCRIPT_PATH="scripts/reinforcement_learning/rsl_rl/train_racing.py"
TASK="Isaac-RANS-Single-v0"
ROBOT_NAME="Leatherback"
TASK_NAME="RaceGates"
ALGORITHM="ppo-memory"
TYPE_OF_TRAINING="hyper"  # Options: "hyper" or "padd"
FIXED_TRACK_ID=-1  # -1 means random track
SAME_TRACK_FOR_ALL_ENVS= False  # If True, all environments will use the same


# Common arguments that apply to all training runs
COMMON_ARGS="--task=${TASK} --headless env.robot_name=${ROBOT_NAME} env.task_name=${TASK_NAME} --algorithm=${ALGORITHM} --type_of_training=${TYPE_OF_TRAINING} --fixed_track_id=${FIXED_TRACK_ID} --same_track_for_all_envs=${SAME_TRACK_FOR_ALL_ENVS}"

# Array of seeds for training (10 seeds)
SEEDS=(42 37 456 789 1337 2048 3141 5678 9999 1111)
# SEEDS=(42)

# Function to run training for a single seed
run_training() {
    local seed="$1"
    
    echo "Starting training with seed: ${seed}"
    echo "Command: ./isaaclab.sh -p ${SCRIPT_PATH} ${COMMON_ARGS} --seed=${seed}"
    
    ./isaaclab.sh -p ${SCRIPT_PATH} ${COMMON_ARGS} --seed=${seed}
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "✓ Training completed successfully for seed: ${seed}"
    else
        echo "✗ Training failed for seed: ${seed}"
    fi
    echo "========================================"
}

# Main execution: iterate through all seeds
echo "Starting racing training for ${#SEEDS[@]} different seeds..."
echo "Task: ${TASK}"
echo "Robot: ${ROBOT_NAME}"
echo "Task Name: ${TASK_NAME}"
echo "Algorithm: ${ALGORITHM}"
echo "Seeds: ${SEEDS[*]}"
echo "========================================"

for seed in "${SEEDS[@]}"; do
    run_training "${seed}"
done

echo "All training runs completed!"
echo "Seeds used: ${SEEDS[*]}"