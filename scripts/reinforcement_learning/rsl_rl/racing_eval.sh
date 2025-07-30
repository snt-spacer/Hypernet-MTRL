
#!/bin/bash

# Common evaluation parameters
SCRIPT_PATH="./scripts/reinforcement_learning/rsl_rl/eval_racing.py"
TASK="Isaac-RANS-Single-v0"
NUM_ENVS=4096
TRACK_ID=2 # 1, 2, 8
BASE_LOG_DIR="logs/rsl_rl/multitask_racing_baseline_noTrackInfo"

# Common arguments that apply to all evaluations
COMMON_ARGS="--task=${TASK} --headless --num_envs=${NUM_ENVS} --track_id=${TRACK_ID}"

# Array of checkpoint paths (relative to BASE_LOG_DIR)
# CHECKPOINTS=(
#     "2025-07-13_12-34-20_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-42/model_999.pt"
#     "2025-07-13_12-23-18_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-42/model_999.pt"
#     "2025-07-13_12-10-01_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-42/model_999.pt"
#     "2025-07-13_11-56-31_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-42/model_1000.pt"
#     "2025-07-13_11-13-22_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-42/model_999.pt"
#     "2025-07-13_10-59-34_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-42/model_999.pt"
# )
# CHECKPOINTS=(
#     "2025-07-15_10-26-40_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-42/model_2300.pt"
#     "2025-07-15_11-18-58_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-42/model_2550.pt"
#     "2025-07-15_12-04-29_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-42/model_2150.pt"
#     "2025-07-15_12-49-51_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-42/model_1700.pt"
#     "2025-07-15_13-37-19_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-42/model_1450.pt"
#     "2025-07-15_14-42-48_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-42/model_2200.pt"
#     "2025-07-15_15-36-48_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-42/model_2500.pt"
#     "2025-07-15_16-54-39_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-42/model_2600.pt"
# )

# Hypernet general obs w/o track info
CHECKPOINTS=(
    "2025-07-28_11-29-34_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-42/model_3200.pt"
    "2025-07-28_12-11-16_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-37/model_3450.pt"
    "2025-07-29_10-52-34_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-456/model_3000.pt"
    "2025-07-30_12-34-08_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-789/model_2800.pt"
    "2025-07-30_13-16-15_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-1337/model_3150.pt"
    "2025-07-30_13-58-38_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-2048/model_2350.pt"
    "2025-07-30_14-40-44_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-3141/model_2050.pt"
    "2025-07-30_15-22-36_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-5678/model_3900.pt"
    "2025-07-30_16-05-09_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-9999/model_2600.pt"
    "2025-07-30_16-47-37_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-1111/model_2500.pt"
)

# Function to run evaluation for a single checkpoint
run_evaluation() {
    local checkpoint_path="$1"
    local full_checkpoint_path="${BASE_LOG_DIR}/${checkpoint_path}"
    
    echo "Running evaluation for checkpoint: ${checkpoint_path}"
    ./isaaclab.sh -p ${SCRIPT_PATH} ${COMMON_ARGS} --checkpoint=${full_checkpoint_path}
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "✓ Evaluation completed successfully for: ${checkpoint_path}"
    else
        echo "✗ Evaluation failed for: ${checkpoint_path}"
    fi
    echo "----------------------------------------"
}

# Main execution: iterate through all checkpoints
echo "Starting racing evaluation for ${#CHECKPOINTS[@]} checkpoints..."
echo "Task: ${TASK}"
echo "Track ID: ${TRACK_ID}"
echo "Number of environments: ${NUM_ENVS}"
echo "========================================"

for checkpoint in "${CHECKPOINTS[@]}"; do
    run_evaluation "${checkpoint}"
done

echo "All evaluations completed!"