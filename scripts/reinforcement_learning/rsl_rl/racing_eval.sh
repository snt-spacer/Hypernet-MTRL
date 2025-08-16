
#!/bin/bash

# Common evaluation parameters
SCRIPT_PATH="./scripts/reinforcement_learning/rsl_rl/eval_racing.py"
TASK="Isaac-RANS-Single-v0"
NUM_ENVS=4096
TRACK_ID=0 # 1, 2, 8, 0 BCN track
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
# CHECKPOINTS=(
#     "2025-07-28_11-29-34_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-42/model_3150.pt"
#     "2025-07-28_12-11-16_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-37/model_1450.pt"
#     "2025-07-29_10-52-34_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-456/model_3000.pt"
#     "2025-07-30_12-34-08_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-789/model_2850.pt"
#     "2025-07-30_13-16-15_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-1337/model_3800.pt"
#     "2025-07-30_13-58-38_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-2048/model_2350.pt"
#     "2025-07-30_14-40-44_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-3141/model_2050.pt"
#     "2025-07-30_15-22-36_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-5678/model_3900.pt"
#     "2025-07-30_16-05-09_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-9999/model_2750.pt"
#     "2025-07-30_16-47-37_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-1111/model_3850.pt"
# )

# Hypernet general obs w/ track info
# CHECKPOINTS=(
#     "2025-07-30_20-41-40_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-42/model_1350.pt"
#     "2025-07-30_21-29-31_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-37/model_850.pt"
#     "2025-07-30_22-17-27_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-456/model_3500.pt"
#     "2025-07-30_23-05-48_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-789/model_2650.pt"
#     "2025-07-30_23-53-53_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-1337/model_2950.pt"
#     "2025-07-31_00-41-57_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-2048/model_3450.pt"
#     "2025-07-31_01-30-11_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-3141/model_1950.pt"
#     "2025-07-31_02-17-57_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-5678/model_3500.pt"
#     "2025-07-31_03-05-57_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-9999/model_1700.pt"
#     "2025-07-31_03-53-55_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-1111/model_1300.pt"
# )

# Baseline w/ track info in general obs
# CHECKPOINTS=(
#     "2025-07-31_08-32-00_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-42/model_3999.pt"
#     "2025-07-31_09-07-09_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-37/model_3000.pt"
#     "2025-07-31_09-42-18_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-456/model_3950.pt"
#     "2025-07-31_10-17-05_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-789/model_3999.pt"
#     "2025-07-31_10-52-15_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-1337/model_1900.pt"
#     "2025-07-31_11-27-09_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-2048/model_2600.pt"
#     "2025-07-31_12-02-13_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-3141/model_3999.pt"
#     "2025-07-31_12-37-16_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-5678/model_3999.pt"
#     "2025-07-31_13-12-31_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-9999/model_3999.pt"
#     "2025-07-31_13-48-01_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-1111/model_3000.pt"
# )

# Baseline w/o track info
# CHECKPOINTS=(
#     "2025-07-31_16-10-49_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-42/model_3900.pt"
#     "2025-07-31_16-45-52_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-37/model_3250.pt"
#     "2025-07-31_17-20-36_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-456/model_3999.pt"
#     "2025-07-31_17-55-38_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-789/model_3400.pt"
#     "2025-07-31_18-30-49_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-1337/model_3500.pt"
#     "2025-07-31_19-06-02_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-2048/model_3400.pt"
#     "2025-07-31_19-41-05_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-3141/model_3650.pt"
#     "2025-07-31_20-16-21_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-5678/model_3200.pt"
#     "2025-07-31_20-51-40_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-9999/model_3999.pt"
#     "2025-07-31_21-26-32_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-1111/model_3800.pt"
# )

# Hypernet 64x64x64 w Tinfo in Gobs
# CHECKPOINTS=(
#     "2025-08-01_14-07-00_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-42/model_2150.pt"
#     "2025-08-01_14-55-56_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-37/model_3150.pt"
#     "2025-08-01_15-44-38_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-456/model_3700.pt"
#     "2025-08-01_16-32-57_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-789/model_1450.pt"
#     "2025-08-01_17-21-33_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-1337/model_650.pt"
#     "2025-08-01_18-10-18_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-2048/model_600.pt"
#     "2025-08-01_18-59-03_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-3141/model_1150.pt"
#     "2025-08-01_19-47-38_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-5678/model_3250.pt"
#     "2025-08-01_20-36-24_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-9999/model_2550.pt"
#     "2025-08-01_21-24-55_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-1111/model_1750.pt"
# )

# Hypernet 32x32 w/o hypernet init w Tinfo in Gobs
# CHECKPOINTS=(
#     "2025-08-02_10-45-16_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-42/model_2250.pt"
#     "2025-08-02_11-33-13_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-37/model_1950.pt"
#     "2025-08-02_12-21-07_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-456/model_3300.pt"
#     "2025-08-02_13-09-07_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-789/model_3900.pt"
#     "2025-08-02_13-56-53_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-1337/model_2050.pt"
#     "2025-08-02_14-45-20_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-2048/model_1500.pt"
#     "2025-08-02_15-33-49_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-3141/model_1550.pt"
#     "2025-08-02_16-22-20_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-5678/model_2400.pt"
#     "2025-08-02_17-10-48_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-9999/model_1500.pt"
#     "2025-08-02_17-58-54_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-1111/model_2650.pt"
# )

# Hypernet 512x256x128x64x32 w Tinfo in Gobs
# CHECKPOINTS=(
#     "2025-08-02_23-23-48_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-42/model_3450.pt"
#     "2025-08-03_00-14-14_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-37/model_3950.pt"
#     "2025-08-03_01-05-00_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-456/model_2150.pt"
#     "2025-08-03_01-55-19_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-789/model_1750.pt"
#     "2025-08-03_02-45-59_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-1337/model_2350.pt"
#     "2025-08-03_03-36-49_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-2048/model_3750.pt"
#     "2025-08-03_04-27-50_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-3141/model_3900.pt"
#     "2025-08-03_07-00-42_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-1111/model_3750.pt"
# )

# Hypernet 32 w Tinfo in Gobs
# CHECKPOINTS=(
#     "2025-08-03_13-57-54_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-42/model_3300.pt"
#     "2025-08-03_14-45-36_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-37/model_2000.pt"
#     "2025-08-03_15-33-14_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-456/model_3200.pt"
#     "2025-08-03_16-20-57_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-789/model_1600.pt"
#     "2025-08-03_17-08-46_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-1337/model_3750.pt"
#     "2025-08-03_17-56-40_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-2048/model_2600.pt"
#     "2025-08-03_18-44-20_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-3141/model_1300.pt"
#     "2025-08-03_19-32-18_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-5678/model_3150.pt"
#     "2025-08-03_20-19-54_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-9999/model_2800.pt"
#     "2025-08-03_21-07-48_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-1111/model_3200.pt"
# )

# Hypernet Critic 32x32 Tinfo in Gobs
# CHECKPOINTS=(
#     "2025-08-04_10-07-04_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-42/model_3550.pt"
#     "2025-08-04_11-07-35_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-37/model_1700.pt"
#     "2025-08-04_12-08-08_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-456/model_1700.pt"
#     "2025-08-04_13-08-34_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-789/model_3650.pt"
#     "2025-08-04_14-09-09_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-1337/model_2950.pt"
#     "2025-08-04_15-09-42_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-2048/model_1300.pt"
#     "2025-08-04_16-10-07_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-3141/model_3300.pt"
#     "2025-08-04_17-26-42_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-5678/model_2500.pt" 
#     "2025-08-04_18-27-27_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-9999/model_3100.pt"
#     "2025-08-04_19-27-54_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-1111/model_1450.pt"
# )

# MAX GATES 35
# Hypernet Critic 32x32 Tinfo in Gobs
# CHECKPOINTS=(
#     "2025-08-05_15-58-12_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-42/model_3650.pt"
#     "2025-08-05_17-48-14_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-37/model_2950.pt"
#     "2025-08-05_19-34-38_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-456/model_600.pt" #NG
#     "2025-08-05_21-24-11_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-789/model_3350.pt"
#     "2025-08-05_23-11-04_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-1337/model_3999.pt"
#     "2025-08-06_00-57-08_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-2048/model_1250.pt"
#     "2025-08-06_02-41-15_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-3141/model_3999.pt"
#     "2025-08-06_04-31-22_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-5678/model_250.pt" #NG
#     "2025-08-06_06-11-58_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-9999/model_650.pt" #NG
#     "2025-08-06_08-04-24_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-1111/model_250.pt" #NG
# )

# MAX GATES 40
# Hypernet 32x32 Tinfo in Gobs 
# CHECKPOINTS=(
#     2025-08-08_08-42-03_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-42/model_1700.pt
#     # 2025-08-08_12-38-13_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-37/model_3800. #NG
#     2025-08-08_20-13-24_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-789/model_2700.pt 
#     2025-08-08_23-57-09_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-1337/model_2300.pt 
#     2025-08-09_03-51-29_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-2048/model_1150.pt #1100
#     # 2025-08-09_07-32-42_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-3141/model_850.pt # NG
#     2025-08-09_11-20-15_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-5678/model_2900.pt #3000
#     2025-08-09_15-14-05_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-9999/model_2900.pt
#     2025-08-09_19-10-52_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-1111/model_350.pt #750
# )

# Baseline 32x32 Gobs w Tinf
CHECKPOINTS=(
    2025-08-12_11-23-04_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-42/model_3999.pt
    2025-08-13_08-16-46_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-37/model_3999.pt
    2025-08-13_15-13-54_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-456/model_2900.pt
    2025-08-13_22-12-09_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-789/model_3999.pt
    2025-08-14_05-03-03_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-1337/model_3999.pt
    2025-08-14_12-12-19_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-2048/model_3999.pt
    2025-08-14_19-12-24_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-3141/model_2800.pt
    2025-08-15_02-08-00_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-5678/model_3999.pt
    2025-08-15_08-57-27_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-9999/model_3999.pt
    2025-08-15_15-48-19_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-1111/model_3999.pt
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