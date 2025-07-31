#!/bin/bash

# --- Configuration ---
OUTPUT_DIR_NAME="multitask_eval_control_hypernet_general_obs_task_id"
BASE_OUTPUT_DIR="/workspace/isaaclab/source/${OUTPUT_DIR_NAME}"
mkdir -p "$BASE_OUTPUT_DIR"
EVALUATION_LOG_FILE="${BASE_OUTPUT_DIR}/evaluation_run_$(date +"%Y-%m-%d_%H-%M-%S").log"

# Define your robot and tasks
robot="ModularFreeflyer"
BASE_TASKS=(GoToPosition GoToPose TrackVelocities Rendezvous)
num_envs=1024 # Base number of environments, adjust if needed
algorithm="ppo" #ppo, ppo-memory, ppo-beta
runs_per_env=1

# Hypernetwork 32x32
# MODEL_PATHS=(
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_15-43-21_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/model_1500.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_16-13-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/model_1100.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_16-44-19_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/model_1050.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_17-14-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/model_1050.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_17-45-18_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/model_1600.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_18-15-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/model_850.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_18-47-07_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/model_850.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_19-18-15_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/model_1500.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_19-48-44_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/model_1850.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_20-19-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/model_1900.pt
# )

# Baseline with info about thrusters
# MODEL_PATHS=(
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_10-21-16_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/model_950.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_10-48-38_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/model_1600.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_11-15-42_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/model_1700.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_11-42-24_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/model_1550.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_07-17-03_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/model_1650.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_07-44-01_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/model_1550.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_08-11-02_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/model_1800.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_08-38-05_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/model_1950.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_09-05-05_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/model_1500.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_09-32-09_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/model_1000.pt
# )

# Hypernetwork 32x32 with general obs task id
MODEL_PATHS=(
    /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-28_18-11-11_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-42/model_2700.pt
    /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-29_10-40-36_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/model_2700.pt
    /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-29_11-41-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/model_3500.pt
    /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-29_12-42-39_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/model_500.pt
    /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-29_13-44-13_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/model_3700.pt
    /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-29_14-45-35_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/model_3000.pt
    /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-29_15-46-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/model_3600.pt
    /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-29_16-48-38_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/model_3650.pt
    /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-29_17-49-29_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/model_250.pt
    /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-29_18-50-32_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/model_3999.pt
    /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-29_19-52-10_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/model_1150.pt
)

# Check if model paths are provided
if [ ${#MODEL_PATHS[@]} -eq 0 ]; then
    echo "Error: No model paths provided. Please edit the script and add your .pth file paths to the MODEL_PATHS array."
    exit 1
fi

total_start_time=$(date +%s)
echo "Starting Evaluation Script..." | tee -a "$EVALUATION_LOG_FILE"
echo "Log file: $EVALUATION_LOG_FILE" | tee -a "$EVALUATION_LOG_FILE"
echo "Found ${#MODEL_PATHS[@]} models to evaluate." | tee -a "$EVALUATION_LOG_FILE"

# Determine the number of tasks for calculating EVAL_NUM_ENVS
NUM_TASKS=${#BASE_TASKS[@]}
EVAL_NUM_ENVS=$((num_envs * NUM_TASKS))

# Convert BASE_TASKS array to a comma-separated string
IFS=, EVAL_TASKS_NAMES="${BASE_TASKS[*]}"

echo "Evaluating all tasks simultaneously: [${EVAL_TASKS_NAMES}]" | tee -a "$EVALUATION_LOG_FILE"
echo "Total environments: ${EVAL_NUM_ENVS} (${num_envs} per task × ${NUM_TASKS} tasks)" | tee -a "$EVALUATION_LOG_FILE"

# --- Main Loop: Evaluation for each model path ---
for model_path in "${MODEL_PATHS[@]}"
do
    run_start_time=$(date +%s)

    # Check if the model file exists
    if [ ! -f "$model_path" ]; then
        echo "--------------------------------------------------------" | tee -a "$EVALUATION_LOG_FILE"
        echo "Error: Model file not found: $model_path" | tee -a "$EVALUATION_LOG_FILE"
        echo "Skipping evaluation for this model." | tee -a "$EVALUATION_LOG_FILE"
        echo "--------------------------------------------------------" | tee -a "$EVALUATION_LOG_FILE"
        continue # Skip to the next model
    fi

    echo "--------------------------------------------------------" | tee -a "$EVALUATION_LOG_FILE"
    echo "Starting evaluation for Model: $model_path" | tee -a "$EVALUATION_LOG_FILE"
    echo "Evaluating all tasks simultaneously: [${EVAL_TASKS_NAMES}]" | tee -a "$EVALUATION_LOG_FILE"

    # Execute the evaluation script for all tasks at once
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/eval.py \
        --task=Isaac-RANS-MultiTask-v0 \
        --headless \
        --num_envs="${EVAL_NUM_ENVS}" \
        --checkpoint="${model_path}" \
        --algorithm="${algorithm}" \
        --runs_per_env="${runs_per_env}" \
        env.robot_name="${robot}" \
        env.tasks_names="[${EVAL_TASKS_NAMES}]" >> "$EVALUATION_LOG_FILE" 2>&1

    # Check if the evaluation command executed successfully
    if [ $? -eq 0 ]; then
        echo "Evaluation completed successfully for all tasks." | tee -a "$EVALUATION_LOG_FILE"
    else
        echo "Evaluation failed. Check logs in $EVALUATION_LOG_FILE for details." | tee -a "$EVALUATION_LOG_FILE"
    fi

    run_end_time=$(date +%s)
    run_duration=$((run_end_time - run_start_time))
    echo "Evaluation for model $model_path finished in $(($run_duration / 60)) minutes and $(($run_duration % 60)) seconds." | tee -a "$EVALUATION_LOG_FILE"
    echo "--------------------------------------------------------" | tee -a "$EVALUATION_LOG_FILE"
    echo "" | tee -a "$EVALUATION_LOG_FILE"

done

total_end_time=$(date +%s)
total_duration=$((total_end_time - total_start_time))
total_hours=$(($total_duration / 3600))
total_minutes=$((($total_duration % 3600) / 60))
total_seconds=$(($total_duration % 60))

echo "All evaluation runs completed." | tee -a "$EVALUATION_LOG_FILE"
echo "Total execution time: $total_hours hours, $total_minutes minutes, and $total_seconds seconds." | tee -a "$EVALUATION_LOG_FILE"