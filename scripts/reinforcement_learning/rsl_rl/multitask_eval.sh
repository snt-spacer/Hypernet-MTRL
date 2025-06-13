#!/bin/bash

# --- Configuration ---
OUTPUT_DIR_NAME="multitask_eval_FP_deep_nets_4x64_4x32_4x128"
BASE_OUTPUT_DIR="/workspace/isaaclab/source/${OUTPUT_DIR_NAME}"
mkdir -p "$BASE_OUTPUT_DIR"
EVALUATION_LOG_FILE="${BASE_OUTPUT_DIR}/evaluation_run_$(date +"%Y-%m-%d_%H-%M-%S").log"

# Define your robot and tasks
robot="FloatingPlatform"
BASE_TASKS=(GoToPose GoToPosition TrackVelocities GoThroughPoses)
num_envs=4096 # Base number of environments, adjust if needed

MODEL_PATHS=(
    /workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64/2025-06-11_17-09-28_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-2/model_2999.pt
    /workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64/2025-06-11_16-30-37_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-1/model_2999.pt
    /workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64/2025-06-11_14-47-23_rsl-rl_ppo_GoToPose_FloatingPlatform_r-0_seed-1/model_2999.pt
    /workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64/2025-06-11_15-02-44_rsl-rl_ppo_GoToPose_FloatingPlatform_r-0_seed-2/model_2999.pt
    /workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64/2025-06-11_15-50-25_rsl-rl_ppo_GoThroughPoses_FloatingPlatform_r-0_seed-1/model_2999.pt
    /workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64/2025-06-11_16-10-22_rsl-rl_ppo_GoThroughPoses_FloatingPlatform_r-0_seed-2/model_2999.pt
    /workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x32/2025-06-11_12-26-10_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-1/model_1999.pt
    /workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x32/2025-06-11_13-16-26_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-2/model_1999.pt
    /workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x32/2025-06-11_11-32-22_rsl-rl_ppo_GoThroughPoses_FloatingPlatform_r-0_seed-1/model_1999.pt
    /workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x32/2025-06-11_11-59-24_rsl-rl_ppo_GoThroughPoses_FloatingPlatform_r-0_seed-2/model_1999.pt
    /workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x32/2025-06-11_10-06-00_rsl-rl_ppo_GoToPose_FloatingPlatform_r-0_seed-1/model_1999.pt
    /workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x32/2025-06-11_10-27-12_rsl-rl_ppo_GoToPose_FloatingPlatform_r-0_seed-2/model_1999.pt
    /workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x128/2025-06-11_18-50-46_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-1/model_2999.pt
    /workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x128/2025-06-11_20-04-55_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-2/model_2999.pt
    /workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x128/2025-06-11_17-30-19_rsl-rl_ppo_GoThroughPoses_FloatingPlatform_r-0_seed-1/model_2999.pt
    /workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x128/2025-06-11_18-10-32_rsl-rl_ppo_GoThroughPoses_FloatingPlatform_r-0_seed-2/model_2999.pt
    /workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x128/2025-06-11_15-22-26_rsl-rl_ppo_GoToPose_FloatingPlatform_r-0_seed-1/model_2999.pt
    /workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x128/2025-06-11_15-53-51_rsl-rl_ppo_GoToPose_FloatingPlatform_r-0_seed-2/model_2999.pt
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

    # Determine the number of tasks for calculating EVAL_NUM_ENVS
    NUM_TASKS=${#BASE_TASKS[@]}
    EVAL_NUM_ENVS=$((num_envs * NUM_TASKS))

    # --- Evaluation Phase Execution ---
    # Iterate through each base task to evaluate the model on it
    for CURRENT_TASK in "${BASE_TASKS[@]}"; do
        temp_tasks_array=("$CURRENT_TASK") # Start with the current task

        # Add the other tasks to the array, ensuring the CURRENT_TASK is first
        for task_name in "${BASE_TASKS[@]}"; do
            if [[ "$task_name" != "$CURRENT_TASK" ]]; then
                temp_tasks_array+=("$task_name")
            fi
        done

        # Convert temp_tasks_array to a comma-separated string
        IFS=, EVAL_ORDERED_TASKS_NAMES="${temp_tasks_array[*]}"

        echo "  Evaluating Task: ${CURRENT_TASK}" | tee -a "$EVALUATION_LOG_FILE"
        echo "  Evaluation Task Order Passed: [${EVAL_ORDERED_TASKS_NAMES}]" | tee -a "$EVALUATION_LOG_FILE"

        # Execute the evaluation script
        ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/eval.py \
            --task=Isaac-RANS-MultiTask-v0 \
            --headless \
            --num_envs="${EVAL_NUM_ENVS}" \
            --checkpoint="${model_path}" \
            env.robot_name="${robot}" \
            env.tasks_names="[${EVAL_ORDERED_TASKS_NAMES}]" >> "$EVALUATION_LOG_FILE" 2>&1

        # Check if the evaluation command executed successfully
        if [ $? -eq 0 ]; then
            echo "  Evaluation completed successfully for Task: ${CURRENT_TASK}." | tee -a "$EVALUATION_LOG_FILE"
        else
            echo "  Evaluation failed for Task: ${CURRENT_TASK}. Check logs in $EVALUATION_LOG_FILE for details." | tee -a "$EVALUATION_LOG_FILE"
        fi
        echo "  ---" | tee -a "$EVALUATION_LOG_FILE"
    done

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