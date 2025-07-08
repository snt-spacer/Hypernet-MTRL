#!/bin/bash

# --- Configuration ---
OUTPUT_DIR_NAME="multitask_eval_rendezvous"
BASE_OUTPUT_DIR="/workspace/isaaclab/source/${OUTPUT_DIR_NAME}"
mkdir -p "$BASE_OUTPUT_DIR"
EVALUATION_LOG_FILE="${BASE_OUTPUT_DIR}/evaluation_run_$(date +"%Y-%m-%d_%H-%M-%S").log"

# Define your robot and tasks
robot="FloatingPlatform"
BASE_TASKS=(GoToPosition GoToPose TrackVelocities Rendezvous)
num_envs=1024 # Base number of environments, adjust if needed
algorithm="ppo-memory" #ppo, ppo-memory, ppo-beta
runs_per_env=1

# MODEL_PATHS=(
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_new-eval/2025-06-29_19-31-21_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-1/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_new-eval/2025-06-29_20-57-10_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-2/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_new-eval/2025-06-29_22-18-33_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-3/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_new-eval/2025-06-29_23-40-03_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-4/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_new-eval/2025-06-30_01-01-29_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-5/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_new-eval/2025-06-30_02-23-58_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-6/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_new-eval/2025-06-30_03-45-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-7/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_new-eval/2025-06-30_05-07-19_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-8/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_new-eval/2025-06-30_06-28-35_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-9/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_new-eval/2025-06-30_07-51-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-10/model_3999.pt
# )
# MODEL_PATHS=(
#     /workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-16_08-51-11_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-1/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-16_10-01-56_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-2/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-16_11-11-50_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-3/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-16_12-22-08_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-4/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-16_13-33-22_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-5/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-16_14-43-28_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-6/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-16_15-54-41_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-7/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-16_17-05-14_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-8/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-16_18-16-00_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-9/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_deep_net_4x64_10xseeds/2025-06-16_19-27-00_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-10/model_3999.pt
# )
# MODEL_PATHS=(
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_new-eval_max-obs_128x512x128/2025-06-30_19-53-58_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-1/model_1499.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_new-eval_max-obs_128x512x128/2025-06-30_20-22-33_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-2/model_1499.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_new-eval_max-obs_128x512x128/2025-06-30_20-51-33_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-3/model_1499.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_new-eval_max-obs_128x512x128/2025-06-30_21-20-03_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-4/model_1499.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_new-eval_max-obs_128x512x128/2025-06-30_21-48-59_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-5/model_1499.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_new-eval_max-obs_128x512x128/2025-06-30_22-17-44_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-6/model_1499.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_new-eval_max-obs_128x512x128/2025-06-30_22-46-29_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-7/model_1499.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_new-eval_max-obs_128x512x128/2025-06-30_23-15-13_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-8/model_1499.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_new-eval_max-obs_128x512x128/2025-06-30_23-44-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-9/model_1499.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_new-eval_max-obs_128x512x128/2025-07-01_00-12-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-GoThroughPoses_FloatingPlatform_r-0_seed-10/model_1499.pt
# )

MODEL_PATHS=(
    /workspace/isaaclab/logs/rsl_rl/multitask_memory_normW_hyperparams_rendezvous/2025-07-05_23-43-33_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_FloatingPlatform_r-0_seed-1/model_1999.pt
    /workspace/isaaclab/logs/rsl_rl/multitask_memory_normW_hyperparams_rendezvous/2025-07-06_00-18-06_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_FloatingPlatform_r-0_seed-2/model_1999.pt
    /workspace/isaaclab/logs/rsl_rl/multitask_memory_normW_hyperparams_rendezvous/2025-07-06_00-52-28_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_FloatingPlatform_r-0_seed-3/model_1999.pt
    /workspace/isaaclab/logs/rsl_rl/multitask_memory_normW_hyperparams_rendezvous/2025-07-06_01-27-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_FloatingPlatform_r-0_seed-4/model_1999.pt
    /workspace/isaaclab/logs/rsl_rl/multitask_memory_normW_hyperparams_rendezvous/2025-07-06_02-01-19_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_FloatingPlatform_r-0_seed-5/model_1999.pt
    /workspace/isaaclab/logs/rsl_rl/multitask_memory_normW_hyperparams_rendezvous/2025-07-06_02-36-21_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_FloatingPlatform_r-0_seed-6/model_1999.pt
    /workspace/isaaclab/logs/rsl_rl/multitask_memory_normW_hyperparams_rendezvous/2025-07-06_03-11-05_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_FloatingPlatform_r-0_seed-7/model_1999.pt
    /workspace/isaaclab/logs/rsl_rl/multitask_memory_normW_hyperparams_rendezvous/2025-07-06_03-45-39_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_FloatingPlatform_r-0_seed-8/model_1999.pt
    /workspace/isaaclab/logs/rsl_rl/multitask_memory_normW_hyperparams_rendezvous/2025-07-06_04-20-24_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_FloatingPlatform_r-0_seed-9/model_1999.pt
    /workspace/isaaclab/logs/rsl_rl/multitask_memory_normW_hyperparams_rendezvous/2025-07-06_04-55-06_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_FloatingPlatform_r-0_seed-10/model_1999.pt
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