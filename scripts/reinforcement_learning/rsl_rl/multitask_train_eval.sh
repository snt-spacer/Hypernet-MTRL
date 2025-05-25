#!/bin/bash

# --- Configuration ---
OUTPUT_DIR_NAME="multitask_train_eval_info"
BASE_OUTPUT_DIR="/workspace/isaaclab/source/${OUTPUT_DIR_NAME}"
mkdir -p "$BASE_OUTPUT_DIR"
MODEL_TRACKER_FILE="${BASE_OUTPUT_DIR}/trained_models_tracker.log"

# Define your robots and tasks
robots=(Turtlebot2)
BASE_TASKS=(GoToPosition GoToPose TrackVelocities)
TRAINING_TASK_CONFIGS=("${BASE_TASKS[@]}" "ALL_COMBINED_TASKS")
num_envs=4096

# Convert the tasks array into a comma-separated string for Python arguments
# IFS=, EVAL_TASKS_NAMES="${tasks[*]}" # Note: IFS is automatically reset for subsequent commands.
# NUM_TASKS=${#tasks[@]}

# --- Main Loop: Training and Evaluation for each robot and seed ---
for robot in "${robots[@]}"
do
    for task_config in "${TRAINING_TASK_CONFIGS[@]}"
    do
        # Determine the actual tasks for the current training run
        if [ "$task_config" == "ALL_COMBINED_TASKS" ]; then
            tasks=("${BASE_TASKS[@]}")
            TRAINING_DESCRIPTION="all_tasks"
        else
            tasks=("$task_config")
            TRAINING_DESCRIPTION="$task_config"
        fi

        # Convert the tasks array into a comma-separated string for Python arguments
        IFS=, EVAL_TASKS_NAMES="${tasks[*]}"  # Note: IFS is automatically reset for subsequent commands.
        NUM_TASKS=${#tasks[@]}

        for seed in {1..6}
        do
            # Generate a unique timestamp for the current training run's log file
            TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
            CLEAN_TASKS_NAMES=$(echo "$EVAL_TASKS_NAMES" | tr ',' '-')
            TRAINING_LOG_FILE="${BASE_OUTPUT_DIR}/training_${robot}_${CLEAN_TASKS_NAMES}_seed-${seed}_${TIMESTAMP}.log"

            echo "--------------------------------------------------------"
            echo "Starting run for Robot: $robot, Seed: $seed, Tasks: $EVAL_TASKS_NAMES"
            echo "Training logs will be written to: $TRAINING_LOG_FILE"

            # --- Training Phase Execution ---
            ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
                --task=Isaac-RANS-MultiTask-v0 \
                --seed="${seed}" \
                --num_envs="${num_envs}" \
                --headless \
                env.robot_name="${robot}" \
                env.tasks_names="[${EVAL_TASKS_NAMES}]" > "$TRAINING_LOG_FILE" 2>&1

            # Check if the training command executed successfully
            if [ $? -eq 0 ]; then
                echo "Training completed successfully for Robot: $robot, Seed: $seed, Tasks: $CLEAN_TASKS_NAMES."
            else
                echo "Training failed for Robot: $robot, Seed: $seed, Tasks: $CLEAN_TASKS_NAMES. Check logs in $TRAINING_LOG_FILE for details."
                echo "--------------------------------------------------------"
                continue
            fi

            # --- Post-Training: Extract Paths and Prepare for Evaluation ---
            LOG_BASE_DIR=$(grep -oP 'Logging experiment in directory: \K[^ ]+' "$TRAINING_LOG_FILE" | tail -1)
            MODEL_RELATIVE_PATH=$(grep -oP 'wandb: Syncing run \K[^ ]+' "$TRAINING_LOG_FILE" | tail -1)
            TOTAL_STEPS_STR=$(grep -oP 'Learning iteration [0-9]+/\K[0-9]+' "$TRAINING_LOG_FILE" | tail -1)

            # --- Error Handling for Path Extractions ---
            if [ -z "$LOG_BASE_DIR" ] || [ -z "$MODEL_RELATIVE_PATH" ] || [ -z "$TOTAL_STEPS_STR" ]; then
                echo "Error: One or more critical pieces of information (LOG_BASE_DIR, MODEL_RELATIVE_PATH, TOTAL_STEPS_STR) could not be extracted from $TRAINING_LOG_FILE."
                echo "  LOG_BASE_DIR: '$LOG_BASE_DIR'"
                echo "  MODEL_RELATIVE_PATH: '$MODEL_RELATIVE_PATH'"
                echo "  TOTAL_STEPS_STR: '$TOTAL_STEPS_STR'"
                echo "Skipping evaluation for Robot: $robot, Seed: $seed, Tasks: $CLEAN_TASKS_NAMES. and continuing to the next run."
                echo "--------------------------------------------------------"
                continue
            fi

            MODEL_NUMBER=$((TOTAL_STEPS_STR - 1))
            EXPERIMENT_DIR="${LOG_BASE_DIR}/${MODEL_RELATIVE_PATH}"
            FINAL_MODEL_CHECKPOINT_PATH="${EXPERIMENT_DIR}/model_${MODEL_NUMBER}.pt"

            echo "Derived Model Checkpoint Path: $FINAL_MODEL_CHECKPOINT_PATH"

            # Save the extracted model path to the overall tracker file
            echo "${TIMESTAMP}: ${robot}_seed${seed}: ${FINAL_MODEL_CHECKPOINT_PATH}" >> "$MODEL_TRACKER_FILE"
            echo "Model path added to: $MODEL_TRACKER_FILE"

            # --- Evaluation Phase Execution ---
            EVAL_NUM_ENVS=$((num_envs * NUM_TASKS))
            # Iterate through each task for individual evaluation
            for i in "${!tasks[@]}"; do
                CURRENT_TASK="${tasks[i]}"
                temp_tasks_array=("${CURRENT_TASK}") # Start with the current task
                for task_name in "${tasks[@]}"; do
                    if [[ "$task_name" != "$CURRENT_TASK" ]]; then
                        temp_tasks_array+=("${task_name}") # Add other tasks
                    fi
                done
                # Convert temp_tasks_array to a comma-separated string
                IFS=, EVAL_ORDERED_TASKS_NAMES="${temp_tasks_array[*]}"
                echo "Starting evaluation for model: $FINAL_MODEL_CHECKPOINT_PATH"
                echo "  Evaluation Task: ${CURRENT_TASK}"
                echo "  Evaluation Task Order Passed: [${EVAL_ORDERED_TASKS_NAMES}]"
                ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/eval.py \
                    --task=Isaac-RANS-MultiTask-v0 \
                    --headless \
                    --num_envs="${EVAL_NUM_ENVS}" \
                    --checkpoint="${FINAL_MODEL_CHECKPOINT_PATH}" \
                    env.robot_name="${robot}" \
                    env.tasks_names="[${EVAL_ORDERED_TASKS_NAMES}]" >> "$TRAINING_LOG_FILE" 2>&1 # '>>' to append to the log

                if [ $? -eq 0 ]; then
                    echo "Evaluation completed successfully for Robot: $robot, Seed: $seed, Tasks: $CLEAN_TASKS_NAMES."
                else
                    echo "Evaluation failed for Robot: $robot, Seed: $seed, Tasks: $CLEAN_TASKS_NAMES. Check logs in $TRAINING_LOG_FILE for details."
                    echo "--------------------------------------------------------"
                    continue
                fi
                echo "Evaluation finished for Task: ${CURRENT_TASK}."
            done
            echo "--------------------------------------------------------"
            echo ""
        done
    done
done

echo "All training and evaluation runs completed."