#!/bin/bash

SCRIPT_PATH="./scripts/reinforcement_learning/rsl_rl/eval_orbital.py"
TASK="Isaac-RANS-Single-v0"
NUM_ENVS=4096
BASE_LOG_DIR="logs/rsl_rl/multitask_orbital"
COMMON_ARGS="--task=${TASK} --headless --num_envs=${NUM_ENVS} env.task_name=GoToPosition env.robot_name=ModularFreeflyer"
METRICS_SAVE_DIR="source/evaluation_metrics/orbital_exp"
THRUSTERS_PATERNS=(
    "[True, True, True, True, True, True, True, True]"
    "[False, False, True, True, True, True, True, True]"
    "[False, True, True, True, False, True, True, True]"
    "[False, True, False, True, False, True, False, True]"
    "[True, True, False, False, False, False, False, False]"
    "[True, False, False, False, False, False, False, False]"
)
CHECKPOINTS=(
    # "2025-07-17_18-34-33_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-42/model_850.pt"
    # "2025-07-18_07-33-48_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-42/model_1100.pt"
    "2025-07-21_06-00-25_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-42/model_10000.pt"
    "2025-07-21_07-54-39_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-37/model_9200.pt"
    "2025-07-21_09-43-20_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-8888/model_11650.pt"
    "2025-07-21_11-38-10_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-653/model_11600.pt"
)

mkdir -p "${METRICS_SAVE_DIR}"
mkdir -p "${METRICS_SAVE_DIR}/logs"

for checkpoint_path in "${CHECKPOINTS[@]}"; do
    for i in "${!THRUSTERS_PATERNS[@]}"; do
        thruster_pattern="${THRUSTERS_PATERNS[$i]}"
        full_checkpoint_path="${BASE_LOG_DIR}/${checkpoint_path}"
        pattern_name="pattern_${i}"
        experiment_dir=$(dirname "${checkpoint_path}")
        experiment_name=$(basename "${experiment_dir}")
        full_experiment_name="${experiment_name}_${pattern_name}"
        log_file="${METRICS_SAVE_DIR}/logs/${full_experiment_name}_${checkpoint_name}.log"
        echo "Running evaluation for checkpoint: ${full_checkpoint_path}"
        echo "Thruster pattern: ${thruster_pattern}"
        echo "Log file: ${log_file}"
        ./isaaclab.sh -p ${SCRIPT_PATH} ${COMMON_ARGS} --eval_thruster_pattern="${thruster_pattern}" --checkpoint=${full_checkpoint_path} > "${log_file}" 2>&1

        # Copy the metrics files to a source directory
        dest_dir="${METRICS_SAVE_DIR}/${experiment_name}/${checkpoint_name}/${pattern_name}"
        mkdir -p "${dest_dir}"
        source_dir="${BASE_LOG_DIR}/${experiment_name}/metrics"
        cp "$source_dir"/*.csv "${dest_dir}"/
        echo "Source directory: ${source_dir}"
        echo "Metrics copied to: ${dest_dir}"
        echo "------------------------------------------------------------------------------"
    done
    echo "Completed evaluations for checkpoint: ${checkpoint_path}"
    echo "------------------------------------------------------------------------------"
done