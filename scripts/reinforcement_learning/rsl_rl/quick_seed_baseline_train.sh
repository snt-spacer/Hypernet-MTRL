#!/bin/bash
TASKS=(
    "GoToPosition"
    "GoToPose"
    "TrackVelocities"
    "Rendezvous"
)
#Train
for task in "${TASKS[@]}"; do
    BASE_TASKS+=("Isaac-RANS-MultiTask-${task}-v0")
    for SEED in {1..10}
    do
        ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-RANS-Single-v0 --headless env.robot_name=ModularFreeflyer env.task_name=${task} --algorithm=ppo --seed=$SEED
        sleep 5
    done
done