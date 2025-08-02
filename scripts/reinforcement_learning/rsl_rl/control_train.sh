#!/bin/bash
ALGORITHM="ppo-memory"
for SEED in {1..10}
do
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-RANS-MultiTask-v0 --headless env.robot_name=ModularFreeflyer env.tasks_names='[GoToPosition, GoToPose, TrackVelocities, Rendezvous]' --algorithm=$ALGORITHM --seed=$SEED
    sleep 5
done