robots=(Jetbot Turtlebot Kingfisher Leatherback)
tasks=(GoToPosition GoToPose GoThroughPositions GoThroughPoses RaceGates TrackVelocities)

for robot in "${robots[@]}"
do
    for task in "${tasks[@]}"
    do
        for seed in {1..5}
        do
            ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-${robot}-${task}-Direct-v0 --seed=${seed} --headless
        done
    done
done
