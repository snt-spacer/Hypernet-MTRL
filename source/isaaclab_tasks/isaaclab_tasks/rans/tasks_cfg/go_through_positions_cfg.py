# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.utils import configclass

from isaaclab_tasks.rans.domain_randomization import NoisyObservationsCfg

from .task_core_cfg import TaskCoreCfg


@configclass
class GoThroughPositionsCfg(TaskCoreCfg):
    """Configuration for the GoThroughPosition task."""

    # Initial conditions
    spawn_min_dist: float = 0.5
    """Minimal distance between the spawn position and the target position in m. Defaults to 0.5 m."""
    spawn_max_dist: float = 5.0
    """Maximal distance between the spawn position and the target position in m. Defaults to 5.0 m."""
    spawn_min_heading_dist: float = 0.0
    """Minimal angle between the spawn orientation and the angle required to be looking at the target in rad.
    Defaults to 0.0 rad."""
    spawn_max_heading_dist: float = math.pi
    """Maximal angle between the spawn orientation and the angle required to be looking at the target in rad.
    Defaults to pi rad."""
    spawn_min_lin_vel: float = 0.0
    """Minimal linear velocity when spawned in m/s. Defaults to 0.0 m/s."""
    spawn_max_lin_vel: float = 0.0
    """Maximal linear velocity when spawned in m/s. Defaults to 0.0 m/s."""
    spawn_min_ang_vel: float = 0.0
    """Minimal angular velocity when spawned in rad/s. Defaults to 0.0 rad/s."""
    spawn_max_ang_vel: float = 0.0
    """Maximal angular velocity when spawned in rad/s. Defaults to 0.0 rad/s."""

    # Goal spawn
    goal_max_dist_from_origin: float = 0.0
    """Maximal distance from the origin of the environment. Defaults to 0.0."""
    goal_min_dist: float = 1.0
    """Minimal distance between the goals. Defaults to 1.0 m."""
    goal_max_dist: float = 5.0
    """Maximal distance between the goals. Defaults to 5.0 m."""
    max_num_goals: int = 10
    """Maximal number of goals. Defaults to 10."""
    min_num_goals: int = 6
    """Minimal number of goals. Defaults to 6."""
    loop: bool = True
    """Whether the goals should loop or not. Defaults to True."""

    # Observation
    num_subsequent_goals: int = 2
    """Number of subsequent goals available in the observation. Defaults to 2."""

    # Tolerance
    position_tolerance: float = 0.10
    """Tolerance for the position of the robot. Defaults to 1cm."""
    maximum_robot_distance: float = 30.0
    """Maximal distance between the robot and the target position. Defaults to 10 m."""

    # Reward Would be good to have a config for each reward type
    position_heading_exponential_reward_coeff: float = 0.25
    position_heading_weight: float = 0.2
    linear_velocity_min_value: float = 0.5
    linear_velocity_max_value: float = 2.0
    angular_velocity_min_value: float = 0.5
    angular_velocity_max_value: float = 20.0
    boundary_exponential_reward_coeff: float = 1.0
    linear_velocity_weight: float = -0.00
    angular_velocity_weight: float = -0.00
    boundary_weight: float = -10.0
    time_penalty: float = -0.0
    reached_bonus: float = 10.0
    progress_weight: float = 1.0

    # Randomization
    noisy_observation_cfg: NoisyObservationsCfg = NoisyObservationsCfg(
        enable=False,
        randomization_modes=["uniform"],
        slices=[(0, 2), 2, 3, (3, 5)]
        + sum([[5 + i * 3, (6 + i * 3, 8 + i * 3)] for i in range(num_subsequent_goals - 1)], []),
        max_delta=[0.03, 0.01, 0.03, 0.01] + sum([[0.03, 0.01] for _ in range(num_subsequent_goals - 1)], []),
    )

    # Spaces
    observation_space: int = 3 + 3 * num_subsequent_goals
    state_space: int = 0
    action_space: int = 0
    gen_space: int = 6
