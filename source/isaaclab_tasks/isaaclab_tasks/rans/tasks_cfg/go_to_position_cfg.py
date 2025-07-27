# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.utils import configclass

from isaaclab_tasks.rans.domain_randomization import NoisyObservationsCfg

from .task_core_cfg import TaskCoreCfg


@configclass
class GoToPositionCfg(TaskCoreCfg):
    """Configuration for the GoToPosition task."""

    # Initial conditions
    spawn_min_dist: float = 0.5
    """Minimal distance between the spawn pose and the target pose in m. Defaults to 0.5 m."""
    spawn_max_dist: float = 5.0
    """Maximal distance between the spawn pose and the target pose in m. Defaults to 5.0 m."""
    spawn_min_heading_dist: float = 0.0
    """Minimal angle between the spawn orientation and the angle required to be looking at the target in rad.
    Defaults to 0.0 rad."""
    spawn_max_heading_dist: float = math.pi
    """Maximal angle between the spawn orientation and the angle required to be looking at the target in rad.
    Defaults to pi rad."""
    spawn_min_lin_vel: float = 0.0
    """Minimal linear velocity at spawn pose in m/s. Defaults to 0.0 m/s."""
    spawn_max_lin_vel: float = 0.0
    """Maximal linear velocity at spawn pose in m/s. Defaults to 0.0 m/s."""
    spawn_min_ang_vel: float = 0.0
    """Minimal angular velocity at spawn in rad/s. Defaults to 0.0 rad/s."""
    spawn_max_ang_vel: float = 0.0
    """Maximal angular velocity at spawn in rad/s. Defaults to 0.0 rad/s."""

    # Goal spawn
    goal_max_dist_from_origin: float = 0.0
    """Maximal distance from the origin of the environment. Defaults to 0.0."""

    # Tolerance
    position_tolerance: float = 0.01
    """Tolerance for the position of the robot. Defaults to 1cm."""
    maximum_robot_distance: float = 15.0
    """Maximal distance between the robot and the target pose. Defaults to 15 m."""
    reset_after_n_steps_in_tolerance: int = 100
    """Reset the environment after n steps in tolerance. Defaults to 100 steps."""

    # Reward Would be good to have a config for each reward type
    position_exponential_reward_coeff: float = 1.0
    heading_exponential_reward_coeff: float = 0.25
    min_heading_dist_scaler: float = 0.5
    max_heading_dist_scaler: float = 2.5
    linear_velocity_min_value: float = 0.0
    linear_velocity_max_value: float = 0.45
    angular_velocity_min_value: float = 0.0
    angular_velocity_max_value: float = 0.9
    boundary_exponential_reward_coeff: float = 1.0
    position_weight: float = 1.0
    progress_weight: float = 0.2
    heading_weight: float = 0.25
    linear_velocity_weight: float = -0.05
    angular_velocity_weight: float = -0.05
    boundary_weight: float = -10.0

    min_heading_dist_scaler: float = 0.5
    max_heading_dist_scaler: float = 2.5
    heading_weight: float = 0.25

    # Randomization
    noisy_observation_cfg: NoisyObservationsCfg = NoisyObservationsCfg(
        enable=False, randomization_modes=["uniform"], slices=[0, (1, 3), (3, 5), 5], max_delta=[0.03, 0.01, 0.03, 0.01]
    )
    thruster_mask_multiplier: float = 0.5
    """[0,1] The higher the more power to gen_actions for masking the activation of thrusters."""

    # Spaces
    observation_space: int = 6
    state_space: int = 0
    action_space: int = 0
    gen_space: int = 5


    # Evaluation
    eval_mode: bool = True
    """If True, the task is in evaluation mode. Defaults to False."""
    eval_mass: float = 0.0
    """Mass of the robot in kg for evaluation. Defaults to 0.0 kg, which means no mass change."""
    eval_thruster_pattern = [True, False, False, False, False, False, False, False]
    """The thruster activation pattern for evaluation."""