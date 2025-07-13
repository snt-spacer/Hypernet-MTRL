# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.utils import configclass

from isaaclab_tasks.rans.domain_randomization import NoisyObservationsCfg

from .task_core_cfg import TaskCoreCfg


@configclass
class RaceGatesCfg(TaskCoreCfg):
    """Configuration for the RaceWayposes task."""

    # Initial conditions
    spawn_min_dist: float = 0.5
    """Minimal distance between the spawn pose and the target pose in m. Defaults to 0.5 m."""
    spawn_max_dist: float = 2.0
    """Maximal distance between the spawn pose and the target pose in m. Defaults to 2.0 m."""
    spawn_min_cone_spread: float = 0.0
    """When generating an initial position, the robot is spawned in a cone behind (+pi) the target's orientation.
    This parameter defines the minimal angle that cone can have. Defaults to 0.0 rad.
    Spawn formula:
    dx = target_x - robot_x
    dy = target_y - robot_y
    heading_to_target = atan2(dy, dx)
    theta = random(spawn_min_cone_spread, spawn_max_cone_spread) * random_sign() + heading_to_target + pi
    px = d * cos(theta)"""
    spawn_max_cone_spread: float = math.pi / 4
    """When generating an initial position, the robot is spawned in a cone behind (+pi) the target's orientation.
    This parameter defines the maximal angle that cone can have. Defaults to pi rad.
    Spawn formula:
    dx = target_x - robot_x
    dy = target_y - robot_y
    heading_to_target = atan2(dy, dx)
    theta = random(spawn_min_cone_spread, spawn_max_cone_spread) * random_sign() + heading_to_target + pi
    px = d * cos(theta)"""
    spawn_min_heading_dist: float = 0.0
    """Minimal angle between the spawn orientation and the target orientation in rad. Defaults to 0.0 rad."""
    spawn_max_heading_dist: float = math.pi / 4
    """Maximal angle between the spawn orientation and the target orientation in rad. Defaults to pi rad."""
    spawn_min_lin_vel: float = 0.0
    """Minimal linear velocity when spawned in m/s. Defaults to 0.0 m/s."""
    spawn_max_lin_vel: float = 0.0
    """Maximal linear velocity when spawned in m/s. Defaults to 0.0 m/s."""
    spawn_min_ang_vel: float = 0.0
    """Minimal angular velocity when spawned in rad/s. Defaults to 0.0 rad/s."""
    spawn_max_ang_vel: float = 0.0
    """Maximal angular velocity when spawned in rad/s. Defaults to 0.0 rad/s."""
    spawn_at_random_gate: bool = False
    fixed_track_id: int = 10
    """Controls track generation across environments and resets:
    - If -1: Each environment gets a different random track every reset
    - If set to a specific number: All environments get the same track (but new track each reset)
    Combined with same_track_for_all_envs for full control over track behavior."""
    same_track_for_all_envs: bool = True
    """Controls track persistence across resets:
    - If True: The same track is used for all environments and persists across resets
    - If False: New tracks are generated each reset (behavior depends on fixed_track_id)
    Track generation summary:
    1. same_track_for_all_envs=True: Same track across all envs and all resets
    2. same_track_for_all_envs=False + fixed_track_id=-1: Different track per env, new tracks each reset
    3. same_track_for_all_envs=False + fixed_track_id=N: Same track across envs, new track each reset"""

    # Goal spawn
    max_num_corners: int = 13
    """Maximal number of corners. Defaults to 13."""
    min_num_corners: int = 9
    """Minimal number of corners. Defaults to 9."""
    track_rejection_angle: float = (12.5 / 180.0) * math.pi
    """Angle in radians to reject tracks that have too sharp corners. Defaults to 12.5 degrees.
    sharp corners can lead to self-intersecting tracks."""
    min_point_distance: float = 0.1
    """The minimum distance between the points sampled to create the track. Should be between 0 and 1. Smaller values can create more complex tracks."""
    scale: float = 20.0
    """Scale of the track. Defaults to 20.0."""
    rad: float = 0.2
    """A coefficient that affects the smoothness of the track. Defaults to 0.2."""
    edgy: float = 0.0
    """A coefficient that affects the edginess of the track. Defaults to 0.0."""
    loop: bool = False
    num_laps: int = 1
    gate_width: float = 0.75
    

    # Observation
    num_subsequent_goals: int = 3
    """Number of subsequent goals available in the observation. Defaults to 2."""

    # Tolerance
    maximum_robot_distance: float = 30.0
    """Maximal distance between the robot and the target position. Defaults to 10 m."""

    # Reward Would be good to have a config for each reward type
    position_heading_exponential_reward_coeff: float = 0.25
    position_heading_weight: float = 0.0
    boundary_exponential_reward_coeff: float = 1.0
    boundary_weight: float = -10.0
    time_penalty: float = -0.0
    reached_bonus: float = 50.0
    reverse_penalty: float = -100.0
    progress_weight: float = 2.0

    # Randomization
    noisy_observation_cfg: NoisyObservationsCfg = NoisyObservationsCfg(
        enable=False,
        randomization_modes=["uniform"],
        slices=[(0, 2), 2, 3, (3, 5), (5, 7)]
        + sum(
            [[7 + i * 3, (8 + i * 3, 10 + i * 3), (10 + i * 3, 12 + i * 3)] for i in range(num_subsequent_goals - 1)],
            [],
        ),
        max_delta=[0.03, 0.01, 0.03, 0.01, 0.01]
        + sum([[0.03, 0.01, 0.01] for _ in range(num_subsequent_goals - 1)], []),
    )

    # Spaces
    observation_space: int = 3 + 5 * num_subsequent_goals
    state_space: int = 0
    action_space: int = 0
    gen_space: int = 11
