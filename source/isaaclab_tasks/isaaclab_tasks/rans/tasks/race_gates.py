# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch

from isaaclab.markers import BICOLOR_DIAMOND_CFG, GATE_2D_CFG, VisualizationMarkers
from isaaclab.scene import InteractiveScene
from isaaclab.utils.math import sample_random_sign
from isaaclab.markers.config import TRACK_CFG
import numpy as np

from isaaclab_tasks.rans import RaceGatesCfg
from isaaclab_tasks.rans.utils import PerEnvSeededRNG, TrackGenerator

from .task_core import TaskCore

import torch.nn.functional as F

EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)


class RaceGatesTask(TaskCore):
    """
    Implements the GoThroughPosition task. The robot has to reach a target position.
    """

    def __init__(
        self,
        scene: InteractiveScene | None = None,
        task_cfg: RaceGatesCfg = RaceGatesCfg(),
        task_uid: int = 0,
        num_envs: int = 1,
        device: str = "cuda",
        env_ids: torch.Tensor | None = None,
        decimation: int = 1,
        num_tasks: int = 1,
    ) -> None:
        """
        Initializes the GoThroughPoses task.

        Args:
            task_cfg: The configuration of the task.
            task_uid: The unique id of the task.
            num_envs: The number of environments.
            device: The device on which the tensors are stored.
            task_id: The id of the task.
            env_ids: The ids of the environments used by this task."""

        super().__init__(
            scene=scene, task_uid=task_uid, num_envs=num_envs, device=device, env_ids=env_ids, decimation=decimation, num_tasks=num_tasks
        )

        # Task and reward parameters
        self._task_cfg = task_cfg

        # Unique RNG for the tracks
        if self._task_cfg.fixed_track_id == -1:
            self._track_rng = PerEnvSeededRNG(0, self._num_envs, self._device)
        else:
            self._track_rng = PerEnvSeededRNG(self._task_cfg.fixed_track_id, self._num_envs, self._device)

        # Instantiate the track generator
        self._track_generator = TrackGenerator(
            scale=self._task_cfg.scale,
            rad=self._task_cfg.rad,
            edgy=self._task_cfg.edgy,
            max_num_points=self._task_cfg.max_num_corners,
            min_num_points=self._task_cfg.min_num_corners,
            min_point_distance= self._task_cfg.min_point_distance,
            rng=self._track_rng,
        )
        self.num_generations = 0

        # Defines the observation and actions space sizes for this task
        self._dim_task_obs = self._task_cfg.observation_space
        self._dim_gen_act = self._task_cfg.gen_space

        # Buffers
        self.initialize_buffers(env_ids=env_ids)

    @property
    def eval_data_keys(self) -> list[str]:
        """
        Returns the keys of the data used for evaluation.

        Returns:
            list[str]: The keys of the data used for evaluation."""
        
        return [
                "target_positions", 
                "target_headings", 
                "target_rotations", 
                "previous_is_before_gate", 
                "previous_is_after_gate", 
                "target_index", 
                "num_goals",
                "trajectory_completed",
                "position_distance",
                "cos_heading_to_target_error",
                "sin_heading_to_target_error",
                "cos_target_heading_error",
                "sin_target_heading_error",
                "subsequent_goals_distance",
                "cos_heading_to_subsequent_goals_error",
                "sin_heading_to_subsequent_goals_error",
                "cos_target_heading_to_subsequent_goals_error",
                "sin_target_heading_to_subsequent_goals_error",
                "missed_gate",
            ]
    
    @property
    def eval_data_specs(self)->dict[str, list[str]]:
        return {
            "target_positions": [coord for i in range(self._task_cfg.max_num_corners) for coord in (f".x_{i}.m", f".y_{i}.m")],
            "target_headings": [f".target_heading_{i}.rad" for i in self._task_cfg.max_num_corners],
            "target_rotations": [rot_matx_comp for i in range(self._task_cfg.max_num_corners) for rot_matx_comp in (f".target_rot_matx_r0_c0_{i}.m", f".target_rot_maty_r0_c1_{i}.m", f".target_rot_matx_r1_c0_{i}.m", f".target_rot_matx_r1_c1_{i}.m")],
            "previous_is_before_gate": [".u"],
            "previous_is_after_gate": [".u"],
            "target_index": [".u"],
            "num_goals": [".u"],
            "trajectory_completed": [".u"],
            "position_distance": [".distance.m"],
            "cos_heading_to_target_error": [".cos(heading).u"],
            "sin_heading_to_target_error": [".sin(heading).u"],
            "cos_target_heading_error": [".cos(target_heading_error).u"],
            "sin_target_heading_error": [".sin(target_heading_error).u"],
            "subsequent_goals_distance": [f".distance_sub_goal_{i}.m" for i in range(self._task_cfg.max_num_corners - 1)],
            "cos_heading_to_subsequent_goals_error": [f".cos(heading)_sub_goal_{i}.u" for i in range(self._task_cfg.max_num_corners - 1)],
            "sin_heading_to_subsequent_goals_error": [f".sin(heading)_sub_goal_{i}.u" for i in range(self._task_cfg.max_num_corners - 1)],
            "cos_target_heading_to_subsequent_goals_error": [f".cos(target_heading)_sub_goal_{i}.u" for i in range(self._task_cfg.max_num_corners - 1)],
            "sin_target_heading_to_subsequent_goals_error": [f".sin(target_heading)_sub_goal_{i}.u" for i in range(self._task_cfg.max_num_corners - 1)],
            "missed_gate": [".u"],
        }

    @property
    def eval_data(self) -> dict:
        """
        Returns the data used for evaluation.

        Returns:
            dict: The data used for evaluation."""
        
        subsequent_goals_distance = []
        cos_heading_to_subsequent_goals_error = []
        sin_heading_to_subsequent_goals_error = []
        cos_target_heading_to_subsequent_goals_error = []
        sin_target_heading_to_subsequent_goals_error = []

        for i in range(self._task_cfg.num_subsequent_goals - 1):
            subsequent_goals_distance.append(self._task_data[:, 8 + 5 * i])
            cos_heading_to_subsequent_goals_error.append(self._task_data[:, 9 + 5 * i])
            sin_heading_to_subsequent_goals_error.append(self._task_data[:, 10 + 5 * i])
            cos_target_heading_to_subsequent_goals_error.append(self._task_data[:, 11 + 5 * i])
            sin_target_heading_to_subsequent_goals_error.append(self._task_data[:, 12 + 5 * i])

        subsequent_goals_distance = torch.stack(subsequent_goals_distance, dim=-1)
        cos_heading_to_subsequent_goals_error = torch.stack(cos_heading_to_subsequent_goals_error, dim=-1)
        sin_heading_to_subsequent_goals_error = torch.stack(sin_heading_to_subsequent_goals_error, dim=-1)
        cos_target_heading_to_subsequent_goals_error = torch.stack(cos_target_heading_to_subsequent_goals_error, dim=-1)
        sin_target_heading_to_subsequent_goals_error = torch.stack(sin_target_heading_to_subsequent_goals_error, dim=-1)

        reshaped_subsequent_goals_distance = subsequent_goals_distance.view(
            subsequent_goals_distance.shape[0], self._task_cfg.num_subsequent_goals - 1, -1
        ).permute(0, 2, 1).reshape(subsequent_goals_distance.shape)
        reshaped_cos_heading_to_subsequent_goals_error = cos_heading_to_subsequent_goals_error.view(
            cos_heading_to_subsequent_goals_error.shape[0], self._task_cfg.num_subsequent_goals - 1, -1
        ).permute(0, 2, 1).reshape(cos_heading_to_subsequent_goals_error.shape)
        reshaped_sin_heading_to_subsequent_goals_error = sin_heading_to_subsequent_goals_error.view(
            sin_heading_to_subsequent_goals_error.shape[0], self._task_cfg.num_subsequent_goals - 1, -1
        ).permute(0, 2, 1).reshape(sin_heading_to_subsequent_goals_error.shape)
        reshaped_cos_target_heading_to_subsequent_goals_error = cos_target_heading_to_subsequent_goals_error.view(
            cos_target_heading_to_subsequent_goals_error.shape[0], self._task_cfg.num_subsequent_goals - 1, -1
        ).permute(0, 2, 1).reshape(cos_target_heading_to_subsequent_goals_error.shape)
        reshaped_sin_target_heading_to_subsequent_goals_error = sin_target_heading_to_subsequent_goals_error.view(
            sin_target_heading_to_subsequent_goals_error.shape[0], self._task_cfg.num_subsequent_goals - 1, -1
        ).permute(0, 2, 1).reshape(sin_target_heading_to_subsequent_goals_error.shape)
        
        return {
            "target_positions": self._target_positions,
            "target_headings": self._target_heading,
            "target_rotations": self._target_rotations,
            "previous_is_before_gate": self._previous_is_before_gate,
            "previous_is_after_gate": self._previous_is_after_gate,
            "target_index": self._target_index,
            "num_goals": self._num_goals,
            "trajectory_completed": self._trajectory_completed,
            "position_distance": self._task_data[:, 3],
            "cos_heading_to_target_error": self._task_data[:, 4],
            "sin_heading_to_target_error": self._task_data[:, 5],
            "cos_target_heading_error": self._task_data[:, 6],
            "sin_target_heading_error": self._task_data[:, 7],
            "subsequent_goals_distance": reshaped_subsequent_goals_distance,
            "cos_heading_to_subsequent_goals_error": reshaped_cos_heading_to_subsequent_goals_error,
            "sin_heading_to_subsequent_goals_error": reshaped_sin_heading_to_subsequent_goals_error,
            "cos_target_heading_to_subsequent_goals_error": reshaped_cos_target_heading_to_subsequent_goals_error,
            "sin_target_heading_to_subsequent_goals_error": reshaped_sin_target_heading_to_subsequent_goals_error,
            "missed_gate": self._missed_gate,
        }

    def initialize_buffers(self, env_ids: torch.Tensor | None = None) -> None:
        """
        Initializes the buffers used by the task.

        Args:
            env_ids: The ids of the environments used by this task."""

        super().initialize_buffers(env_ids)
        self._position_error = torch.zeros((self._num_envs, 2), device=self._device, dtype=torch.float32)
        self._position_dist = torch.zeros((self._num_envs,), device=self._device, dtype=torch.float32)
        self._previous_position_dist = torch.zeros((self._num_envs,), device=self._device, dtype=torch.float32)
        self._target_rotations = torch.zeros(
            (self._num_envs, self._task_cfg.max_num_corners, 2, 2), device=self._device, dtype=torch.float32
        )
        self._target_positions = torch.zeros(
            (self._num_envs, self._task_cfg.max_num_corners, 2),
            device=self._device,
            dtype=torch.float32,
        )
        self._target_heading = torch.zeros(
            (self._num_envs, self._task_cfg.max_num_corners),
            device=self._device,
            dtype=torch.float32,
        )
        self._previous_is_before_gate = torch.zeros((self._num_envs,), device=self._device, dtype=torch.bool)
        self._previous_is_after_gate = torch.zeros((self._num_envs,), device=self._device, dtype=torch.bool)
        self._target_index = torch.zeros((self._num_envs,), device=self._device, dtype=torch.long)
        self._trajectory_completed = torch.zeros((self._num_envs,), device=self._device, dtype=torch.bool)
        self._num_goals = torch.zeros((self._num_envs,), device=self._device, dtype=torch.long)
        self._ALL_INDICES = torch.arange(self._num_envs, dtype=torch.long, device=self._device)
        self._num_resets_per_env = torch.zeros((self._num_envs,), device=self._device, dtype=torch.long)
        self._laps_completed = torch.zeros((self._num_envs,), device=self._device, dtype=torch.long)
        self._missed_gate = torch.zeros((self._num_envs,), device=self._device, dtype=torch.bool)

    def create_logs(self) -> None:
        """
        Creates a dictionary to store the training statistics for the task."""

        super().create_logs()

        self.scalar_logger.add_log("task_state", "AVG/normed_linear_velocity", "mean")
        self.scalar_logger.add_log("task_state", "MAX/normed_linear_velocity", "max")
        self.scalar_logger.add_log("task_state", "AVG/absolute_angular_velocity", "mean")
        self.scalar_logger.add_log("task_state", "AVG/position_distance", "mean")
        self.scalar_logger.add_log("task_state", "AVG/boundary_distance", "mean")
        self.scalar_logger.add_log("task_state", "MAX/num_tracks(resets)", "max")
        self.scalar_logger.add_log("task_state", "MIN/num_tracks(resets)", "min")
        self.scalar_logger.add_log("task_state", "AVG/laps_completed", "mean")
        self.scalar_logger.add_log("task_state", "SUM/num_goals", "sum")
        self.scalar_logger.add_log("task_state", "AVG/num_gates", "mean")

        self.scalar_logger.add_log("task_reward", "AVG/boundary", "mean")
        self.scalar_logger.add_log("task_reward", "AVG/heading", "mean")
        self.scalar_logger.add_log("task_reward", "AVG/progress", "mean")
        self.scalar_logger.add_log("task_reward", "AVG/goals_reward", "mean")
        self.scalar_logger.add_log("task_reward", "AVG/time_penalty", "mean")
        self.scalar_logger.add_log("task_reward", "AVG/reverse_penalty", "mean")
        self.scalar_logger.add_log("task_reward", "AVG/missed_gate_penalty", "mean")

    def get_observations(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the observation tensor from the current state of the robot.

        The observation is given in the robot's frame. The observation is composed of the following elements:
        - The linear velocity of the robot.
        - The angular velocity of the robot.
        - The distance to the target position.
        - The angle between the robot's heading and the target position.
        - The angle between the robot's heading and the target heading.
        - Depending on the task configuration, a number of subsequent poses are added to the observation. For each of
            them, the following elements are added:
            - The distance between the n th and the n+1 th goal.
            - The angle between the n th goal and the n+1 th goal's position (in the current's goal frame).
            - The angle between the n th goal and the n+1 th goal's heading.

        Angle measurements are converted to a cosine and a sine to avoid discontinuities in 0 and 2pi.
        This provides a continuous representation of the angle.

        self._task_data[:, 0] = The linear velocity of the robot along the x-axis.
        self._task_data[:, 1] = The linear velocity of the robot along the y-axis.
        self._task_data[:, 2] = The angular velocity of the robot.
        self._task_data[:, 3] = The distance between the robot and the target position.
        self._task_data[:, 4] = The cosine of the angle between the robot's heading and the target position.
        self._task_data[:, 5] = The sine of the angle between the robot's heading and the target position.
        self._task_data[:, 6] = The cosine of the angle between the robot's heading and the target heading.
        self._task_data[:, 7] = The sine of the angle between the robot's heading and the target heading.
        self._task_data[:, 8 + i*5] = The distance between the n th and the n+1 th goal.
        self._task_data[:, 9 + i*5] = The cosine of the angle between the n th goal and the n+1 th goal's position.
        self._task_data[:, 10 + i*5] = The sine of the angle between the n th goal and the n+1 th goal's position.
        self._task_data[:, 11 + i*5] = The cosine of the angle between the n th goal and the n+1 th goal's heading.
        self._task_data[:, 12 + i*5] = The sine of the angle between the n th goal and the n+1 th goal's heading.

        Returns:
            torch.Tensor: The observation tensor."""

        # position error
        position_error = (
            self._target_positions[self._ALL_INDICES, self._target_index]
            - self._robot.root_link_pos_w[self._env_ids, :2]
        )
        position_dist = torch.linalg.norm(position_error, dim=-1)

        # position error expressed as distance and angular error (to the position)
        heading = self._robot.heading_w[self._env_ids]
        target_heading_w = torch.atan2(
            self._target_positions[self._ALL_INDICES, self._target_index, 1]
            - self._robot.root_link_pos_w[self._env_ids, 1],
            self._target_positions[self._ALL_INDICES, self._target_index, 0]
            - self._robot.root_link_pos_w[self._env_ids, 0],
        )
        target_heading_error = torch.atan2(torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading))
        # Heading of the gate in the robot frame
        heading_error = torch.atan2(
            torch.sin(self._target_heading[self._ALL_INDICES, self._target_index] - heading),
            torch.cos(self._target_heading[self._ALL_INDICES, self._target_index] - heading),
        )

        # Store in buffer
        self._task_data[:, 0:2] = self._robot.root_com_lin_vel_b[self._env_ids, :2]
        self._task_data[:, 2] = self._robot.root_com_ang_vel_w[self._env_ids, -1]
        self._task_data[:, 3] = position_dist
        self._task_data[:, 4] = torch.cos(target_heading_error)
        self._task_data[:, 5] = torch.sin(target_heading_error)
        self._task_data[:, 6] = torch.cos(heading_error)
        self._task_data[:, 7] = torch.sin(heading_error)

        # We compute the observations of the subsequent goals in the previous goal's frame.
        for i in range(self._task_cfg.num_subsequent_goals - 1):
            # Check if the index is looking beyond the number of goals
            overflowing = (self._target_index + i + 1) >= self._num_goals
            # If it is, then set the next index to 0 (Loop around)
            indices = torch.remainder(self._target_index + i + 1, self._num_goals)
            # Compute the distance between the nth-1 goal, and the nth goal
            goal_distance = torch.linalg.norm(
                self._target_positions[self._ALL_INDICES, indices - 1]
                - self._target_positions[self._ALL_INDICES, indices],
                dim=-1,
            )
            # Compute the angular distance between the nth-1 goal, and the nth goal (world goal frame)
            target_heading_w = torch.atan2(
                self._target_positions[self._ALL_INDICES, indices, 1]
                - self._target_positions[self._ALL_INDICES, indices - 1, 1],
                self._target_positions[self._ALL_INDICES, indices, 0]
                - self._target_positions[self._ALL_INDICES, indices - 1, 0],
            )
            # Compute the heading in the nth-1 goal frame
            target_heading_error = torch.atan2(
                torch.sin(target_heading_w - self._target_heading[self._ALL_INDICES, indices - 1]),
                torch.cos(target_heading_w - self._target_heading[self._ALL_INDICES, indices - 1]),
            )
            # Compute the heading delta between the nth-1 goal, and the nth goal
            heading_error = torch.atan2(
                torch.sin(
                    self._target_heading[self._ALL_INDICES, indices]
                    - self._target_heading[self._ALL_INDICES, indices - 1]
                ),
                torch.cos(
                    self._target_heading[self._ALL_INDICES, indices]
                    - self._target_heading[self._ALL_INDICES, indices - 1]
                ),
            )
            # If the task is not set to loop, we set the next goal to be 0.
            if not self._task_cfg.loop:
                goal_distance = goal_distance * torch.logical_not(overflowing)
                target_heading_error = target_heading_error * torch.logical_not(overflowing)
            # Add to buffer
            self._task_data[:, 8 + 5 * i] = goal_distance
            self._task_data[:, 9 + 5 * i] = torch.cos(target_heading_error)
            self._task_data[:, 10 + 5 * i] = torch.sin(target_heading_error)
            self._task_data[:, 11 + 5 * i] = torch.cos(heading_error)
            self._task_data[:, 12 + 5 * i] = torch.sin(heading_error)

        for randomizer in self.randomizers:
            randomizer.observations(observations=self._task_data)
        
        # Gates positions normalization & padding
        max_actual_goals = int(torch.max(self._num_goals).item()) + 1
        # Initialize with zeros for all environments
        points_with_padding = torch.zeros(
            (self._num_envs, self._task_cfg.max_num_corners, 2),
            device=self._device,
            dtype=torch.float32,
        )
        headings_with_padding = torch.zeros(
            (self._num_envs, self._task_cfg.max_num_corners),
            device=self._device,
            dtype=torch.float32,
        )

        # Fill in the actual points and headings for each environment
        for env_idx in range(self._num_envs):
            num_goals_env = int(self._num_goals[env_idx].item()) + 1
            if num_goals_env > 0:
                points = self._target_positions[env_idx, :num_goals_env] - self._env_origins[env_idx, :2].unsqueeze(0)
                normalized_points = points / self._task_cfg.scale
                points_with_padding[env_idx, :num_goals_env] = normalized_points
                
                headings = self._target_heading[env_idx, :num_goals_env]
                normalized_headings = torch.atan2(torch.sin(headings), torch.cos(headings)) / math.pi
                headings_with_padding[env_idx, :num_goals_env] = normalized_headings
        
        gates_positions = points_with_padding.view(self._num_envs, -1)
        normalized_num_goals = (self._num_goals.float() - self._task_cfg.min_num_corners) / (self._task_cfg.max_num_corners - self._task_cfg.min_num_corners)

        # Reshape and concatenate gates positions and headings
        gates_positions_reshaped = points_with_padding
        headings_with_padding_reshaped = headings_with_padding.view(self._num_envs, self._task_cfg.max_num_corners, 1)
        gates_and_headings_combined = torch.cat((gates_positions_reshaped, headings_with_padding_reshaped), dim=-1)
        flattened_gates_and_headings = gates_and_headings_combined.view(self._num_envs, -1)
        
        track_info = torch.concat((flattened_gates_and_headings, normalized_num_goals.unsqueeze(-1)), dim=-1)
        
        # combined_task_data = torch.concat((self._task_data, self._robot.get_observations(env_ids=self._env_ids), gates_positions), dim=-1)

        # Concatenate the task observations with the robot observations
        return torch.concat((self._task_data, self._robot.get_observations(env_ids=self._env_ids)), dim=-1), track_info

    def compute_rewards(self) -> torch.Tensor:
        """
        Computes the reward for the current state of the robot.

        Returns:
            torch.Tensor: The reward for the current state of the robot."""

        # position error
        self._position_error = (
            self._target_positions[self._ALL_INDICES, self._target_index]
            - self._robot.root_link_pos_w[self._env_ids, :2]
        )
        self._position_dist = torch.linalg.norm(self._position_error, dim=-1)
        # position error expressed as distance and angular error (to the position)
        heading = self._robot.heading_w[self._env_ids]
        heading_error = torch.atan2(
            torch.sin(self._target_heading[self._ALL_INDICES, self._target_index] - heading),
            torch.cos(self._target_heading[self._ALL_INDICES, self._target_index] - heading),
        )
        heading_dist = torch.abs(heading_error)

        # boundary distance
        boundary_dist = torch.abs(self._task_cfg.maximum_robot_distance - self._position_dist)
        # normed linear velocity
        linear_velocity = torch.linalg.norm(self._robot.root_com_vel_w[self._env_ids, :2], dim=-1)
        # normed angular velocity
        angular_velocity = torch.abs(self._robot.root_com_vel_w[self._env_ids, -1])
        # progress
        progress_rew = self._previous_position_dist - self._position_dist

        # heading reward (encourages the robot to face the target)
        heading_rew = torch.exp(-heading_dist / self._task_cfg.position_heading_exponential_reward_coeff)

        # boundary rew
        boundary_rew = torch.exp(-boundary_dist / self._task_cfg.boundary_exponential_reward_coeff)

        # Project the robot position into the target frame
        pos_proj = torch.matmul(
            self._target_rotations[self._ALL_INDICES, self._target_index], self._position_error.unsqueeze(-1)
        ).squeeze(-1)
        # Fix: Swap "before" and "after" gate definitions to match expected behavior
        # Now "before" means in front of the gate, and "after" means behind the gate
        is_before_gate = torch.logical_and(
            torch.logical_and(
                pos_proj[:, 0] > 0,
                pos_proj[:, 0] < 1,
            ),
            torch.abs(pos_proj[:, 1]) < self._task_cfg.gate_width / 2,
        )
        is_after_gate = torch.logical_and(
            torch.logical_and(
                pos_proj[:, 0] < 0,
                pos_proj[:, 0] > -1,
            ),
            torch.abs(pos_proj[:, 1]) < self._task_cfg.gate_width / 2,
        )

        self._missed_gate = torch.logical_and(
            torch.logical_and(
                pos_proj[:, 0] < 0,
                pos_proj[:, 0] > -1,
            ),
            torch.abs(pos_proj[:, 1]) > self._task_cfg.gate_width / 2,
        )
        
        # Checks if the goal is reached (robot has moved from being in front of the gate to behind it)
        goal_reached = torch.logical_and(is_after_gate, self._previous_is_before_gate).int()
        goal_reverse = torch.logical_and(is_before_gate, self._previous_is_after_gate).int()
        reached_ids = goal_reached.nonzero(as_tuple=False).squeeze(-1)
        # if the goal is reached, the target index is updated
        self._target_index = self._target_index + goal_reached
        # Check if the trajectory is completed
        close_loop = 0 if self._task_cfg.num_laps > 1 else 1
        self._trajectory_completed = self._target_index > self._num_goals + close_loop  # +1 so starts and finish on the same gate
        
        # Track lap completion: when trajectory is completed, increment lap counter
        lap_completed = self._trajectory_completed.int()
        self._laps_completed = self._laps_completed + lap_completed
        
        # To avoid out of bounds errors, set the target index to 0 if the trajectory is completed
        # If the task loops, then the target index is set to 0 which will make the robot go back to the first goal
        # The episode termination is handled in the get_dones method (looping or not)
        self._target_index = self._target_index * (~self._trajectory_completed)

        # If goal is reached make next progress null
        self._previous_position_dist[reached_ids] = 0

        # Update logs
        self.scalar_logger.log("task_state", "AVG/position_distance", self._position_dist)
        self.scalar_logger.log("task_state", "AVG/boundary_distance", boundary_dist)
        self.scalar_logger.log("task_state", "AVG/normed_linear_velocity", linear_velocity)
        self.scalar_logger.log("task_state", "MAX/normed_linear_velocity", linear_velocity)
        self.scalar_logger.log("task_state", "AVG/absolute_angular_velocity", angular_velocity)
        self.scalar_logger.log("task_state", "MAX/num_tracks(resets)", self._num_resets_per_env)
        self.scalar_logger.log("task_state", "MIN/num_tracks(resets)", self._num_resets_per_env)
        self.scalar_logger.log("task_state", "AVG/laps_completed", self._laps_completed)
        self.scalar_logger.log("task_state", "SUM/num_goals", goal_reached)
        self.scalar_logger.log("task_state", "AVG/num_gates", torch.mean(self._num_goals.float()))
        
        self.scalar_logger.log("task_reward", "AVG/boundary", boundary_rew * self._task_cfg.boundary_weight)
        self.scalar_logger.log("task_reward", "AVG/heading", heading_rew * self._task_cfg.position_heading_weight)
        self.scalar_logger.log("task_reward", "AVG/progress", progress_rew * self._task_cfg.progress_weight)
        self.scalar_logger.log("task_reward", "AVG/time_penalty", self._task_cfg.time_penalty)
        self.scalar_logger.log("task_reward", "AVG/goals_reward", goal_reached * self._task_cfg.reached_bonus)
        self.scalar_logger.log("task_reward", "AVG/reverse_penalty", goal_reverse * self._task_cfg.reverse_penalty)
        self.scalar_logger.log("task_reward", "AVG/missed_gate_penalty", self._missed_gate.int() * self._task_cfg.missed_gate_penalty)

        

        # Set the previous values
        self._previous_is_before_gate = is_before_gate.clone()
        self._previous_is_after_gate = is_after_gate.clone()
        self._previous_position_dist = self._position_dist.clone()

        # Return the reward by combining the different components and adding the robot rewards
        return (
            progress_rew * self._task_cfg.progress_weight
            + heading_rew * self._task_cfg.position_heading_weight
            + boundary_rew * self._task_cfg.boundary_weight
            + self._task_cfg.time_penalty
            + self._task_cfg.reached_bonus * goal_reached
            + self._task_cfg.reverse_penalty * goal_reverse
            + self._missed_gate.int() * self._task_cfg.missed_gate_penalty
        ) + self._robot.compute_rewards(env_ids=self._env_ids)

    def reset(
        self,
        env_ids: torch.Tensor,
        gen_actions: torch.Tensor | None = None,
        env_seeds: torch.Tensor | None = None,
    ) -> None:
        """
        Resets the task to its initial state.

        If gen_actions is None, then the environment is generated at random. This is the default mode.
        If env_seeds is None, then the seed is generated at random. This is the default mode.

        The environment actions for this task are the following all belong to the [0,1] range:
        - gen_actions[0]: The lower bound of the range used to sample the difference in heading between the goals.
        - gen_actions[1]: The range used to sample the difference in heading between the goals.
        - gen_actions[2]: The lower bound of the range used to sample the distance between the goals.
        - gen_actions[3]: The range used to sample the distance between the goals.
        - gen_actions[4]: The lower bound of the range used to sample the spread of the cone in which the goals are.
        - gen_actions[5]: The range used to sample the spread of the cone in which the goals are.
        - gen_actions[6]: The value used to sample the distance between the spawn position and the first goal.
        - gen_actions[7]: The value used to sample the angle between the spawn position and the first goal.
        - gen_actions[8]: The value used to sample the angle between the spawn heading and the first goal's heading.
        - gen_actions[9]: The value used to sample the linear velocity of the robot at spawn.
        - gen_actions[10]: The value used to sample the angular velocity of the robot at spawn.

        Args:
            env_ids (torch.Tensor): The ids of the environments.
            gen_actions (torch.Tensor | None): The actions for the task. Defaults to None.
            env_seeds (torch.Tensor | None): The seeds for the environments. Defaults to None.
        """

        super().reset(env_ids, gen_actions=gen_actions, env_seeds=env_seeds)

        # Reset the target index and trajectory completed
        # self._target_index[env_ids] = 0
        self._trajectory_completed[env_ids] = False
        
        # Reset the lap counter
        self._laps_completed[env_ids] = 0

        # The first 6 env actions define ranges, we need to make sure they don't exceed the [0,1] range.
        # They are given as [min, delta] we will convert them to [min, max] that is max = min + delta
        # Note that they are defined as [min, delta] to make sure the min is the min and the max is the max. This
        # is always true as they are strictly positive.
        self._gen_actions[env_ids, 1] = torch.clip(self._gen_actions[env_ids, 0] + self._gen_actions[env_ids, 1], max=1)
        self._gen_actions[env_ids, 3] = torch.clip(self._gen_actions[env_ids, 2] + self._gen_actions[env_ids, 3], max=1)
        self._gen_actions[env_ids, 5] = torch.clip(self._gen_actions[env_ids, 4] + self._gen_actions[env_ids, 5], max=1)

        # Set the track id if using non-random tracks
        if self._task_cfg.fixed_track_id == -1:
            track_ids = self._rng.sample_integer_torch(0, 2147483648, (1,), env_ids)
            self._track_rng.set_seeds(track_ids, env_ids)
        else:
            track_ids = torch.ones_like(env_ids) * self._task_cfg.fixed_track_id
            self._track_rng.set_seeds(track_ids, env_ids)

        # Randomizes goals and initial conditions
        self.set_goals(env_ids)
        self.set_initial_conditions(env_ids)

        # Resets the goal reached flag
        self._goal_reached[env_ids] = 0

        # Make sure the position error and position dist are up to date after the reset
        self._position_error[env_ids] = (
            self._target_positions[env_ids, self._target_index[env_ids]]
            - self._robot.root_link_pos_w[self._env_ids, :2][env_ids]
        )
        self._position_dist[env_ids] = torch.linalg.norm(self._position_error[env_ids], dim=-1)
        self._previous_position_dist[env_ids] = self._position_dist[env_ids].clone()

        # Reset the gate states
        # is_before_gate: robot is in front of the gate (positive x in gate frame)
        # is_after_gate: robot is behind the gate (negative x in gate frame)
        self._previous_is_before_gate[env_ids] = False
        self._previous_is_after_gate[env_ids] = False
        self._missed_gate[env_ids] = False

    def get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates if the platforms should be killed or not.

        Returns:
            torch.Tensor: Whether the platforms should be killed or not."""

        # Kill robots that would stray too far from the target.
        self._position_error = (
            self._target_positions[self._ALL_INDICES, self._target_index]
            - self._robot.root_link_pos_w[self._env_ids, :2]
        )
        self._previous_position_dist = self._position_dist.clone()
        self._position_dist = torch.linalg.norm(self._position_error, dim=-1)
        ones = torch.ones_like(self._goal_reached, dtype=torch.long)
        task_failed = torch.zeros_like(self._goal_reached, dtype=torch.long)
        task_failed = torch.where(
            self._position_dist > self._task_cfg.maximum_robot_distance,
            ones,
            task_failed,
        )
        # # Add missed gate as a failure condition
        task_failed = torch.where(
            self._missed_gate,
            ones,
            task_failed,
        )

        task_completed = torch.zeros_like(self._goal_reached, dtype=torch.long)
        # If the task is set to loop, don't terminate the episode early.
        # If not looping, terminate after completing the specified number of laps
        if not self._task_cfg.loop:
            task_completed = torch.where(self._laps_completed >= self._task_cfg.num_laps, ones, task_completed)
        return task_failed, task_completed

    def set_goals(self, env_ids: torch.Tensor):
        """
        Generates a random sequence of oriented goals for the task.
        If self._task_cfg.same_track_for_all_envs is True, all envs get the same track.
        Otherwise, each env gets its own random track.

        These goals are generated in a way allowing to precisely control the difficulty of the task through the
        environment action. This is done by randomizing ranges within which they can be generated. More information
        below:

        - The first goal is picked randomly in a square centered on the origin. Its orientation is picked randomly. This
            goal is the starting point of the trajectory and it cannot be changed through the environment action. We
            recommend setting that square to be 0. This way, the trajectory will always start at the origin.
        - The next goals are picked randomly in a cone originating from the previous goal.
            - For the heading, the environment action selects the range within which the goal will be picked for
                the whole trajectory. The new heading is picked randomly in a cone aligned with the direction to the
                previous goal. It uses the parameters `minimal_heading_distance` and `maximal_heading_distance`,
                and env_action[0], env_action[1] to set the range. The formula is the following:
                delta_heading = (U[env_action[0],env_action[1]] * (maximal_heading_distance - minimal_heading_distance) + minimal_heading_distance) * rand_sign()
            - For the position, we want to randomize at a given distance from the previous goal, and within a cone aligned
                with the direction to the previous goal. The environment action selects both the distance and the spread of
                the cone. The formula is the following:
                radius = U[env_action[2],e655362147483648nv_action[3]] * (maximal_goal_radius - minimal_goal_radius) + minimal_spawn_radius
                spawn_angle_delta = (U[env_action[4],env_action[5]] * (maximal_cone_spread - minimal_cone_spread) + minimal_cone_spread) * rand_sign()
                position_x = radius * cos(spawn_angle_delta + previous_goal_heading) + previous_goal_x
                position_y = radius * sin(spawn_angle_delta + previous_goal_heading) + previous_goal_y

        Args:
            env_ids (torch.Tensor): The ids of the environments.
            step (int, optional): The current step. Defaults to 0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The target positions and orientations."""
        
        if self._task_cfg.same_track_for_all_envs:
            # Per-env random tracks
            num_goals = len(env_ids)
            if self.num_generations < 1:
                if self._task_cfg.fixed_track_id == 0:
                    # BCN track
                    print("Generating BCN track for all environments")
                    self.points, self.tangents, self.num_goals = self._track_generator.generate_bcn_track(env_ids)
                else:
                    self.points, self.tangents, self.num_goals = self._track_generator.generate_tracks_points_non_fixed_points(env_ids)
            self._target_positions[env_ids] = self.points[env_ids] + self._env_origins[env_ids, :2].unsqueeze(1)
            self._target_heading[env_ids] = self.tangents[env_ids]
            self._target_rotations[env_ids, :, 0, 0] = torch.cos(self.tangents[env_ids])
            self._target_rotations[env_ids, :, 0, 1] = torch.sin(self.tangents[env_ids])
            self._target_rotations[env_ids, :, 1, 0] = -torch.sin(self.tangents[env_ids])
            self._target_rotations[env_ids, :, 1, 1] = torch.cos(self.tangents[env_ids])
            self._num_goals[env_ids] = self.num_goals[env_ids] - 1
            if self._task_cfg.spawn_at_random_gate:
                self._target_index[env_ids] = self._rng.sample_integer_torch(
                    torch.zeros_like(env_ids), self.num_goals[env_ids] - 1, (1,), env_ids
                ).long()
            else:
                self._target_index[env_ids] = 0

            self.num_generations += 1
        else:
            # Per-env random tracks
            num_goals = len(env_ids)
            points, tangents, num_goals = self._track_generator.generate_tracks_points_non_fixed_points(env_ids)
            self._target_positions[env_ids] = points + self._env_origins[env_ids, :2].unsqueeze(1)
            self._target_heading[env_ids] = tangents
            self._target_rotations[env_ids, :, 0, 0] = torch.cos(tangents)
            self._target_rotations[env_ids, :, 0, 1] = torch.sin(tangents)
            self._target_rotations[env_ids, :, 1, 0] = -torch.sin(tangents)
            self._target_rotations[env_ids, :, 1, 1] = torch.cos(tangents)
            self._num_goals[env_ids] = num_goals - 1
            if self._task_cfg.spawn_at_random_gate:
                self._target_index[env_ids] = self._rng.sample_integer_torch(
                    torch.zeros_like(env_ids), num_goals - 1, (1,), env_ids
                ).long()
            else:
                self._target_index[env_ids] = 0

        self._num_resets_per_env[env_ids] += 1

    def set_initial_conditions(self, env_ids: torch.Tensor) -> None:
        """
        Generates the initial conditions for the robots. The initial conditions are randomized based on the
        environment actions. The generation of the initial conditions is done so that if the environment actions are
        close to 0 then the task is the easiest, if they are close to 1 then the task is hardest. The configuration of
        the task defines the ranges within which the initial conditions are randomized.

        Args:
            env_ids (torch.Tensor): The ids of the environments.
            step (int, optional): The current step. Defaults to 0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The initial position,
            orientation and velocity of the robot."""

        num_resets = len(env_ids)

        # Randomizes the initial pose of the platform
        initial_pose = torch.zeros((num_resets, 7), device=self._device, dtype=torch.float32)

        # Position, the position is picked in a cone behind the first target.
        r = (
            self._gen_actions[env_ids, 6] * (self._task_cfg.spawn_max_dist - self._task_cfg.spawn_min_dist)
            + self._task_cfg.spawn_min_dist
        )
        # We add pi to make sure the robot is behind the first target
        # if the env_action is 0, then the robot is perfectly aligned with the target
        theta = (
            (
                self._gen_actions[env_ids, 7]
                * (self._task_cfg.spawn_max_cone_spread - self._task_cfg.spawn_min_cone_spread)
                + self._task_cfg.spawn_min_cone_spread
            )
            * sample_random_sign((num_resets,), device=self._device)
            + self._target_heading[env_ids, self._target_index[env_ids]]
            + math.pi
        )
        initial_pose[:, 0] = r * torch.cos(theta) + self._target_positions[env_ids, self._target_index[env_ids], 0]
        initial_pose[:, 1] = r * torch.sin(theta) + self._target_positions[env_ids, self._target_index[env_ids], 1]
        initial_pose[:, 2] = self._robot_origins[env_ids, 2]

        # Orientation
        delta_heading = (
            (
                self._gen_actions[env_ids, 8]
                * (self._task_cfg.spawn_max_heading_dist - self._task_cfg.spawn_min_heading_dist)
            )
            + self._task_cfg.spawn_min_heading_dist
        ) * sample_random_sign((num_resets,), device=self._device)
        theta = delta_heading + self._target_heading[env_ids, self._target_index[env_ids]]
        initial_pose[:, 3] = torch.cos(theta * 0.5)
        initial_pose[:, 6] = torch.sin(theta * 0.5)

        # Randomizes the velocity of the platform
        initial_velocity = torch.zeros((num_resets, 6), device=self._device, dtype=torch.float32)

        # Linear velocity
        velocity_norm = (
            self._gen_actions[env_ids, 9] * (self._task_cfg.spawn_max_lin_vel - self._task_cfg.spawn_min_lin_vel)
            + self._task_cfg.spawn_min_lin_vel
        )
        theta = torch.rand((num_resets,), device=self._device) * 2 * math.pi
        initial_velocity[:, 0] = velocity_norm * torch.cos(theta)
        initial_velocity[:, 1] = velocity_norm * torch.sin(theta)

        # Angular velocity of the platform
        angular_velocity = (
            self._gen_actions[env_ids, 10] * (self._task_cfg.spawn_max_ang_vel - self._task_cfg.spawn_min_ang_vel)
            + self._task_cfg.spawn_min_ang_vel
        )
        initial_velocity[:, 5] = angular_velocity

        # Apply to articulation
        self._robot.set_pose(initial_pose, self._env_ids[env_ids])
        self._robot.set_velocity(initial_velocity, self._env_ids[env_ids])

    def create_task_visualization(self) -> None:
        """Adds the visual marker to the scene.

        There are 3 makers for the goals:
        - The next goal is marked in red.
        - The passed goals are marked in grey.
        - The current goals are marked in green.

        They are represented by a pin with an arrow on top of it. The arrow's orientation is the same as the goal's.
        The pin is here to precisely visualize the position of the goal.

        The robot is represented by a diamond with two colors. The colors are used to represent the orientation of the
        robot. The green color represents the front of the robot, and the red color represents the back of the robot.
        """

        # Define the visual markers and edit their properties
        gate_marker_cfg = GATE_2D_CFG.copy()
        gate_marker_cfg.markers["gate_2d"].width = self._task_cfg.gate_width
        gate_marker_cfg.markers["gate_2d"].height = self._task_cfg.gate_width
        gate_marker_cfg_grey = GATE_2D_CFG.copy()
        gate_marker_cfg_grey.markers["gate_2d"].width = self._task_cfg.gate_width
        gate_marker_cfg_grey.markers["gate_2d"].height = self._task_cfg.gate_width
        gate_marker_cfg_grey.markers["gate_2d"].corner_material.diffuse_color = (
            0.5,
            0.5,
            0.5,
        )
        gate_marker_cfg_grey.markers["gate_2d"].front_material.diffuse_color = (
            0.5,
            0.5,
            0.5,
        )
        gate_marker_cfg_grey.markers["gate_2d"].back_material.diffuse_color = (
            0.5,
            0.5,
            0.5,
        )
        robot_marker_cfg = BICOLOR_DIAMOND_CFG.copy()
        gate_marker_cfg.prim_path = f"/Visuals/Command/task_{self._task_uid}/next_goal"
        gate_marker_cfg_grey.prim_path = f"/Visuals/Command/task_{self._task_uid}/passed_goals"
        robot_marker_cfg.prim_path = f"/Visuals/Command/task_{self._task_uid}/robot_pose"
        # We should create only one of them.
        self.next_goals_visualizer = VisualizationMarkers(gate_marker_cfg)
        self.passed_goals_visualizer = VisualizationMarkers(gate_marker_cfg_grey)
        self.robot_pos_visualizer = VisualizationMarkers(robot_marker_cfg)
        # Add track visualizer
        self.track_visualizer = VisualizationMarkers(
            TRACK_CFG.replace(prim_path=f"/Visuals/Command/task_{self._task_uid}/track")
        )

    def update_task_visualization(self) -> None:
        """Updates the visual marker to the scene.
        This implements the logic to check to use the appropriate colors. It also discards duplicate goals.
        Since the number of goals is flexible, but the length of the tensor is fixed, we need to discard some of the
        goals."""

        # For the reminder of the goals, we need to check if they are passed or not.
        # We do this by iterating over the 2nd axis of the self._target_position tensor.
        # The update time scales linearly with the number of goals.
        passed_goals_pos_list = []
        passed_goals_heading_list = []
        next_goals_pos_list = []
        next_goals_heading_list = []
        for i in range(self._task_cfg.max_num_corners):
            ok_goals = self._num_goals >= i
            passed_goals = torch.logical_and(self._target_index > i, ok_goals)
            passed_goals_pos_list.append(self._target_positions[passed_goals, i])
            passed_goals_heading_list.append(self._target_heading[passed_goals, i])
            next_goals = torch.logical_and(self._target_index <= i, ok_goals)
            next_goals_pos_list.append(self._target_positions[next_goals, i])
            next_goals_heading_list.append(self._target_heading[next_goals, i])
        passed_pos_goals = torch.cat(passed_goals_pos_list, dim=0)
        passed_heading_goals = torch.cat(passed_goals_heading_list, dim=0)
        next_pos_goals = torch.cat(next_goals_pos_list, dim=0)
        next_heading_goals = torch.cat(next_goals_heading_list, dim=0)

        # Assign the positions to the visual markers (They need to be dynamically allocated)
        # Under the hood, these are converted to numpy arrays, so that's definitely a waste, but since it's
        # only for visualization, it's not a big deal.
        passed_goals_pos = torch.zeros((passed_pos_goals.shape[0], 3), device=self._device)
        passed_goals_pos[:, :2] = passed_pos_goals
        next_goals_pos = torch.zeros((next_pos_goals.shape[0], 3), device=self._device)
        next_goals_pos[:, :2] = next_pos_goals

        # Assign the orientations to the visual markers (They need to be dynamically allocated)
        passed_goals_quat = torch.zeros((passed_heading_goals.shape[0], 4), device=self._device)
        passed_goals_quat[:, 0] = torch.cos(passed_heading_goals * 0.5)
        passed_goals_quat[:, 3] = torch.sin(passed_heading_goals * 0.5)
        next_goals_quat = torch.zeros((next_heading_goals.shape[0], 4), device=self._device)
        next_goals_quat[:, 0] = torch.cos(next_heading_goals * 0.5)
        next_goals_quat[:, 3] = torch.sin(next_heading_goals * 0.5)

        # If there are no goals of a given type, we should hide the markers.
        if passed_goals_pos.shape[0] == 0:
            self.passed_goals_visualizer.set_visibility(False)
        else:
            self.passed_goals_visualizer.set_visibility(True)
            self.passed_goals_visualizer.visualize(passed_goals_pos, orientations=passed_goals_quat)
        if next_goals_pos.shape[0] == 0:
            self.next_goals_visualizer.set_visibility(False)
        else:
            self.next_goals_visualizer.set_visibility(True)
            self.next_goals_visualizer.visualize(next_goals_pos, orientations=next_goals_quat)

        # Update the robot visualization. TODO Ideally we should lift the diamond a bit.
        self.robot_pos_visualizer.visualize(self._robot.root_link_pos_w[self._env_ids], self._robot.root_link_quat_w[self._env_ids])

        # --- Track visualization ---
        # Visualize the track for all environments
        all_translations = []
        all_orientations = []
        all_scales = []
        all_marker_indices = []
        for env_idx in range(self._target_positions.shape[0]):
            num_goals = int(self._num_goals[env_idx].item()) + 1
            points = self._target_positions[env_idx, :num_goals]  # (num_goals, 2)
            if points.shape[0] > 1:
                # Convert to 3D
                points3d = torch.zeros((points.shape[0], 3), device=points.device)
                points3d[:, :2] = points
                points_np = points3d.cpu().numpy()
                # --- CLOSE THE LOOP ---
                if points_np.shape[0] > 2:
                    points_np = np.concatenate([points_np, points_np[:1]], axis=0)
                num_segments = points_np.shape[0] - 1
                translations = np.zeros((num_segments, 3))
                orientations = np.zeros((num_segments, 4))
                scales = np.ones((num_segments, 3))
                marker_indices = np.zeros(num_segments, dtype=int)
                for i in range(num_segments):
                    p0 = points_np[i]
                    p1 = points_np[i + 1]
                    mid = (p0 + p1) / 2
                    vec = p1 - p0
                    length = np.linalg.norm(vec)
                    if length > 1e-6:
                        z_axis = np.array([0, 0, 1])
                        axis = np.cross(z_axis, vec)
                        axis_norm = np.linalg.norm(axis)
                        if axis_norm < 1e-6:
                            if np.dot(z_axis, vec) > 0:
                                quat = np.array([1, 0, 0, 0])
                            else:
                                quat = np.array([0, 1, 0, 0])
                        else:
                            axis = axis / axis_norm
                            angle = np.arccos(np.dot(z_axis, vec) / length)
                            qw = np.cos(angle / 2)
                            qx, qy, qz = axis * np.sin(angle / 2)
                            quat = np.array([qw, qx, qy, qz])
                    else:
                        quat = np.array([1, 0, 0, 0])
                    translations[i] = mid
                    orientations[i] = quat
                    scales[i] = [1.0, 1.0, length]
                all_translations.append(translations)
                all_orientations.append(orientations)
                all_scales.append(scales)
                all_marker_indices.append(marker_indices)
        if all_translations:
            translations = np.concatenate(all_translations, axis=0)
            orientations = np.concatenate(all_orientations, axis=0)
            scales = np.concatenate(all_scales, axis=0)
            marker_indices = np.concatenate(all_marker_indices, axis=0)
            self.track_visualizer.set_visibility(True)
            self.track_visualizer.visualize(translations=translations, orientations=orientations, scales=scales, marker_indices=marker_indices)
        else:
            self.track_visualizer.set_visibility(False)