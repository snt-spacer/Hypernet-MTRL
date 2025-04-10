# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch
from gymnasium import spaces, vector

from isaaclab.actuator_force.actuator_force import PropellerActuator
from isaaclab.assets import Articulation
from isaaclab.physics.hydrodynamics import Hydrodynamics
from isaaclab.physics.hydrostatics import Hydrostatics
from isaaclab.scene import InteractiveScene

from isaaclab_tasks.rans import KingfisherRobotCfg

from .robot_core import RobotCore


class KingfisherRobot(RobotCore):

    def __init__(
        self,
        scene: InteractiveScene | None = None,
        robot_cfg: KingfisherRobotCfg = KingfisherRobotCfg(),
        robot_uid: int = 0,
        num_envs: int = 1,
        decimation: int = 5,
        device: str = "cuda",
    ):
        super().__init__(scene=scene, robot_uid=robot_uid, num_envs=num_envs, decimation=decimation, device=device)
        self._robot_cfg = robot_cfg
        self._dim_robot_obs = self._robot_cfg.observation_space
        self._dim_robot_act = self._robot_cfg.action_space
        self._dim_gen_act = self._robot_cfg.gen_space

        # Hydrostatics, Hydrodynamics, and Thruster Dynamics
        self._hydrostatics = Hydrostatics(num_envs, device, self._robot_cfg.hydrostatics_cfg)
        self._hydrodynamics = Hydrodynamics(num_envs, device, self._robot_cfg.hydrodynamics_cfg)
        # TODO: the dt should not be hardcoded here, should be passed from the simulation instead.
        self._thruster_dynamics = PropellerActuator(
            num_envs=num_envs, device=device, dt=self._physics_dt, cfg=self._robot_cfg.propeller_cfg
        )

        # Buffers
        self.initialize_buffers()

    @property
    def eval_data_keys(self) -> list[str]:
        return [
            "position", 
            "heading", 
            "linear_velocity", 
            "angular_velocity", 
            "hydrostatic_force", 
            "hydrodynamic_force",
            "thruster_force",
            "no_torque", 
            "actions",
            "unaltered_actions",
        ]
    
    @property
    def eval_data_specs(self)->dict[str, list[str]]:
        return {
            "position": [".robot_pos.x.m", ".robot_pos.y.m", ".robot_pos.z.m"],
            "heading": [".robot_heading.rad"],
            "linear_velocity": [".robot_lin_vel.x.m/s", ".robot_lin_vel.y.m/s", ".robot_lin_vel.z.m/s"],
            "angular_velocity": [".robot_ang_vel.x.rad/s", ".robot_ang_vel.y.rad/s", ".robot_ang_vel.z.rad/s"],
            "hydrostatic_force": [".robot_hydrostatic_force.x.u", ".robot_hydrostatic_force.y.u", ".robot_hydrostatic_force.z.u", ".robot_hydrostatic_torque.x.u", ".robot_hydrostatic_torque.y.u", ".robot_hydrostatic_torque.z.u"],
            "hydrodynamic_force": [".robot_hydrodynamic_force.x.u", ".robot_hydrodynamic_force.y.u", ".robot_hydrodynamic_force.z.u", ".robot_hydrodynamic_torque.x.u", ".robot_hydrodynamic_torque.y.u", ".robot_hydrodynamic_torque.z.u"],
            "thruster_force": [".robot_thruster_force.x.u", ".robot_thruster_force.y.u", ".robot_thruster_force.z.u", ".robot_thruster_torque.x.u", ".robot_thruster_torque.y.u", ".robot_thruster_torque.z.u"],
            "no_torque": [".robot_no_torque.x.u", ".robot_no_torque.y.u", ".robot_no_torque.z.u"],
            "actions": [f".robot_actions{i}.u" for i in range(self._robot_cfg.action_space)],
            "unaltered_actions": [f".robot_unaltered_actions{i}.u" for i in range(self._robot_cfg.action_space)],
        }
    
    @property
    def eval_data(self) -> dict:
        return {
            "position": self.root_pos_w,
            "heading": self.heading_w,
            "linear_velocity": self.root_lin_vel_b,
            "angular_velocity": self.root_ang_vel_b,
            "hydrostatic_force": self._hydrostatic_force,
            "hydrodynamic_force": self._hydrodynamic_force,
            "thruster_force": self._thruster_force,
            "no_torque": self._no_torque,
            "actions": self._actions,
            "unaltered_actions": self._unaltered_actions,
        }

    def initialize_buffers(self, env_ids=None):
        super().initialize_buffers(env_ids)
        self._actions = torch.zeros((self._num_envs, self._dim_robot_act), device=self._device, dtype=torch.float32)
        self._previous_actions = torch.zeros(
            (self._num_envs, self._dim_robot_act), device=self._device, dtype=torch.float32
        )
        self._hydrostatic_force = torch.zeros((self._num_envs, 1, 6), device=self._device, dtype=torch.float32)
        self._hydrodynamic_force = torch.zeros((self._num_envs, 1, 6), device=self._device, dtype=torch.float32)
        self._thruster_force = torch.zeros((self._num_envs, 1, 6), device=self._device, dtype=torch.float32)
        self._no_torque = torch.zeros(self._num_envs, 1, 3, device=self._device, dtype=torch.float32)

    def run_setup(self, robot: Articulation):
        super().run_setup(robot)
        self._root_idx, _ = self._robot.find_bodies([self._robot_cfg.root_id_name])
        self._left_thruster_idx = self._robot.find_bodies("thruster_left")[0]
        self._right_thruster_idx = self._robot.find_bodies("thruster_right")[0]
        self._thrusters_dof_idx, _ = self._robot.find_bodies(self._robot_cfg.thrusters_dof_name)  # TODO: duplicated?
        self.default_joint_pos = self._robot.data.default_joint_pos.clone()

    def create_logs(self):
        super().create_logs()

        self.scalar_logger.add_log("robot_state", "AVG/thrusters", "mean")
        self.scalar_logger.add_log("robot_state", "AVG/action_rate", "mean")
        self.scalar_logger.add_log("robot_state", "AVG/joint_acceleration", "mean")
        self.scalar_logger.add_log("robot_reward", "AVG/action_rate", "mean")
        self.scalar_logger.add_log("robot_reward", "AVG/joint_acceleration", "mean")
        self.scalar_logger.add_log("robot_reward", "AVG/energy_norm", "mean")

    def get_observations(self) -> torch.Tensor:
        return self._unaltered_actions

    def compute_rewards(self):
        # TODO: DT should be factored in?

        # Compute
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        joint_accelerations = torch.sum(torch.square(self.joint_acc), dim=1)
        energy = torch.sum(torch.square(self._actions), dim=1)
        energy_norm = energy * self._step_dt / self._robot_cfg.max_energy

        # Log data
        self.scalar_logger.log("robot_state", "AVG/action_rate", action_rate)
        self.scalar_logger.log("robot_state", "AVG/joint_acceleration", joint_accelerations)
        self.scalar_logger.log("robot_reward", "AVG/action_rate", action_rate)
        self.scalar_logger.log("robot_reward", "AVG/joint_acceleration", joint_accelerations)
        self.scalar_logger.log("robot_reward", "AVG/energy_norm", energy_norm)

        # TODO: REVIEW
        return (
            action_rate * self._robot_cfg.rew_action_rate_scale
            + joint_accelerations * self._robot_cfg.rew_joint_accel_scale
            + energy_norm * self._robot_cfg.energy_reward_scale
        )

    def get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        task_failed = torch.zeros(self._num_envs, dtype=torch.int32, device=self._device)
        task_done = torch.zeros(self._num_envs, dtype=torch.int32, device=self._device)
        return task_failed, task_done

    def reset(
        self,
        env_ids: torch.Tensor,
        gen_actions: torch.Tensor | None = None,
        env_seeds: torch.Tensor | None = None,
    ):
        super().reset(env_ids, gen_actions, env_seeds)
        self._previous_actions[env_ids] = 0
        self._thruster_dynamics.reset()

    def set_initial_conditions(self, env_ids: torch.Tensor | None = None):
        locking_joints = torch.zeros((len(env_ids), 1), device=self._device)

        self._robot.set_joint_velocity_target(locking_joints, env_ids=env_ids)
        self._robot.set_joint_position_target(locking_joints, env_ids=env_ids)

    def process_actions(self, actions: torch.Tensor):
        # Enforce action limits at the robot level
        actions = actions.float()
        actions.clip_(min=-1.0, max=1.0)
        # Store the unaltered actions, by default the robot should only observe the unaltered actions.
        self._previous_unaltered_actions = self._unaltered_actions.clone()
        self._unaltered_actions = actions.clone()

        # Apply action randomizers
        for randomizer in self.randomizers:
            randomizer.actions(dt=self.scene.physics_dt, actions=actions)

        self._previous_actions = self._actions.clone()
        self._actions = actions
        self._thruster_dynamics.set_target_cmd(self._actions)

        # TODO: fix log
        self.scalar_logger.log("robot_state", "AVG/thrusters", torch.linalg.norm(actions, dim=-1))

    def compute_physics(self):
        # Compute hydrostatics
        self._hydrostatic_force[:, 0, :] = self._hydrostatics.compute_archimedes_metacentric_local(
            self.root_pos_w, self.root_quat_w
        )

        # Compute hydrodynamics
        self._hydrodynamic_force[:, 0, :] = self._hydrodynamics.ComputeHydrodynamicsEffects(
            self.root_quat_w, self.root_vel_w
        )

        # Compute thruster dynamics.
        self._thruster_force[:, 0, :] = self._thruster_dynamics.update_forces()

    def apply_actions(self):
        # Compute the physics
        super().apply_actions()
        for randomizer in self.randomizers:
            randomizer.update(dt=self.scene.physics_dt, actions=self._actions)

        combined = self._hydrostatic_force + self._hydrodynamic_force
        self._robot.set_external_force_and_torque(combined[..., :3], combined[..., 3:], body_ids=self._root_idx)
        # only apply thruster forces if they are not zero, otherwise it disables external previous forces.
        lft_thruster_force = self._thruster_force[..., :3]
        rgt_thruster_force = self._thruster_force[..., 3:]
        if lft_thruster_force.any():
            self._robot.set_external_force_and_torque(
                lft_thruster_force, self._no_torque, body_ids=self._left_thruster_idx
            )
        if rgt_thruster_force.any():
            self._robot.set_external_force_and_torque(
                rgt_thruster_force, self._no_torque, body_ids=self._right_thruster_idx
            )

    def configure_gym_env_spaces(self):
        single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        action_space = vector.utils.batch_space(single_action_space, self._num_envs)

        return single_action_space, action_space
