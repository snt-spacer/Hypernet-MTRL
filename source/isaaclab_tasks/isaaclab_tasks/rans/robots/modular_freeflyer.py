# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch
from gymnasium import spaces, vector

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene
from isaaclab.utils import math as math_utils

from isaaclab_tasks.rans import ModularFreeflyerRobotCfg

from .robot_core import RobotCore

import numpy as np

from isaaclab.markers import ARROW_CFG, VisualizationMarkers
from isaaclab.utils.math import quat_from_angle_axis, quat_mul, quat_apply


class ModularFreeflyerRobot(RobotCore):
    def __init__(
        self,
        scene: InteractiveScene | None = None,
        robot_cfg: ModularFreeflyerRobotCfg = ModularFreeflyerRobotCfg(),
        robot_uid: int = 0,
        num_envs: int = 1,
        decimation: int = 6,
        device: str = "cuda",
    ) -> None:
        super().__init__(scene=scene, robot_uid=robot_uid, num_envs=num_envs, decimation=decimation, device=device)
        self._robot_cfg = robot_cfg
        self._dim_robot_obs = self._robot_cfg.observation_space
        self._dim_robot_act = self._robot_cfg.action_space
        self._dim_gen_act = self._robot_cfg.gen_space

        # Buffers
        self.initialize_buffers()

    @property
    def eval_data_keys(self) -> list[str]:
        """
        Returns the keys of the data used for evaluation.

        Returns:
            list[str]: The keys of the data used for evaluation."""
        
        return [
            "position",
            "heading",
            "linear_velocity",
            "angular_velocity",
            "thruster_action",
            "reaction_wheel_actions",
            "thrust_forces",
            "thrust_torques",
            "thrust_positions",
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
            "thruster_action": [f".robot_thrust_action_{i}.x.u" for i in range(self._robot_cfg.num_thrusters)],
            "reaction_wheel_actions": [".reaction_wheel_action.u"],
            "thrust_forces": [".robot_thrust_forces.x.m/s", ".robot_thrust_forces.y.m/s", ".robot_thrust_forces.z.m/s"],
            "thrust_torques": [".robot_thrust_torques.x.m/s", ".robot_thrust_torques.y.m/s", ".robot_thrust_torques.z.m/s"],
            "thrust_positions": [".robot_thrust_positions.x.m", ".robot_thrust_positions.y.m", ".robot_thrust_positions.z.m"],
            "actions": [f".robot_actions{i}.u" for i in range(self._robot_cfg.action_space)],
            "unaltered_actions": [f".robot_unaltered_actions{i}.u" for i in range(self._robot_cfg.action_space)],
        }
    
    @property
    def eval_data(self) -> dict:
        """
        Returns the data used for evaluation.

        Returns:
            dict: The data used for evaluation."""
        
        return {
            "position": self.root_pos_w,
            "heading": self.heading_w,
            "linear_velocity": self.root_lin_vel_b,
            "angular_velocity": self.root_ang_vel_b,
            "thruster_action": self._thrust_actions,
            "reaction_wheel_actions": self._reaction_wheel_actions,
            "thrust_forces": self._thrust_forces,
            "thrust_torques": self._thrust_torques,
            "thrust_positions": self._thrust_positions,
            "actions": self._actions,
            "unaltered_actions": self._unaltered_actions,
        }

    def initialize_buffers(self, env_ids=None) -> None:
        super().initialize_buffers(env_ids)
        self._previous_actions = torch.zeros(
            (self._num_envs, self._dim_robot_act),
            device=self._device,
            dtype=torch.float32,
        )
        self._thruster_action = torch.zeros(
            (self._num_envs, self._robot_cfg.num_thrusters),
            device=self._device,
            dtype=torch.float32,
        )
        self._reaction_wheel_actions = torch.zeros((self._num_envs, 1), device=self._device, dtype=torch.float32)
        self._transforms = torch.zeros((self._num_envs, 3, 4), device=self._device, dtype=torch.float32)
        self._thrust_forces = torch.zeros(
            (self._num_envs, self._robot_cfg.num_thrusters, 3),
            device=self._device,
            dtype=torch.float32,
        )
        self._thrust_torques = torch.zeros(
            (self._num_envs, self._robot_cfg.num_thrusters, 3),
            device=self._device,
            dtype=torch.float32,
        )
        self._thrust_positions = torch.zeros(
            (self._num_envs, self._robot_cfg.num_thrusters, 3),
            device=self._device,
            dtype=torch.float32,
        )

        self._thrusters_active_mask = torch.full((self._num_envs, self._robot_cfg.num_thrusters), True, dtype=torch.bool, device=self._device)
        self.random_values_thruster_activation= self._rng.sample_uniform_torch(0, 1, (self._robot_cfg.num_thrusters,), env_ids)

    def run_setup(self, robot: Articulation) -> None:
        """Loads the robot into the task. After it has been loaded."""
        super().run_setup(robot)
        # Sets the articulation to be our overloaded articulation with improved force application
        self._robot = robot

        # Get the indices of the lock joints
        self._lock_ids, _ = self._robot.find_joints(
            [self._robot_cfg.x_lock_name, self._robot_cfg.y_lock_name, self._robot_cfg.z_lock_name]
        )
        self._thrusters_ids, _ = self._robot.find_bodies("thruster_.*")
        # Get the index of the root body (used to get the state of the robot)
        self._root_idx = self._robot.find_bodies(self._robot_cfg.root_body_name)[0]
        # Get the thrust generator
        self._thrust_generator = ThrustGenerator(self._robot_cfg, self._num_envs, self._device)

        if self._robot_cfg.is_reaction_wheel:
            self._reaction_wheel_dof_idx, _ = self._robot.find_joints(self._robot_cfg.reaction_wheel_dof_name)

        self.asset = self.scene[self._robot_cfg.robot_name]
        self.default_com: torch.Tensor = self.asset.root_physx_view.get_coms().to(self._device)

    def create_logs(self) -> None:
        super().create_logs()

        self.scalar_logger.add_log("robot_state", "AVG/thrust", "mean")
        self.scalar_logger.add_log("robot_state", "AVG/action_rate", "mean")
        self.scalar_logger.add_log("robot_state", "AVG/joint_acceleration", "mean")
        self.scalar_logger.add_log("robot_reward", "AVG/action_rate", "mean")
        self.scalar_logger.add_log("robot_reward", "AVG/joint_acceleration", "mean")

    def get_observations(self, env_ids: torch.Tensor) -> torch.Tensor:
        return self._unaltered_actions[env_ids]

    def compute_rewards(self, env_ids: torch.Tensor) -> torch.Tensor:
        # TODO: DT should be factored in?

        # Compute
        action_rate = torch.sum(torch.square(self._unaltered_actions - self._previous_unaltered_actions), dim=1)
        joint_accelerations = torch.sum(torch.square(self.joint_acc), dim=1)

        # Log data
        self.scalar_logger.log("robot_state", "AVG/action_rate", action_rate)
        self.scalar_logger.log("robot_state", "AVG/joint_acceleration", joint_accelerations)
        self.scalar_logger.log("robot_reward", "AVG/action_rate", action_rate * self._robot_cfg.rew_joint_accel_scale)
        self.scalar_logger.log("robot_reward", "AVG/joint_acceleration", joint_accelerations * self._robot_cfg.rew_joint_accel_scale)

        return (
            action_rate[env_ids] * self._robot_cfg.rew_action_rate_scale
            + joint_accelerations[env_ids] * self._robot_cfg.rew_joint_accel_scale
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
    ) -> None:
        super().reset(env_ids, gen_actions, env_seeds)
        self._previous_actions[env_ids] = 0
        self._thrust_forces[env_ids] = 0
        self._thrust_positions[env_ids] = 0

    def set_initial_conditions(self, env_ids: torch.Tensor | None = None) -> None:
        # Create zero tensor
        zeros = torch.zeros(
            (len(env_ids), len(self._lock_ids)),
            device=self._device,
            dtype=torch.float32,
        )
        # Sets the joints to zero
        self._robot.set_joint_position_target(zeros, joint_ids=self._lock_ids, env_ids=env_ids)
        self._robot.set_joint_velocity_target(zeros, joint_ids=self._lock_ids, env_ids=env_ids)

        if self._robot_cfg.is_reaction_wheel:
            rw_reset = torch.zeros(
                (len(env_ids), len(self._reaction_wheel_dof_idx)),
                device=self._device,
                dtype=torch.float32,
            )
            self._robot.set_joint_velocity_target(rw_reset, joint_ids=self._reaction_wheel_dof_idx, env_ids=env_ids)


    def process_actions(self, actions: torch.Tensor) -> None:
        """Process the actions for the robot.

        Expects either binary actions: 0 or 1, or continuous actions: [0, 1].

        - First, clip the actions to the action space limits. This is done to avoid violating the robot's limits.
        - Second, apply the action randomizers to the actions. This is done to add noise to the actions, apply
          different scaling factors to the actions, etc.
        - Third, format the actions to send to the actuators.

        Args:
            actions (torch.Tensor): The actions to process."""

        # Enforce action limits at the robot level
        # actions = actions.float()  # RuntimeError: result type Float can't be cast to the desired output type long int
        # actions.clip_(min=0.0, max=1.0)
        # Store the unaltered actions, by default the robot should only observe the unaltered actions.
        self._previous_unaltered_actions = self._unaltered_actions.clone()
        self._unaltered_actions = actions.clone()

        # Apply action randomizers
        for randomizer in self.randomizers:
            randomizer.actions(dt=self.scene.physics_dt, actions=actions)

        self._previous_actions = self._actions.clone()
        self._actions = actions
        # self._actions = (actions > 0.5).float() #TODO: Remove when rsl_rl supports multidiscrete actions -> self._actions = actions
        self._actions = self._actions

        # Assumes the action space is [-1, 1]
        if self._robot_cfg.action_mode == "continuous":
            self._thrust_actions = self._actions[:, : self._robot_cfg.num_thrusters]
            self._thrust_actions = (self._thrust_actions + 1) / 2.0
            self._thrust_actions[~self._thrusters_active_mask] = 0.0
        else:
            self._thrust_actions = (self._actions[:, : self._robot_cfg.num_thrusters] > 0.0).float()
        
        if self._robot_cfg.is_reaction_wheel:
            self._reaction_wheel_actions = self._actions[:, -1].unsqueeze(-1) * self._robot_cfg.reaction_wheel_scale

        # Log data
        self.scalar_logger.log("robot_state", "AVG/thrust", torch.sum(self._thrust_actions, dim=-1))

    def compute_physics(self) -> None:
        self._thrust_positions, self._thrust_forces = self._thrust_generator.cast_actions_to_thrust(
            self._thrust_actions
        )

    def apply_actions(self) -> None:
        # Compute the physics
        super().apply_actions()
        for randomizer in self.randomizers:
            randomizer.update(dt=self.scene.physics_dt, actions=self._actions)

        self._robot.set_external_force_and_torque(
            self._thrust_forces, self._thrust_torques, positions=self._thrust_positions, body_ids=self._thrusters_ids
        )

        if self._robot_cfg.is_reaction_wheel:
            self._robot.set_joint_velocity_target(
                self._reaction_wheel_actions, joint_ids=self._reaction_wheel_dof_idx
            )

    def set_pose(
        self,
        pose: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        self._robot.write_root_pose_to_sim(pose, env_ids)

    def set_velocity(
        self,
        velocity: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        zeros = torch.zeros(
            (len(env_ids), len(self._lock_ids)),
            device=self._device,
            dtype=torch.float32,
        )
        self._robot.write_joint_state_to_sim(zeros, zeros, joint_ids=self._lock_ids, env_ids=env_ids)

    def configure_gym_env_spaces(self):
        # single_action_space = spaces.MultiDiscrete([2] * self._robot_cfg.num_thrusters)
        # action_space = vector.utils.batch_space(single_action_space, self._num_envs)

        # return single_action_space, action_space
        # TODO: Multidiscrete on rsl_rl
        single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(self._dim_robot_act,), dtype=np.float32)
        action_space = vector.utils.batch_space(single_action_space, self._num_envs)

        return single_action_space, action_space

    def create_robot_visualization(self) -> None:
        self._thruster_markers = []
        for i in range(self._robot_cfg.num_thrusters):
            cfg = ARROW_CFG.copy().replace(  # type: ignore
                prim_path=f"/Visuals/thrusters/thruster{i}"
            )
            cfg.markers["arrow"].tail_radius = 0.1
            cfg.markers["arrow"].tail_length = 1.0
            cfg.markers["arrow"].head_radius = 0.2
            cfg.markers["arrow"].head_length = 0.5

            # Use a different color for each thruster (gradient from red to blue)
            blue = i / max(self._robot_cfg.num_thrusters - 1, 1)
            cfg.markers["arrow"].visual_material = sim_utils.PreviewSurfaceCfg(
                emissive_color=(1.0 - blue, 0.2, blue)
            )

            # Create the marker and add to list
            self._thruster_markers.append(VisualizationMarkers(cfg))

    def update_robot_visualization(self) -> None:
        # This is a visualization of the force, not the trhusters, basically the arrow points in the direction of the force not the thrust
        N = self.root_link_pos_w.shape[0]       # batch size
        base_quat = self.root_link_quat_w       # (N,4)
        pos_w    = self.root_link_pos_w         # (N,3)
        thrust_actions = self._thrust_actions   # (N, num_thrusters)

        for i, marker in enumerate(self._thruster_markers):
            # —— compute world position & orientation exactly as before ——
            local_offset = self._thrust_positions[:, i]               # (N,3)
            world_offset = quat_apply(base_quat, local_offset)        # (N,3)
            thrust_pos   = pos_w + world_offset                       # (N,3)

            theta  = self._robot_cfg.thruster_transforms[i][2]
            angle  = torch.full((N,), theta, device=base_quat.device)
            axis   = torch.tensor([0.0,0.0,1.0], device=base_quat.device).repeat(N,1)
            local_q        = quat_from_angle_axis(angle, axis)        # (N,4)
            thruster_quat_w = quat_mul(base_quat, local_q)            # (N,4)

            # —— NEW: build a uniform XYZ scale from the thrust magnitude ——
            t = thrust_actions[:, i].clamp(0.0, 1.0)                  # (N,)  in [0,1]
            # if your base arrow was 1.0 long, this will shorten it down toward 0
            scales = torch.stack([t, t, t], dim=1)                    # (N,3)

            # visualize with per‑instance scales
            marker.visualize(thrust_pos, thruster_quat_w, scales=scales)


    ##
    # Derived base properties
    ##

    @property
    def heading_w(self):
        """Yaw heading of the base frame (in radians). Shape is (num_instances,).

        Note:
            This quantity is computed by assuming that the forward-direction of the base
            frame is along x-direction, i.e. :math:`(1, 0, 0)`.
        """
        forward_w = math_utils.quat_apply(self.root_link_quat_w, self._robot.data.FORWARD_VEC_B)
        return torch.atan2(forward_w[:, 1], forward_w[:, 0])

    ##
    # Derived root properties
    ##

    @property
    def root_state_w(self):
        """Root state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 13).

        The position and quaternion are of the articulation root's actor frame. Meanwhile, the linear and angular
        velocities are of the articulation root's center of mass frame.
        """
        return self._robot.data.body_state_w[:, self._root_idx]

    @property
    def root_pos_w(self) -> torch.Tensor:
        """Root position in simulation world frame. Shape is (num_instances, 3).

        This quantity is the position of the actor frame of the articulation root.
        """
        return self._robot.data.body_pos_w[:, self._root_idx].squeeze()

    @property
    def root_quat_w(self) -> torch.Tensor:
        """Root orientation (w, x, y, z) in simulation world frame. Shape is (num_instances, 4).

        This quantity is the orientation of the actor frame of the articulation root.
        """
        return self._robot.data.body_quat_w[:, self._root_idx].squeeze()

    @property
    def root_vel_w(self) -> torch.Tensor:
        """Root velocity in simulation world frame. Shape is (num_instances, 6).

        This quantity contains the linear and angular velocities of the articulation root's center of
        mass frame.
        """
        return self._robot.data.body_vel_w[:, self._root_idx].squeeze()

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        """Root linear velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the articulation root's center of mass frame.
        """
        return self._robot.data.body_lin_vel_w[:, self._root_idx].squeeze()

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        """Root angular velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the articulation root's center of mass frame.
        """
        return self._robot.data.body_ang_vel_w[:, self._root_idx].squeeze()

    @property
    def root_lin_vel_b(self) -> torch.Tensor:
        """Root linear velocity in base frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the articulation root's center of mass frame with
        respect to the articulation root's actor frame.
        """
        return math_utils.quat_rotate_inverse(self.root_quat_w, self.root_lin_vel_w)

    @property
    def root_ang_vel_b(self) -> torch.Tensor:
        """Root angular velocity in base world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the articulation root's center of mass frame with respect to the
        articulation root's actor frame.
        """
        return math_utils.quat_rotate_inverse(self.root_quat_w, self.root_ang_vel_w)

    ##
    # Derived Root Link Frame properties
    ##

    @property
    def root_link_pos_w(self) -> torch.Tensor:
        """Root link position in simulation world frame. Shape is (num_instances, 3).

        This quantity is the position of the actor frame of the root rigid body relative to the world.
        """
        return self._robot.data.body_link_pos_w[:, self._root_idx].squeeze()

    @property
    def root_link_quat_w(self) -> torch.Tensor:
        """Root link orientation (w, x, y, z) in simulation world frame. Shape is (num_instances, 4).

        This quantity is the orientation of the actor frame of the root rigid body.
        """
        return self._robot.data.body_link_quat_w[:, self._root_idx].squeeze()

    @property
    def root_link_vel_w(self) -> torch.Tensor:
        """Root linear velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's actor frame relative to the world.
        """
        return self._robot.data.body_link_vel_w[:, self._root_idx].squeeze()

    @property
    def root_link_lin_vel_w(self) -> torch.Tensor:
        """Root linear velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's actor frame relative to the world.
        """
        return self._robot.data.body_link_lin_vel_w[:, self._root_idx].squeeze()

    @property
    def root_link_ang_vel_w(self) -> torch.Tensor:
        """Root link angular velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the actor frame of the root rigid body relative to the world.
        """
        return self._robot.data.body_link_ang_vel_w[:, self._root_idx].squeeze()

    @property
    def root_link_lin_vel_b(self) -> torch.Tensor:
        """Root link linear velocity in base frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the actor frame of the root rigid body frame with respect to the
        rigid body's actor frame.
        """
        return math_utils.quat_rotate_inverse(self.root_link_quat_w, self.root_link_lin_vel_w)

    @property
    def root_link_ang_vel_b(self) -> torch.Tensor:
        """Root link angular velocity in base world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the actor frame of the root rigid body frame with respect to the
        rigid body's actor frame.
        """
        return math_utils.quat_rotate_inverse(self.root_link_quat_w, self.root_link_ang_vel_w)

    ##
    # Derived CoM frame properties
    ##

    @property
    def root_com_pos_w(self) -> torch.Tensor:
        """Root center of mass position in simulation world frame. Shape is (num_instances, 3).

        This quantity is the position of the actor frame of the root rigid body relative to the world.
        """
        return self._robot.data.body_com_pos_w[:, self._root_idx].squeeze()

    @property
    def root_com_quat_w(self) -> torch.Tensor:
        """Root center of mass orientation (w, x, y, z) in simulation world frame. Shape is (num_instances, 4).

        This quantity is the orientation of the actor frame of the root rigid body relative to the world.
        """
        return self._robot.data.body_com_quat_w[:, self._root_idx].squeeze()

    @property
    def root_com_vel_w(self) -> torch.Tensor:
        """Root center of mass velocity in simulation world frame. Shape is (num_instances, 6).

        This quantity contains the linear and angular velocities of the root rigid body's center of mass frame relative to the world.
        """
        return self._robot.data.body_com_vel_w[:, self._root_idx].squeeze()

    @property
    def root_com_lin_vel_w(self) -> torch.Tensor:
        """Root center of mass linear velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's center of mass frame relative to the world.
        """
        return self._robot.data.body_com_lin_vel_w[:, self._root_idx].squeeze()

    @property
    def root_com_ang_vel_w(self) -> torch.Tensor:
        """Root center of mass angular velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the root rigid body's center of mass frame relative to the world.
        """
        return self._robot.data.body_com_ang_vel_w[:, self._root_idx].squeeze()

    @property
    def root_com_lin_vel_b(self) -> torch.Tensor:
        """Root center of mass linear velocity in base frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's center of mass frame with respect to the
        rigid body's actor frame.
        """
        return math_utils.quat_rotate_inverse(self.root_com_quat_w, self.root_com_lin_vel_w)

    @property
    def root_com_ang_vel_b(self) -> torch.Tensor:
        """Root center of mass angular velocity in base world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the root rigid body's center of mass frame with respect to the
        rigid body's actor frame.
        """
        return math_utils.quat_rotate_inverse(self.root_com_quat_w, self.root_com_ang_vel_w)


class ThrustGenerator:
    def __init__(self, robot_cfg: ModularFreeflyerRobotCfg, num_envs: int, device: str):

        self._num_envs = num_envs
        self._device = device
        self._robot_cfg = robot_cfg

        self.initialize_buffers()
        self.get_transforms_from_cfg()

    def initialize_buffers(self):
        self._transforms2D = torch.zeros(
            (self._num_envs, self._robot_cfg.num_thrusters, 3, 3),
            dtype=torch.float,
            device=self._device,
        )
        self._transforms = torch.zeros(
            (self._num_envs, self._robot_cfg.num_thrusters, 5),
            dtype=torch.float,
            device=self._device,
        )
        self._thrust_force = torch.zeros(
            (self._num_envs, self._robot_cfg.num_thrusters),
            device=self._device,
            dtype=torch.float32,
        )
        self.unit_vector = torch.zeros(
            (self._num_envs, self._robot_cfg.num_thrusters, 2),
            device=self._device,
            dtype=torch.float32,
        )
        self.unit_vector[:, :, 0] = 1.0

    def get_transforms_from_cfg(self):
        assert (
            len(self._robot_cfg.thruster_transforms) == self._robot_cfg.num_thrusters
        ), "Number of thruster transforms does not match the number of thrusters"
        # Transforms are stored in [x,y,theta,F] format, they need to be converted to 2D transforms, and a compact representation
        for i, trsfrm in enumerate(self._robot_cfg.thruster_transforms):
            # 2D transforms used to project the forces
            self._transforms2D[:, i, 0, 0] = math.cos(trsfrm[2])
            self._transforms2D[:, i, 0, 1] = math.sin(-trsfrm[2])
            self._transforms2D[:, i, 1, 0] = math.sin(trsfrm[2])
            self._transforms2D[:, i, 1, 1] = math.cos(trsfrm[2])
            self._transforms2D[:, i, 2, 0] = trsfrm[0]
            self._transforms2D[:, i, 2, 1] = trsfrm[1]
            self._transforms2D[:, i, 2, 2] = 1.0
            # Compact transform representation to inform the network
            self._transforms[:, i, 0] = math.cos(trsfrm[2])
            self._transforms[:, i, 1] = math.sin(trsfrm[2])
            self._transforms[:, i, 2] = trsfrm[0]
            self._transforms[:, i, 3] = trsfrm[1]
            self._transforms[:, i, 4] = self._robot_cfg.thruster_max_thrust[i]
            # Maximum thrust force
            self._thrust_force[:, i] = self._robot_cfg.thruster_max_thrust[i]

    def cast_actions_to_thrust(self, actions):
        """
        Projects the forces on the platform."""

        rand_forces = actions * self._thrust_force
        # Split transforms into translation and rotation
        self.R = self._transforms2D[:, :, :2, :2].reshape(-1, 2, 2)
        self.T = self._transforms2D[:, :, 2, :2].reshape(-1, 2)
        # Create a zero tensor to add 3rd dimension
        zero = torch.zeros((self.T.shape[0], 1), device=self._device, dtype=torch.float32)
        # Generate positions
        positions = torch.cat([self.T, zero], dim=-1)
        # Project forces
        force_vector = self.unit_vector * rand_forces.view(-1, self._robot_cfg.num_thrusters, 1)
        rotated_forces = torch.matmul(self.R.reshape(-1, 2, 2), force_vector.view(-1, 2, 1))
        projected_forces = torch.cat([rotated_forces[:, :, 0], zero], dim=-1)
        return positions.reshape(-1, self._robot_cfg.num_thrusters, 3), projected_forces.reshape(
            -1, self._robot_cfg.num_thrusters, 3
        )

    @property
    def compact_transforms(self):
        return self._transforms

    @property
    def transforms2D(self):
        return self._transforms2D
