# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.scene import InteractiveScene
from isaaclab.utils import configclass

from isaaclab_tasks.rans.domain_randomization import RandomizationCore, RandomizationCoreCfg, Registerable
from isaaclab_tasks.rans.utils import PerEnvSeededRNG


@configclass
class CoMRandomizationCfg(RandomizationCoreCfg):
    enable: bool = False
    """Whether to enable the randomization or not."""

    randomization_modes: list = ["none"]
    """The randomization modes to apply to the center of mass of the rigid body."""

    body_name: str | None = None
    """The name of the body to randomize the center of mass of. This body must be part of the asset. The asset is set by
    the caller of the randomization."""

    # 2D or 3D randomization
    dimension: int = 2
    """The dimension of the randomization. Can be 2D or 3D. If 2D, the randomization is applied in the x and y
    directions. If 3D, the randomization is applied in the x, y, and z directions."""

    # Uniform randomization
    max_delta: float = 0.00001
    """The maximum delta to apply to the center of mass of the rigid body. (in meters) Default is 0.0001."""

    # Normal randomization
    std: float = 0.00001
    """The standard deviation of the normal distribution to sample the delta from. (in meters) Default is 0.0001."""

    # Define the size of the generative space associated with the randomization
    gen_space: int = 1 if enable else 0  # DO NOT EDIT


class CoMRandomization(Registerable, RandomizationCore):
    _cfg: CoMRandomizationCfg

    def __init__(
        self,
        cfg: CoMRandomizationCfg,
        rng: PerEnvSeededRNG,
        scene: InteractiveScene,
        asset_name: str = "MISSING",
        num_envs: int = 1,
        device: str = "cuda",
        **kwargs
    ):
        """CoM randomization class.

        Args:
            cfg: The configuration for the randomization.
            rng: The random number generator.
            scene: The scene to get the assets from.
            asset_name: The name of the asset to randomize.
            num_envs: The number of environments.
            device: The device on which the tensors are stored."""

        super().__init__(cfg, rng, scene, num_envs, device)
        self._asset_name = asset_name

    @property
    def data(self) -> dict[str, torch.Tensor]:
        """The current center of mass of the rigid body."""
        super().data
        return {"CoM": self._current_com[self._ALL_INDICES, self._body_id]}

    @property
    def data_shape(self) -> dict[str, tuple[int]]:
        super().data
        return {"CoM": (3,)}

    def check_cfg(self) -> None:
        """Check the configuration of the randomization."""

        super().check_cfg()

        assert self._cfg.dimension in [2, 3], "Invalid dimension, must be 2 or 3."

        if "uniform" in self._cfg.randomization_modes:
            assert "normal" not in self._cfg.randomization_modes, "The 'uniform' and 'normal' modes cannot be combined."

        if "normal" in self._cfg.randomization_modes:
            assert (
                "uniform" not in self._cfg.randomization_modes
            ), "The 'uniform' and 'normal' modes cannot be combined."

    def pre_setup(self) -> None:
        """Setup the com randomization."""

        super().pre_setup()

        if self._cfg.enable:
            self._asset: Articulation | RigidObject = self._scene[self._asset_name]
            self._body_id, _ = self._asset.find_bodies(self._cfg.body_name)
            self._default_com: torch.Tensor = self._asset.root_physx_view.get_coms().to(self._device)
            self._current_com = self._default_com.clone()
            self._default_inertias: torch.Tensor = self._asset.root_physx_view.get_inertias().to(self._device)
            self._current_inertias = self._default_inertias.clone()
            # self._masses: torch.Tensor = self._asset.root_physx_view.get_masses().to(self._device)

    def default_reset(self, env_ids: torch.Tensor | None = None, **kwargs) -> None:
        """The default reset function for the randomization."""

        if env_ids is None:
            env_ids = self._ALL_INDICES

        # By default, the com is set to the default com
        self._current_com[env_ids, self._body_id] = self._default_com[env_ids, self._body_id]
        # Reset inertias to default
        self._current_inertias[env_ids, self._body_id] = self._default_inertias[env_ids, self._body_id]

    def default_update(self, **kwargs) -> None:
        """The default update function for the randomization."""

        # Do nothing
        pass

    def _update_inertias(self, env_ids: torch.Tensor) -> None:
        """Update the inertia tensor based on the change in center of mass using the parallel axis theorem.
        
        Args:
            env_ids: The ids of the environments to update.
        """
        # Calculate the CoM shift vector
        com_shift = self._current_com[env_ids, self._body_id, :3] - self._default_com[env_ids, self._body_id, :3]
        
        # Get the mass for the body
        mass = self._asset.root_physx_view.get_masses().to(self._device)[env_ids, self._body_id]
        
        # Apply parallel axis theorem: I_new = I_old + m * (d^2 * I - d ⊗ d)
        # where d is the CoM shift vector, I is the identity matrix, and ⊗ is outer product
        
        # Calculate d^2 (squared magnitude of the shift)
        d_squared = torch.sum(com_shift ** 2, dim=-1, keepdim=True)  # Shape: [num_envs, 1]
        
        # Create identity matrix for each environment
        identity = torch.eye(3, device=self._device).unsqueeze(0).expand(len(env_ids), -1, -1)  # Shape: [num_envs, 3, 3]
        
        # Calculate outer product d ⊗ d for each environment
        com_shift_expanded = com_shift.unsqueeze(-1)  # Shape: [num_envs, 3, 1]
        outer_product = torch.bmm(com_shift_expanded, com_shift_expanded.transpose(-2, -1))  # Shape: [num_envs, 3, 3]
        
        # Calculate the inertia correction term: m * (d^2 * I - d ⊗ d)
        mass_expanded = mass.unsqueeze(-1).unsqueeze(-1)  # Shape: [num_envs, 1, 1]
        d_squared_expanded = d_squared.unsqueeze(-1)  # Shape: [num_envs, 1, 1]
        inertia_correction = mass_expanded * (d_squared_expanded * identity - outer_product)
        flat_inertia_correction = inertia_correction.view(len(env_ids), 9)  # Flatten to [num_envs, 9]

        # Update the inertias
        self._current_inertias[env_ids, self._body_id] = self._default_inertias[env_ids, self._body_id] + flat_inertia_correction

    def fn_on_reset_uniform(
        self, env_ids: torch.Tensor | None, gen_actions: torch.Tensor | None = None, **kwargs
    ) -> None:
        """Randomize the com of the rigid bodies uniformly. The function applies a delta to the com of the
        rigid bodies.

        Note: The generative actions are given in the range [0, 1] and are scaled to the range [-1, 1].

        Args:
            env_ids: The ids of the environments.
            gen_actions: The actions taken by the agent."""

        if gen_actions is None:
            gen_actions = self._rng.sample_uniform_torch(0, 1, (1,), env_ids)

        if env_ids is None:
            env_ids = self._ALL_INDICES

        theta = self._rng.sample_uniform_torch(-torch.pi, torch.pi, (1,), ids=env_ids)
        if self._cfg.dimension == 2:
            self._current_com[env_ids, self._body_id, 0] += (
                torch.cos(theta) * (gen_actions - 0.5) * 2 * self._cfg.max_delta
            )
            self._current_com[env_ids, self._body_id, 1] += (
                torch.sin(theta) * (gen_actions - 0.5) * 2 * self._cfg.max_delta
            )
        else:
            phi = self._rng.sample_uniform_torch(-torch.pi, torch.pi, (1,), ids=env_ids)
            self._current_com[env_ids, self._body_id, 0] += (
                torch.cos(theta) * torch.sin(phi) * (gen_actions - 0.5) * 2 * self._cfg.max_delta
            )
            self._current_com[env_ids, self._body_id, 1] += (
                torch.sin(theta) * torch.sin(phi) * (gen_actions - 0.5) * 2 * self._cfg.max_delta
            )
            self._current_com[env_ids, self._body_id, 2] += (
                torch.cos(phi) * (gen_actions - 0.5) * 2 * self._cfg.max_delta
            )
        
        # Update inertias based on the CoM change
        self._update_inertias(env_ids)

    def fn_on_reset_normal(
        self, env_ids: torch.Tensor | None = None, gen_actions: torch.Tensor | None = None, **kwargs
    ) -> None:
        """Randomize the com of the rigid bodies normally. The function applies a delta to the com of the
        rigid bodies.

        Note: The generative actions are given in the range [0, 1] and are scaled to the range [-1, 1].

        Args:
            env_ids: The ids of the environments.
            gen_actions: The actions taken by the agent."""

        if gen_actions is None:
            gen_actions = self._rng.sample_normal_torch(0, 1, (1,), env_ids)

        if env_ids is None:
            env_ids = self._ALL_INDICES

        normal = gen_actions * self._cfg.std
        theta = self._rng.sample_uniform_torch(-torch.pi, torch.pi, (1,), ids=env_ids)
        if self._cfg.dimension == 2:
            phi = self._rng.sample_uniform_torch(-torch.pi, torch.pi, (1,), ids=env_ids)
            self._current_com[env_ids, self._body_id, 0] += torch.cos(theta) * normal
            self._current_com[env_ids, self._body_id, 1] += torch.sin(theta) * normal
        else:
            phi = self._rng.sample_uniform_torch(-torch.pi, torch.pi, (1,), ids=env_ids)
            self._current_com[env_ids, self._body_id, 0] += torch.cos(theta) * torch.sin(phi) * normal
            self._current_com[env_ids, self._body_id, 1] += torch.sin(theta) * torch.sin(phi) * normal
            self._current_com[env_ids, self._body_id, 2] += torch.cos(phi) * normal
        
        # Update inertias based on the CoM change
        self._update_inertias(env_ids)

    def fn_on_update_spring(self, dt: float = 0.0, accelerations: torch.Tensor | None = None, **kwargs) -> None:
        """Change the mass of the rigid bodies by decaying it throughout the episode.

        Args:
            dt: The time step. (in seconds)"""

        raise NotImplementedError

    def apply_randomization(self, ids: torch.Tensor | None = None) -> None:
        """Updates the com and inertias of the robot.

        Args:
            ids: The ids of the environments."""

        if ids is None:
            ids_cpu = self._ALL_INDICES_CPU
        else:
            ids_cpu = ids.to("cpu")

        self._asset.root_physx_view.set_coms(self._current_com.to("cpu"), ids_cpu)
        self._asset.root_physx_view.set_inertias(self._current_inertias.to("cpu"), indices=ids_cpu)
