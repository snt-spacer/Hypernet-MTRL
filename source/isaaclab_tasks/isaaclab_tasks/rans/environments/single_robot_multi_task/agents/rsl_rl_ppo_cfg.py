# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class SingleRobotMultiTaskPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 4000
    save_interval = 200
    experiment_name = "multitask_deep_net_4x64m128_10xseeds"
    logger = "wandb"
    wandb_kwargs = {
        "project": "multitask_deep_net_4x64m128_10xseeds",
        "entity": "spacer-rl",
        "group": "zeroG",
    }
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[64, 64, 128, 64, 64],
        critic_hidden_dims=[64, 64, 128, 64, 64],
        activation="elu",
        clip_actions=True,
        clip_actions_range=[0, 1],
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive", #adaptive, fixed
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
