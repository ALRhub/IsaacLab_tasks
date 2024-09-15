# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class BoxPushingPPORunnerCfg_Step_RL_IsaacLab_HP(RslRlOnPolicyRunnerCfg):
    seed = -1
    num_steps_per_env = 24
    max_iterations = 800
    save_interval = 50
    experiment_name = "step_rl_IsaacLab_HP"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=10,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="fixed",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class BoxPushingPPORunnerCfg_Step_RL_Fancy_Gym_HP(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 100
    max_iterations = 1042
    save_interval = 50
    experiment_name = "step_rl_Fancy_Gym_HP"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 512],
        critic_hidden_dims=[512, 512],
        activation="tanh",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=10,
        num_mini_batches=40,
        learning_rate=1.0e-4,
        schedule="fixed",
        gamma=1.0,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
