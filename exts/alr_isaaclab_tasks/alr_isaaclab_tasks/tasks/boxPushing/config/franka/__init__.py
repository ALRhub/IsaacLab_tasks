# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import gymnasium as gym

from . import agents, joint_effort_env_cfg, mp_wrapper

import fancy_gym.envs.registry as fancy_gym_registry

##
# Register Gym environments.
##

##
# Joint Effort Control
##

# Dense reward
for reward_type in ["Dense", "TemporalSparse"]:
    for rl_type in ["step", "bbrl"]:
        gym.register(
            id=f"Isaac-Box-Pushing-{reward_type}-{rl_type}-Franka-v0",
            entry_point="alr_isaaclab_tasks.tasks.boxPushing.box_pushing_env:BoxPushingEnv",
            kwargs={
                "env_cfg_entry_point": getattr(
                    joint_effort_env_cfg, f"FrankaBoxPushingEnvCfg_{reward_type}"
                ),
                "rsl_rl_cfg_entry_point": getattr(
                    agents.rsl_rl_cfg, f"BoxPushingPPORunnerCfg_{rl_type}"
                ),
                "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
                "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg_{rl_type}.yaml",
            },
            disable_env_checker=True,
        )

    fancy_gym_registry.upgrade(
        id=f"Isaac-Box-Pushing-{reward_type}-{rl_type}-Franka-v0",
        mp_wrapper=mp_wrapper.FrankaBoxPushingMPWrapper,
    )
