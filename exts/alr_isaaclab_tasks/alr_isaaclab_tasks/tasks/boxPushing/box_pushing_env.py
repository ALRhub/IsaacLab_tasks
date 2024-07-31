# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: np.ndarray | None
from __future__ import annotations

import gymnasium as gym

from omni.isaac.lab.envs.manager_based_rl_env import ManagerBasedRLEnv
from alr_isaaclab_tasks.tasks.boxPushing.box_pushing_env_cfg import BoxPushingEnvCfg


class BoxPushingEnv(ManagerBasedRLEnv):

    def __init__(self, cfg: BoxPushingEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the environment.

        Args:
            cfg: The configuration for the environment.
            render_mode: The render mode for the environment. Defaults to None, which
                is similar to ``"human"``.
        """
        # initialize the base class to setup the scene.
        super().__init__(cfg, render_mode, **kwargs)

        action_dim = sum(self.action_manager.action_term_dim)
        self.single_action_space = gym.spaces.Box(low=-1, high=1, shape=(action_dim,))

        # batch the spaces for vectorized environments
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
