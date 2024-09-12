# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg

import omni.isaac.lab.utils.math as math_utils
from math import sqrt


if TYPE_CHECKING:
    from omni.isaac.lab.envs.manager_based_env import ManagerBasedEnv


def reset_root_state_with_random_yaw_orientation_and_no_collision(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    cube_size: float = 0.025,
    max_sample_tries: int = 10,
    top_cube_cfg: SceneEntityCfg = SceneEntityCfg("top_cube"),
    bot_cube_cfg: SceneEntityCfg = SceneEntityCfg("bot_cube"),
):
    # extract the used quantities (to enable type-hinting)
    top_cube: RigidObject = env.scene[top_cube_cfg.name]
    bot_cube: RigidObject = env.scene[bot_cube_cfg.name]
    # get default root state
    top_root_states = top_cube.data.default_root_state[env_ids].clone()
    bot_root_states = bot_cube.data.default_root_state[env_ids].clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=top_root_states.device)
    top_rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=top_cube.device)
    bot_rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=bot_cube.device)

    top_positions = top_root_states[:, 0:3] + env.scene.env_origins[env_ids] + top_rand_samples
    bot_positions = bot_root_states[:, 0:3] + env.scene.env_origins[env_ids] + bot_rand_samples

    # checking if the cubes are colliding and resample if true
    min_dist = sqrt(cube_size**2 * 2) + 0.001
    for i in range(max_sample_tries):

        distances = torch.norm(top_positions[env_ids, :3] - bot_positions[env_ids, :3], dim=1)
        mask = distances >= min_dist

        if mask.all():
            break
        else:
            bot_rand_samples = math_utils.sample_uniform(
                ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=bot_cube.device
            )
            bot_positions[env_ids, :] = torch.where(
                mask.unsqueeze(1),
                bot_positions[env_ids, :],
                bot_root_states[:, 0:3] + env.scene.env_origins[env_ids] + bot_rand_samples[env_ids, :],
            )

    top_orientations = math_utils.random_yaw_orientation(len(env_ids), device=top_cube.device)
    bot_orientations = math_utils.random_yaw_orientation(len(env_ids), device=bot_cube.device)

    # set into the physics simulation
    top_cube.write_root_pose_to_sim(torch.cat([top_positions, top_orientations], dim=-1), env_ids=env_ids)
    bot_cube.write_root_pose_to_sim(torch.cat([bot_positions, bot_orientations], dim=-1), env_ids=env_ids)
