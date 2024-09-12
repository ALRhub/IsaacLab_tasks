# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import FrameTransformer
from omni.isaac.lab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

CUBE_HALF_SIZE = 0.025


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def cube_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("top_cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.linalg.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


# TODO only if grasped
def cube_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    top_cube_cfg: SceneEntityCfg = SceneEntityCfg("top_cube"),
    bot_cube_cfg: SceneEntityCfg = SceneEntityCfg("bot_cube"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    top_cube: RigidObject = env.scene[top_cube_cfg.name]
    bot_cube: RigidObject = env.scene[bot_cube_cfg.name]
    # compute the desired position in the world frame
    des_pos_w = bot_cube.data.root_pos_w
    des_pos_w[:, 2] += CUBE_HALF_SIZE * 2
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.linalg.norm(top_cube.data.root_pos_w - des_pos_w, dim=1)
    # rewarded if the object is lifted above the threshold
    return 1 - torch.tanh(distance / std)


def cube_static(
    env: ManagerBasedRLEnv,
    std: float,
    top_cube_cfg: SceneEntityCfg = SceneEntityCfg("top_cube"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    top_cube: RigidObject = env.scene[top_cube_cfg.name]

    lin_vel = torch.linalg.norm(top_cube.data.root_lin_vel_b, dim=1)
    ang_vel = torch.linalg.norm(top_cube.data.root_ang_vel_b, dim=1)
    return 1 - torch.tanh(lin_vel / std + ang_vel)
