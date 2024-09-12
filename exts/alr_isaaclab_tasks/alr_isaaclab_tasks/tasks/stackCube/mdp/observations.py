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
from omni.isaac.lab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )

    return object_pos_b


def cube_ee_vector(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = cube_pos_w - ee_w

    return object_ee_distance


def cube_to_cube_vector(
    env: ManagerBasedRLEnv,
    top_cube_cfg: SceneEntityCfg = SceneEntityCfg("top_cube"),
    bot_cube_cfg: SceneEntityCfg = SceneEntityCfg("bot_cube"),
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    top_cube: RigidObject = env.scene[top_cube_cfg.name]
    bot_cube: RigidObject = env.scene[bot_cube_cfg.name]
    # Target object position: (num_envs, 3)
    top_cube_pos_w = top_cube.data.root_pos_w
    bot_cube_pos_w = bot_cube.data.root_pos_w
    # End-effector position: (num_envs, 3)
    # Distance of the end-effector to the object: (num_envs,)
    cube_to_cube_vector = top_cube_pos_w - bot_cube_pos_w

    return cube_to_cube_vector
