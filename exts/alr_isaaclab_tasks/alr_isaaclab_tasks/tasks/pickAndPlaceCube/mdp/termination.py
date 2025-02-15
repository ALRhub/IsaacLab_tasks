# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

from .rewards import position_command_error, orientation_command_error

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def object_reached_goal_position(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    threshold: dict[str, float] = {"position": 0.95, "orientation": 0.02},
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    
) -> torch.Tensor:
    """Termination condition for the object reaching the goal position.

    Args:
        env: The environment.
        std: Control the saturation rate of the tanh-kernel, lower values correspond to faster saturation, high values to slower saturation.
        command_name: The name of the command that is used to control the object.
        threshold: The thresholds for the object to reach the goal pose. Defaults for position is 0.95 and for orientation 0.02 rad (~ 1.15 degrees).
        robot_cfg: The robot configuration. Defaults to SceneEntityCfg("robot").
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").

    """
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)

    # obtain the error between the desired and current object positions in the world coordinate frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    curr_pos_w = object.data.root_pos_w
    position_error = torch.norm(curr_pos_w - des_pos_w, dim=1)
    position_error = 1 - torch.tanh(position_error / std)

    return position_error > threshold["position"]


def object_reached_goal_pose(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    threshold: dict[str, float] = {"position": 0.95, "orientation": 0.02},
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"), 
) -> torch.Tensor:
    """Termination condition for the object reaching the goal pose.

    Args:
        env: The environment.
        std: Control the saturation rate of the tanh-kernel, lower values correspond to faster saturation, high values to slower saturation.
        command_name: The name of the command that is used to control the object.
        threshold: The thresholds for the object to reach the goal pose. Defaults for position is 0.95 and for orientation 0.02 rad (~ 1.15 degrees).
        robot_cfg: The robot configuration. Defaults to SceneEntityCfg("robot").
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").

    """
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)

    position_error = object_reached_goal_position(env, std, command_name, threshold, robot_cfg, object_cfg)

    # obtain the error between the desired and current object orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(robot.data.root_quat_w, des_quat_b)
    curr_quat_w = object.data.root_quat_w
    # print(f"DESIRED QUAT_W: {des_quat_b}, CURRENT QUAT_W: {curr_quat_w}")
    orientation_error = quat_error_magnitude(curr_quat_w, des_quat_w)
    
    return (position_error > threshold["position"]) and (orientation_error < threshold["orientation"])


def is_success(
    env: ManagerBasedRLEnv,
    command_name: str,
    limit_pose_dist: float = 0.05,
    limit_or_dist: float = 0.5,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    terminated = env.termination_manager.dones
    return torch.where(
        terminated,
        torch.logical_and(
            position_command_error(env, command_name, robot_cfg=robot_cfg, object_cfg=object_cfg)
            < limit_pose_dist,
            orientation_command_error(env, command_name, robot_cfg=robot_cfg, object_cfg=object_cfg)
            < limit_or_dist,
        ),
        False,
    )
