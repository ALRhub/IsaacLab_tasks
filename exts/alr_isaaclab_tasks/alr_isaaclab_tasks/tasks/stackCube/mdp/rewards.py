# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import FrameTransformer
from alr_isaaclab_tasks.tasks.stackCube.mdp.utils import is_grasping, is_cube_on_top, CUBE_HALF_SIZE

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


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
    # compute the distance to the target
    des_pos_w = bot_cube.data.root_pos_w
    des_pos_w[:, 2] += CUBE_HALF_SIZE * 2
    distance = torch.linalg.norm(top_cube.data.root_pos_w - des_pos_w, dim=1)

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


def dense_stepped_reward(
    env: ManagerBasedRLEnv,
    pos_threshold: float = 0.005,
    min_force: float = 0.5,
    max_grasp_angle: float = 85,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_forces_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    top_cube_cfg: SceneEntityCfg = SceneEntityCfg("top_cube"),
    bot_cube_cfg: SceneEntityCfg = SceneEntityCfg("bot_cube"),
):
    # rewarding the reaching of the cube
    reward = 2 * cube_ee_distance(env, std=0.2)

    # rewarding only if the cube is grasped
    placed_reward = cube_goal_distance(env, std=0.2)
    grasping = is_grasping(
        env,
        min_force,
        max_grasp_angle,
        robot_cfg,
        contact_forces_cfg,
    )
    reward[grasping] = (4 + placed_reward)[grasping]

    # rewarding 1 only if the cube is ungrasped, else reward if grippers are far from each other
    robot_joints: Articulation = env.scene[robot_cfg.name]
    gripper_width = robot_joints.data.joint_limits[:, -1, 1] * 2
    ungrasped_reward = torch.sum(robot_joints.data.joint_pos[:, -2:], dim=1) / gripper_width
    ungrasped_reward[~grasping] = 1.0

    static_reward = cube_static(env, std=0.1, top_cube_cfg=top_cube_cfg)

    is_stacked = is_cube_on_top(env, pos_threshold, top_cube_cfg, bot_cube_cfg)
    reward[is_stacked] = (6 + (ungrasped_reward + static_reward) / 2)[is_stacked]

    # reward for task success
    success = env.termination_manager.terminated
    reward[success] = 8

    return reward
