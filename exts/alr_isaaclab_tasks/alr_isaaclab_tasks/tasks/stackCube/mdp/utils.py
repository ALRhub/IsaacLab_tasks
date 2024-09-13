from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.sensors import ContactSensor
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import matrix_from_quat

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

CUBE_HALF_SIZE = 0.025


def is_cube_on_top(
    env: ManagerBasedRLEnv,
    pos_threshold: float = 0.005,
    top_cube_cfg: SceneEntityCfg = SceneEntityCfg("top_cube"),
    bot_cube_cfg: SceneEntityCfg = SceneEntityCfg("bot_cube"),
):
    top_cube: RigidObject = env.scene[top_cube_cfg.name]
    bot_cube: RigidObject = env.scene[bot_cube_cfg.name]

    top_cube_pos = top_cube.data.root_pos_w
    bot_cube_pos = bot_cube.data.root_pos_w
    cube_offset = top_cube_pos - bot_cube_pos

    xy_mask = torch.linalg.norm(cube_offset[..., :2], dim=1) <= CUBE_HALF_SIZE + pos_threshold
    z_mask = torch.abs(cube_offset[..., 2] - CUBE_HALF_SIZE * 2) <= pos_threshold
    cubes_stacked = torch.logical_and(xy_mask, z_mask)
    return cubes_stacked


def is_grasping(
    env: ManagerBasedRLEnv,
    min_force: float = 0.5,
    max_grasp_angle: float = 85,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_forces_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
):
    # robot: RigidObject = env.scene[robot_cfg.name]

    contact_forces: ContactSensor = env.scene[contact_forces_cfg.name]
    left_contact_force = contact_forces.data.force_matrix_w[:, 0, 0, :]
    # right_contact_force = contact_forces.data.force_matrix_w[:, 1, 0, :]
    left_force = torch.linalg.norm(left_contact_force, dim=1)
    # right_force = torch.linalg.norm(right_contact_force, dim=1)

    # left_finger_mat = torch.zeros((env.num_envs, 4, 4), device=robot.device)
    # left_finger_mat[..., :3, :3] = matrix_from_quat(robot.data.body_state_w[:, -2, 3:7])
    # left_finger_mat[..., :3, 3] = robot.data.body_state_w[:, -2, :3]
    # left_finger_mat[..., 3, 3] = 1

    # right_finger_mat = torch.zeros((env.num_envs, 4, 4), device=robot.device)
    # right_finger_mat[..., :3, :3] = matrix_from_quat(robot.data.body_state_w[:, -1, 3:7])
    # right_finger_mat[..., :3, 3] = robot.data.body_state_w[:, -1, :3]
    # right_finger_mat[..., 3, 3] = 1
    # # y axis direction, TODO test -1
    # left_direction = left_finger_mat[..., :3, 1]
    # right_direction = -right_finger_mat[..., :3, 1]

    # left_angle = _compute_angle_between(left_direction, left_contact_force)
    # right_angle = _compute_angle_between(right_direction, right_contact_force)

    # left_mask = torch.logical_and(left_force >= min_force, torch.rad2deg(left_angle) <= max_grasp_angle)
    # right_mask = torch.logical_and(right_force >= min_force, torch.rad2deg(right_angle) <= max_grasp_angle)

    # # TODO replace
    # is_grasping = torch.logical_and(left_mask, right_mask)
    # is_grasping = (left_force + right_force) > 0
    is_grasping = (left_force) > 0

    # print(f"LEFT FORCE: {left_force}")
    # print(f"LEFT DIRECTION: {left_direction}")
    # print(f"LEFT ANGLE: {left_angle}")
    # print(f"LEFT MASK: {left_mask}")
    # print(f"IS GRASPING: {is_grasping}")
    # print("==================================================")

    return is_grasping


def _compute_angle_between(x1: torch.Tensor, x2: torch.Tensor):
    """Compute angle (radian) between two torch tensors"""
    x1, x2 = _normalize_vector(x1), _normalize_vector(x2)
    dot_prod = torch.clip(torch.einsum("ij,ij->i", x1, x2), -1, 1)
    return torch.arccos(dot_prod)


def _normalize_vector(x: torch.Tensor, eps=1e-6):
    """normalizes a given torch tensor x and if the norm is less than eps, set the norm to 0"""
    norm = torch.linalg.norm(x, axis=1)
    norm[norm < eps] = 1
    norm = 1 / norm
    return torch.multiply(x, norm[:, None])
