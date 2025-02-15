# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import FrameTransformer
from omni.isaac.lab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul
from omni.isaac.lab.utils.array import convert_to_torch


if TYPE_CHECKING:
    from exts.alr_isaaclab_tasks.alr_isaaclab_tasks.tasks.pickAndPlaceCube.pick_and_place_cube_env import PickAndPlaceCubeEnv


# TODO resolve import
def action_scaled_l2(env: PickAndPlaceCubeEnv) -> torch.Tensor:

    return torch.sum(
        torch.square(
            env.action_manager.action * env.cfg.actions.body_joint_effort.scale
        ),
        dim=1,
    )

def object_ee_distance(
    env: PickAndPlaceCubeEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.linalg.norm(cube_pos_w - ee_w, dim=1)

    return torch.clamp(object_ee_distance, min=0.05, max=100)


def object_ee_distance_tanh(
    env: PickAndPlaceCubeEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel.
    Using a tanh-kernel results in a smooth, clamped reward value range between [0, 1]."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_grasped(
    env: PickAndPlaceCubeEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    proximity_threshold: float = 0.95,
    gripper_open_val: torch.tensor = torch.tensor([0.04]),
    gripper_threshold: float = 0.005,
) -> torch.Tensor:
    """Check if an object is grasped by the specified robot.
    The function calculates the tanh distance between the target object and the robot's end-effector as well as the amount of closing the gripper.
    """

    robot: Articulation = env.scene[robot_cfg.name]

    # Difference between the end-effector's pose and the object's pose
    pose_distance = object_ee_distance(env, object_cfg, ee_frame_cfg)

    # grasp decision is based on the pose distance as well as how much both gripper fingers are closed.
    # TODO: later include contact information for improving generalization across different objects.
    grasped = torch.logical_and(
        pose_distance > proximity_threshold,
        torch.abs(robot.data.joint_pos[:, -1] - gripper_open_val.to(env.device)) > gripper_threshold,
    )
    grasped = torch.logical_and(
        grasped, torch.abs(robot.data.joint_pos[:, -2] - gripper_open_val.to(env.device)) > gripper_threshold
    )

    return grasped


def position_command_error(
    env: PickAndPlaceCubeEnv,
    command_name: str,
    end_ep: bool = False,
    end_ep_weight: float = 100.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3], command[:, 3:]
    )
    # distance of the object to the target: (num_envs,)
    distance = torch.linalg.norm(object.data.root_pos_w - des_pos_w, dim=1)

    #  If there is a different weighting only to be computed at the end of an episode
    if end_ep:
        #  compute only for terminated envs
        terminated = env.termination_manager.dones
        distance = torch.where(terminated, distance, 0.0) * end_ep_weight

    return distance


def position_command_error_tanh(
    env: PickAndPlaceCubeEnv, 
    std: float, 
    command_name: str, 
    end_ep: bool = False,
    end_ep_weight: float = 100.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    diff_threshold: float = 0.95, 
    gripper_open_val: torch.tensor = torch.tensor([0.04]), 
    gripper_threshold: float = 0.005) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame) and maps it with a tanh kernel.
    """
    # extract the asset (to enable type hinting)
    object: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)

    # check if the object is grasped
    grasp_mask = object_grasped(env, robot_cfg, object_cfg, ee_frame_cfg, diff_threshold, gripper_open_val, gripper_threshold)

    # calculate the goal position error only if the object is grasped

    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    curr_pos_w = object.data.root_pos_w
    distance_error = torch.norm(curr_pos_w - des_pos_w, dim=1)
    distance_error = 1 - torch.tanh(distance_error / std)
    masked_distance_error = torch.where(grasp_mask, distance_error, torch.zeros_like(distance_error))

    #  If there is a different weighting only to be computed at the end of an episode
    if end_ep:
        #  compute only for terminated envs
        terminated = env.termination_manager.dones
    masked_distance_error = torch.where(terminated, masked_distance_error, 0.0) * end_ep_weight

    return masked_distance_error
            

def orientation_command_error(
    env: PickAndPlaceCubeEnv,
    command_name: str, 
    end_ep: bool = False,
    end_ep_weight: float = 100.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    diff_threshold: float = 0.95, 
    gripper_open_val: torch.tensor = torch.tensor([0.04]), 
    gripper_threshold: float = 0.005) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    object: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)

    # check if the object is grasped
    grasp_mask = object_grasped(env, robot_cfg, object_cfg, ee_frame_cfg, diff_threshold, gripper_open_val, gripper_threshold)

    # calculate the goal position error only if the object is grasped

    # obtain the desired and current orientations (in world frame)
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(robot.data.root_quat_w, des_quat_b)
    curr_quat_w = object.data.root_quat_w
    orientation_error = quat_error_magnitude(curr_quat_w, des_quat_w)
    masked_orientation_error = torch.where(grasp_mask, orientation_error, torch.zeros_like(orientation_error))

    #  If there is a different weighting only to be computed at the end of an episode
    if end_ep:
        #  compute only for terminated envs
        terminated = env.termination_manager.dones
        masked_orientation_error = torch.where(terminated, masked_orientation_error, 0.0) * end_ep_weight

    return masked_orientation_error
    

# TODO somehow asset_cfg.joint_ids is None so has to be replaced with :
def joint_pos_limits_bp(
    env: PickAndPlaceCubeEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = -(
        asset.data.joint_pos[:, :] - asset.data.soft_joint_pos_limits[:, :, 0]
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos[:, :] - asset.data.soft_joint_pos_limits[:, :, 1]
    ).clip(min=0.0)

    return torch.sum(out_of_limits, dim=1)


def end_ep_vel(
    env: PickAndPlaceCubeEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    #  retrieving velocity
    asset: Articulation = env.scene[asset_cfg.name]
    vel = torch.abs(asset.data.joint_vel[:, :7])

    reward = torch.linalg.norm(vel, dim=1)

    #  compute only for terminated envs
    terminated = env.termination_manager.dones
    reward = torch.where(terminated, reward, 0.0)

    return reward


def joint_vel_limits_bp(
    env: PickAndPlaceCubeEnv,
    soft_ratio: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint velocities if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint velocity and the soft limits.

    Args:
        soft_ratio: The ratio of the soft limits to be used.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # max joint velocities
    arm_dof_vel_max = convert_to_torch(
        [2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100], device=env.device
    )
    # compute out of limits constraints
    out_of_limits = torch.abs(asset.data.joint_vel[:, :7]) - arm_dof_vel_max

    mask = out_of_limits > 0
    out_of_limits = torch.where(mask, out_of_limits, 0)

    return soft_ratio * torch.sum(out_of_limits, dim=1)


def rod_inclined_angle(
    env: PickAndPlaceCubeEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    desired_rod_quat = convert_to_torch([0.0, 1.0, 0.0, 0.0], device=env.device).repeat(
        env.num_envs, 1
    )
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_quat = ee_frame.data.target_quat_w[..., 0, :]
    rot_dist = quat_error_magnitude(desired_rod_quat, ee_quat)

    return torch.where(rot_dist > torch.pi / 4.0, rot_dist / torch.pi, rot_dist * 0)
