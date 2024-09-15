# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from pytorch_kinematics.transforms import Transform3d
import pytorch_kinematics as pk

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import quat_from_euler_xyz, sample_uniform
from omni.isaac.lab.utils.array import convert_to_torch


if TYPE_CHECKING:
    from alr_isaaclab_tasks.tasks.boxPushing.box_pushing_env import BoxPushingEnv


def sample_box_poses(
    env: BoxPushingEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state to a random position uniformly within the given ranges.

    This function randomizes the root position of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of position and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = convert_to_torch(range_list, device=asset.device)
    rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    positions = root_states[:, :3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])

    # set into the physics simulation
    box_poses = torch.cat([positions, orientations], dim=-1)
    asset.write_root_pose_to_sim(box_poses, env_ids=env_ids)


def reset_robot_cfg_with_IK(
    env: BoxPushingEnv,
    env_ids: torch.Tensor,
):

    ######
    # IK #
    ######

    robot: Articulation = env.scene["robot"]
    boxes: Articulation = env.scene["object"]

    # processing target box poses
    PUSH_ROD_OFFSET = 0.27
    target_poses = boxes.data.body_state_w[env_ids, 0, :7] + convert_to_torch(
        [0.0, 0.0, PUSH_ROD_OFFSET, 0.0, 0.0, 0.0, 0.0], device=robot.device
    )
    target_poses[:, :3] -= robot.data.root_pos_w[env_ids]
    target_poses[:, 3:7] = convert_to_torch([0.0, 1.0, 0.0, 0.0], device=robot.device)
    target_transforms = Transform3d(pos=target_poses[:, :3], rot=target_poses[:, 3:7], device=robot.device)

    # solving IK

    ik = pk.PseudoInverseIK(
        env.chain,
        retry_configs=robot.data.joint_pos[0, :7].unsqueeze(0),  # initial config
        joint_limits=robot.data.joint_limits[0, :7],
        lr=0.2,
    )
    sol = ik.solve(target_transforms)
    joint_pos_des = sol.solutions[:, 0]

    # setting desired joint angles in simulation
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    robot_entity_cfg.resolve(env.scene)

    # Setting data into buffers (not done when sending to sim)
    robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids, env_ids=env_ids)
    robot.set_joint_velocity_target(
        torch.zeros(joint_pos_des.shape, device=robot.device), joint_ids=robot_entity_cfg.joint_ids, env_ids=env_ids
    )

    robot.write_joint_state_to_sim(
        joint_pos_des,
        torch.zeros(joint_pos_des.shape, device=robot.device),
        joint_ids=robot_entity_cfg.joint_ids,
        env_ids=env_ids,
    )


def reset_box_root_state_uniform_with_precomputed_robot_IK(
    env: BoxPushingEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):

    box_poses = sample_box_poses(env, env_ids, pose_range, asset_cfg)

    robot: Articulation = env.scene["robot"]
    box_poses[:, :3] -= robot.data.root_pos_w[env_ids]

    ######
    # IK #
    ######

    # getting IK

    joint_pos_des = env.get_ik_solutions(box_poses)

    # setting desired joint angles in simulation
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    robot_entity_cfg.resolve(env.scene)

    # Setting data into buffers (not done when sending to sim)
    robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids, env_ids=env_ids)
    robot.set_joint_velocity_target(
        torch.zeros(joint_pos_des.shape, device=robot.device), joint_ids=robot_entity_cfg.joint_ids, env_ids=env_ids
    )

    robot.write_joint_state_to_sim(
        joint_pos_des,
        torch.zeros(joint_pos_des.shape, device=robot.device),
        joint_ids=robot_entity_cfg.joint_ids,
        env_ids=env_ids,
    )
