# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the stack task.

The functions can be passed to the :class:`omni.isaac.lab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from alr_isaaclab_tasks.tasks.stackCube.mdp.utils import is_grasping, is_cube_on_top, CUBE_HALF_SIZE

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def cubes_stacked(
    env: ManagerBasedRLEnv,
    pos_threshold: float = 0.005,
    lin_threshold: float = 1e-2,
    ang_threshold: float = 0.5,
    min_force: float = 0.5,
    max_grasp_angle: float = 85,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_forces_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    top_cube_cfg: SceneEntityCfg = SceneEntityCfg("top_cube"),
    bot_cube_cfg: SceneEntityCfg = SceneEntityCfg("bot_cube"),
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    top_cube: RigidObject = env.scene[top_cube_cfg.name]

    # checking if the cubes are stacked
    cubes_stacked = is_cube_on_top(env, pos_threshold, top_cube_cfg, bot_cube_cfg)

    # checking if the top cube is static
    lin_vel = torch.linalg.norm(top_cube.data.root_lin_vel_b, dim=1) <= lin_threshold
    ang_vel = torch.linalg.norm(top_cube.data.root_ang_vel_b, dim=1) <= ang_threshold
    top_cube_is_static = torch.logical_and(lin_vel, ang_vel)

    # check if robot is grasping
    grasping = is_grasping(
        env,
        min_force,
        max_grasp_angle,
        robot_cfg,
        contact_forces_cfg,
    )

    success = cubes_stacked * top_cube_is_static * (~grasping)

    return success.bool()
