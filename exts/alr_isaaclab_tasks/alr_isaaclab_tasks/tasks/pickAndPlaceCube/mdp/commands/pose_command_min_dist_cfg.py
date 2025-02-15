# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

from omni.isaac.lab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg
from omni.isaac.lab.utils import configclass

from .pose_command_min_dist import UniformPoseWithMinDistCommand


@configclass
class UniformPoseWithMinDistCommandCfg(UniformPoseCommandCfg):
    """Configuration for uniform pose command generator."""

    class_type: type = UniformPoseWithMinDistCommand

    box_name: str = MISSING
    """Name of the box for which the commands are generated."""

    min_dist: float = 0.0
    """Minimal distance to the box to respect during the sampling"""
