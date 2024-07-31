# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.sensors import FrameTransformerCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.utils import configclass

from alr_isaaclab_tasks.tasks.boxPushing import mdp
from alr_isaaclab_tasks.tasks.boxPushing.box_pushing_env_cfg import (
    BoxPushingEnvCfg,
    DenseRewardCfg,
    TemporalSparseRewardCfg,
)


##
# Pre-defined configs
##
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip
from alr_isaaclab_tasks.tasks.boxPushing.assets.franka import FRANKA_PANDA_CFG  # isort: skip


@configclass
class FrankaBoxPushingEnvCfg(BoxPushingEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        # self.actions.body_joint_pos = mdp.JointPositionActionCfg(
        #     asset_name="robot",
        #     joint_names=["panda_joint.*"],
        #     use_default_offset=True,
        # )
        self.actions.body_joint_effort = mdp.JointEffortActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=10.0, debug_vis=True
        )
        self.commands.object_pose.body_name = "panda_hand"

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.41, 0.0, 0.055],
                rot=[1, 0, 0, 0],
            ),
            spawn=UsdFileCfg(
                usd_path=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../assets/box.usda"),
                scale=(0.001, 0.001, 0.001),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.27],
                    ),
                ),
            ],
        )


@configclass
class FrankaBoxPushingEnvCfg_Dense(FrankaBoxPushingEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.rewards = DenseRewardCfg()


@configclass
class FrankaBoxPushingEnvCfg_TemporalSparse(FrankaBoxPushingEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.rewards = TemporalSparseRewardCfg()


@configclass
class FrankaBoxPushingEnvCfg_PLAY(FrankaBoxPushingEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
