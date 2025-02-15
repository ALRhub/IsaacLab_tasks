import torch
import numpy as np

from omni.isaac.lab_tasks.utils.wrappers.mp_wrapper import MPWrapper
from omni.isaac.lab.assets import Articulation


class FrankaBoxPushingMPWrapper(MPWrapper):
    mp_config = {
        "ProDMP": {
            "black_box_kwargs": {
                "verbose": 2,
                "backend": "torch",
                "device": "cuda:0",
                "reward_aggregation": torch.sum,
            },
            "controller_kwargs": {
                "p_gains": 0.01
                * torch.tensor(
                    [120.0, 120.0, 120.0, 120.0, 50.0, 30.0, 10.0], device="cuda:0"
                ),
                "d_gains": 0.01
                * torch.tensor(
                    [10.0, 10.0, 10.0, 10.0, 6.0, 5.0, 3.0], device="cuda:0"
                ),
            },
            "trajectory_generator_kwargs": {
                # "num_dof": 7,
                "weights_scale": 0.3,
                "goal_scale": 0.3,
                "auto_scale_basis": True,
                "relative_goal": False,
                "disable_goal": False,
                "device": "cuda:0",
            },
            "basis_generator_kwargs": {
                "num_basis": 8,
                "basis_bandwidth_factor": 3,
                "num_basis_outside": 0,
                "alpha": 10,
                "device": "cuda:0",
            },
            "phase_generator_kwargs": {
                "phase_generator_type": "exp",
                "tau": 2.0,
                "alpha_phase": 3.0,
                "device": "cuda:0",
            },
        }
    }

    # Random x goal + random init pos
    @property
    def context_mask(self):
        return np.hstack(
            [
                [True] * 7,  # joints position relative
                [False] * 7,  # joints velocity
                [True] * 7,  # pose of box
                [True] * 7,  # pose of target
            ]
        )

    @property
    def current_pos(self) -> torch.Tensor:
        scene = self.env.unwrapped.scene
        asset: Articulation = scene["robot"]
        return asset.data.joint_pos[:, :7]

    @property
    def current_vel(self) -> torch.Tensor:
        scene = self.env.unwrapped.scene
        asset: Articulation = scene["robot"]
        return asset.data.joint_vel[:, :7]
