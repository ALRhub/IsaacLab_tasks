# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import numpy as np
import fancy_gym
from scripts.utils import plot_joint_trajectories


# Import extensions to set up environment tasks
import alr_isaaclab_tasks.tasks  # noqa: F401

from omni.isaac.lab_tasks.utils import parse_env_cfg


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env_fg = gym.make("fancy/BoxPushingDense-v0", render_mode="human")
    env_fg.reset(seed=42)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    # simulate environment

    # joint position lists
    joint_positons = []
    joint_positons_fg = []

    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # compute zero actions
            actions_fg = env_fg.action_space.sample()
            actions = torch.tensor(actions_fg, device=env.unwrapped.device)
            actions = actions.repeat(args_cli.num_envs, 1)
            # print("=====")
            # print("IsaacLab: ", actions[0])
            # print("Fancy gym: ", actions_fg)

            # apply actions
            obs, _, _, _, _ = env.step(actions)
            joint_positons.append(obs["policy"][0, :7].tolist())

            obs_fg, _, terminated, truncated, _ = env_fg.step(actions_fg)
            joint_positons_fg.append(obs_fg[:7].tolist())
            env_fg.render()

            if terminated or truncated:
                env_fg.reset()

                plot_joint_trajectories(joint_positons, joint_positons_fg)
                joint_positons = []
                joint_positons_fg = []

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
