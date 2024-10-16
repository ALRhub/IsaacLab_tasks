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
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--motion_primitive",
    type=str,
    default=None,
    help=(
        "Whether to use a motion primitive for the training. The supported ones depend in the environment: ProMP"
        " etc..."
    ),
)
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

# Import extensions to set up environment tasks
import alr_isaaclab_tasks.tasks  # noqa: F401

from omni.isaac.lab_tasks.utils import parse_env_cfg


def main():
    """Zero actions agent with Isaac Lab environment."""

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    task_name = args_cli.task
    if args_cli.motion_primitive is not None:
        task_name = "gym_" + args_cli.motion_primitive + "/" + task_name
    # create environment
    env = gym.make(task_name, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # compute zero actions
            actions = (
                torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                if not args_cli.motion_primitive
                else torch.zeros(
                    (args_cli.num_envs, env.action_space.shape[0]),
                    device=env.unwrapped.device,
                )
            )
            # apply actions
            env.step(actions)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
