# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with Stable Baselines3.

Since Stable-Baselines3 does not support buffers living on GPU directly,
we recommend using smaller number of environments. Otherwise,
there will be significant overhead in GPU->CPU transfer.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Train an RL agent with Stable-Baselines3."
)
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--video_interval",
    type=int,
    default=2000,
    help="Interval between video recordings (in steps).",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--seed", type=int, default=None, help="Seed used for the environment"
)
parser.add_argument(
    "--max_iterations", type=int, default=None, help="RL Policy training iterations."
)
parser.add_argument(
    "--logger",
    type=str,
    default=None,
    choices={"wandb"},
    help="Logger module to use.",
)
parser.add_argument(
    "--log_project_name",
    type=str,
    default=None,
    help="Name of the logging project when using wandb or neptune.",
)
parser.add_argument(
    "--log_run_group",
    type=str,
    default=None,
    help="Name of the logging group when using wandb.",
)
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
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
from datetime import datetime

# Import extensions to set up environment tasks
import alr_isaaclab_tasks.tasks  # noqa: F401

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize

import wandb
from wandb.integration.sb3 import WandbCallback

from omni.isaac.lab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config
from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper, process_sb3_cfg


@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict
):
    """Train with stable-baselines agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = (
        args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    )
    agent_cfg["seed"] = (
        args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    )
    # max iterations for training
    if args_cli.max_iterations is not None:
        agent_cfg["n_timesteps"] = (
            args_cli.max_iterations * agent_cfg["n_steps"] * env_cfg.scene.num_envs
        )

    task_name = args_cli.task
    if args_cli.motion_primitive is not None:
        task_name = "gym_" + args_cli.motion_primitive + "/" + task_name

    # directory for logging into
    log_dir = os.path.join(
        "logs", "sb3", task_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )

    # processing task name for MP

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg)
    # read configurations about the agent-training
    policy_arch = agent_cfg.pop("policy")
    n_timesteps = agent_cfg.pop("n_timesteps")

    n_steps_per_update = agent_cfg.pop("n_steps_per_update")
    if n_steps_per_update % env_cfg.scene.num_envs != 0:
        raise ValueError(
            f"n_steps_per_update: {n_steps_per_update} not dividable by n_envs: {env_cfg.scene.num_envs}"
        )
    n_steps = int(n_steps_per_update / env_cfg.scene.num_envs)
    agent_cfg["n_steps"] = n_steps
    n_minibatches = agent_cfg.pop("n_minibatches")
    agent_cfg["batch_size"] = int(env_cfg.scene.num_envs * n_steps / n_minibatches)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )

    # create isaac environment
    env = gym.make(
        task_name, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for stable baselines
    env = Sb3VecEnvWrapper(env, args_cli.motion_primitive)

    if "normalize_input" in agent_cfg:
        env = VecNormalize(
            env,
            training=True,
            norm_obs="normalize_input" in agent_cfg
            and agent_cfg.pop("normalize_input"),
            norm_reward="normalize_value" in agent_cfg
            and agent_cfg.pop("normalize_value"),
            clip_obs="clip_obs" in agent_cfg and agent_cfg.pop("clip_obs"),
            gamma=agent_cfg["gamma"],
            clip_reward="clip_rew" in agent_cfg and agent_cfg.pop("clip_rew"),
        )

    if args_cli.logger == "wandb":
        wandb.init(
            project=args_cli.log_project_name,
            group=args_cli.log_run_group,
            config={
                "policy_type": policy_arch,
                "total_timesteps": n_timesteps,
                "env_name": task_name,
            },
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
        )

    # create agent from stable baselines
    agent = PPO(policy_arch, env, verbose=1, **agent_cfg)
    # configure the logger
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    agent.set_logger(new_logger)

    # callbacks for agent
    if args_cli.logger == "wandb":
        checkpoint_callback = WandbCallback(
            gradient_save_freq=10000,
            model_save_freq=10000,
            model_save_path=os.path.join(log_dir, "model"),
            log="all",
            verbose=2,
        )
    else:
        checkpoint_callback = CheckpointCallback(
            save_freq=10000, save_path=log_dir, name_prefix="model", verbose=2
        )
    # train the agent
    agent.learn(total_timesteps=n_timesteps, callback=checkpoint_callback)
    # save the final model
    agent.save(os.path.join(log_dir, "model"))

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
