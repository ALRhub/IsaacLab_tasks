import fancy_gym

import yaml
import os
from datetime import datetime
import argparse


from parse_sb3_hp import process_sb3_cfg
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (
    VecNormalize,
    VecVideoRecorder,
    SubprocVecEnv,
)

import wandb
from wandb.integration.sb3 import WandbCallback

from omni.isaac.lab.utils.io import dump_yaml

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
# parser.add_argument(
#     "--seed", type=int, default=None, help="Seed used for the environment"
# )
# parser.add_argument(
#     "--max_iterations", type=int, default=None, help="RL Policy training iterations."
# )
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
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()


def main():

    agent_cfg = yaml.safe_load(
        open(
            "/home/johann/hiwi/alr_tasks/exts/alr_isaaclab_tasks/alr_isaaclab_tasks/tasks/boxPushing/config/franka/agents/sb3_ppo_cfg.yaml"
        )
    )
    agent_cfg = process_sb3_cfg(agent_cfg)
    policy_arch = agent_cfg.pop("policy")
    n_timesteps = agent_cfg.pop("n_timesteps")
    seed = agent_cfg.pop("seed")
    n_envs = args_cli.num_envs
    n_steps_per_update = agent_cfg.pop("n_steps_per_update")
    if n_steps_per_update % n_envs != 0:
        raise ValueError(
            f"n_steps_per_update: {n_steps_per_update} not dividable by n_envs: {n_envs}"
        )
    n_steps = int(n_steps_per_update / n_envs)
    agent_cfg["n_steps"] = n_steps
    n_minibatches = agent_cfg.pop("n_minibatches")
    agent_cfg["batch_size"] = int(n_envs * n_steps / n_minibatches)

    if args_cli.logger == "wandb":
        wandb.init(
            project=args_cli.log_project_name,
            group=args_cli.log_run_group,
            config={
                "policy_type": policy_arch,
                "total_timesteps": n_timesteps,
                "env_name": args_cli.task,
            },
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
        )

    # directory for logging into
    log_dir = os.path.join(
        "logs",
        "sb3",
        args_cli.task,
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # Parallel environments
    vec_env = make_vec_env(args_cli.task, vec_env_cls=SubprocVecEnv, n_envs=n_envs)
    vec_env.seed(seed)  # Explicitly seed without passing to reset

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "record_video_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
        }
        print("[INFO] Recording videos during training.")
        vec_env = VecVideoRecorder(vec_env, **video_kwargs)

    if "normalize_input" in agent_cfg:
        vec_env = VecNormalize(
            vec_env,
            training=True,
            norm_obs="normalize_input" in agent_cfg
            and agent_cfg.pop("normalize_input"),
            norm_reward="normalize_value" in agent_cfg
            and agent_cfg.pop("normalize_value"),
            clip_obs="clip_obs" in agent_cfg and agent_cfg.pop("clip_obs"),
            gamma=agent_cfg["gamma"],
            clip_reward="clip_rew" in agent_cfg and agent_cfg.pop("clip_rew"),
        )

    model = PPO(
        policy_arch,
        vec_env,
        verbose=1,
        tensorboard_log=os.path.join(log_dir, "tensorboard"),
        **agent_cfg,
    )
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

    model.learn(total_timesteps=n_timesteps, callback=checkpoint_callback)


if __name__ == "__main__":
    main()
