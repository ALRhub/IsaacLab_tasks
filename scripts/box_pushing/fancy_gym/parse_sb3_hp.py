from typing import Any

from stable_baselines3.common.utils import constant_fn
import torch.nn as nn  # noqa: F401


def process_sb3_cfg(cfg: dict) -> dict:
    """Convert simple YAML types to Stable-Baselines classes/components.

    Args:
        cfg: A configuration dictionary.

    Returns:
        A dictionary containing the converted configuration.

    Reference:
        https://github.com/DLR-RM/rl-baselines3-zoo/blob/0e5eb145faefa33e7d79c7f8c179788574b20da5/utils/exp_manager.py#L358
    """

    def update_dict(hyperparams: dict[str, Any]) -> dict[str, Any]:
        for key, value in hyperparams.items():
            if isinstance(value, dict):
                update_dict(value)
            else:
                if key in [
                    "policy_kwargs",
                    "replay_buffer_class",
                    "replay_buffer_kwargs",
                ]:
                    hyperparams[key] = eval(value)
                elif key in [
                    "learning_rate",
                    "clip_range",
                    "clip_range_vf",
                    "delta_std",
                ]:
                    if isinstance(value, str):
                        _, initial_value = value.split("_")
                        initial_value = float(initial_value)
                        hyperparams[key] = (
                            lambda progress_remaining: progress_remaining
                            * initial_value
                        )
                    elif isinstance(value, (float, int)):
                        # Negative value: ignore (ex: for clipping)
                        if value < 0:
                            continue
                        hyperparams[key] = constant_fn(float(value))
                    else:
                        raise ValueError(f"Invalid value for {key}: {hyperparams[key]}")

        return hyperparams

    # parse agent configuration and convert to classes
    return update_dict(cfg)
