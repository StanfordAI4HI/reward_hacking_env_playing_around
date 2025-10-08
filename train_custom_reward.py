"""
Train PPO with custom reward function (e.g., written by LLM) while tracking ground truth reward.

This script enables reward hacking experiments by:
1. Training with a custom reward function (the one being optimized)
2. Tracking ground truth reward in parallel (for measuring reward hacking)
3. Logging both metrics to detect when custom reward increases but GT reward stagnates
"""

import argparse
import datetime
import importlib.util
import pickle
from pathlib import Path
from typing import Optional

import ray
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env

from rl_utils.reward_wrapper import RewardFunction, RewardWrapper


def load_reward_from_file(file_path: str, class_name: Optional[str] = None) -> RewardFunction:
    """Load reward function class from Python file (e.g., written by LLM).

    Args:
        file_path: Path to Python file containing reward function
        class_name: Optional specific class name to load

    Returns:
        Instance of RewardFunction subclass
    """
    spec = importlib.util.spec_from_file_location("custom_reward", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if class_name:
        return getattr(module, class_name)()

    # Auto-detect: find first class with calculate_reward method
    # (more robust than checking inheritance since RewardFunction is defined in multiple places)
    for name in dir(module):
        obj = getattr(module, name)
        if (isinstance(obj, type) and
            hasattr(obj, 'calculate_reward') and
            callable(getattr(obj, 'calculate_reward')) and
            not name.startswith('_') and
            name != 'RewardFunction'):  # Skip the base class itself
            return obj()

    raise ValueError(f"No RewardFunction found in {file_path}")


class DualRewardCallback(DefaultCallbacks):
    """Callback to track both custom reward (for training) and GT reward (for measuring hacking)."""

    def on_episode_step(self, *, episode, **kwargs):
        info = episode.last_info_for()

        # Track custom reward (what PPO optimizes - the reward function written by LLM)
        episode.user_data.setdefault("custom_reward_sum", 0)
        episode.user_data["custom_reward_sum"] += episode.prev_reward_for()

        # Track GT reward (for measuring hacking)
        if "ground_truth_reward" in info:
            episode.user_data.setdefault("gt_reward_sum", 0)
            episode.user_data["gt_reward_sum"] += info["ground_truth_reward"]

    def on_episode_end(self, *, episode, **kwargs):
        episode.custom_metrics["custom_reward"] = episode.user_data.get("custom_reward_sum", 0)
        episode.custom_metrics["gt_reward"] = episode.user_data.get("gt_reward_sum", 0)


class DualTrackingRewardWrapper(RewardWrapper):
    """Extended RewardWrapper that tracks both custom reward (for training) and GT reward (for measuring hacking)."""

    def __init__(self, env, env_name: str, custom_reward_fn: RewardFunction, gt_reward_fn: RewardFunction):
        super().__init__(env, env_name, custom_reward_fn)
        self.gt_reward_fn = gt_reward_fn

    def step(self, action):
        # Save previous observation before super().step() updates it
        prev_obs = self.last_obs

        obs_np, reward, terminated, truncated, info = super().step(action)

        # Compute GT reward for tracking
        # After super().step(), self.last_obs has been updated to current observation
        gt_reward = self.gt_reward_fn.calculate_reward(prev_obs, action, self.last_obs)
        info["ground_truth_reward"] = gt_reward

        return obs_np, reward, terminated, truncated, info


def create_env_with_custom_reward(env_config: dict):
    """Create environment with custom reward function (e.g., written by LLM) + GT reward for tracking.

    Args:
        env_config: Dictionary containing env_type, custom_reward_file, and environment-specific config

    Returns:
        Environment wrapped with DualTrackingRewardWrapper
    """
    env_type = env_config["env_type"]
    custom_reward = load_reward_from_file(env_config["custom_reward_file"])

    # Load GT reward based on env type
    if env_type == "pandemic":
        from pandemic_simulator.environment.pandemic_env import PandemicPolicyGymEnv

        from reward_functions.pandemic_gt_rew_fns import (
            PanEtAlTruePandemicRewardFunction,
        )

        base_env = PandemicPolicyGymEnv(env_config)
        gt_reward = PanEtAlTruePandemicRewardFunction()
        return DualTrackingRewardWrapper(base_env, "pandemic", custom_reward, gt_reward)

    elif env_type == "traffic":
        from flow.utils.registry import make_create_env

        from reward_functions.traffic_gt_rew_fns import PanEtAlTrueTrafficRewardFunction

        create_env_fn, _ = make_create_env(
            params=env_config["flow_params_default"],
            reward_specification=env_config["reward_specification"],
            reward_fun=env_config["reward_fun"],
            reward_scale=env_config["reward_scale"],
        )
        base_env = create_env_fn()
        gt_reward = PanEtAlTrueTrafficRewardFunction()
        return DualTrackingRewardWrapper(base_env, "traffic", custom_reward, gt_reward)

    else:
        raise ValueError(f"Unknown env type: {env_type}")


def main():
    parser = argparse.ArgumentParser(description="Train PPO with custom reward function")
    parser.add_argument(
        "--env-type",
        type=str,
        required=True,
        choices=["pandemic", "traffic"],
        help="Type of environment to train on"
    )
    parser.add_argument(
        "--custom-reward-file",
        type=str,
        required=True,
        help="Path to Python file defining custom reward function (e.g., written by LLM)"
    )
    parser.add_argument("--num-workers", type=int, default=2, help="Number of rollout workers")
    parser.add_argument("--num-gpus", type=int, default=0, help="Number of GPUs")
    parser.add_argument("--num-iterations", type=int, default=100, help="Number of training iterations")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--checkpoint-freq", type=int, default=10, help="Save checkpoint every N iterations")

    args = parser.parse_args()

    # Initialize Ray
    ray.init()

    # Get environment-specific config
    if args.env_type == "pandemic":
        from utils.pandemic_config import get_config, get_ppo_config

        env_config = get_config()
        env_config["env_type"] = "pandemic"
        env_config["custom_reward_file"] = args.custom_reward_file

        register_env("custom_reward_env", create_env_with_custom_reward)

        ppo_config = get_ppo_config(env_config, args.num_gpus, args.seed, args.num_workers)
        ppo_config = ppo_config.environment("custom_reward_env", env_config=env_config)

    elif args.env_type == "traffic":
        from utils.traffic_config import get_config, get_ppo_config

        env_config = get_config()
        env_config["env_type"] = "traffic"
        env_config["custom_reward_file"] = args.custom_reward_file

        register_env("custom_reward_env", create_env_with_custom_reward)

        ppo_config = get_ppo_config("custom_reward_env", env_config, args.num_gpus, args.seed, args.num_workers)
        ppo_config = ppo_config.environment("custom_reward_env", env_config=env_config)

    # Enable dual reward tracking callback
    ppo_config = ppo_config.callbacks(DualRewardCallback)

    # Build algorithm
    algo = ppo_config.build()

    # Create save directory
    save_root = Path("logs") / "custom_reward" / args.env_type / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_root.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_file = save_root / "config.txt"
    with open(config_file, 'w') as f:
        f.write(f"Environment: {args.env_type}\n")
        f.write(f"Custom reward file: {args.custom_reward_file}\n")
        f.write(f"Num workers: {args.num_workers}\n")
        f.write(f"Num iterations: {args.num_iterations}\n")
        f.write(f"Seed: {args.seed}\n")

    print("\nTraining configuration:")
    print(f"  Environment: {args.env_type}")
    print(f"  Custom reward file: {args.custom_reward_file}")
    print(f"  Save directory: {save_root}")
    print(f"  Num workers: {args.num_workers}")
    print(f"  Num iterations: {args.num_iterations}\n")

    # Training loop
    for iteration in range(args.num_iterations):
        # Save checkpoint periodically
        if iteration % args.checkpoint_freq == 0:
            checkpoint = algo.save(checkpoint_dir=str(save_root / f"checkpoint_{iteration}"))
            print(f"Saved checkpoint to {checkpoint}")

        # Train one iteration
        result = algo.train()

        # Extract metrics
        custom_reward_mean = result.get("custom_metrics", {}).get("custom_reward_mean", float('nan'))
        gt_reward_mean = result.get("custom_metrics", {}).get("gt_reward_mean", float('nan'))
        episode_reward_mean = result.get("env_runners", {}).get("episode_reward_mean", float('nan'))

        print(f"Iteration {iteration + 1}/{args.num_iterations}")
        print(f"  Custom Reward (optimized): {custom_reward_mean:.2f}")
        print(f"  GT Reward (tracked):       {gt_reward_mean:.2f}")
        print(f"  Episode Reward Mean:       {episode_reward_mean:.2f}")

        # Save results
        with open(save_root / f"iter_{iteration}.pkl", 'wb') as f:
            pickle.dump(result, f)

    # Save final checkpoint
    final_checkpoint = algo.save(checkpoint_dir=str(save_root / "checkpoint_final"))
    print(f"\nTraining complete! Final checkpoint: {final_checkpoint}")
    print(f"Results saved to: {save_root}")

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
