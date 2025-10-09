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
import sys
from pathlib import Path
from typing import Optional

import ray
from loguru import logger
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.policy.policy import Policy
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


def setup_logging(log_dir: Path, log_level: str = "INFO"):
    """Set up logging to both file and console using loguru."""
    log_file = log_dir / "training.log"

    # Remove default logger
    logger.remove()

    # Add console handler with specified log level
    logger.add(
        sys.stdout,
        format="<level>{message}</level>",
        level=log_level.upper(),
        colorize=True,
    )

    # Add file handler (captures everything including DEBUG)
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="DEBUG",
        rotation=None,
    )

    return log_file


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
    parser.add_argument("--init-checkpoint", action="store_true",
                        help="Initialize policy from BC-trained base policy checkpoint")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Console logging level (file always logs DEBUG)")

    args = parser.parse_args()

    # Create save directory first
    save_root = Path("logs") / "custom_reward" / args.env_type / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_root.mkdir(parents=True, exist_ok=True)

    # Set up logging
    log_file = setup_logging(save_root, args.log_level)
    logger.info("="*60)
    logger.info("Starting PPO training with custom reward function")
    logger.info("="*60)

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

    # Load initial checkpoint if provided
    if args.init_checkpoint:
        # Automatically find the correct base policy checkpoint based on env_type
        checkpoint_map = {
            "pandemic": "data/base_policy_checkpoints/pandemic_base_policy/checkpoint_000100",
            "traffic": "data/base_policy_checkpoints/traffic_base_policy/checkpoint_000025",
        }

        if args.env_type not in checkpoint_map:
            logger.warning(f"No base policy checkpoint found for env_type={args.env_type}")
        else:
            base_checkpoint = Path(checkpoint_map[args.env_type])

            if not base_checkpoint.exists():
                logger.warning(f"Base policy checkpoint not found at {base_checkpoint}")
            else:
                # Path to the default policy inside the checkpoint
                pol_ckpt = (
                    base_checkpoint
                    / "policies"
                    / "default_policy"
                )

                pretrained_policy = Policy.from_checkpoint(pol_ckpt)  # env-free load
                algo.get_policy().set_weights(pretrained_policy.get_weights())  # weights only
                algo.workers.sync_weights()  # push to remote workers
                logger.info(f"✔ Warm-started policy from {pol_ckpt}")

    # Save configuration
    config_file = save_root / "config.txt"
    with open(config_file, 'w') as f:
        f.write(f"Environment: {args.env_type}\n")
        f.write(f"Custom reward file: {args.custom_reward_file}\n")
        f.write(f"Num workers: {args.num_workers}\n")
        f.write(f"Num GPUs: {args.num_gpus}\n")
        f.write(f"Num iterations: {args.num_iterations}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Checkpoint frequency: {args.checkpoint_freq}\n")
        f.write(f"Init checkpoint (BC policy): {args.init_checkpoint}\n")
        f.write(f"Log level: {args.log_level}\n")

    logger.info("")
    logger.info("Training configuration:")
    logger.info(f"  Environment: {args.env_type}")
    logger.info(f"  Custom reward file: {args.custom_reward_file}")
    logger.info(f"  Save directory: {save_root}")
    logger.info(f"  Num workers: {args.num_workers}")
    logger.info(f"  Num GPUs: {args.num_gpus}")
    logger.info(f"  Num iterations: {args.num_iterations}")
    logger.info(f"  Seed: {args.seed}")
    logger.info(f"  Init checkpoint: {args.init_checkpoint}")
    logger.info("")

    # Training loop
    logger.info("Starting training loop...")
    for iteration in range(args.num_iterations):
        # Save checkpoint periodically
        if iteration % args.checkpoint_freq == 0:
            checkpoint = algo.save(checkpoint_dir=str(save_root / f"checkpoint_{iteration}"))
            logger.info(f"Saved checkpoint to {checkpoint}")

        # Train one iteration
        logger.debug(f"Starting training iteration {iteration + 1}/{args.num_iterations}")
        result = algo.train()

        # Extract metrics
        custom_reward_mean = result.get("custom_metrics", {}).get("custom_reward_mean", float('nan'))
        gt_reward_mean = result.get("custom_metrics", {}).get("gt_reward_mean", float('nan'))
        episode_reward_mean = result.get("env_runners", {}).get("episode_reward_mean", float('nan'))
        episode_len_mean = result.get("env_runners", {}).get("episode_len_mean", float('nan'))
        num_episodes = result.get("env_runners", {}).get("num_episodes", 0)

        logger.info(f"Iteration {iteration + 1}/{args.num_iterations}")
        logger.info(f"  Custom Reward (optimized): {custom_reward_mean:.2f}")
        logger.info(f"  GT Reward (tracked):       {gt_reward_mean:.2f}")
        logger.info(f"  Episode Reward Mean:       {episode_reward_mean:.2f}")
        logger.debug(f"  Episode Length Mean:       {episode_len_mean:.2f}")
        logger.debug(f"  Num Episodes:              {num_episodes}")

        # Save results
        with open(save_root / f"iter_{iteration}.pkl", 'wb') as f:
            pickle.dump(result, f)
        logger.debug(f"Saved iteration results to {save_root / f'iter_{iteration}.pkl'}")

    # Save final checkpoint
    final_checkpoint = algo.save(checkpoint_dir=str(save_root / "checkpoint_final"))
    logger.info("")
    logger.info("="*60)
    logger.info("Training complete!")
    logger.info(f"Final checkpoint: {final_checkpoint}")
    logger.info(f"Results saved to: {save_root}")
    logger.info(f"Training log: {log_file}")
    logger.info("="*60)

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
