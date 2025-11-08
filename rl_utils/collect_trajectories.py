import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import Policy
import argparse
import numpy as np
import pickle
from pathlib import Path
import os
import warnings
from pandemic_simulator.environment.pandemic_env import PandemicPolicyGymEnv
from ray.rllib.utils.checkpoints import get_checkpoint_info

from rl_utils.env_setups import setup_pandemic_env, setup_glucose_env, setup_traffic_env
from utils.pandemic_config import get_ppo_config as get_pandemic_ppo_config
from utils.glucose_config import get_ppo_config as get_glucose_ppo_config
from utils.traffic_config import get_ppo_config as get_traffic_ppo_config

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore", category=UserWarning)


def create_env(env_config, wrap_env=True):
    """Create environment based on the specified type."""
    env_type = env_config.get("env_type")
    
    if env_type == "pandemic":
        return setup_pandemic_env(env_config, wrap_env)
    if env_type == "glucose":
        return setup_glucose_env(env_config, wrap_env)
    if env_type == "traffic":
        return setup_traffic_env(env_config, wrap_env)
    raise ValueError(f"Unknown environment type: {env_type}")


def get_rewards_from_info(info, env_type):
    """Extract true_reward and modified_reward from info dict based on env_type."""
    if env_type == "pandemic":
        # Pandemic uses 'true_rew' and 'proxy_rew'
        true_reward = info.get("true_rew", 0)
    else:
        # Glucose and traffic use 'true_reward'
        true_reward = info.get("true_reward", 0)
    
    modified_reward = info.get("modified_reward", 0)
    return true_reward, modified_reward


def collect_trajectory(policy, env, env_type):
    """Collect a single trajectory.
    
    Returns:
        trajectory: List of (state, action, true_reward, modified_reward) tuples
    """
    trajectory = []
    # obs, info = env.reset()
    obs, obs_np, info = env.reset_keep_obs_obj()
    done = False
    truncated = False
    step = 0
    while not (done or truncated):
        # Get action from policy (no exploration)

       
        # action = policy.compute_single_action(obs_np, explore=False)
        action = policy.compute_single_action(obs_np)        
        
        # Take step in environment
        # next_obs, reward, done, truncated, info = env.step(action)
        next_obs, next_obs_np, reward, done, truncated, info = env.step_keep_obs_obj(action)
        
        # Extract rewards from info
        true_reward, modified_reward = get_rewards_from_info(info, env_type)
        
        # Store transition (state is the observation, not the obs wrapper object)
        trajectory.append((obs_np.copy(), action, true_reward, modified_reward))
        
        obs_np = next_obs_np
        step+= 1
        print (f"Step {step} of trajectory")
    
    return trajectory


def collect_trajectories(checkpoint_path, env_type, reward_fun_type, num_trajectories, output_dir, seed=0):
    """Collect multiple trajectories from a trained policy.
    
    Args:
        checkpoint_path: Path to the checkpoint directory
        env_type: Type of environment ('pandemic', 'glucose', or 'traffic')
        reward_fun_type: Type of reward function ('gt_reward_fn' or 'proxy_reward_fn')
        num_trajectories: Number of trajectories to collect
        output_dir: Directory to save trajectories
        seed: Random seed
    """
    # Initialize Ray
    ray.init()
    
    try:
        # Get environment config
        if env_type == "pandemic":
            from utils.pandemic_config import get_config as get_env_config
            env_config = get_env_config()
            env_config["env_type"] = "pandemic"
            env_config["reward_fun_type"] = reward_fun_type
            ppo_config = get_pandemic_ppo_config(env_config, 1, seed, 1)
            register_env("pandemic_env", create_env)
            ppo_config = ppo_config.environment("pandemic_env", env_config=env_config)
            
        elif env_type == "traffic":
            from utils.traffic_config import get_config as get_env_config
            env_config = get_env_config()
            env_config["env_type"] = "traffic"
            env_config["reward_fun_type"] = reward_fun_type
            ppo_config = get_traffic_ppo_config("traffic_env", env_config, 1, seed, 1)
            register_env("traffic_env", create_env)
            ppo_config = ppo_config.environment("traffic_env", env_config=env_config)
            
        elif env_type == "glucose":
            from utils.glucose_config import get_config as get_env_config
            env_config = get_env_config()
            env_config["env_type"] = "glucose"
            env_config["gt_reward_fn"] = "magni_rew"
            env_config["reward_fun_type"] = reward_fun_type
            ppo_config = get_glucose_ppo_config(env_config, 1, seed, 1)
            register_env("glucose_env", create_env)
            ppo_config = ppo_config.environment("glucose_env", env_config=env_config)
        else:
            raise ValueError(f"Unknown environment type: {env_type}")
        
        # # Create algorithm to load policy
        algo = ppo_config.build()
        
        # Load policy from checkpoint
        checkpoint_path = Path(checkpoint_path)
        pol_ckpt = checkpoint_path / "policies" / "default_policy"
        
        if not pol_ckpt.exists():
            raise ValueError(f"Policy checkpoint not found at {pol_ckpt}")
        
        print(f"Loading policy from {pol_ckpt}...")
        pretrained_policy = Policy.from_checkpoint(pol_ckpt)
        algo.get_policy().set_weights(pretrained_policy.get_weights())
        algo.workers.sync_weights()
        print("✔ Policy loaded successfully")

        #-------------------------------
        # checkpoint_info = get_checkpoint_info(checkpoint_path)
        # state = Algorithm._checkpoint_info_to_algorithm_state(checkpoint_info)
        # # Extract the config and override GPU settings
        # config = state["config"].copy()
        # config["num_gpus"] = 0
        # config["num_gpus_per_worker"] = 0
        # config["num_rollout_workers"] = 1
        # config["evaluation_num_workers"] = 1
        # config["input_"]=checkpoint_path
        # algo = ORPO(config=config)
        # algo.restore(checkpoint_path)
        
        # Create evaluation environment
        # eval_env = create_env(env_config, wrap_env=True)
        eval_env = PandemicPolicyGymEnv(
            config=env_config,
            obs_history_size=3,
            num_days_in_obs=8
        )
            
         # Save trajectories
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        output_file = output_path / f"trajectories_{env_type}_{reward_fun_type}_n{num_trajectories}_seed{seed}.pkl"

        # Collect trajectories
        trajectories = []
        print(f"\nCollecting {num_trajectories} trajectories...")
        for i in range(num_trajectories):
            trajectory = collect_trajectory(algo, eval_env, env_type)
            trajectories.append(trajectory)
            
            # Print statistics
            total_true_reward = sum(t[2] for t in trajectory)  # t[2] is true_reward
            total_modified_reward = sum(t[3] for t in trajectory)  # t[3] is modified_reward
            print(f"Trajectory {i+1}/{num_trajectories}: "
                  f"Length={len(trajectory)}, "
                  f"True Reward={total_true_reward:.2f}, "
                  f"Modified Reward={total_modified_reward:.2f}")
        
       
            with open(output_file, 'wb') as f:
                pickle.dump(trajectories, f)
        
        print(f"\n✔ Saved {num_trajectories} trajectories to {output_file}")
        
        # Print summary statistics
        all_true_rewards = [sum(t[2] for t in traj) for traj in trajectories]
        all_modified_rewards = [sum(t[3] for t in traj) for traj in trajectories]
        traj_lengths = [len(traj) for traj in trajectories]
        
        print("\n=== Summary Statistics ===")
        print(f"Number of trajectories: {num_trajectories}")
        print(f"Average trajectory length: {np.mean(traj_lengths):.2f} ± {np.std(traj_lengths):.2f}")
        print(f"Average true reward per trajectory: {np.mean(all_true_rewards):.2f} ± {np.std(all_true_rewards):.2f}")
        print(f"Average modified reward per trajectory: {np.mean(all_modified_rewards):.2f} ± {np.std(all_modified_rewards):.2f}")
        
        # Cleanup
        eval_env.close()
        algo.stop()
        
    finally:
        ray.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="Collect trajectories from a trained policy checkpoint"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to the checkpoint directory (containing policies/default_policy)"
    )
    parser.add_argument(
        "--env-type",
        type=str,
        required=True,
        choices=["pandemic", "glucose", "traffic"],
        help="Type of environment"
    )
    parser.add_argument(
        "--reward-fun-type",
        type=str,
        default="gt_reward_fn",
        choices=["gt_reward_fn", "proxy_reward_fn"],
        help="Type of reward function to use"
    )
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=10,
        help="Number of trajectories to collect"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="collected_trajectories",
        help="Directory to save collected trajectories"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    collect_trajectories(
        checkpoint_path=args.checkpoint_path,
        env_type=args.env_type,
        reward_fun_type=args.reward_fun_type,
        num_trajectories=args.num_trajectories,
        output_dir=args.output_dir,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

