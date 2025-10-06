import os
import pickle
import numpy as np
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.utils.checkpoints import get_checkpoint_info
from ray.rllib.policy.sample_batch import SampleBatch
from typing import List, Dict, Any
import torch
from ray.rllib.algorithms.ppo import PPO  # or whatever algorithm you use

from pandemic_simulator.environment.interfaces import PandemicObservation
from pandemic_simulator.environment.pandemic_env import PandemicPolicyGymEnv
from occupancy_measures.agents.orpo import ORPO
from utils.pandemic_config import get_config
from utils.trajectory_types import TrajectoryStep
from ray.tune.registry import register_env
from rl_utils.reward_wrapper import RewardWrapper
from utils.pandemic_gt_rew_fns import TruePandemicRewardFunction

def _pandemic_env_creator(cfg):
    # Merge your own config with extras you pass via env_config
    base_cfg = get_config()
    return PandemicPolicyGymEnv(
        config=base_cfg,
        obs_history_size=cfg.get("obs_history_size", 3),
        num_days_in_obs=cfg.get("num_days_in_obs", 8),
    )

def rollout_and_save(
    checkpoint_path: str,
    policy_name: str,
    save_dir: str,
    num_episodes: int = 100,
    max_steps: int = 192,  # Default pandemic horizon
    obs_history_size: int = 3,
    num_days_in_obs: int = 8
) -> None:
    """
    Roll out a policy and save trajectories to disk using pickle.
    
    Args:
        checkpoint_path: Path to the policy checkpoint
        save_dir: Directory to save trajectories
        num_episodes: Number of episodes to roll out
        max_steps: Maximum number of steps per episode
        obs_history_size: Number of timesteps of history to include
        num_days_in_obs: Number of days of data in each observation
    """
    # Extract policy name from checkpoint path
    if checkpoint_path == "pandemic-uniform-policy":
        # policy_name = "pandemic-uniform-policy"
        checkpoint_info = get_checkpoint_info("rollout_data/2025-07-10_11-40-34/checkpoint_000000/")
        input_path= "rollout_data/2025-07-10_11-40-34"  # placeholders
        restore_checkpoint_path = "rollout_data/2025-07-10_11-40-34/checkpoint_000000/"
    else:
        # policy_name = checkpoint_path.split("/")[-3]
         # Load the policy
        checkpoint_info = get_checkpoint_info(checkpoint_path)  # This returns a dict
        input_path = checkpoint_path
        #"rollout_data/"+policy_name
        restore_checkpoint_path= checkpoint_path

    print("policy_name:", policy_name)
    
    #make directory input_path
    # os.makedirs(input_path, exist_ok=True)
    
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init()

    register_env("pandemic_env", _pandemic_env_creator)

   
    state = Algorithm._checkpoint_info_to_algorithm_state(checkpoint_info)
    # Extract the config and override GPU settings
    config = state["config"].copy()
    config["num_gpus"] = 0
    config["num_gpus_per_worker"] = 0
    config["num_rollout_workers"] = 1
    config["evaluation_num_workers"] = 1
    config["input_"]=input_path

    algo = ORPO(config=config)
    # Load the checkpoint
    algo.restore(restore_checkpoint_path)
    
    # Get config and non-essential business locations
    env_config = get_config()
    
    
    # Create environment with non-essential business location tracking
    env = PandemicPolicyGymEnv(
        config=env_config,
        obs_history_size=obs_history_size,
        num_days_in_obs=num_days_in_obs
    )

    gt_reward_set = TruePandemicRewardFunction()
    gt_reward_set.set_specific_reward(int(policy_name.replace("policy_","")))
    # env = RewardWrapper(env, env_name="pandemic", reward_function=gt_reward_set)
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Roll out episodes
    for episode in range(0,num_episodes,1):
        print(f"Rolling out episode {episode + 1}/{num_episodes}")
        
        # Reset environment
        obs, obs_np, info = env.reset_keep_obs_obj()

        done = False
        steps = 0
        trajectory: List[TrajectoryStep] = []
        episode_reward = 0
        while not done:
            print (steps)
            # print(f"Unlocked businesses: {obs.unlocked_non_essential_business_locations}")
            # Get action from policy
            if checkpoint_path == "pandemic-uniform-policy":
                action = env.action_space.sample()
            elif "base_policy" in policy_name:
                action = algo.compute_single_action(obs_np, policy_id="safe_policy0")
            else:
                # print (algo.compute_single_action(obs_np, policy_id="default_policy"))
                action = algo.compute_single_action(obs_np, policy_id="default_policy")
            
            # Step environment
            next_obs, next_obs_np, reward, terminated, truncated, info = env.step_keep_obs_obj(action)
            done = terminated or truncated

            # assert False
            
            # Get true reward from info

            true_reward = info["true_rew"]
            proxy_reward = info["proxy_rew"]
            modified_reward = gt_reward_set.calculate_reward(obs, action, next_obs)
           
            # Store step
            trajectory.append(TrajectoryStep(
                obs=obs,
                action=action,
                next_obs=next_obs,
                true_reward=true_reward,
                proxy_reward=proxy_reward,
                done=done
            ))

            episode_reward += modified_reward
            
            obs = next_obs
            obs_np = next_obs_np
            steps += 1
            
        print(f"Episode {episode} reward: {episode_reward}")
        # Save trajectory using pickle with policy name in filename
        save_path = os.path.join(save_dir, f"{policy_name}_trajectory_{episode}_full.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(trajectory, f)
    
    # Shutdown Ray
    ray.shutdown()

def load_trajectory(file_path: str) -> List[TrajectoryStep]:
    """
    Load a trajectory from disk.
    
    Args:
        file_path: Path to the trajectory file
        
    Returns:
        List of TrajectoryStep objects
    """
    with open(file_path, 'rb') as f:
        trajectory = pickle.load(f)
    return trajectory

if __name__ == "__main__":
    #this is a safe policy
    # checkpoint_path_hack = "rollout_data/pandemic_base_policy/checkpoint_000100/"
    # # #this is a reward-hacking policy
    # #/next/u/stephhk/orpo/data/logs/pandemic/ORPO/proxy/model_128-128/weights_10.0_0.1_0.01/seed_0/2025-05-05_21-29-00
    # checkpoint_path_safe = "rollout_data/2025-05-05_21-29-00/checkpoint_000200/"

    # checkpoint_path_opt = "rollout_data/2025-06-24_13-49-08/checkpoint_000200/"
    
    # checkpoint_1 = "rollout_data/2025-07-09_16-58-20/checkpoint_000050/"
    # checkpoint_2 = "rollout_data/2025-07-10_11-40-34/checkpoint_000000/"
    
    # paths = ["pandemic-uniform-policy"] #checkpoint_path_hack
    with open("data/gt_rew_fn_data/pandemic_gt_rew_fns2checkpoint_paths.pkl", "rb") as f:
        paths = pickle.load(f)
    for rm_id, checkpoint_path in paths.items():
        save_dir = "rollout_data/trajectories/"
        
        # policy_name = checkpoint_path.split("/")[-3]
        # print ("checkpoint_path:", checkpoint_path)
        # print("policy_name:", policy_name)
        # continue
        print (f"=================={rm_id}========================")
        rollout_and_save(
            checkpoint_path=checkpoint_path,
            policy_name = f"policy_{rm_id}",
            save_dir=save_dir,
            num_episodes=20,
            max_steps=193
        )
        
        # Load and verify a trajectory
        # trajectory = load_trajectory(os.path.join(save_dir, "trajectory_0.pkl"))
        # print(f"Loaded trajectory with {len(trajectory)} steps")
        # print(f"First step action: {trajectory[0].action}")
        # print(f"First step reward: {trajectory[0].true_reward}")
        # print(f"First step observation shape: {trajectory[0].obs.hourly_data.shape}") 