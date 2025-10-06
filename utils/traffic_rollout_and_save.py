import os
import pickle
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.utils.checkpoints import get_checkpoint_info
from typing import List
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from ray.rllib.env.multi_agent_env import make_multi_agent
import torch

from flow.utils.registry import make_create_env
from utils.traffic_config import get_config
from utils.trajectory_types import TrajectoryStep
from utils.traffic_gt_rew_fns import merge_true_reward_fn,commute_time, TrueTrafficRewardFunction, penalize_accel, penalize_headway

from flow_reward_misspecification.flow.envs.traffic_obs_wrapper import TrafficObservation

def rollout_and_save(
    checkpoint_path: str,
    save_dir: str,
    num_episodes: int = 100,
    max_steps: int = 1000,  # Default traffic horizon
) -> None:
    """
    Roll out a policy and save trajectories to disk using pickle.
    
    Args:
        checkpoint_path: Path to the policy checkpoint
        save_dir: Directory to save trajectories
        num_episodes: Number of episodes to roll out
        max_steps: Maximum number of steps per episode
    """
    # Get config and create environment
    env_configs = get_config()
    
    create_env, env_name = make_create_env(
        params=env_configs["flow_params_default"],
        reward_specification=env_configs["reward_specification"],
        reward_fun=env_configs["reward_fun"],
        reward_scale=env_configs["reward_scale"],
    )

    print("env_name:", env_name)
    
    # # Register the environment with Ray
    # register_env(env_name, make_multi_agent(create_env))
    # register_env("MergePOEnv", make_multi_agent(create_env))

    #MergePOEnv
    env = create_env()
    
    # Extract policy name from checkpoint path
    if checkpoint_path == "traiffc-uniform-policy":
        policy_name = "traiffc-uniform-policy"
        checkpoint_info = get_checkpoint_info("rollout_data/2025-06-17_16-14-06/checkpoint_000100/")  # This returns a dict
        input_path = "rollout_data_pan_et_al_rew_fns/2025-06-17_16-14-06" #placeholders
        restore_checkpoint_path = "rollout_data/2025-06-17_16-14-06/checkpoint_000100/"
    else:
        policy_name = checkpoint_path.split("/")[-3] + "/" + checkpoint_path.split("/")[-2]
        checkpoint_info = get_checkpoint_info(checkpoint_path)  # This returns a dict
        input_path = "rollout_data_pan_et_al_rew_fns/"+policy_name
        restore_checkpoint_path = checkpoint_path
    
    print("policy_name:", policy_name)
    
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init()
    
    # Load the policy
    state = Algorithm._checkpoint_info_to_algorithm_state(checkpoint_info)
    # Extract the config and override GPU settings
    config = state["config"].copy()
    config["num_gpus"] = 0
    config["num_gpus_per_worker"] = 0
    config["num_rollout_workers"] = 1
    config["evaluation_num_workers"] = 1
    config["input_"]=input_path
    config["env"] = make_multi_agent(create_env)

    # config["env_name"] = env_name
    # config["env_config"]["env_name"] = env_name
    # print(config.get("env_config"))
    # print("============")

    algo = PPO(config=config)
    # Load the checkpoint
    algo.restore(restore_checkpoint_path)
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Roll out episodes
    for episode in range(0, num_episodes, 1):
        print(f"Rolling out episode {episode + 1}/{num_episodes}")
        
        # Reset environment
        obs_np,last_info = env.reset()
        obs = TrafficObservation()
        obs.update_obs_with_sim_state(env)

        done = False
        steps = 0
        trajectory: List[TrajectoryStep] = []
        episode_reward = 0
        proxy_episode_reward = 0
        while not done:
            print(steps)
            # print (obs_np)
            # Get action from policy
            if checkpoint_path == "traiffc-uniform-policy":
                action = env.action_space.sample()
            elif "base_policy" in policy_name:
                action = algo.compute_single_action(torch.tensor(obs_np), policy_id="safe_policy0")
            else:
                action = algo.compute_single_action(torch.tensor(obs_np), policy_id="current")
            
            # Step environment
            next_obs_np, reward, terminated, truncated, info = env.step(action)
            # next_obs_np = next_obs_np[0]
            # assert isinstance(next_obs_np, np.ndarray), "next_obs_np is not a NumPy array"

            next_obs = TrafficObservation()
            next_obs.update_obs_with_sim_state(env)

            # print (len(next_obs.rl_vehicles))
            # if len(next_obs.rl_vehicles) == 0:
            #     print(last_info)
            #     print (next_obs)
            #     print (info)
            #     print (next_obs_np)
            #     # If no RL vehicles, we can skip this step
            #     continue
            # assert len(next_obs.rl_vehicles) > 0, "No RL vehicles found in the next observation"

            done = terminated or truncated
            
            # Get true reward from info
            true_reward = info["true_reward"]
            proxy_reward = info["proxy_reward"]

            # print ("true_reward:", true_reward)
            # print ("proxy_reward:", proxy_reward)
            # print ("conputed true reward:", merge_true_reward_fn(env, action))
            # print ("computed :", commute_time(env, action) + penalize_accel(env, action) + 0.1 * penalize_headway(env, action))

            rf = TrueTrafficRewardFunction()
            # print ("obs.max_speed:", obs.max_speed)
            # print ("next_obs.max_speed:", next_obs.max_speed)
            # print ("obs.max_speed:", obs.max_length)
            # print ("next_obs.max_speed:", next_obs.max_length)


            rf.sample_linear_reward(prev_obs=obs, action=action, obs=next_obs, weights=rf.weights[0])
            pred_rew = 0
            # pred_rew = rf.calculate_reward(obs,action, next_obs)
            print ("conputed true reward with custom obs:", pred_rew)
            print ("\n")
           
            # Store step
            trajectory.append(TrajectoryStep(
                obs=obs,
                action=action,
                next_obs=next_obs,
                true_reward=true_reward,
                proxy_reward=proxy_reward,
                done=done
            ))

            episode_reward += true_reward
            proxy_episode_reward += proxy_reward
            
            obs = next_obs
            obs_np = next_obs_np
            last_info = info
            steps += 1
            
        print(f"Episode {episode} return: {episode_reward}")
        print(f"Episode {episode} proxy return: {proxy_episode_reward}")
        # assert False
        # Save trajectory using pickle with policy name in filename
        save_path = os.path.join(save_dir, f"{policy_name.split('/')[0]}_trajectory_{episode}_full.pkl")
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
    checkpoint_path_hack = "rollout_data_pan_et_al_rew_fns/traffic_base_policy/checkpoint_000025/"
    #this is a reward-hacking policy
    #/next/u/stephhk/orpo/data/logs/traffic/2025-06-17_16-14-06
    checkpoint_path_safe = "rollout_data_pan_et_al_rew_fns/2025-06-17_16-14-06/checkpoint_000100/"
    
    checkpoint_path_opt= "rollout_data_pan_et_al_rew_fns/2025-06-24_13-51-42/checkpoint_000250/"

    # checkpoint_1 = "rollout_data/2025-07-09_16-57-36/checkpoint_000025/"
    # checkpoint_2 = "rollout_data/2025-07-10_13-33-33/checkpoint_000005/"
    # paths = [checkpoint_1, checkpoint_2]

    
    for checkpoint_path in [checkpoint_path_opt, checkpoint_path_hack, "traiffc-uniform-policy"]:
        save_dir = "rollout_data_pan_et_al_rew_fns/trajectories/"
        
        rollout_and_save(
            checkpoint_path=checkpoint_path,
            save_dir=save_dir,
            num_episodes=2,
        )
        
        # Load and verify a trajectory
        # trajectory = load_trajectory(os.path.join(save_dir, "trajectory_0.pkl"))
        # print(f"Loaded trajectory with {len(trajectory)} steps")
        # print(f"First step action: {trajectory[0].action}")
        # print(f"First step reward: {trajectory[0].true_reward}")
        # print(f"First step observation shape: {trajectory[0].obs.shape}")







