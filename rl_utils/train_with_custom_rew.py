"""
Training script using CleanRL's PPO implementation with PufferLib vectorization.
Based on CleanRL's ppo_continuous_action.py: https://docs.cleanrl.dev/rl-algorithms/ppo/
"""

import os
import random
import warnings

import numpy as np
import torch
import pufferlib
import pufferlib.vector
from pufferlib.emulation import GymnasiumPufferEnv
from pufferlib import pufferl

from rl_utils.env_setups import setup_pandemic_env, setup_glucose_env, setup_traffic_env
from models.default_policy import Policy

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore", category=UserWarning)



def evaluate_with_pandemic_metrics(trainer, env_name, num_episodes=10):
    """
    Custom evaluation function that tracks pandemic-specific metrics
    similar to RLlib's PandemicCallbacks.
    
    Tracks:
    - true_reward and proxy_reward per episode
    - Breakdown of reward components (true_rew_breakdown, proxy_rew_breakdown)
    - Correlation between true and proxy rewards
    - Modified reward if present
    """
    if env_name != "pandemic":
        # For non-pandemic envs, use default evaluate
        return trainer.evaluate()
    
    # Manually run evaluation episodes to collect detailed metrics
    eval_env = trainer.vecenv.driver_env
    policy = trainer.policy
    device = next(policy.parameters()).device
    
    episode_metrics = []
    
    for ep in range(num_episodes):
        obs, info = eval_env.reset()
        done = False
        truncated = False
        
        # Episode accumulators
        total_true_reward = 0
        total_proxy_reward = 0
        total_modified_reward = 0
        timestep_true_rewards = []
        timestep_proxy_rewards = []
        true_rew_breakdown_accum = {}
        proxy_rew_breakdown_accum = {}
        step_count = 0
        
        while not (done or truncated):
            # Get action from policy
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, value = policy.forward_eval(obs_tensor)
                # For discrete actions, sample from logits
                if hasattr(eval_env.single_action_space, 'n'):
                    action_probs = torch.softmax(logits, dim=-1)
                    action = torch.argmax(action_probs, dim=-1).cpu().numpy()[0]
                else:
                    action = logits.cpu().numpy()[0]
            
            # Step environment
            obs, reward, done, truncated, info = eval_env.step(action)
            step_count += 1
            
            # Extract metrics from info
            if isinstance(info, dict):
                true_rew = info.get("true_rew", 0)
                proxy_rew = info.get("proxy_rew", 0)
                modified_rew = info.get("modified_reward", 0)
                
                total_true_reward += true_rew
                total_proxy_reward += proxy_rew
                total_modified_reward += modified_rew
                timestep_true_rewards.append(true_rew)
                timestep_proxy_rewards.append(proxy_rew)
                
                # Accumulate breakdown components
                if "true_rew_breakdown" in info:
                    for rew_id, rew_val in info["true_rew_breakdown"].items():
                        if rew_id not in true_rew_breakdown_accum:
                            true_rew_breakdown_accum[rew_id] = 0
                        true_rew_breakdown_accum[rew_id] += rew_val
                
                if "proxy_rew_breakdown" in info:
                    for rew_id, rew_val in info["proxy_rew_breakdown"].items():
                        if rew_id not in proxy_rew_breakdown_accum:
                            proxy_rew_breakdown_accum[rew_id] = 0
                        proxy_rew_breakdown_accum[rew_id] += rew_val
        
        # Compute episode metrics
        ep_metrics = {
            "episode_length": step_count,
            "true_reward": total_true_reward,
            "proxy_reward": total_proxy_reward,
            "modified_reward": total_modified_reward,
        }
        
        # Add averaged breakdown components
        for rew_id, rew_val in true_rew_breakdown_accum.items():
            ep_metrics[f"true_{rew_id}"] = rew_val / step_count if step_count > 0 else 0
        
        for rew_id, rew_val in proxy_rew_breakdown_accum.items():
            ep_metrics[f"proxy_{rew_id}"] = rew_val / step_count if step_count > 0 else 0
        
        # Compute correlation between true and proxy rewards
        if len(timestep_true_rewards) > 1:
            try:
                corr = np.corrcoef(timestep_true_rewards, timestep_proxy_rewards)[0, 1]
                ep_metrics["corr_btw_rewards"] = corr if not np.isnan(corr) else 0
            except:
                ep_metrics["corr_btw_rewards"] = 0
        else:
            ep_metrics["corr_btw_rewards"] = 0
        
        episode_metrics.append(ep_metrics)
    
    # Aggregate metrics across episodes
    aggregated_metrics = {}
    if episode_metrics:
        for key in episode_metrics[0].keys():
            values = [ep[key] for ep in episode_metrics]
            aggregated_metrics[f"eval/{key}_mean"] = np.mean(values)
            aggregated_metrics[f"eval/{key}_std"] = np.std(values)
    
    return aggregated_metrics


def create_env(env_config, wrap_env=True):
    """Create environment based on the specified type."""
    env_type = env_config.get("env_type")
    reward_fun_type = env_config.get("reward_fun_type")
   
    if env_type == "pandemic":
        # return setup_pandemic_env(env_config, wrap_env)
        env = setup_pandemic_env(env_config, wrap_env)
        # Test that the environment works
        obs, info = env.reset()  # This should set _current_sim_time
        return env
    if env_type == "glucose":
        return setup_glucose_env(env_config, wrap_env)
    if env_type == "traffic":
        return setup_traffic_env(env_config, wrap_env)
    raise ValueError(f"Unknown environment type: {env_type}")


def main():
    #python3 -m rl_utils.train_with_custom_rew  --vec.seed 0 --vec.num-envs 10 --vec.num-workers 2 --train.name pandemic_gt_reward_fn --train.update-epochs 100

    default_config  = pufferl.load_config('default')
    
    env_name = default_config["train"]["name"].split("_")[0]
    reward_fun_type = default_config["train"]["name"].replace(env_name+"_", "")

    # Seeding
    random.seed(default_config["vec"]["seed"])
    np.random.seed(default_config["vec"]["seed"])
    torch.manual_seed(default_config["vec"]["seed"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print (f"reward_fun_type: {reward_fun_type}")
    print (f"env_name: {env_name}")
    # env_name = 
    
    # Get environment-specific config and override PPO params from config
    if env_name == "pandemic":
        from utils.pandemic_config import get_config as get_env_config
        from utils.pandemic_config import get_ppo_config
        env_config = get_env_config()
        env_config["env_type"] = "pandemic"
        env_config["reward_fun_type"] = reward_fun_type
        
        # Get PPO config for architecture (no parameters needed for pandemic)
        ppo_config = get_ppo_config()
        
    elif env_name == "traffic":
        from utils.traffic_config import get_config as get_env_config
        from utils.traffic_config import get_ppo_config
        env_config = get_env_config()
        env_config["env_type"] = "traffic"
        env_config["reward_fun_type"] = reward_fun_type
        ppo_config = get_ppo_config()
        
    elif env_name == "glucose":
        from utils.glucose_config import get_config as get_env_config
        from utils.glucose_config import get_ppo_config
        env_config = get_env_config()
        env_config["env_type"] = "glucose"
        env_config["gt_reward_fn"] = "magni_rew"
        env_config["reward_fun_type"] = reward_fun_type
        
        ppo_config = get_ppo_config()
    # Create vectorized environments with PufferLib
    def make_env_fn(**kwargs):
        # print ("env config:")
        # print (env_config)
        return GymnasiumPufferEnv(create_env(env_config, wrap_env=True))
    
    envs = pufferlib.vector.make(
        make_env_fn,
        num_envs=default_config["vec"]["num_envs"],
        backend="Multiprocessing",
        num_workers=default_config["vec"]["num_workers"]
    )

    ppo_config["num_workers"] = default_config["vec"]["num_workers"]
    ppo_config["num_envs"] = default_config["vec"]["num_envs"]
    ppo_config["seed"] = default_config["vec"]["seed"]

    # Create policy using config
    if env_name == "glucose":
        from models.glucose_policy import GlucosePolicy
        # For Box action space, use np.prod to get the total number of action dimensions
        policy = GlucosePolicy(ppo_config["model"]).to(device)
    else:   
        policy = Policy(envs.driver_env, ppo_config).to(device)    

    print ("action space: ", envs.driver_env.action_space)

    for k in default_config["train"].keys():
        if k in ppo_config:
            default_config["train"][k] = ppo_config[k]
    
    default_config["train"]["env"] = env_name
    trainer = pufferl.PuffeRL(default_config["train"], envs, policy)
    print ("# of epochs: ", default_config['train']['update_epochs'])
    for epoch in range(default_config['train']['update_epochs']):
        print ("on epoch ", epoch)
        
        # Use custom evaluation for pandemic to track reward breakdowns
        if env_name == "pandemic":
            eval_metrics = evaluate_with_pandemic_metrics(trainer, env_name, num_episodes=5)
            print(f"Evaluation metrics: {eval_metrics}")
        else:
            trainer.evaluate()
        
        logs = trainer.train()
    
    print("Done!")
    envs.close()

if __name__ == "__main__":
    main()
    

