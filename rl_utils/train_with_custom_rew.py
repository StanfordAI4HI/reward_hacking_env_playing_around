"""
Training script using CleanRL's PPO implementation with PufferLib vectorization.
Based on CleanRL's ppo_continuous_action.py: https://docs.cleanrl.dev/rl-algorithms/ppo/
"""

import os
import random
import warnings
import json
from datetime import datetime

import numpy as np
import torch
import pufferlib
import pufferlib.vector
from pufferlib.emulation import GymnasiumPufferEnv
from pufferlib import pufferl

from rl_utils.env_setups import setup_pandemic_env, setup_glucose_env, setup_traffic_env
from models.default_policy import Policy
from utils.logging_utils import NumpyEncoder, evaluate_with_pandemic_metrics

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore", category=UserWarning)


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
    
    # Create log file path with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "result_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{default_config['train']['name']}_{timestamp}.jsonl")
    
    print(f"Logging results to: {log_file}")
    print(f"# of epochs: {default_config['train']['update_epochs']}")
    
    for epoch in range(default_config['train']['update_epochs']):
        print(f"on epoch {epoch}")
        
        # Use custom evaluation for pandemic to track reward breakdowns
        if env_name == "pandemic":
            eval_metrics = evaluate_with_pandemic_metrics(trainer, env_name, num_episodes=5)
            print(f"Evaluation metrics: {eval_metrics}")
        else:
            eval_metrics = trainer.evaluate()
        
        logs = trainer.train()
        
        # Prepare log entry with epoch info and metrics
        log_entry = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "eval_metrics": eval_metrics if eval_metrics else {},
            "train_logs": logs if logs else {}
        }
        
        # Append to log file (JSONL format - one JSON object per line)
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry, cls=NumpyEncoder) + "\n")
    
    print("Done!")
    print(f"Results saved to: {log_file}")
    envs.close()

if __name__ == "__main__":
    main()
    

