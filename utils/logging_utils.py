import json
import numpy as np
import torch

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


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
