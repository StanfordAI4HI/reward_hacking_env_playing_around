import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple, Optional
from ray.tune.registry import register_env


class GlucoseWrapper(gym.Wrapper):
    """
    Wrapper for the glucose environment that allows different termination penalties
    for training reward vs true reward evaluation.
    """
    
    def __init__(self, env: gym.Env, true_reward_termination_penalty: Optional[float] = 200_000):
        """
        Args:
            env: The wrapped glucose environment
            true_reward_termination_penalty: Termination penalty to apply only to true reward evaluation.
                                           If None, uses the same penalty as training reward.
        """
        super().__init__(env)
        self.true_reward_termination_penalty = true_reward_termination_penalty
        
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Apply different termination penalties to training reward vs true reward.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Get the original true reward before any termination penalty was applied
        original_true_reward = info.get("true_reward", 0)
        
        # Apply the true reward termination penalty if specified and different from training
        if (terminated and 
            self.true_reward_termination_penalty != self.env.termination_penalty and self.env.t < self.env.horizon):
            
            # Remove the training termination penalty from true reward if it was applied
            if self.env.termination_penalty is not None:
                original_true_reward = original_true_reward + self.env.termination_penalty
            
            # Apply the true reward specific termination penalty
            original_true_reward = original_true_reward - self.true_reward_termination_penalty
            
            # Update the info dictionary with the corrected true reward
            info["true_reward"] = original_true_reward
        # print ("called in wrapper:", ( self.env.t, self.env.horizon))
        return obs, reward, terminated, truncated, info


# def make_glucose_wrapper_env(config: Dict[str, Any]) -> GlucoseWrapper:
#     """
#     Factory function to create a wrapped glucose environment.
    
#     Args:
#         config: Environment configuration dictionary. Must contain:
#                 - All standard glucose environment parameters
#                 - true_reward_termination_penalty: Optional termination penalty for true reward
#     """
#     from glucose.bgp.simglucose.envs.simglucose_gym_env import SimglucoseEnv
    
#     # Extract the true reward termination penalty from config
#     true_reward_termination_penalty = config.pop("true_reward_termination_penalty", None)
    
#     # Create the base glucose environment
#     base_env = SimglucoseEnv(config)
    
#     # Wrap it with our custom wrapper
#     wrapped_env = GlucoseWrapper(base_env, true_reward_termination_penalty)
    
#     return wrapped_env


