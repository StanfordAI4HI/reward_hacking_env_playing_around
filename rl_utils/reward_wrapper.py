from gymnasium import Wrapper
from gymnasium.spaces import Box
import numpy as np
from typing import List, Sequence
from abc import ABCMeta, abstractmethod
import numpy as np
from bgp.simglucose.envs.glucose_obs_wrapper import GlucoseObservation
from flow_reward_misspecification.flow.envs.traffic_obs_wrapper import TrafficObservation


class RewardFunction(metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def calculate_reward(
        self, prev_obs: GlucoseObservation, action: float, obs: GlucoseObservation
    ) -> float:
        pass

class SumReward:
    def __init__(self, reward_functions: Sequence[RewardFunction], weights: dict[str, float]):
        """Initialize a sum reward function.
        
        Args:
            reward_functions: List of reward functions to combine
            weights: Dictionary mapping reward function class names to their weights
        """
        if not all(rf.__class__.__name__ in weights for rf in reward_functions):
            raise ValueError("Each reward function's class name must have a corresponding weight")
        
        self._reward_fns = reward_functions
        self._weights = weights

    def calculate_reward(
        self, prev_obs, action: int, obs
    ) -> float:
        """Calculate the weighted sum of all reward functions.
        
        Args:
            prev_obs: Previous observation
            action: Action taken
            obs: Current observation
            
        Returns:
            float: Weighted sum of all reward values
        """
        total_reward = 0.0
        for reward_fn in self._reward_fns:
            reward = reward_fn.calculate_reward(prev_obs, action, obs)
            weight = self._weights[reward_fn.__class__.__name__]
            total_reward += weight * reward
        return total_reward


class RewardWrapper(Wrapper):
    def __init__(self, env, env_name, reward_function):
        super().__init__(env)
        self.env_name = env_name
        self.reward_function = reward_function #is of type SumReward
        # print ("created reward wrapper...")
        self.ep_return = 0
      
  
    def reset(self, **kwargs):
        # print ("comptued return inside RewardWrapper:", self.ep_return)
        self.ep_return = 0
        if "pandemic" in self.env_name:
            obs, obs_np, info = self.env.reset_keep_obs_obj()
        elif "glucose" in self.env_name:
            obs_np, info = self.env.reset()
            obs = GlucoseObservation()
            obs.update_obs_with_sim_state(self.env)
        elif "traffic" in self.env_name:
            obs_np,info = self.env.reset()
            obs = TrafficObservation()
            obs.update_obs_with_sim_state(self.env)
        else:
            raise ValueError("Env not recognized")
        
        #obs_np,info = self.env.reset(**kwargs)


        self.last_obs_np = obs_np
        self.last_obs = obs
        return obs_np,info

    def step(self, action):
        # Get the original step result
        #obs_obj = self.env.
        # obs, original_reward, terminated, truncated, info = self.env.step(action)
        if "pandemic" in self.env_name:
            obs, obs_np, original_reward, terminated, truncated, info = self.env.step_keep_obs_obj(action)
            
        elif "glucose" in self.env_name:
            obs_np, original_reward, terminated, truncated, info = self.env.step(action)

            obs = GlucoseObservation()
            obs.update_obs_with_sim_state(self.env)

            # obs.bg = np.array(obs.bg)
            # obs.insulin = np.array(obs.insulin)

            obs.bg = np.asarray(obs.bg)
            obs.insulin = np.asarray(obs.insulin)
            obs.cho = np.asarray(obs.cho)

            self.last_obs.bg = np.asarray(self.last_obs.bg)
            self.last_obs.insulin = np.asarray(self.last_obs.insulin)
            self.last_obs.cho = np.asarray(self.last_obs.cho)
        elif "traffic" in self.env_name:
            obs_np, original_reward, terminated, truncated, info = self.env.step(action)
            obs = TrafficObservation()
            obs.update_obs_with_sim_state(self.env)
        else:
            raise ValueError("Env not recognized")
        
        done = terminated or truncated
        reward = self.reward_function.calculate_reward(self.last_obs, action, obs)

        if type(reward) is np.ndarray:
            reward = reward.item()
        if np.isnan(reward):
            print("Warning: NaN reward encountered. Setting to 0.")
            reward = 0.0
        
        # print ("reward:", reward)
        info["modified_reward"] = reward
        #no overwriting reward for now
        # reward = original_reward
        self.ep_return += original_reward
        
        # Store original reward in info for reference
        info["original_reward"] = original_reward
        # print ("overwriting reward...")
        self.last_obs_np = obs_np
        self.last_obs = obs

        # print ("new reward:",reward)
        # print ("default reward:",original_reward)
        return obs_np, reward, terminated, truncated, info

