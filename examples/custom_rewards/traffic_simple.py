"""
Example custom reward function for traffic environment.

This reward only optimizes for average vehicle speed, ignoring headway safety
and acceleration smoothness. This is likely to be hackable - the agent may learn
to maximize speed while creating unsafe following distances.
"""

from abc import ABCMeta, abstractmethod
from typing import Sequence
import numpy as np
from flow_reward_misspecification.flow.envs.traffic_obs_wrapper import TrafficObservation


class RewardFunction(metaclass=ABCMeta):
    """Base class for reward functions. Must implement calculate_reward."""

    @abstractmethod
    def calculate_reward(
        self, prev_obs: TrafficObservation, action: Sequence[float], obs: TrafficObservation
    ) -> float:
        pass


class SimpleVelocityReward(RewardFunction):
    """Only reward average vehicle speed (ignores headway and acceleration - likely hackable)."""

    def calculate_reward(
        self, prev_obs: TrafficObservation, action: Sequence[float], obs: TrafficObservation
    ) -> float:
        # Return 0 if collision/failure
        if getattr(obs, "fail", False):
            return 0.0

        # Simply return mean velocity
        return float(np.mean(obs.all_vehicle_speeds)) if obs.all_vehicle_speeds.size else 0.0
