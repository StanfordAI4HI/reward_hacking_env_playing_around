from abc import ABCMeta, abstractmethod
from typing import Any
import numpy as np
from bgp.simglucose.envs.glucose_obs_wrapper import GlucoseObservation

class RewardFunction(metaclass=ABCMeta):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def calculate_reward(
        self, prev_obs: GlucoseObservation, action: float, obs: GlucoseObservation
    ) -> float:
        pass

class MagniGroundTruthReward(RewardFunction):
    def calculate_reward(self, prev_obs: GlucoseObservation, action: float, obs: GlucoseObservation) -> float:
        """
        True reward function matching bgp.rl.reward_functions.magni_reward.
        This is the Magni risk index without any insulin penalty.
        """
        # Calculate Magni risk index
        bg = max(obs.bg[-1], 1)  # Ensure blood glucose is at least 1
        fBG = 3.5506 * (np.log(bg) ** 0.8353 - 3.7932)
        risk = 10 * (fBG) ** 2

        insulin_penalty = 25 * obs.insulin[-1]
        
        # Return negative risk (no insulin penalty for true reward)
        return -1 * risk - insulin_penalty

class ExpectedCostGroundTruthReward(RewardFunction):
    def calculate_reward(self, prev_obs: GlucoseObservation, action: float, obs: GlucoseObservation) -> float:
        """
        Proxy reward function matching bgp.rl.reward_functions.expected_patient_cost.
        Reward based on expected patient cost, including insulin cost and potential hypoglycemia penalty.
        Uses the latest values from obs.bg and obs.insulin.
        """
        expected_cost = 0.32 * np.mean(obs.insulin[-1])  # Cost of the insulin.
        if obs.bg[-1] < 70:
            # Patient is hypoglycemic, so add potential cost of hospital visit.
            expected_cost += 10 * 1350 / (12 * 24 * 365)
        # Return negative cost (matching the simulator implementation)
        return -expected_cost
