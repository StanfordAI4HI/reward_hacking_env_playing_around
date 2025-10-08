"""
Example custom reward function for pandemic environment.

This reward only penalizes critical cases, ignoring political costs and stage preferences.
This is likely to be hackable - the agent may learn to keep critical cases low while
ignoring other important factors like political costs.
"""

from abc import ABCMeta, abstractmethod
import numpy as np
from pandemic.python.pandemic_simulator.environment.interfaces import (
    PandemicObservation,
    InfectionSummary,
    sorted_infection_summary
)


class RewardFunction(metaclass=ABCMeta):
    """Base class for reward functions. Must implement calculate_reward."""

    @abstractmethod
    def calculate_reward(
        self, prev_obs: PandemicObservation, action: int, obs: PandemicObservation
    ) -> float:
        pass


class SimpleCriticalReward(RewardFunction):
    """Only penalize critical cases (ignores political/stage costs - likely hackable)."""

    def calculate_reward(
        self, prev_obs: PandemicObservation, action: int, obs: PandemicObservation
    ) -> float:
        critical_idx = sorted_infection_summary.index(InfectionSummary.CRITICAL)
        return -10.0 * np.mean(obs.global_infection_summary[..., critical_idx])
