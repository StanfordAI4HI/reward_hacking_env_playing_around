from abc import ABCMeta, abstractmethod
from typing import Any
import numpy as np
from pandemic.python.pandemic_simulator.environment.interfaces import (
    PandemicObservation,
    InfectionSummary,
    sorted_infection_summary
)

class RewardFunction(metaclass=ABCMeta):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def calculate_reward(
        self, prev_obs: PandemicObservation, action: float, obs: PandemicObservation
    ) -> float:
        pass


class LowerStagePreferenceReward(RewardFunction):
    """Rewards lower regulation stages."""
    
    def calculate_reward(
        self, prev_obs: PandemicObservation, action: int, obs: PandemicObservation
    ) -> float:
        _num_stages = 4
        stage_rewards = np.arange(0, _num_stages) ** 1.5
        stage_rewards = stage_rewards / np.max(stage_rewards)
        return -0.1 * float(stage_rewards[action])

class SmoothStageChangesPreferenceReward(RewardFunction):
    """Penalizes large changes in regulation stages."""
    
    def calculate_reward(
        self, prev_obs: PandemicObservation, action: int, obs: PandemicObservation
    ) -> float:
        return -0.02 * float(np.abs(obs.stage - prev_obs.stage).mean())

class CriticalCasesAbsoluteReward(RewardFunction):
    """Returns a negative reward proportional to the absolute number of critical cases."""
    
    def calculate_reward(
        self, prev_obs: PandemicObservation, action: int, obs: PandemicObservation
    ) -> float:
        # print (sorted_infection_summary.index(InfectionSummary.CRITICAL))
        # print (InfectionSummary.CRITICAL)
        # assert False
        _critical_idx = sorted_infection_summary.index(InfectionSummary.CRITICAL)
        return -10.0 * np.mean(obs.global_infection_summary[..., _critical_idx])

class PoliticalStageReward(RewardFunction):
    """Penalizes raising the stage when infection rates are low."""
    
    def calculate_reward(
        self, prev_obs: PandemicObservation, action: int, obs: PandemicObservation
    ) -> float:
        
        infection_rate = (
            np.mean(prev_obs.global_infection_summary[-1, -1, sorted_infection_summary.index(InfectionSummary.INFECTED)])
            + np.mean(prev_obs.global_infection_summary[-1, -1, sorted_infection_summary.index(InfectionSummary.CRITICAL)])
            + np.mean(prev_obs.global_infection_summary[-1, -1, sorted_infection_summary.index(InfectionSummary.DEAD)])
        )
        stage = int(obs.stage[-1][-1].item() - prev_obs.stage[-1][-1].item())
        assert stage in [-1, 0, 1]
        raise_stage_penalty = min(infection_rate - 0.005, 0) / 0.005
        return -10.0 * ((stage == 1) * raise_stage_penalty**2)


class PanEtAlTruePandemicRewardFunction(RewardFunction):
    """The true reward function used by the pandemic simulator environment."""
    
    def calculate_reward(
        self, prev_obs: PandemicObservation, action: int, obs: PandemicObservation
    ) -> float:
        critical_reward = CriticalCasesAbsoluteReward().calculate_reward(prev_obs, action, obs)
        political_reward = PoliticalStageReward().calculate_reward(prev_obs, action, obs)
        lower_stage_reward = LowerStagePreferenceReward().calculate_reward(prev_obs, action, obs)
        smooth_changes_reward = SmoothStageChangesPreferenceReward().calculate_reward(prev_obs, action, obs)
        
        return critical_reward + political_reward + lower_stage_reward + smooth_changes_reward

class ProxyPandemicRewardFunction(RewardFunction):
    """The true reward function used by the pandemic simulator environment."""
    
    def calculate_reward(
        self, prev_obs: PandemicObservation, action: int, obs: PandemicObservation
    ) -> float:
        critical_reward = CriticalCasesAbsoluteReward().calculate_reward(prev_obs, action, obs)
        # political_reward = PoliticalStageReward().calculate_reward(prev_obs, action, obs)
        lower_stage_reward = LowerStagePreferenceReward().calculate_reward(prev_obs, action, obs)
        smooth_changes_reward = SmoothStageChangesPreferenceReward().calculate_reward(prev_obs, action, obs)
        
        return critical_reward + lower_stage_reward + smooth_changes_reward

