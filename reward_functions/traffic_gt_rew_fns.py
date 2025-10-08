from abc import ABCMeta, abstractmethod
from typing import Any, Sequence
import numpy as np

from flow_reward_misspecification.flow.envs.traffic_obs_wrapper import TrafficObservation


class RewardFunction(metaclass=ABCMeta):
    """Minimal interface expected by the rest of the code base."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
        """Create the reward object (sub-classes decide what to store)."""
        pass

    @abstractmethod
    def calculate_reward(
        self,
        prev_obs: TrafficObservation,
        action: Sequence[float],
        obs: TrafficObservation,
    ) -> float:
        """Return the scalar reward produced by `action` in `prev_obs → obs`."""
        ...

class PenEtAlProxyTrafficRewardFunction(RewardFunction):
   
    def __init__(
        self,
        eta1: float = 1.0,
        eta2: float = 1.0,
        eta3: float = 0.1,
        t_min: float = 1.0,
        accel_threshold: float = 0.0,
    ) -> None:
        
        self.eta1 = eta1
        self.eta2 = eta2
        self.eta3 = eta3
        self.t_min = t_min
        self.accel_threshold = accel_threshold
        self._eps = np.finfo(np.float32).eps


    def global_velocity(self, velocities):
        return float(np.mean(velocities)) if velocities.size else 0.0

    def penalize_accel(self, rl_actions):
        if rl_actions is None:
            return 0
        mean_actions = np.mean(np.abs(np.array(rl_actions)))
        return min(0, self.accel_threshold - mean_actions)

    def penalize_headway(self, rl_id2lead_id, rl_ids, ego_speeds, leader_headways):

        # cost = 0.0
        # for speed, headway in zip(ego_speeds, leader_headways):
        #     if speed > 0:  # avoid divide-by-zero
        #         t_headway = max(headway / speed, 0.0)
        #         cost += min((t_headway - self.t_min) / self.t_min, 0.0)  # ≤ 0
        # # # print (".  cost2:", cost2)

        cost = 0
        t_min = 1  # smallest acceptable time headway
        for  i,rl_id in enumerate(rl_ids):
            # print ("rl_id2lead_id:", rl_id2lead_id)
            lead_id = rl_id2lead_id.get(rl_id)
            if lead_id not in ["", None] and ego_speeds[i] > 0:
                t_headway = max(
                    leader_headways[i] / ego_speeds[i], 0
                )
                cost += min((t_headway - t_min) / t_min, 0)

        # cost = 0
        # t_min = 1  # smallest acceptable time headway
        # for rl_id in env.rl_veh:
        #     lead_id = env.k.vehicle.get_leader(rl_id)
        #     if lead_id not in ["", None] and env.k.vehicle.get_speed(rl_id) > 0:
        #         t_headway = max(
        #             env.k.vehicle.get_headway(rl_id) / env.k.vehicle.get_speed(rl_id), 0
        #         )
        #         cost += min((t_headway - t_min) / t_min, 0)
        return cost


    def calculate_reward(
        self,
        prev_obs: TrafficObservation,
        action: Sequence[float],
        obs: TrafficObservation,
    ) -> float:
        
       
        if getattr(obs, "fail", False):
            # print (".    fail=True")
            return 0.0

        # ------------------------------------------------------------------ #
        # 1. commmute statistic                #
        # ------------------------------------------------------------------ #
        commute_time = self.global_velocity(obs.all_vehicle_speeds)
        # ------------------------------------------------------------------ #
        # 2. acceleration statistic                                          #
        # ------------------------------------------------------------------ #
        accel_penalty = self.penalize_accel(action)
        # ------------------------------------------------------------------ #
        # 3. headway statistic                                            #
        # ------------------------------------------------------------------ #
        headway_penalty = self.penalize_headway(obs.rl_id2lead_id, obs.rl_ids, obs.ego_speeds, obs.leader_headways)
        
        reward = (
            self.eta1 * commute_time + self.eta2 * accel_penalty + self.eta3 * headway_penalty
        )
        return reward



class PanEtAlTrueTrafficRewardFunction(RewardFunction):
   
    def __init__(
        self,
        eta1: float = 1.0,
        eta2: float = 1.0,
        eta3: float = 0.1,
        t_min: float = 1.0,
        accel_threshold: float = 0.0,
    ) -> None:
       
        self.eta1 = eta1
        self.eta2 = eta2
        self.eta3 = eta3
        self.t_min = t_min
        self.accel_threshold = accel_threshold
        self._eps = np.finfo(np.float32).eps


    def commute_time(self, velocities, rl_actions):
        # vel = np.array(env.k.vehicle.get_speed(env.k.vehicle.get_ids()))

        if any(velocities < -100):
            return -10000.0
        if len(velocities) == 0:
            return -10000.0

        commute = np.array([(v + 0.001) ** -1 for v in velocities])
        commute = commute[commute > 0]
        return -np.mean(commute)

    def penalize_accel(self, rl_actions):
        if rl_actions is None:
            return 0
        mean_actions = np.mean(np.abs(np.array(rl_actions)))
        return min(0, self.accel_threshold - mean_actions)

    def penalize_headway(self, rl_id2lead_id, rl_ids, ego_speeds, leader_headways):

        # cost = 0.0
        # for speed, headway in zip(ego_speeds, leader_headways):
        #     if speed > 0:  # avoid divide-by-zero
        #         t_headway = max(headway / speed, 0.0)
        #         cost += min((t_headway - self.t_min) / self.t_min, 0.0)  # ≤ 0
        # # # print (".  cost2:", cost2)

        cost = 0
        t_min = 1  # smallest acceptable time headway
        for  i,rl_id in enumerate(rl_ids):
            # print ("rl_id2lead_id:", rl_id2lead_id)
            lead_id = rl_id2lead_id.get(rl_id)
            if lead_id not in ["", None] and ego_speeds[i] > 0:
                t_headway = max(
                    leader_headways[i] / ego_speeds[i], 0
                )
                cost += min((t_headway - t_min) / t_min, 0)

        # cost = 0
        # t_min = 1  # smallest acceptable time headway
        # for rl_id in env.rl_veh:
        #     lead_id = env.k.vehicle.get_leader(rl_id)
        #     if lead_id not in ["", None] and env.k.vehicle.get_speed(rl_id) > 0:
        #         t_headway = max(
        #             env.k.vehicle.get_headway(rl_id) / env.k.vehicle.get_speed(rl_id), 0
        #         )
        #         cost += min((t_headway - t_min) / t_min, 0)
        return cost


    def calculate_reward(
        self,
        prev_obs: TrafficObservation,
        action: Sequence[float],
        obs: TrafficObservation,
    ) -> float:
        """
        Compute the ground-truth reward given *prev_obs* → *obs* transition.

        Parameters
        ----------
        prev_obs : TrafficObservation
            Observation before the action (unused here, but part of the API).
        action : Sequence[float]
            Vector of bounded accelerations (one per RL vehicle).
        obs : TrafficObservation
            Observation **after** the environment step.

        Returns
        -------
        float
            Non-negative reward value.
        """
       
        if getattr(obs, "fail", False):
            # print (".    fail=True")
            return 0.0

        # ------------------------------------------------------------------ #
        # 1. commmute statistic                #
        # ------------------------------------------------------------------ #
        commute_time = self.commute_time(obs.all_vehicle_speeds, action)
        # ------------------------------------------------------------------ #
        # 2. acceleration statistic                                          #
        # ------------------------------------------------------------------ #
        accel_penalty = self.penalize_accel(action)
        # ------------------------------------------------------------------ #
        # 3. headway statistic                                            #
        # ------------------------------------------------------------------ #
        headway_penalty = self.penalize_headway(obs.rl_id2lead_id, obs.rl_ids, obs.ego_speeds, obs.leader_headways)
        
        reward = (
            self.eta1 * commute_time + self.eta2 * accel_penalty + self.eta3 * headway_penalty
        )
        return reward

