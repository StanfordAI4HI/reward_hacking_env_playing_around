from abc import ABCMeta, abstractmethod
from typing import Any, Sequence

import numpy as np

from flow_reward_misspecification.flow.envs.traffic_obs_wrapper import (
    TrafficObservation,
)


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
    def __init__(self) -> None:
        pass

    def calculate_reward(
        self,
        prev_obs: TrafficObservation,
        action: Sequence[float],
        obs: TrafficObservation,
    ) -> float:

        if getattr(obs, "fail", False):
                # print (".    fail=True")
            return 0.0
        return float(np.mean(obs.all_vehicle_speeds)) if obs.all_vehicle_speeds.size else 0.0


class PanEtAlTrueTrafficRewardFunction(RewardFunction):
    """
    Re-implementation of the Flow `compute_reward` logic, but expressed in
    observation-space so it can be reused outside the simulator loop.

    • **Term 1 (`cost1`)** – system-level desired-velocity reward
      Matches `rewards.desired_velocity`: 1 when all vehicles drive exactly
      at `target_velocity`, 0 when they are far from it.

    • **Term 2 (`cost2`)** – headway penalty for each RL vehicle
      Linear penalty when time-headway < `t_min` seconds.

    • **Term 3 (`cost3`)** – acceleration penalty
      Linear penalty on the mean |accel| sent to RL vehicles.

    The final reward is `max(η₁·cost1 + η₂·cost2 + η₃·cost3, 0)`, unless
    a failure/ collision occurred, in which case it is **0**.
    """

    def __init__(
        self,
        eta1: float = 1.0,
        eta2: float = 0.10,
        eta3: float = 1.0,
        t_min: float = 1.0,
        accel_threshold: float = 0.0,
        evaluate: bool = False,
    ) -> None:
        """
        Args
        ----
        eta1 / eta2 / eta3 : weights for the three cost terms (see above)
        t_min              : minimum acceptable time-headway (s)
        accel_threshold    : |a| below which no acceleration penalty applies
        evaluate           : if *True*, reward is simply average speed of
                             **all** vehicles ( Flow’s evaluation mode )
        """
        self.eta1 = eta1
        self.eta2 = eta2
        self.eta3 = eta3
        self.t_min = t_min
        self.accel_threshold = accel_threshold
        self.evaluate = evaluate
        self._eps = np.finfo(np.float32).eps

    # --------------------------------------------------------------------- #
    # Public API                                                            #
    # --------------------------------------------------------------------- #
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
        # ------------------------------------------------------------------ #
        # 0. Early exit – evaluation mode or collision                       #
        # ------------------------------------------------------------------ #
        if self.evaluate:
            # mean speed of *all* vehicles (m/s)
            return float(np.mean(obs.all_vehicle_speeds)) if obs.all_vehicle_speeds.size else 0.0

        if getattr(obs, "fail", False):
            # print (".    fail=True")
            return 0.0

        # ------------------------------------------------------------------ #
        # 1. Desired-velocity term (system level)                            #
        # ------------------------------------------------------------------ #
        vel = obs.all_vehicle_speeds
        if vel.size == 0:
            cost1 = 0.0
        else:
            target = obs.target_velocity
            max_cost = np.linalg.norm(np.full_like(vel, target))
            cost = np.linalg.norm(vel - target)
            cost1 = max(max_cost - cost, 0.0) / (max_cost + self._eps)  # ∈ [0, 1]
            # print (".  cost1:", cost1)
        # ------------------------------------------------------------------ #
        # 2. Headway penalty (RL vehicles only)                              #
        # ------------------------------------------------------------------ #
        cost2 = 0.0
        for speed, headway in zip(obs.ego_speeds, obs.leader_headways):
            if speed > 0:  # avoid divide-by-zero
                t_headway = max(headway / speed, 0.0)
                cost2 += min((t_headway - self.t_min) / self.t_min, 0.0)  # ≤ 0
        # print (".  cost2:", cost2)

        # ------------------------------------------------------------------ #
        # 3. Acceleration penalty                                            #
        # ------------------------------------------------------------------ #
        if action is None or len(action) == 0:
            cost3 = 0.0
        else:
            mean_abs_accel = float(np.mean(np.abs(action)))
            cost3 = (
                self.accel_threshold - mean_abs_accel
                if mean_abs_accel > self.accel_threshold
                else 0.0
            )  # ≤ 0
        # print (".  cost3:", cost3)
        # ------------------------------------------------------------------ #
        # 4. Weighted sum, clamped to be non-negative                        #
        # ------------------------------------------------------------------ #
        reward = (
            self.eta1 * cost1 + self.eta2 * cost2 + self.eta3 * cost3
        )
        return reward

