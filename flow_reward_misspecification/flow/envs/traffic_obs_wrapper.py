from dataclasses import dataclass
import numpy as np
from flow.envs.merge import MergePOEnv
from dataclasses import dataclass, field
from typing import List
import numpy as np
from flow.envs.merge import MergePOEnv

@dataclass
class TrafficObservation:
    """Dataclass that updates numpy arrays with information from MergePOEnv. This observation is
    used by the reinforcement learning (RL) interface for traffic control. RL vehicles are those
    controlled by the RL algorithm, while non-RL vehicles are other drivers on the road."""
    # ── Per-RL-vehicle arrays ────────────────────────────────────────────────
    ego_speeds: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # **Un-normalised speed (m/s) of each RL-controlled vehicle.**

    leader_headways: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # **Bumper-to-bumper distance (m) from every RL vehicle to its current leader.**

    leader_speed_diffs: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # **(Leader speed − ego speed) in m/s for each RL vehicle.**

    follower_headways: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # **Distance (m) between each RL vehicle and its follower (if any).**

    follower_speed_diffs: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # **(Ego speed − follower speed) in m/s.**

    rl_ids: List[str] = field(
        default_factory=list
    )  # **String IDs of the RL-controlled vehicles present at this timestep.**
      # Maintains a stable mapping from array index → vehicle id.

    # ── Global variables ─────────────────────────────────────────────────────
    all_vehicle_speeds: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # **Vector of raw speeds (m/s) for *every* vehicle in the network.**

    target_velocity: float = (
        0.0  # **Speed set-point (m/s) that the reward encourages vehicles to match.**
    )

    fail: bool = (
        False  # **True if a collision / failure occurred in the current rollout.**
    )

    # Normalisation constants kept for convenient re-scaling in policy code
    max_speed: float = (
        0.0  # **Network’s absolute speed limit (m/s).** Used for normalising features.
    )

    max_length: float = (
        0.0  # **Approx. longest possible bumper-to-bumper gap (m) on the road.**
        # Used to normalise headway distances.
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Update from env ─────────────────────────────────────────────────────────
    def update_obs_with_sim_state(self, env: MergePOEnv) -> None:
        """Populate all fields from the current simulator state."""
        # ---------- global -----------
        self.rl_ids          = env.rl_veh.copy()
        self.target_velocity = env.env_params.additional_params["target_velocity"]
        self.max_speed       = env.k.network.max_speed()
        self.max_length      = env.k.network.length()

        veh_ids              = env.k.vehicle.get_ids()
        self.all_vehicle_speeds = np.array(env.k.vehicle.get_speed(veh_ids), dtype=np.float32)

        # Simple collision indicator used by Flow environments
        self.fail = getattr(env, "failure", False) or bool(getattr(env.k.vehicle, "num_crashed", 0))

        # ---------- per-RL-vehicle ----------
        num_rl = env.num_rl
        self.ego_speeds           = np.zeros(num_rl, dtype=np.float32)
        self.leader_headways      = np.zeros(num_rl, dtype=np.float32)
        self.leader_speed_diffs   = np.zeros(num_rl, dtype=np.float32)
        self.follower_headways    = np.zeros(num_rl, dtype=np.float32)
        self.follower_speed_diffs = np.zeros(num_rl, dtype=np.float32)

        for i, rl_id in enumerate(self.rl_ids):
            if rl_id not in veh_ids:
                continue  # RL vehicle has left the network

            ego_speed = env.k.vehicle.get_speed(rl_id)
            leader_id = env.k.vehicle.get_leader(rl_id)
            follower_id = env.k.vehicle.get_follower(rl_id)

            # Leader ----------------------------------------------------------
            if leader_id in ("", None):
                leader_speed = self.max_speed
                headway = self.max_length
            else:
                leader_speed = env.k.vehicle.get_speed(leader_id)
                headway = (
                    env.k.vehicle.get_x_by_id(leader_id)
                    - env.k.vehicle.get_x_by_id(rl_id)
                    - env.k.vehicle.get_length(rl_id)
                )

            # Follower --------------------------------------------------------
            if follower_id in ("", None):
                follower_speed = 0.0
                follow_headway = self.max_length
            else:
                follower_speed = env.k.vehicle.get_speed(follower_id)
                follow_headway = env.k.vehicle.get_headway(follower_id)

            # Store raw (unnormalised) values so the reward can be recovered
            self.ego_speeds[i]           = ego_speed
            self.leader_headways[i]      = headway
            self.leader_speed_diffs[i]   = leader_speed - ego_speed
            self.follower_headways[i]    = follow_headway
            self.follower_speed_diffs[i] = ego_speed - follower_speed

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers -----------------------------------------------------------------
    def as_flat_array(self, normalise: bool = False) -> np.ndarray:
        """
        Return a 1-D observation suitable for an RL policy.
        If `normalise` is True, the same normalisation scheme as before is used.
        """
        if normalise:
            # Scale speeds by max_speed and distances by max_length
            scale_v = lambda v: v / (self.max_speed + 1e-8)
            scale_d = lambda d: d / (self.max_length + 1e-8)
            feats = [
                scale_v(self.ego_speeds),
                scale_v(self.leader_speed_diffs),
                scale_d(self.leader_headways),
                scale_v(self.follower_speed_diffs),
                scale_d(self.follower_headways),
            ]
        else:
            feats = [
                self.ego_speeds,
                self.leader_speed_diffs,
                self.leader_headways,
                self.follower_speed_diffs,
                self.follower_headways,
            ]
        return np.concatenate(feats, dtype=np.float32)

    # Alias kept for backwards compatibility with older code
    def flatten(self) -> np.ndarray:
        return self.as_flat_array()

# @dataclass
# class TrafficObservation:
#     """Dataclass that updates numpy arrays with information from MergePOEnv. This observation is
#     used by the reinforcement learning interface for traffic control."""
    
#     ego_speeds: np.ndarray = None #the speed of the ego vehicle, normalized by max_speed
#     lead_speed_diffs: np.ndarray = None #the speed difference between the ego vehicle and its leader, normalized by max_speed
#     lead_headways: np.ndarray = None #the headway to the leader, normalized by max_length
#     follow_speed_diffs: np.ndarray = None #the speed difference between the ego vehicle and its follower, normalized by max_speed
#     follow_headways: np.ndarray = None #the headway to the follower, normalized by max_length
#     rl_vehicles: list = None #the list of RL vehicles in the network
#     leader_vehicles: list = None #the list of leader vehicles in the network
#     follower_vehicles: list = None #the list of follower vehicles in the network

#     def update_obs_with_sim_state(
#         self,
#         env: MergePOEnv,
#     ) -> None:
#         """
#         Update the TrafficObservation with the information from the simulation environment.
#         The observation consists of the speeds and bumper-to-bumper headways of the vehicles immediately preceding and following autonomous vehicle, as well as the ego speed of the autonomous vehicles.

#         Observations contain normalized information about each RL vehicle and its immediate neighbors:
#         - Ego vehicle speed (normalized by max_speed)
#         - Speed difference with leader (normalized by max_speed)
#         - Headway to leader (normalized by max_length)
#         - Speed difference with follower (normalized by max_speed)
#         - Headway to follower (normalized by max_length)

#         The observation space is fixed to accommodate up to num_rl vehicles, with zeros filling
#         unused slots when fewer vehicles are present.


#         Args:
#             env: MergePOEnv instance containing the current simulation state
#         """
#         # Get the current RL vehicles being controlled
#         self.rl_vehicles = env.rl_veh.copy()
#         self.leader_vehicles = env.leader.copy()
#         self.follower_vehicles = env.follower.copy()
        
#         # Get normalizing constants
#         max_speed = env.k.network.max_speed()
#         max_length = env.k.network.length()
        
#         # Initialize observation arrays
#         num_rl = env.num_rl # maximum number of controllable vehicles in the network
#         self.ego_speeds = np.zeros(num_rl)
#         self.lead_speed_diffs = np.zeros(num_rl)
#         self.lead_headways = np.zeros(num_rl)
#         self.follow_speed_diffs = np.zeros(num_rl)
#         self.follow_headways = np.zeros(num_rl)

#         # desired velocity for all vehicles in the network, in m/s
#         self.target_velocity = env.env_params.additional_params["target_velocity"] 
        
#         # Extract observation data for each RL vehicle by looping through the rl_vehicles list
#         for i, rl_id in enumerate(env.rl_veh):
#             if rl_id not in env.k.vehicle.get_rl_ids():
#                 continue
                
#             this_speed = env.k.vehicle.get_speed(rl_id)
#             lead_id = env.k.vehicle.get_leader(rl_id)
#             follower = env.k.vehicle.get_follower(rl_id)

#             # Handle leader information
#             if lead_id in ["", None]:
#                 # In case leader is not visible
#                 lead_speed = max_speed
#                 lead_head = max_length
#             else:
#                 lead_speed = env.k.vehicle.get_speed(lead_id)
#                 lead_head = (
#                     env.k.vehicle.get_x_by_id(lead_id)
#                     - env.k.vehicle.get_x_by_id(rl_id)
#                     - env.k.vehicle.get_length(rl_id)
#                 )

#             # Handle follower information
#             if follower in ["", None]:
#                 # In case follower is not visible
#                 follow_speed = 0
#                 follow_head = max_length
#             else:
#                 follow_speed = env.k.vehicle.get_speed(follower)
#                 follow_head = env.k.vehicle.get_headway(follower)

#             # Store normalized observations
#             self.ego_speeds[i] = this_speed / max_speed
#             self.lead_speed_diffs[i] = (lead_speed - this_speed) / max_speed
#             self.lead_headways[i] = lead_head / max_length
#             self.follow_speed_diffs[i] = (this_speed - follow_speed) / max_speed
#             self.follow_headways[i] = follow_head / max_length

#     def get_flat_observation(self) -> np.ndarray:
#         """
#         Get the observation as a flat array for RL algorithms.
        
#         Returns:
#             np.ndarray: Flattened observation array of shape (5 * num_rl,)
#         """
#         observation = []
#         for i in range(len(self.ego_speeds)):
#             observation.extend([
#                 self.ego_speeds[i],
#                 self.lead_speed_diffs[i],
#                 self.lead_headways[i],
#                 self.follow_speed_diffs[i],
#                 self.follow_headways[i]
#             ])
#         return np.array(observation, dtype=np.float32)
    
#     def flatten(self):
#         return self.get_flat_observation()


