import numpy as np
from bgp.simglucose.envs.simglucose_gym_env import SimglucoseEnv
from pandemic_simulator.environment.pandemic_env import PandemicPolicyGymEnv
from occupancy_measures.experiments.traffic_experiments import create_traffic_config
from flow.utils.registry import make_create_env
import json
from rl_utils.reward_wrapper import RewardWrapper
from rl_utils.reward_wrapper import SumReward
from occupancy_measures.envs.glucose_true_rew_wrapper import GlucoseWrapper

def create_env_pandemic(config,reward_function,wrap_env):
    base_env = PandemicPolicyGymEnv(config)
    if not wrap_env:
        return base_env
    return RewardWrapper(base_env, env_name="pandemic", reward_function=reward_function)

def create_env_traffic(config,reward_function,wrap_env):
   
    create_env, env_name = make_create_env(
        params=config["flow_params_default"],
        reward_specification=config["reward_specification"],
        reward_fun=config["reward_fun"],
        reward_scale=config["reward_scale"],
    )
    base_env = create_env()
    if not wrap_env:
        return base_env
    return RewardWrapper(base_env, env_name="traffic", reward_function=reward_function)

def create_env_glucose(config, reward_function,wrap_env):
    base_env = SimglucoseEnv(config)
    if not wrap_env:
        return base_env
    return GlucoseWrapper(RewardWrapper(base_env, env_name="glucose", reward_function=reward_function))

def setup_glucose_env(config,wrap_env):
    if config.get("reward_fun_type") == "gt_reward_fn":
        from reward_functions.glucose_gt_rew_fns import MagniGroundTruthReward
        reward_fn = MagniGroundTruthReward()
    elif config.get("reward_fun_type") == "proxy_reward_fn":
        from reward_functions.glucose_gt_rew_fns import ExpectedCostGroundTruthReward
        reward_fn = ExpectedCostGroundTruthReward()
    return create_env_glucose(config,reward_function=reward_fn,wrap_env=wrap_env)

def setup_traffic_env(config,wrap_env):
    if config.get("reward_fun_type") == "gt_reward_fn":
        from reward_functions.traffic_gt_rew_fns import PanEtAlTrueTrafficRewardFunction
        reward_fn = PanEtAlTrueTrafficRewardFunction()
    elif config.get("reward_fun_type") == "proxy_reward_fn":
        from reward_functions.traffic_gt_rew_fns import PenEtAlProxyTrafficRewardFunction
        reward_fn = PenEtAlProxyTrafficRewardFunction()
    return create_env_traffic(config, reward_function=reward_fn,wrap_env=wrap_env)

def setup_pandemic_env(config,wrap_env):
    if config.get("reward_fun_type") == "gt_reward_fn":
        from reward_functions.pandemic_gt_rew_fns import PanEtAlTruePandemicRewardFunction
        reward_fn = PanEtAlTruePandemicRewardFunction()
    elif config.get("reward_fun_type") == "proxy_reward_fn":
        from reward_functions.pandemic_gt_rew_fns import ProxyPandemicRewardFunction
        reward_fn = ProxyPandemicRewardFunction()
    return create_env_pandemic(config,reward_function=reward_fn,wrap_env=wrap_env)