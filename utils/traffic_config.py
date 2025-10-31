import torch
from flow.flow_cfg.get_experiment import get_exp
import json
from flow.utils.rllib import FlowParamsEncoder
# from occupancy_measures.envs.traffic_callbacks import TrafficCallbacks
from occupancy_measures.models.model_with_discriminator import ModelWithDiscriminatorConfig
from ray.rllib.utils.typing import AlgorithmConfigDict
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.ppo import PPOConfig

def get_ppo_config(env_name,env_config, num_gpus, seed,num_rollout_workers=1):
    """
    PPO configuration for traffic environment.
    
    CleanRL PPO Parameter Mapping:
    ================================
    DIRECTLY MAPPED TO CLEANRL:
    - lr -> learning_rate
    - gamma -> gamma
    - use_gae -> (always True in CleanRL PPO)
    - gae_lambda -> gae_lambda
    - vf_loss_coeff -> vf_coef
    - grad_clip -> max_grad_norm (None here, means no clipping)
    - num_sgd_iter -> update_epochs
    - sgd_minibatch_size -> num_minibatches (computed as train_batch_size / minibatch_size)
    - entropy_coeff -> ent_coef
    
    RLLIB-SPECIFIC (NO DIRECT CLEANRL EQUIVALENT):
    - kl_target: RLlib's adaptive KL penalty target. CleanRL has target_kl but used differently.
    - vf_clip_param: RLlib's value clipping (10000, essentially disabled). CleanRL has --clip-vloss.
    - rollout_fragment_length: RLlib fragment size (4000). Use num_steps in CleanRL.
    - train_batch_size: RLlib total batch. CleanRL computes as num_envs * num_steps.
    - batch_mode: RLlib-specific. CleanRL always truncates.
    - lr_decay_schedule: RLlib schedule. CleanRL has anneal_lr flag but simpler implementation.
    - entropy_coeff_schedule: RLlib schedule. CleanRL doesn't support entropy scheduling.
    - num_rollout_workers: RLlib parallelization. Use PufferLib num_envs.
    
    CLEANRL PARAMETERS NOT IN RLLIB:
    - num_envs: Number of parallel environments (use PufferLib for vectorization)
    - num_steps: Steps per env before update (maps to rollout_fragment_length)
    - anneal_lr: Whether to anneal learning rate (RLlib has lr_schedule but more complex)
    - norm_adv: Normalize advantages (RLlib does by default)
    - clip_vloss: Clip value loss (different from vf_clip_param)
    - target_kl: Early stopping KL (different from kl_target usage)
    - clip_coef: Clip coefficient (not directly specified here, uses default)
    """
    # Training
    rollout_fragment_length = 4000  # default: scenario.N_ROLLOUTS * horizon, maps to num_steps
    train_batch_size = max(
        10 * rollout_fragment_length, rollout_fragment_length
    )
    sgd_minibatch_size = min(16 * 1024, train_batch_size)
    num_training_iters = 250  # noqa: F841
    lr = 5e-5
    lr_start = lr
    lr_end = lr
    lr_horizon = 1000000
    lr_decay_schedule = [  # RLLIB-SPECIFIC: More complex than CleanRL's anneal_lr
        [0, lr_start],
        [lr_horizon, lr_end],
    ]
    batch_mode = "truncate_episodes"  # RLLIB-SPECIFIC: CleanRL always truncates

    # PPO
    gamma = 0.99
    use_gae = True
    gae_lambda = 0.97
    kl_target = 0.02  # RLLIB-SPECIFIC: Adaptive KL penalty target
    vf_clip_param = 10000  # RLLIB-SPECIFIC: Effectively disabled (very high value)
    vf_loss_coeff = 0.5
    grad_clip = None  # NOTE: No gradient clipping in this config
    num_sgd_iter = 5
    entropy_coeff = 0.01
    entropy_coeff_start = entropy_coeff
    entropy_coeff_end = entropy_coeff
    entropy_coeff_horizon = 1000000
    entropy_coeff_schedule = [  # RLLIB-SPECIFIC: CleanRL doesn't support schedules
        [0, entropy_coeff_start],
        [entropy_coeff_horizon, entropy_coeff_end],
    ]

    # model
    width = 512
    depth = 4
    fcnet_hiddens = [width] * depth
    discriminator_width = 256
    discriminator_depth = 2
    use_action_for_disc = True
    vf_share_layers = True
    custom_model_config: ModelWithDiscriminatorConfig = {
        "discriminator_depth": discriminator_depth,
        "discriminator_width": discriminator_width,
        "use_action_for_disc": use_action_for_disc,
    }
    model_config = {
        "custom_model": "model_with_discriminator",
        "fcnet_hiddens": fcnet_hiddens,
        "custom_model_config": custom_model_config,
        "custom_action_dist": "TrafficBeta",
        "vf_share_layers": vf_share_layers,
    }

    # config = AlgorithmConfig().rl_module(_enable_rl_module_api=False)
    config = PPOConfig().rl_module(_enable_rl_module_api=False)

    config_updates: AlgorithmConfigDict = {  # noqa: F841
        "env": env_name,
        "env_config": env_config,
        # "callbacks": TrafficCallbacks,
        "num_rollout_workers": num_rollout_workers,
        "train_batch_size": train_batch_size,
        "sgd_minibatch_size": sgd_minibatch_size,
        "num_sgd_iter": num_sgd_iter,
        "gamma": gamma,
        "use_gae": use_gae,
        "lambda": gae_lambda,
        "kl_target": kl_target,
        "vf_clip_param": vf_clip_param,
        "vf_loss_coeff": vf_loss_coeff,
        "grad_clip": grad_clip,
        "lr": lr,
        "lr_schedule": lr_decay_schedule,
        "num_gpus": num_gpus,
        "rollout_fragment_length": rollout_fragment_length,
        "entropy_coeff": entropy_coeff,
        "entropy_coeff_schedule": entropy_coeff_schedule,
        "model": model_config,
        "normalize_actions": False,
        "framework_str": "torch",
        "batch_mode": batch_mode,
        "seed": seed,
    }
    config.update_from_dict(config_updates)

    config._enable_rl_module_api = False
    config._enable_learner_api = False
    config.enable_connectors = False
    
    return config


def get_config():
    # Environment
    exp_tag = "singleagent_merge_bus"  # horizon might need to be updated for bottleneck to 1040
    scenario = get_exp(exp_tag)
    flow_params = scenario.flow_params
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4
    )
    exp_algo = "PPO"
    reward_fun = "true"
    assert reward_fun in ["true", "proxy"]
    # callbacks = TrafficCallbacks
    use_safe_policy_actions = False
    # Rewards and weights
    proxy_rewards = ["vel", "accel", "headway"]
    proxy_weights = [1, 1, 0.1]
    true_rewards = ["commute", "accel", "headway"]
    true_weights = [1, 1, 0.1]
    true_reward_specification = [
        (r, float(w)) for r, w in zip(true_rewards, true_weights)
    ]
    proxy_reward_specification = [
        (r, float(w)) for r, w in zip(proxy_rewards, proxy_weights)
    ]
    reward_specification = {
        "true": true_reward_specification,
        "proxy": proxy_reward_specification,
    }
    reward_scale = 0.0001
    horizon = flow_params["env"].horizon
    flow_params["env"].horizon = horizon
    env_config = {
        "flow_params": flow_json,
        "flow_params_default": flow_params,
        "reward_specification": reward_specification,
        "reward_fun": reward_fun,
        "run": exp_algo,
        "use_safe_policy_actions": use_safe_policy_actions,
        "reward_scale": reward_scale,
    }

    return env_config


