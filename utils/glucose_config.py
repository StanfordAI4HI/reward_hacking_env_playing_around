import torch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.utils.typing import AlgorithmConfigDict
# from occupancy_measures.envs.glucose_callbacks import GlucoseCallbacks
from occupancy_measures.models.glucose_models import GlucoseModelConfig
from occupancy_measures.models.model_with_discriminator import ModelWithDiscriminatorConfig

def get_ppo_config():
    
    env_name = "glucose_env"
    num_envs_per_worker=5  # RLLIB-SPECIFIC: Use PufferLib num_envs instead
    lr = 1e-4
    grad_clip = 10
    gamma = 0.99
    gae_lambda = 0.98
    vf_loss_coeff = 1e-4  # NOTE: Very small compared to CleanRL default 0.5
    vf_clip_param = 100  # RLLIB-SPECIFIC: Different from CleanRL's clip_vloss
    entropy_coeff = 0.01
    
    rollout_fragment_length = 2000  # default: horizon, maps to num_steps in CleanRL
    kl_target = 1e-3  # RLLIB-SPECIFIC: Target KL divergence
    clip_param = 0.05

    # Model
    vf_share_layers = True
    num_layers = 3
    hidden_size = 64
    
   
    use_action_for_disc = True
    use_history_for_disc = False
    # time_dim = 2
    # disc_history = 48  # how many intervals of five minutes of history to use for discriminator
    # if not use_history_for_disc:
    #     disc_history = 1
    # history_range = (disc_history * -1, 0)
    model_action_scale = 10
    # glucose_custom_model_config = {
    #     "num_layers": num_layers,
    #     "hidden_size": hidden_size,
    #     "action_scale": model_action_scale,
    # }
   

    # model_config = {
    #     "custom_model": "glucose",
    #     "custom_model_config": custom_model_config,
    #     "vf_share_layers": vf_share_layers,
    #     "custom_action_dist": "GlucoseBeta",
    # }
    
    fcnet_hiddens = [hidden_size] * num_layers

    config = {
        "env": env_name,
        "bptt_horizon":rollout_fragment_length,
        "lr": lr,
        "gamma": gamma,
        "kl_target": kl_target,
        "ent_coef": entropy_coeff,
        "vf_coef": vf_loss_coeff,
        "vf_clip_param": vf_clip_param,
        "vf_clip_coef": clip_param,
        "gae_lambda": gae_lambda,
        # "update_epochs": num_sgd_iter,
        "max_grad_norm": grad_clip,
        "torch_deterministic": False,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "batch_size":"auto",
        "minibatch_size":rollout_fragment_length,
        # "callbacks": callbacks,
        "model": {
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "action_scale": model_action_scale,
            "fcnet_activation": "relu",
            "custom_action_dist": "GlucoseBeta",
            "vf_share_layers": vf_share_layers,
        }
    }
    
    return config



def get_config():
    # Environment
    env_name = "glucose_env"
    proxy_reward_fun = "expected_patient_cost"
    true_reward_fun = "magni_bg" #"magni_bg"
    reward_fun = "true"
    patient_name = "adult#001"
    seeds = None  # default: {"numpy": 0, "sensor": 0, "scenario": 0}
    reset_lim = {"lower_lim": 10, "upper_lim": 1000}
    time = False
    meal = False
    bw_meals = True
    load = False
    use_pid_load = False
    hist_init = True
    gt = False
    n_hours = 4
    norm = False
    time_std = None
    use_old_patient_env = False
    action_cap = None
    action_bias = 0
    action_scale = "basal"
    basal_scaling = 43.2
    meal_announce = None
    residual_basal = False
    residual_bolus = False
    residual_PID = False
    fake_gt = False
    fake_real = False
    suppress_carbs = False
    limited_gt = False
    weekly = False
    update_seed_on_reset = True
    deterministic_meal_size = False
    deterministic_meal_time = False
    deterministic_meal_occurrence = False
    harrison_benedict = True
    restricted_carb = False
    meal_duration = 5
    rolling_insulin_lim = None
    universal = False
    reward_bias = 0
    carb_error_std = 0
    carb_miss_prob = 0
    source_dir = ""
    noise_scale = 0
    model = None
    model_device = "cuda" if torch.cuda.is_available() else "cpu"
    use_model = False
    unrealistic = False
    use_custom_meal = False
    custom_meal_num = 3
    custom_meal_size = 1
    start_date = None
    use_only_during_day = False
    horizon_days = 20
    horizon = horizon_days * 12 * 24
    reward_scale = 1e-3 if reward_fun == "true" else 1
    termination_penalty = 1e2 / reward_scale
    use_safe_policy_actions = False
    safe_policy_noise_std_dev = 0.003

    env_config = {
        "proxy_reward_fun": proxy_reward_fun,
        "true_reward_fun": true_reward_fun,
        "reward_fun": reward_fun,
        "patient_name": patient_name,
        "seeds": seeds,
        "reset_lim": reset_lim,
        "time": time,
        "meal": meal,
        "bw_meals": bw_meals,
        "load": load,
        "use_pid_load": use_pid_load,
        "hist_init": hist_init,
        "gt": gt,
        "n_hours": n_hours,
        "norm": norm,
        "time_std": time_std,
        "use_old_patient_env": use_old_patient_env,
        "action_cap": action_cap,
        "action_bias": action_bias,
        "action_scale": action_scale,
        "basal_scaling": basal_scaling,
        "meal_announce": meal_announce,
        "residual_basal": residual_basal,
        "residual_bolus": residual_bolus,
        "residual_PID": residual_PID,
        "fake_gt": fake_gt,
        "fake_real": fake_real,
        "suppress_carbs": suppress_carbs,
        "limited_gt": limited_gt,
        "termination_penalty": termination_penalty,
        "weekly": weekly,
        "update_seed_on_reset": update_seed_on_reset,
        "deterministic_meal_size": deterministic_meal_size,
        "deterministic_meal_time": deterministic_meal_time,
        "deterministic_meal_occurrence": deterministic_meal_occurrence,
        "harrison_benedict": harrison_benedict,
        "restricted_carb": restricted_carb,
        "meal_duration": meal_duration,
        "rolling_insulin_lim": rolling_insulin_lim,
        "universal": universal,
        "reward_bias": reward_bias,
        "carb_error_std": carb_error_std,
        "carb_miss_prob": carb_miss_prob,
        "source_dir": source_dir,
        "model": model,
        "model_device": model_device,
        "use_model": use_model,
        "unrealistic": unrealistic,
        "noise_scale": noise_scale,
        "use_custom_meal": use_custom_meal,
        "custom_meal_num": custom_meal_num,
        "custom_meal_size": custom_meal_size,
        "start_date": start_date,
        "horizon": horizon,
        "use_only_during_day": use_only_during_day,
        "reward_scale": reward_scale,
        "use_safe_policy_actions": use_safe_policy_actions,
        "safe_policy_noise_std_dev": safe_policy_noise_std_dev,
    }

    return env_config