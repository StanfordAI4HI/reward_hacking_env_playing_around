# Standalone Envs for Yoonho — README

This workspace contains several standalone environment codebases used for reward-misspecification experiments and RL work. Major components include:

- `flow_reward_misspecification/` — a revived fork of the Flow traffic environment (networks, scenarios, envs, examples).
- `bgp/` — an embedded simglucose (glucose / blood glucose) environment with RL wrappers and controllers.
- `pandemic/` — pandemic simulator code (Python package under `pandemic/python`).
- `rl_utils/` — helpers and entrypoints for training (including `train_with_custom_rew.py`).
- `reward_functions/` — template and concrete reward function classes for glucose, pandemic, and traffic domains.

## Quick start

1. Install the three editable packages in the workspace (this ensures imports like `flow` and `pandemic` work):

```bash
cd bgp
pip install -e .

cd ../flow_reward_misspecification
pip install -e .

cd ../pandemic
pip install -e .
```

2. Install the requirements.txt file.

Notes:
- `flow_reward_misspecification/setup.py` runs a build step which attempts to install a SUMO-related wheel; if you don't have SUMO installed or don't want the wheel step to run, install the packages manually and/or install SUMO separately (see `flow_reward_misspecification/README.md` for Flow-specific setup).

## Training an RL agent

- The main training entrypoint provided here is `rl_utils/train_with_custom_rew.py`.
- This script uses **PufferLib** for vectorized environment execution and PPO training.
- Example usage to train a PPO policy using the pandemic environment:

```bash
python3 -m rl_utils.train_with_custom_rew --vec.seed 0 --vec.num-envs 10 --vec.num-workers 2 --train.name pandemic_gt_reward_fn --train.update-epochs 100
```

**Important Note**: Currently, only the **pandemic environment** is fully supported with PufferLib. The **traffic** and **glucose** environments use custom action distributions (continuous Beta distributions) that are not yet compatible with PufferLib's sampling interface.

Key CLI options (see PufferLib config for full details):
- `--train.name` : Environment and reward type, format: `{env_name}_{reward_type}` (e.g., `pandemic_gt_reward_fn`, `pandemic_proxy_reward_fn`)
  - Environment name can be: `pandemic` (currently supported), `glucose` (in progress), `traffic` (in progress)
  - Reward type can be: `gt_reward_fn` (ground truth), `proxy_reward_fn` (proxy)
- `--vec.seed` : Random seed
- `--vec.num-envs` : Number of parallel environments
- `--vec.num-workers` : Number of worker processes for environment parallelization
- `--train.update-epochs` : Number of training epochs/updates

## Adding a new reward function

1. Implement the reward class following the template in `reward_functions/`:
   - For glucose: see `reward_functions/glucose_gt_rew_fns.py` (subclass `RewardFunction`, implement `calculate_reward(prev_obs, action, obs)` which returns a scalar).
   - For pandemic: see `reward_functions/pandemic_gt_rew_fns.py`.
   - For traffic: see `reward_functions/traffic_gt_rew_fns.py`.

2. Register the new reward in `rl_utils/env_setups.py` so the environment creation code will choose your new reward class when building the environment.
   - Open `rl_utils/env_setups.py` and follow existing patterns (there are helper functions like `setup_pandemic_env`, `setup_glucose_env`, and `setup_traffic_env`). Add a branch or mapping so `env_config['reward_fun_type']` selects your new reward class and wires it into the environment construction.

Notes about reward templates and LLM-editability
- The reward function and observation templates in this workspace have been structured in a way that makes it straightforward for programmatic editing (e.g., an LLM making small edits). For example:
  - Rewards are classes with a single `calculate_reward(prev_obs, action, obs)` method that expects typed observation objects (see `PandemicObservation`, `GlucoseObservation`, `TrafficObservation`).
  - Observations are small typed containers under `bgp/simglucose/envs/` and `flow_reward_misspecification/flow/envs/` that expose arrays and scalars (LLMs can find and use these fields when modifying reward code).
- You're free to change these templates or observation formats if you prefer a different integration approach for automated editing; just update `rl_utils/env_setups.py` so environment creation uses the new format.

## Training outputs — what to expect

- During training (`train_with_custom_rew.py`), the RL loop periodically saves RLlib `result` dictionaries and stores them under `logs/data/<env_type>/<timestamp>/checkpoint_<iter>`.
- When the RL agent is evaluated or training results are computed, a dictionary is expected to contain policy evaluation metrics. In particular, this workspace prints or produces (as part of the logged results) values with these keys under a `default_policy` namespace:

- `default_policy/true_reward_mean` : mean episodic return under the environment's ground truth reward function (as implemented by this project and based on the referenced methods in the repo).
- `default_policy/proxy_reward_mean` : mean episodic return under the environment's proxy reward function.
- `default_policy/modified_reward_mean` : mean episodic return under the reward function that was used to train the RL agent (this is chosen via the `--reward-fun-type` CLI argument and may be the proxy or true reward).

These keys are produced by evaluation and logging routines used with RLlib in `rl_utils/train_with_custom_rew.py` and by code that computes the alternative reward returns during rollouts. If you'd like these saved in a custom file or included in more detailed CSVs, we can add explicit logging hooks into the training loop.

## References

- Paper that implements the ground truth / proxy reward approach used here: https://arxiv.org/abs/2403.03185