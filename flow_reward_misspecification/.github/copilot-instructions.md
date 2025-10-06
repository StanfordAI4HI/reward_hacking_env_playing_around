<!-- Copilot / AI agent helper for the flow_reward_misspecification repo -->
# Quick instructions for AI coding agents

Goal: Help a developer quickly understand, edit, and extend this fork of the Flow traffic simulation and RL code (the "reward-misspecification" variant).

What this repo is: a revived fork of the Flow traffic environment adapted for experiments on reward misspecification and occupancy-measure regularization. It mixes core Flow components (network/topology/scenario definitions), RL entry points (examples/train.py, train wrappers), and domain-specific environments (glucose / simglucose components under `bgp/`).

Important locations (start here):
- `README.md` — high-level project purpose and install hints.
- `flow/` — the original Flow package: envs, networks, controllers, scenarios and utility helpers. Key files:
  - `flow/setup.py` — builds / installs the package and installs SUMO wheels during build_ext.
  - `flow/flow_cfg/get_experiment.py` — experiment config loader used by `examples/train.py` / `simulate.py`.
  - `flow/envs/` — environment classes and wrappers, e.g. `reward_wrapper.py`, `base.py`, `traffic_obs_wrapper.py`.
  - `flow/networks/` — network/topology definitions (e.g., `ring.py`, `bottleneck.py`).
- `examples/` — runnable scripts for simulation and RL training. Use `simulate.py` and `train.py` here.
- `reward_functions/` — project-specific reward definitions (where proxy/true reward logic lives in this fork).
- `utils/` — utility scripts for launching experiments and glucose-specific configs (e.g. `glucose_config.py`, `glucose_rollout_and_save.py`).
- `bgp/` — a separate glucose (simglucose) codebase embedded here. Look at `bgp/simglucose/envs/simglucose_gym_env.py` and `bgp/rl/reward_functions.py` for how the glucose envs choose reward functions and surface safe-policy actions.

Architecture & data flows (concise):
- Experiment entrypoints: `examples/simulate.py` (non-RL) and `examples/train.py` (RL). They call into `flow/flow_cfg/get_experiment.py` to build an `Env` and `EnvParams`.
- Flow env composition: a `Network` (from `flow/networks`) + `Scenario` (from `flow/scenarios`) -> `flow/envs/*` constructs gym-style envs. RL wrappers and reward wrappers live under `flow/envs/` and `flow/visualize/` contains logging/plot helpers.
- Reward selection: Many envs accept a reward name and map it to a function in a reward module. In glucose code, see `bgp/simglucose/envs/simglucose_gym_env.py` where `reward_fun = reward_functions.<name>` is chosen. The fork computes both true and proxy rewards in places (this fork's README calls this out).
- External sims: SUMO (and optionally Aimsun) are required by many networks. The package `flow/setup.py` installs a bundled `sumotools` wheel during build_ext. Tests or local runs require SUMO binaries installed separately.

Developer workflows (what to run):
- Install python deps (this repo keeps an ancient commented `requirements.txt`; prefer creating a venv and installing the requirements used by your experiment). Typical install used by this repo's setup:
  - pip install -r requirements.txt
  - For SUMO: follow Flow docs (https://flow.readthedocs.io/en/latest/flow_setup.html) to install SUMO system packages and add SUMO to PATH.
- Run examples (from repo root):
  - Non-RL simulate: `python examples/simulate.py ring` (or other experiment name in `flow/flow_cfg/exp_configs/non_rl`).
  - RL train: `python examples/train.py singleagent_ring --rl_trainer rllib` (see `examples/README.md` for trainer switches and extra args).
- Package install (editable) for local dev: `pip install -e .` (this will run `flow/setup.py` build_ext step — it tries to install sumo-related wheel). If SUMO wheel install fails, you can skip build_ext by installing python packages first and installing SUMO separately.

Project-specific patterns & conventions (concrete):
- Reward lookup pattern: Many envs select reward functions by name. Example: `bgp/simglucose/envs/simglucose_gym_env.py` contains lines like `reward_fun = reward_functions.risk_diff` and later calls `reward_fun(obs)`; search for `reward_fun = reward_functions` to find other uses. When adding a reward, export it from the appropriate reward module and add the key used by configs.
- Experiment configs: Experiment names map to files under `flow/flow_cfg/exp_configs/` (singleagent/multiagent/non_rl). `examples/train.py` and `examples/simulate.py` pass the config name to `get_experiment.py` which constructs envs and controllers. To add a new scenario, add a config there and mirror naming in `examples/` calls.
- Logging & training data: `examples/train.py` writes per-run logs into `examples/training_data/<exp_name>/<date_time>`; TensorBoard expects logs in those folders.
- Glucose envs: the glucose (BG) environment is embedded under `bgp/` and uses a separate set of reward functions and controllers. `utils/glucose_rollout_and_save.py` provides a simple runner to create rollouts using `SimglucoseEnv`.

Integration & external dependencies to watch for:
- SUMO (required for many `flow/networks` & `flow/envs` experiments).
- Optional: Aimsun support via `flow/utils/aimsun/*` — requires Aimsun binaries and the `aimsun_flow` environment.
- RL libraries: RLlib (ray), stable-baselines, and a fork `h-baselines` are used by `examples/train.py`. Which trainer is available depends on environment setup.
- The `bgp/` simglucose code expects CSV parameter files under `simglucose/params` — paths are sometimes hardcoded; check `bgp/simglucose/controller/basal_bolus_ctrller.py` and `bgp/simglucose/simulation/user_interface.py` for `PATIENT_PARA_FILE` placeholders.

Editing patterns / small guidance for agents working on code changes:
- When adding a new experiment flag or config key, update `flow/flow_cfg/get_experiment.py` and the matching `exp_configs/*` files. Example: to add `--use_subcutaneous_glucose_obs` support in config, see how `utils/glucose_config.py` builds `glucose_custom_model_config`.
- Keep reward functions pure and small. The project expects functions that accept an observation vector and return a scalar. Place new reward fns under `reward_functions/` or `bgp/rl/reward_functions.py` depending on domain.
- For changes that affect package import (top-level modules), run `pip install -e .` locally to ensure package boundaries are correct and `flow` package imports resolve.

Files to inspect for more context (examples to reference):
- `flow/setup.py` — build/install behaviour and SUMO wheel install.
- `examples/README.md` and `examples/train.py`, `examples/simulate.py` — how experiments are launched.
- `flow/flow_cfg/get_experiment.py` — central experiment builder.
- `flow/envs/reward_wrapper.py` — reward wrapping conventions.
- `bgp/simglucose/envs/simglucose_gym_env.py` — domain-specific env and reward selection.
- `utils/glucose_config.py` and `utils/glucose_rollout_and_save.py` — quick glucose experiment runners and config examples.

What AI agents should avoid doing automatically:
- Running system-level installers for SUMO or Aimsun without prompting (these require system access and external downloads).
- Changing experiment defaults without verifying how training scripts parse args (check `examples/train.py` and `flow/flow_cfg/get_experiment.py`).

If you modify or add reward functions, tests, or experiment configs:
- Update the nearest example in `examples/` to show how to run the new config.
- Add a short unit test or a small runnable example in `examples/` that demonstrates the new reward returning expected scalar ranges.

Questions for the maintainer before larger edits:
1. Which RL trainer is the canonical one for experiments in this fork (RLlib, stable-baselines, or h-baselines)?
2. Are SUMO wheel installs expected to work during `pip install -e .` on CI, or is SUMO installed separately on CI machines?

If anything above is unclear or you'd like the instructions tailored to a specific contributor flow (e.g., CI, Docker, or minimal dev environment), tell me which and I'll iterate.
