<!-- Copilot / AI agent helper for the standalone_envs_for_yoonho workspace -->
# Quick instructions for AI coding agents

Goal: Help a developer quickly understand, edit, and extend this workspace which contains multiple embedded projects (notably Flow traffic environment and an embedded simglucose project under `bgp/`).

What this workspace is: a collection of standalone environment code used for reward-misspecification experiments and other RL/controls tasks. The main codebases present are:
- `flow_reward_misspecification/` — a revived fork of the Flow traffic environment (core Flow package, examples, and configs).
- `bgp/` — a glucose/simglucose environment with RL wrappers and controllers.
- `examples/`, `utils/`, `reward_functions/` — utility and experiment runner scripts used across these projects.

Important locations (start here):
- `flow_reward_misspecification/README.md` — high-level project purpose and install hints for the Flow fork.
- `flow_reward_misspecification/flow/` — Flow package: envs, networks, controllers, scenarios and utility helpers. Key files:
  - `flow_reward_misspecification/flow/setup.py` — builds / installs the package and installs SUMO wheels during build_ext.
  - `flow_reward_misspecification/flow/flow_cfg/get_experiment.py` — experiment config loader used by `examples/train.py` / `simulate.py`.
  - `flow_reward_misspecification/flow/envs/` — environment classes and wrappers, e.g. `reward_wrapper.py`, `base.py`, `traffic_obs_wrapper.py`.
  - `flow_reward_misspecification/flow/networks/` — network/topology definitions (e.g., `ring.py`, `bottleneck.py`).
- `flow_reward_misspecification/examples/` — runnable scripts for simulation and RL training. Use `simulate.py` and `train.py` here.
- `flow_reward_misspecification/reward_functions/` — project-specific reward definitions (proxy/true reward logic used by the fork).
- `utils/` — workspace utility scripts, including glucose helpers like `utils/glucose_config.py` and `utils/glucose_rollout_and_save.py`.
- `bgp/` — embedded simglucose code. Look at `bgp/simglucose/envs/simglucose_gym_env.py` and `bgp/rl/reward_functions.py` for how the glucose envs pick reward functions and expose safe-policy actions.

Architecture & data flows (concise):
- Experiment entrypoints: `flow_reward_misspecification/examples/simulate.py` (non-RL) and `flow_reward_misspecification/examples/train.py` (RL). They call into `flow_reward_misspecification/flow/flow_cfg/get_experiment.py` to build an `Env` and `EnvParams`.
- Flow env composition: a `Network` (from `flow/networks`) + `Scenario` (from `flow/scenarios`) -> `flow/envs/*` constructs gym-style envs. RL wrappers and reward wrappers live under `flow/envs/` and `flow/visualize/` contains logging/plot helpers.
- Reward selection: Many envs accept a reward name and map it to a function in a reward module. In glucose code, see `bgp/simglucose/envs/simglucose_gym_env.py` where `reward_fun = reward_functions.<name>` is chosen. The Flow fork computes both true and proxy rewards in places (this repo is designed for reward misspecification experiments).
- External sims: SUMO (and optionally Aimsun) are required by many networks. The package `flow_reward_misspecification/flow/setup.py` includes a build_ext hook that installs a bundled `sumotools` wheel; CI and local setups often install SUMO separately.

Developer workflows (what to run):
- Install python deps (the repo has a commented `requirements.txt`; use a virtualenv and install the packages you need). Typical steps:
  - pip install -r flow_reward_misspecification/requirements.txt
  - Install SUMO separately (Flow docs: https://flow.readthedocs.io/en/latest/flow_setup.html) and add SUMO to PATH.
- Run examples (from workspace root):
  - Non-RL simulate: `python flow_reward_misspecification/examples/simulate.py ring` (or other experiment name in `flow_reward_misspecification/flow/flow_cfg/exp_configs/non_rl`).
  - RL train: `python flow_reward_misspecification/examples/train.py singleagent_ring --rl_trainer rllib` (see `examples/README.md` for trainer switches and extra args).
- Package install (editable) for local dev: `pip install -e flow_reward_misspecification/` (this runs `flow/setup.py` build_ext which attempts to install a SUMO-related wheel). If the wheel install fails, install SUMO separately and rerun.

Project-specific patterns & conventions (concrete):
- Reward lookup pattern: Many envs select reward functions by name. Example: `bgp/simglucose/envs/simglucose_gym_env.py` contains `reward_fun = reward_functions.risk_diff`. When adding a reward, export it from the appropriate reward module and reference the same key from experiment configs.
- Experiment configs: Experiment names map to files under `flow_reward_misspecification/flow/flow_cfg/exp_configs/`. `examples/train.py` and `examples/simulate.py` pass the config name to `get_experiment.py`. To add a new scenario add a file there and update example calls.
- Logging & training data: `flow_reward_misspecification/examples/train.py` writes per-run logs into `flow_reward_misspecification/examples/training_data/<exp_name>/<date_time>`; TensorBoard expects logs in those folders.
- Glucose env specifics: the glucose environment is under `bgp/` and uses separate reward functions and controllers. `utils/glucose_rollout_and_save.py` shows how to create rollouts using `SimglucoseEnv` and `utils/glucose_config.py` shows a short config builder for glucose RL runs.

Integration & external dependencies to watch for:
- SUMO (required by many `flow/networks` and `flow/envs`).
- Optional: Aimsun support in `flow_reward_misspecification/flow/utils/aimsun/*` — requires Aimsun binaries and a special environment.
- RL libraries: RLlib (ray), stable-baselines, and `h-baselines` are referenced from `examples/train.py` and example configs; availability depends on your environment.
- The `bgp/` simglucose code expects CSV parameter files under `simglucose/params` — some paths are placeholders and may need updating for local runs (see `bgp/simglucose/controller/basal_bolus_ctrller.py`).

Editing patterns / small guidance for agents working on code changes:
- When adding a new experiment flag or config key, update `flow_reward_misspecification/flow/flow_cfg/get_experiment.py` and the matching `exp_configs/*` file.
- Keep reward functions pure and small. Reward functions should accept observations and return scalars; place new reward fns under `flow_reward_misspecification/reward_functions/` or `bgp/rl/reward_functions.py` depending on domain.
- For changes that affect package import (top-level modules), run `pip install -e flow_reward_misspecification/` to ensure imports resolve.

Files to inspect for more context (examples to reference):
- `flow_reward_misspecification/flow/setup.py` — build/install behaviour and SUMO wheel install.
- `flow_reward_misspecification/examples/README.md` and `flow_reward_misspecification/examples/train.py`, `flow_reward_misspecification/examples/simulate.py` — how experiments are launched.
- `flow_reward_misspecification/flow/flow_cfg/get_experiment.py` — central experiment builder.
- `flow_reward_misspecification/flow/envs/reward_wrapper.py` — reward wrapping conventions.
- `bgp/simglucose/envs/simglucose_gym_env.py` — domain-specific env and reward selection.
- `utils/glucose_config.py` and `utils/glucose_rollout_and_save.py` — glucose config and rollout examples.

What AI agents should avoid doing automatically:
- Running system-level installers for SUMO or Aimsun without prompting (these require system access and external downloads).
- Changing experiment defaults without verifying how training scripts parse args (check `flow_reward_misspecification/examples/train.py` and `flow_reward_misspecification/flow/flow_cfg/get_experiment.py`).

If you modify or add reward functions, tests, or experiment configs:
- Update the nearest example in `flow_reward_misspecification/examples/` to show how to run the new config.
- Add a short unit test or a small runnable example in `examples/` that demonstrates the new reward returning expected scalar ranges.

Questions for the maintainer before larger edits:
1. Which RL trainer is the canonical one for experiments in this workspace (RLlib, stable-baselines, or h-baselines)?
2. Are SUMO wheel installs expected to work during `pip install -e flow_reward_misspecification/` on CI, or is SUMO installed separately on CI machines?

If you want the instructions adapted to CI, Docker, or a minimal dev environment, tell me which and I will iterate.
