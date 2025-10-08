# LLM-Based Reward Editing System

## Goal

Enable LLM to edit reward functions by writing Python code. Train PPO using that custom reward function, and measure reward hacking by comparing the custom reward vs ground truth reward over training iterations.

**Clarification**: The LLM outputs Python code that defines a reward function. PPO then trains using that function. We're NOT calling an LLM at runtime to compute rewards.

## Current Architecture

```
train_with_custom_rew.py
  └─> setup_<env>_env(config)
       ├─ if "gt_reward_fn": load GT reward class
       ├─ if "proxy_reward_fn": load proxy reward class
       └─> RewardWrapper(base_env, reward_function)
            └─ step(): reward = reward_function.calculate_reward(prev_obs, action, obs)
                       returns reward to PPO
                       stores info["modified_reward"], info["original_reward"]
```

**Key insight**: `RewardWrapper` already tracks two rewards, but `original_reward` is the base environment reward (not a GT reward function). We need to track a GT reward function separately.

**Reward interface**: All rewards implement `calculate_reward(prev_obs, action, obs) -> float`

## SIMPLEST Design: One Standalone Script (ZERO codebase modifications)

Instead of modifying 3-5 files across the codebase, create **one self-contained script** that does everything.

### File: `rl_utils/train_custom_reward.py` (~150 lines total)

The script contains:

1. **Reward loader** (~15 lines) - dynamically load reward class from Python file
2. **Dual reward tracking callback** (~15 lines) - track both custom reward (for training) and GT reward (for measuring hacking)
3. **Modified env setup** (~20 lines) - custom `create_env()` that uses custom reward + tracks GT reward
4. **Training loop** (~100 lines) - mostly copy-paste from `train_with_custom_rew.py`

#### 1. Reward Loader (in same file)

```python
import importlib.util
from rl_utils.reward_wrapper import RewardFunction

def load_reward_from_file(file_path: str, class_name: str = None):
    """Load reward function class from Python file (e.g., written by LLM)."""
    spec = importlib.util.spec_from_file_location("custom_reward", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if class_name:
        return getattr(module, class_name)()

    # Auto-detect: find first RewardFunction subclass
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and issubclass(obj, RewardFunction) and obj != RewardFunction:
            return obj()

    raise ValueError(f"No RewardFunction found in {file_path}")
```

#### 2. Dual Tracking Callback (in same file)

```python
from ray.rllib.algorithms.callbacks import DefaultCallbacks

class DualRewardCallback(DefaultCallbacks):
    def on_episode_step(self, *, episode, **kwargs):
        info = episode.last_info_for()

        # Track custom reward (what PPO optimizes - the reward function written by LLM)
        episode.user_data.setdefault("custom_reward_sum", 0)
        episode.user_data["custom_reward_sum"] += episode.prev_reward_for()

        # Track GT reward (for measuring hacking)
        if "ground_truth_reward" in info:
            episode.user_data.setdefault("gt_reward_sum", 0)
            episode.user_data["gt_reward_sum"] += info["ground_truth_reward"]

    def on_episode_end(self, *, episode, **kwargs):
        episode.custom_metrics["custom_reward"] = episode.user_data.get("custom_reward_sum", 0)
        episode.custom_metrics["gt_reward"] = episode.user_data.get("gt_reward_sum", 0)
```

#### 3. Custom Environment Setup (in same file)

```python
from rl_utils.reward_wrapper import RewardWrapper

class DualTrackingRewardWrapper(RewardWrapper):
    """Extended RewardWrapper that tracks both custom reward (for training) and GT reward (for measuring hacking)."""

    def __init__(self, env, env_name, custom_reward_fn, gt_reward_fn):
        super().__init__(env, env_name, custom_reward_fn)
        self.gt_reward_fn = gt_reward_fn

    def step(self, action):
        obs_np, reward, terminated, truncated, info = super().step(action)

        # Compute GT reward for tracking (using the obs objects from parent)
        gt_reward = self.gt_reward_fn.calculate_reward(self.last_obs, action, self.last_obs)
        info["ground_truth_reward"] = gt_reward

        return obs_np, reward, terminated, truncated, info

def create_env_with_custom_reward(env_config):
    """Create environment with custom reward function (e.g., written by LLM) + GT reward for tracking."""
    env_type = env_config["env_type"]
    custom_reward = load_reward_from_file(env_config["custom_reward_file"])

    # Load GT reward based on env type
    if env_type == "pandemic":
        from reward_functions.pandemic_gt_rew_fns import PanEtAlTruePandemicRewardFunction
        from pandemic_simulator.environment.pandemic_env import PandemicPolicyGymEnv

        base_env = PandemicPolicyGymEnv(env_config)
        gt_reward = PanEtAlTruePandemicRewardFunction()
        return DualTrackingRewardWrapper(base_env, "pandemic", custom_reward, gt_reward)

    elif env_type == "traffic":
        from reward_functions.traffic_gt_rew_fns import PanEtAlTrueTrafficRewardFunction
        from flow.utils.registry import make_create_env

        create_env_fn, _ = make_create_env(
            params=env_config["flow_params_default"],
            reward_specification=env_config["reward_specification"],
            reward_fun=env_config["reward_fun"],
            reward_scale=env_config["reward_scale"],
        )
        base_env = create_env_fn()
        gt_reward = PanEtAlTrueTrafficRewardFunction()
        return DualTrackingRewardWrapper(base_env, "traffic", custom_reward, gt_reward)

    else:
        raise ValueError(f"Unknown env type: {env_type}")
```

#### 4. Training Loop (in same file)

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-type", type=str, required=True,
                      choices=["pandemic", "traffic"],
                      help="Type of environment to train on")
    parser.add_argument("--custom-reward-file", type=str, required=True,
                      help="Path to Python file defining custom reward function (e.g., written by LLM)")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-gpus", type=int, default=0)
    parser.add_argument("--num-iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    # Initialize Ray
    ray.init()

    # Get environment-specific config (copy-paste from train_with_custom_rew.py)
    if args.env_type == "pandemic":
        from utils.pandemic_config import get_config, get_ppo_config
        env_config = get_config()
        env_config["env_type"] = "pandemic"
        env_config["custom_reward_file"] = args.custom_reward_file
        ppo_config = get_ppo_config(env_config, args.num_gpus, args.seed, args.num_workers)
        register_env("custom_reward_env", create_env_with_custom_reward)
        ppo_config = ppo_config.environment("custom_reward_env", env_config=env_config)

    elif args.env_type == "traffic":
        from utils.traffic_config import get_config, get_ppo_config
        env_config = get_config()
        env_config["env_type"] = "traffic"
        env_config["custom_reward_file"] = args.custom_reward_file
        ppo_config = get_ppo_config("custom_reward_env", env_config, args.num_gpus, args.seed, args.num_workers)
        register_env("custom_reward_env", create_env_with_custom_reward)
        ppo_config = ppo_config.environment("custom_reward_env", env_config=env_config)

    # Enable dual reward tracking callback
    ppo_config = ppo_config.callbacks(DualRewardCallback)

    # Build and train (copy-paste from train_with_custom_rew.py)
    algo = ppo_config.build()

    save_root = Path("logs") / "custom_reward" / args.env_type / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_root.mkdir(parents=True, exist_ok=True)

    for iteration in range(args.num_iterations):
        if iteration % 10 == 0:
            checkpoint = algo.save(checkpoint_dir=str(save_root / f"checkpoint_{iteration}"))
            print(f"Saved checkpoint to {checkpoint}")

        result = algo.train()
        print(f"Iteration {iteration + 1}/{args.num_iterations}")
        print(f"Custom Reward: {result['custom_metrics']['custom_reward_mean']:.2f}")
        print(f"GT Reward: {result['custom_metrics']['gt_reward_mean']:.2f}")

        with open(save_root / f"iter_{iteration}.pkl", 'wb') as f:
            pickle.dump(result, f)

    algo.stop()

if __name__ == "__main__":
    main()
```

## Example Custom Reward File (e.g., written by LLM)

**`examples/custom_rewards/pandemic_simple.py`**:

```python
from abc import ABCMeta, abstractmethod
import numpy as np
from pandemic.python.pandemic_simulator.environment.interfaces import (
    PandemicObservation, InfectionSummary, sorted_infection_summary
)

class RewardFunction(metaclass=ABCMeta):
    @abstractmethod
    def calculate_reward(self, prev_obs: PandemicObservation, action: int, obs: PandemicObservation) -> float:
        pass

class SimpleCriticalReward(RewardFunction):
    """Only penalize critical cases (ignores political/stage costs - likely hackable)."""

    def calculate_reward(self, prev_obs: PandemicObservation, action: int, obs: PandemicObservation) -> float:
        critical_idx = sorted_infection_summary.index(InfectionSummary.CRITICAL)
        return -10.0 * np.mean(obs.global_infection_summary[..., critical_idx])
```

## Usage

```bash
# Train with custom reward function (e.g., written by LLM)
uv run python rl_utils/train_custom_reward.py \
  --env-type pandemic \
  --custom-reward-file examples/custom_rewards/pandemic_simple.py \
  --num-iterations 100 \
  --num-workers 10

# Results contain both rewards:
# - custom_metrics["custom_reward"]: Custom reward function (used for training)
# - custom_metrics["gt_reward"]: GT reward (for measuring hacking)
```

## Measuring Reward Hacking

After training, load pickled results and plot:

- **Custom reward** (custom_metrics["custom_reward_mean"]): Should increase as PPO optimizes the custom function
- **GT reward** (custom_metrics["gt_reward_mean"]): Ideally increases too
- **Reward hacking**: Custom reward increases while GT reward stagnates/decreases

## LLM Workflow

1. LLM sees current reward code (e.g., `pandemic_gt_rew_fns.py`)
2. LLM gets textual feedback (e.g., "reduce critical cases faster")
3. LLM writes edited reward function to `custom_rewards/iteration_1.py`
4. Train: `python train_custom_reward.py --custom-reward-file custom_rewards/iteration_1.py`
5. Analyze: Compare custom reward vs GT reward curves
6. Generate feedback for LLM based on gap
7. Repeat

## Implementation Plan

1. Create `train_custom_reward.py` with all components in one file
2. Create `examples/custom_rewards/` directory
3. Create example reward file
4. Test end-to-end training run
5. Verify metrics are logged correctly

## Summary

**Total code**: ~150 lines in ONE file

**Files created**: 1 (`rl_utils/train_custom_reward.py`)

**Files modified**: 0 (ZERO codebase modifications!)

**Key advantages**:

- Self-contained, easy to understand
- No risk of breaking existing code
- Easy to iterate and debug
- All logic in one place
- Can easily extend for other environments
