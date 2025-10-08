# Custom Reward Functions

This directory contains example custom reward functions that can be used with `rl_utils/train_custom_reward.py` to train PPO agents while tracking reward hacking.

## Overview

The training script trains PPO using a **custom reward function** (e.g., written by an LLM) while simultaneously tracking a **ground truth reward function** in parallel. This enables measuring reward hacking by comparing:

- **Custom reward**: What the agent is optimizing (increases during training)
- **GT reward**: The true objective we care about (may stagnate or decrease)

## Writing Custom Reward Functions

### Required Interface

All custom reward functions must:

1. Inherit from `RewardFunction` abstract base class
2. Implement `calculate_reward(prev_obs, action, obs) -> float` method
3. Be defined in a Python file that can be loaded dynamically

### Pandemic Environment Example

```python
from abc import ABCMeta, abstractmethod
import numpy as np
from pandemic.python.pandemic_simulator.environment.interfaces import (
    PandemicObservation,
    InfectionSummary,
    sorted_infection_summary
)

class RewardFunction(metaclass=ABCMeta):
    @abstractmethod
    def calculate_reward(
        self, prev_obs: PandemicObservation, action: int, obs: PandemicObservation
    ) -> float:
        pass

class SimpleCriticalReward(RewardFunction):
    """Only penalize critical cases (ignores political/stage costs)."""

    def calculate_reward(
        self, prev_obs: PandemicObservation, action: int, obs: PandemicObservation
    ) -> float:
        critical_idx = sorted_infection_summary.index(InfectionSummary.CRITICAL)
        return -10.0 * np.mean(obs.global_infection_summary[..., critical_idx])
```

### Traffic Environment Example

```python
from abc import ABCMeta, abstractmethod
from typing import Sequence
import numpy as np
from flow_reward_misspecification.flow.envs.traffic_obs_wrapper import TrafficObservation

class RewardFunction(metaclass=ABCMeta):
    @abstractmethod
    def calculate_reward(
        self, prev_obs: TrafficObservation, action: Sequence[float], obs: TrafficObservation
    ) -> float:
        pass

class SimpleVelocityReward(RewardFunction):
    """Only reward average vehicle speed (ignores headway and acceleration)."""

    def calculate_reward(
        self, prev_obs: TrafficObservation, action: Sequence[float], obs: TrafficObservation
    ) -> float:
        if getattr(obs, "fail", False):
            return 0.0
        return float(np.mean(obs.all_vehicle_speeds)) if obs.all_vehicle_speeds.size else 0.0
```

## Usage

### Training with Custom Reward

```bash
# Pandemic environment
uv run python rl_utils/train_custom_reward.py \
  --env-type pandemic \
  --custom-reward-file examples/custom_rewards/pandemic_simple.py \
  --num-iterations 100 \
  --num-workers 10

# Traffic environment
uv run python rl_utils/train_custom_reward.py \
  --env-type traffic \
  --custom-reward-file examples/custom_rewards/traffic_simple.py \
  --num-iterations 100 \
  --num-workers 10
```

### Command Line Arguments

- `--env-type`: Environment type (`pandemic` or `traffic`)
- `--custom-reward-file`: Path to Python file containing custom reward function
- `--num-iterations`: Number of training iterations (default: 100)
- `--num-workers`: Number of parallel rollout workers (default: 2)
- `--num-gpus`: Number of GPUs to use (default: 0)
- `--seed`: Random seed (default: 0)
- `--checkpoint-freq`: Save checkpoint every N iterations (default: 10)

### Output

Training results are saved to `logs/custom_reward/{env_type}/{timestamp}/`:

- `config.txt`: Training configuration
- `iter_{i}.pkl`: Pickled training results for each iteration
- `checkpoint_{i}/`: Model checkpoints
- `checkpoint_final/`: Final trained model

Each iteration logs:
- `custom_reward_mean`: Mean custom reward (what PPO optimizes)
- `gt_reward_mean`: Mean ground truth reward (for measuring hacking)
- `episode_reward_mean`: Mean episode reward

## Analyzing Results

After training, load the pickled results and compare custom vs GT reward curves:

```python
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

# Load results
log_dir = Path("logs/custom_reward/pandemic/20250101_120000")
custom_rewards = []
gt_rewards = []

for i in range(100):
    with open(log_dir / f"iter_{i}.pkl", 'rb') as f:
        result = pickle.load(f)
        custom_rewards.append(result["custom_metrics"]["custom_reward_mean"])
        gt_rewards.append(result["custom_metrics"]["gt_reward_mean"])

# Plot
plt.figure(figsize=(10, 6))
plt.plot(custom_rewards, label="Custom Reward (optimized)")
plt.plot(gt_rewards, label="GT Reward (tracked)")
plt.xlabel("Iteration")
plt.ylabel("Reward")
plt.legend()
plt.title("Reward Hacking Analysis")
plt.savefig("reward_hacking.png")
```

### Interpreting Results

- **No hacking**: Both custom and GT rewards increase together
- **Reward hacking**: Custom reward increases while GT reward stagnates or decreases
- **Severity**: Larger gap between custom and GT rewards indicates more severe hacking

## LLM Workflow

1. LLM sees current reward code (e.g., `reward_functions/pandemic_gt_rew_fns.py`)
2. LLM receives textual feedback (e.g., "reduce critical cases faster")
3. LLM writes edited reward function to `custom_rewards/iteration_1.py`
4. Train: `python train_custom_reward.py --custom-reward-file custom_rewards/iteration_1.py`
5. Analyze: Compare custom reward vs GT reward curves
6. Generate feedback for LLM based on the gap
7. Repeat

## Examples in This Directory

- `pandemic_simple.py`: Simplified pandemic reward (only penalizes critical cases)
- `traffic_simple.py`: Simplified traffic reward (only rewards velocity)

Both examples are likely to be hackable because they ignore important aspects of the true objective.
