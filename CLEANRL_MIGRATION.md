# CleanRL + PufferLib Migration Guide

## Overview
This document describes the migration from RLlib to CleanRL PPO with PufferLib vectorization in `train_with_custom_rew.py`.

## Key Changes

### 1. Dependencies
**Old (RLlib):**
- `ray`
- `ray.rllib`

**New (CleanRL + PufferLib):**
- `torch` (PyTorch)
- `pufferlib` (version 3.0.0)
- `cleanrl` (version 1.2.0) - for reference implementation

### 2. Vectorization
**Old:** RLlib's `num_rollout_workers` and `num_envs_per_worker`
**New:** PufferLib's `pufferlib.vector.make()` with `num_envs` parameter

PufferLib provides high-speed local vectorization without the overhead of Ray's distributed system.

### 3. Agent Architecture
**Old:** RLlib's built-in PPO models
**New:** Custom PyTorch `Agent` class with:
- Separate actor and critic networks
- Continuous action space with Gaussian policy
- Orthogonal initialization (CleanRL standard)

### 4. Training Loop
**Old:** RLlib's `algo.train()` abstraction
**New:** Explicit PPO implementation:
- Manual rollout collection
- GAE (Generalized Advantage Estimation)
- Minibatch updates with clipped surrogate objective
- Value function clipping
- Gradient clipping

### 5. Checkpointing
**Old:** RLlib checkpoint format with policy serialization
**New:** PyTorch `.pt` files with:
```python
{
    'update': update_number,
    'model_state_dict': agent.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'args': args,
}
```

## PPO Parameter Mapping

### Directly Mapped Parameters

| RLlib Config | CleanRL Arg | Notes |
|-------------|-------------|-------|
| `lr` | `--learning-rate` | Learning rate |
| `gamma` | `--gamma` | Discount factor |
| `lambda` (gae_lambda) | `--gae-lambda` | GAE lambda |
| `clip_param` | `--clip-coef` | PPO clipping coefficient |
| `vf_loss_coeff` | `--vf-coef` | Value function coefficient |
| `entropy_coeff` | `--ent-coef` | Entropy coefficient |
| `grad_clip` | `--max-grad-norm` | Gradient clipping |
| `num_sgd_iter` | `--update-epochs` | Number of update epochs |
| `sgd_minibatch_size` | `--num-minibatches` | Number of minibatches |

### RLlib-Specific (Not Directly Mapped)

**Parameters that don't have CleanRL equivalents:**

1. **`kl_target` / `kl_coeff`**: RLlib's adaptive KL penalty system
   - CleanRL uses `target_kl` for early stopping, but it's conceptually different
   - RLlib dynamically adjusts KL coefficient during training

2. **`vf_clip_param`**: RLlib's value function clipping threshold
   - CleanRL has `--clip-vloss` flag but uses `clip_coef` value
   - Different implementation philosophy

3. **`rollout_fragment_length`**: RLlib's fragment collection size
   - Maps conceptually to `--num-steps` in CleanRL
   - CleanRL: steps per environment before update
   - RLlib: total steps across all workers

4. **`train_batch_size`**: RLlib's total batch size
   - CleanRL computes as `num_envs * num_steps`
   - No direct parameter

5. **`batch_mode`**: RLlib-specific ("truncate_episodes", "complete_episodes")
   - CleanRL always truncates at `num_steps`

6. **`callbacks`**: RLlib's callback hooks
   - No equivalent in CleanRL
   - Custom logging must be implemented inline

7. **`entropy_coeff_schedule`** / **`lr_schedule`**: RLlib's schedule system
   - CleanRL has simpler `--anneal-lr` flag for linear annealing
   - No entropy scheduling support

8. **`num_rollout_workers`** / **`num_envs_per_worker`**: RLlib parallelization
   - Replaced by PufferLib's `num_envs`
   - Local vectorization instead of distributed

### CleanRL-Specific (Not in RLlib)

**Parameters unique to CleanRL:**

1. **`--anneal-lr`**: Simple linear learning rate annealing
   - RLlib has more complex `lr_schedule`
   
2. **`--norm-adv`**: Normalize advantages
   - RLlib does this by default (not configurable)
   
3. **`--clip-vloss`**: Clip value loss
   - Different from RLlib's `vf_clip_param`
   
4. **`--target-kl`**: Early stopping based on KL divergence
   - Different from RLlib's `kl_target` (adaptive penalty)
   
5. **`--torch-deterministic`**: Set PyTorch deterministic mode
   - Framework-specific

## Environment-Specific Parameter Usage

### Pandemic Environment
```python
# From utils/pandemic_config.py
lr = 0.0003
gamma = 0.99
gae_lambda = 0.95
clip_param = 0.3
vf_loss_coeff = 0.5
entropy_coeff = 0.01
grad_clip = 10
num_sgd_iter = 5

# NOT MAPPED:
# - kl_target = 0.01
# - vf_clip_param = 20
# - entropy_coeff_schedule (0.1 -> 0.01 over 500k steps)
```

### Glucose Environment
```python
# From utils/glucose_config.py
lr = 1e-4
gamma = 0.99
gae_lambda = 0.98
clip_param = 0.05
vf_loss_coeff = 1e-4  # NOTE: Very small!
entropy_coeff = 0.01
grad_clip = 10
num_sgd_iter = 4

# NOT MAPPED:
# - kl_coeff = 0.2
# - kl_target = 1e-3
# - vf_clip_param = 100
# - normalize_actions = False
```

### Traffic Environment
```python
# From utils/traffic_config.py
lr = 5e-5
gamma = 0.99
gae_lambda = 0.97
vf_loss_coeff = 0.5
entropy_coeff = 0.01
grad_clip = None  # No clipping
num_sgd_iter = 5

# NOT MAPPED:
# - kl_target = 0.02
# - vf_clip_param = 10000 (effectively disabled)
# - lr_decay_schedule (constant in this case)
```

## Usage Examples

### Basic Training
```bash
python rl_utils/train_with_custom_rew.py \
    --env-type pandemic \
    --reward-fun-type gt_reward_fn \
    --num-envs 8 \
    --num-steps 2048 \
    --total-timesteps 1000000
```

### With Custom Hyperparameters
```bash
python rl_utils/train_with_custom_rew.py \
    --env-type glucose \
    --reward-fun-type proxy_reward_fn \
    --num-envs 4 \
    --num-steps 2000 \
    --learning-rate 1e-4 \
    --clip-coef 0.05 \
    --vf-coef 1e-4 \
    --update-epochs 4 \
    --total-timesteps 2000000
```

### Loading from Checkpoint
```bash
python rl_utils/train_with_custom_rew.py \
    --env-type traffic \
    --reward-fun-type gt_reward_fn \
    --init-checkpoint logs/data/traffic/20250101_120000/checkpoint_100.pt \
    --total-timesteps 500000
```

## Important Notes

### 1. Checkpoint Compatibility
- Old RLlib checkpoints **cannot** be directly loaded
- Network architectures may differ significantly
- If you need to migrate old policies, you'll need custom conversion scripts

### 2. Performance Considerations
- PufferLib vectorization is typically faster than Ray for local training
- Memory usage is different (all envs in same process)
- GPU utilization is more efficient with local vectorization

### 3. Hyperparameter Tuning
- Parameters extracted from RLlib configs are starting points
- May need retuning for CleanRL implementation
- Notable differences:
  - No adaptive KL penalty (consider manual tuning)
  - Simpler value clipping (may affect stability)
  - No entropy scheduling (fixed coefficient)

### 4. Logging and Monitoring
- No built-in TensorBoard integration (add manually if needed)
- Results saved as pickle files in `{env}_running_results/`
- Checkpoints saved in `logs/data/{env}/{timestamp}/`

### 5. Custom Models
- Current implementation uses simple MLP
- For custom architectures (like the discriminator models in RLlib configs):
  - Modify the `Agent` class in `train_with_custom_rew.py`
  - Ensure forward pass matches CleanRL interface

## Migration Checklist

- [x] Replace RLlib imports with PyTorch/PufferLib
- [x] Implement CleanRL-style Agent class
- [x] Rewrite training loop with explicit PPO
- [x] Map hyperparameters from config files
- [x] Flag unmapped parameters in config files
- [x] Update checkpoint format
- [x] Add evaluation function
- [ ] (Optional) Add TensorBoard logging
- [ ] (Optional) Implement custom model architectures
- [ ] (Optional) Add entropy coefficient scheduling
- [ ] (Optional) Implement adaptive KL penalty

## References

- CleanRL PPO implementation: https://github.com/vwxyzjn/cleanrl
- PufferLib documentation: https://github.com/PufferAI/PufferLib
- Original paper: Schulman et al., "Proximal Policy Optimization Algorithms"
