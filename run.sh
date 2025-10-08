#!/bin/zsh

uv run python train_custom_reward.py \
    --env-type pandemic \
    --custom-reward-file examples/custom_rewards/pandemic_simple.py \
    --num-iterations 100 \
    --num-workers 2 \
    --seed 0
