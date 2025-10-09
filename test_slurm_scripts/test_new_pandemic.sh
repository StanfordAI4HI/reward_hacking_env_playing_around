#!/bin/zsh
#SBATCH --partition=iris
#SBATCH --account=iris
#SBATCH --time=100:00:00
#SBATCH --nodelist=iris5,iris6,iris7,iris8,iris9,iris10
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=pandemic
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --mail-user=yoonho@stanford.edu
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --output=slurm_logs/slurmjob_%j.out

source /iris/u/yoonho/slurm_init.sh
uv sync

# Set threading to 1 for better multiprocessing performance
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

uv run python train_custom_reward.py \
    --env-type pandemic \
    --custom-reward-file examples/custom_rewards/pandemic_simple.py \
    --num-iterations 100 \
    --num-workers 10 \
    --init-checkpoint \
    --num-gpus 1 \
    --seed 0
