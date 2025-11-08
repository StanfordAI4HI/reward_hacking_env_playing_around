#!/bin/bash
#SBATCH --partition=next
#SBATCH --account=next
#SBATCH --nodelist=next5
#SBATCH --time=200:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=pandemic_rew_learning
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G

### Logging
#SBATCH --output=/next/u/stephhk/reward_hacking_env_playing_around/run_logs/slurmjob_%j.out                    # Name of stdout output file (%j expands to jobId)
#SBATCH --error=/next/u/stephhk/reward_hacking_env_playing_around/run_logs/slurmjob_%j.err                        # Name of stderr output file (%j expands to jobId)
#SBATCH --mail-user=stephhk@stanford.edu # Email of notification
#SBATCH --mail-type=END,FAIL,REQUEUE

cd /next/u/stephhk/reward_hacking_env_playing_around/
conda activate orpo

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
exec 1>&2  # Redirect stdout (fd 1) to stderr (fd 2)

python3 -m rl_utils.collect_trajectories \
    --checkpoint-path /next/u/stephhk/orpo/data/base_policy_checkpoints/pandemic_base_policy/checkpoint_000100/ \
    --env-type pandemic \
    --num-trajectories 10 \
    --output-dir bc_pol_trajectories