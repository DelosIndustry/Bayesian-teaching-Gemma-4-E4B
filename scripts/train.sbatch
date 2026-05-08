#!/bin/bash
#SBATCH --job-name=e4b-sft-bay
#SBATCH --partition=p02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=48
#SBATCH --mem=220G
#SBATCH --time=168:00:00
#SBATCH --nodelist=DIS02
#SBATCH --output=/abr/coss36/bayesian-teaching/logs/train_bay_%j.out
#SBATCH --error=/abr/coss36/bayesian-teaching/logs/train_bay_%j.err

source ~/.bashrc
conda activate ai
cd /abr/coss36/bayesian-teaching

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"
echo "=========================================="

ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file configs/accelerate_zero3.yaml \
    scripts/run_sft.py configs/e4b_bayesian_sft.yaml

echo "=========================================="
echo "End: $(date)"
echo "=========================================="
