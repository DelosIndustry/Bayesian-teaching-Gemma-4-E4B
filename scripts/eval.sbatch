#!/bin/bash
#SBATCH --job-name=e4b-eval
#SBATCH --partition=p02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/abr/coss36/bayesian-teaching/logs/eval_%j.out
#SBATCH --error=/abr/coss36/bayesian-teaching/logs/eval_%j.err

source ~/.bashrc
conda activate ai
cd /abr/coss36/bayesian-teaching

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"
echo "=========================================="

# TODO: Fill in evaluation command (Task 7)
# Example:
# python -m src.evaluation \
#     --model_path /abr/coss36/bayesian-teaching/models/e4b-full-bayesian \
#     --teaching bayesian \
#     --domain flight_4features \
#     --quantization bf16 \
#     --output_dir results/

echo "Evaluation placeholder - will be filled in Task 7"

echo "=========================================="
echo "End: $(date)"
echo "=========================================="
