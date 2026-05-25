#!/bin/bash

#SBATCH --account=3dllms
#SBATCH --partition=mb-h100
#SBATCH --gres=gpu:2
#SBATCH --time=0-06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128gb
#SBATCH --output=/gscratch/tkishore/MLLM-Judge/Experiments/slurm/_%j.out
#SBATCH --error=/gscratch/tkishore/MLLM-Judge/Experiments/slurm/_%j.err

module --force purge
module load arcc/1.0
module load gcc/14.2.0
module load miniconda3
module load cuda-toolkit/12.8.0

source activate /gscratch/tkishore/vllm 



export XDG_CACHE_HOME=/gscratch/tkishore/cache
export HF_HOME=/gscratch/tkishore/cache/hf_cache
export HF_HUB_CACHE=/gscratch/tkishore/cache/hf_cache/hub
export TORCH_HOME=/gscratch/tkishore/cache/torch
export FLASHINFER_CACHE_DIR=/gscratch/tkishore/cache/flashinfer

set -euo pipefail

PROJECT_DIR="/gscratch/tkishore/MLLM-Judge"
cd "$PROJECT_DIR"

# One shared base run id for this Slurm job; each model gets a suffix.
BASE_RUN_ID="${RUN_ID:-slurm_${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)}"

MODELS=(
  "Qwen/Qwen3-VL-4B-Instruct"
  "google/gemma-3-4b-it"
  "Qwen/Qwen3-VL-8B-Instruct"
  "google/gemma-3-12b-it"
  "Qwen/Qwen3-VL-30B-A3B-Instruct"
  "google/gemma-3-27b-it"
)

echo "=========================================="
echo "Slurm Job: ${SLURM_JOB_ID:-N/A}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Running ${#MODELS[@]} models sequentially"
echo "=========================================="

for model in "${MODELS[@]}"; do
  safe_model="${model//\//_}"
  model_run_id="${BASE_RUN_ID}_${safe_model}"

  echo ""
  echo "=========================================="
  echo "Starting model: ${model}"
  echo "RUN_ID: ${model_run_id}"
  echo "=========================================="

  MODEL_NAME="$model" RUN_ID="$model_run_id" bash "$PROJECT_DIR/scripts/run.sh"

done

echo ""
echo "=========================================="
echo "All model runs finished successfully"
echo "=========================================="
