#!/bin/bash
#SBATCH --job-name=comparing-models-gpqa
#SBATCH --gres=gpu:a100:4
#SBATCH -n 4 
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=comparing-models-taco-gpqa.out
#SBATCH --error=comparing-models-taco-gpqa.err

set -x

whoami

source /om/user/tiffany8/miniconda/etc/profile.d/conda.sh
conda activate zero

echo "check gpu status before running job"
nvidia-smi

cd /om/user/tiffany8/grpo-urop/TinyZero

export PYTHONPATH=$PWD:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
export TORCH_USE_CUDA_DSA=1
export HYDRA_FULL_ERROR=1

python scripts/comparing_models_gpqa.py
