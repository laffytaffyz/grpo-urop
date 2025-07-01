#!/bin/bash
#SBATCH --job-name=gpt2_xl_eval_tiff
#SBATCH --gres=gpu:a100:4
#SBATCH -n 4 
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=gpt2_xl_eval_tiff.out
#SBATCH --error=gpt2_xl_eval_tiff.err
#SBATCH --partition=cbmm

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

python scripts/eval_tiff_gpt_xl.py
