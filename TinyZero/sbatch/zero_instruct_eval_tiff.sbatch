#!/bin/bash
#SBATCH --job-name=zero_instruct_eval_tiff
#SBATCH --gres=gpu:a100:2
#SBATCH -n 2 
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=zero_instruct_eval_tiff.out
#SBATCH --error=zero_instruct_eval_tiff.err

set -x

whoami

source /om/user/tiffany8/miniconda/etc/profile.d/conda.sh
conda activate zero

echo "check gpu status before running job"
nvidia-smi

cd /om/user/tiffany8/grpo-urop/TinyZero

python scripts/eval_tiff.py
