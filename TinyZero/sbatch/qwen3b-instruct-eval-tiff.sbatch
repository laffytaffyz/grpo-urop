#!/bin/bash
#SBATCH --job-name=qwen_instruct_eval_tiff
#SBATCH --gres=gpu:a100:4
#SBATCH -n 4 
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=qwen_instruct_eval_tiff.out
#SBATCH --error=qwen_instruct_eval_tiff.err

set -x

whoami

source /om/user/tiffany8/miniconda/etc/profile.d/conda.sh
conda activate zero

echo "check gpu status before running job"
nvidia-smi

cd /om/user/tiffany8/grpo-urop/TinyZero

export PYTHONPATH=$PWD:$PYTHONPATH

python scripts/eval_tiff_qwen.py