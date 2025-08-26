#!/bin/bash
#SBATCH --job-name=quicktest
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --mem=128G
#SBATCH --output=quicktest.out
#SBATCH --error=quicktest.err

set -x
whoami
source /om/user/tiffany8/miniconda/etc/profile.d/conda.sh
conda activate zero 
nvidia-smi

python /om/user/tiffany8/grpo-urop/TinyZero/scripts/quicktest.py