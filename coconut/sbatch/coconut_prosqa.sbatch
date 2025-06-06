#!/bin/bash                     
#SBATCH --job-name=coconut_prosqa   # Name of the job
#SBATCH --output=/om/user/tiffany8/coconut_prosqa_output.log
#SBATCH --error=/om/user/tiffany8/coconut_prosqa_error.log
#SBATCH -n 1                     # 1 CPU core
#SBATCH --gres=gpu:a100:4             # 4 A100 GPU 
#SBATCH --mem=64G
#SBATCH --time=30:00:00               # Time limit (HH:MM:SS)
#SBATCH --partition=cbmm

set -x

whoami

source /om/user/tiffany8/miniconda/etc/profile.d/conda.sh
conda activate coconut

echo "check gpu status before running job"
nvidia-smi

cd /om/user/tiffany8/grpo-urop/coconut

torchrun --nnodes 1 --nproc_per_node 4 /om/user/tiffany8/grpo-urop/coconut/run.py /om/user/tiffany8/grpo-urop/coconut/args/prosqa_coconut.yaml
