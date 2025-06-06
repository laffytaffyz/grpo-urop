#!/bin/bash                     
#SBATCH --job-name=coconut_prontoqa_dataset   # Name of the job
#SBATCH --output=/om/user/tiffany8/coconut_prontoqa_dataset_output.log
#SBATCH --error=/om/user/tiffany8/coconut_prontoqa_dataset_error.log
#SBATCH -n 1                     # 1 CPU core
#SBATCH --gres=gpu:a100:1             # 1 A100 GPU 
#SBATCH --mem=64G
#SBATCH --time=2:00:00               # Time limit (HH:MM:SS)
#SBATCH --partition=cbmm

set -x

whoami

source /om/user/tiffany8/miniconda/etc/profile.d/conda.sh
conda activate coconut

echo "check gpu status before running job"
nvidia-smi

cd /om/user/tiffany8/grpo-urop/coconut/prontoqa_github/prontoqa
python run_experiment.py --model-name json --model-size dummy --ordering random --num-trials 10000 --few-shot-examples 0 --ontology fictional --min-hops 5 --max-hops 5 --hops-skip 1

