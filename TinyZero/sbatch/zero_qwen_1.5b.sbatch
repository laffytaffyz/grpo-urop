#!/bin/bash                     
#SBATCH --job-name=zero_qwen_1.5b   # Name of the job
#SBATCH --output=/om/user/tiffany8/zero_qwen_1.5b_output.log
#SBATCH --error=/om/user/tiffany8/zero_qwen_1.5b_error.log
#SBATCH -n 4                          # 4 CPU core
#SBATCH --gres=gpu:a100:4             # 6 A100 GPU 
#SBATCH --mem=128G
#SBATCH --time=48:00:00               # Time limit (HH:MM:SS)

set -x

whoami

source /om/user/tiffany8/miniconda/etc/profile.d/conda.sh
conda activate zero 

echo "check gpu status before running job"
nvidia-smi

cd /om/user/tiffany8/grpo-urop/TinyZero

export N_GPUS=4
export BASE_MODEL=/om/user/tiffany8/TinyZero/model/Qwen2.5-1.5B
export DATA_DIR=/om/user/tiffany8/TinyZero/dataset
export ROLLOUT_TP_SIZE=4
export EXPERIMENT_NAME=countdown-qwen2.5-1.5b
export VLLM_ATTENTION_BACKEND=XFORMERS

bash ./scripts/train_tiny_zero.sh
