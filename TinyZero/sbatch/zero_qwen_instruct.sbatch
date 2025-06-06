#!/bin/bash                     
#SBATCH --job-name=zero_qwen_instruct   # Name of the job
#SBATCH --output=/om/user/tiffany8/zero_qwen_instruct_output.log
#SBATCH --error=/om/user/tiffany8/zero_qwen_instruct_error.log
#SBATCH -n 2                     # 2 CPU core
#SBATCH --gres=gpu:2             # 2 A100 GPU 
#SBATCH --mem=64G
#SBATCH --time=168:00:00               # Time limit (HH:MM:SS)
#SBATCH --partition=cbmm

set -x

whoami

source /om/user/tiffany8/miniconda/etc/profile.d/conda.sh
conda activate zero 

echo "check gpu status before running job"
nvidia-smi

cd /om/user/tiffany8/grpo-urop/TinyZero

export N_GPUS=2
export BASE_MODEL=/om/user/tiffany8/TinyZero/model/Qwen2.5-3B-Instruct
export DATA_DIR=/om/user/tiffany8/TinyZero/qwen_dataset
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=countdown-qwen2.5-3b-instruct
export VLLM_ATTENTION_BACKEND=XFORMERS

bash ./scripts/train_tiny_zero.sh
