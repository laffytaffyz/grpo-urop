#!/bin/bash                     
#SBATCH --job-name=qwen3b-instruct-ppo  # Name of the job
#SBATCH --output=/om/user/tiffany8/qwen3b_instruct_ppo.out
#SBATCH --error=/om/user/tiffany8/qwen3b_instruct_ppo.err
#SBATCH -n 4                     # 4 CPU core
#SBATCH --gres=gpu:a100:4             # 4 A100 GPU 
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
export BASE_MODEL=/om/user/tiffany8/grpo-urop/TinyZero/model/Qwen2.5-3B-Instruct
export DATA_DIR=/om/user/tiffany8/grpo-urop/TinyZero/qwen_dataset
export ROLLOUT_TP_SIZE=1
export EXPERIMENT_NAME=qwen2.5-3b-instruct-ppo
export VLLM_ATTENTION_BACKEND=XFORMERS
export CKPT_DIR=/om/user/tiffany8/grpo-urop/TinyZero/checkpoints/TinyZero/qwen-3b-instruct-ppo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

bash ./scripts/train_qwen3b_instruct_ppo.sh
