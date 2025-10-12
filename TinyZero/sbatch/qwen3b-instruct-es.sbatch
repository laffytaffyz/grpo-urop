#!/bin/bash                     
#SBATCH --job-name=qwen3b-instruct-es  # Name of the job
#SBATCH --output=/om/user/tiffany8/qwen3b_instruct_es_test.out
#SBATCH --error=/om/user/tiffany8/qwen3b_instruct_es_test.err
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:4             # 4 A100 GPU 
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
export DATA_DIR=/om/user/tiffany8/grpo-urop/TinyZero/dataset/gsm
export ROLLOUT_TP_SIZE=1
export EXPERIMENT_NAME=qwen2.5-3b-instruct-es
export VLLM_ATTENTION_BACKEND=XFORMERS
export CKPT_DIR=/om/user/tiffany8/grpo-urop/TinyZero/checkpoints/TinyZero/qwen-3b-instruct-es
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1
export RAY_DEBUG_MODE=1

bash ./scripts/train_es.sh
