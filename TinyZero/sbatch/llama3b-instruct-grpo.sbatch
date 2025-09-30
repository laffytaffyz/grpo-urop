#!/bin/bash                     
#SBATCH --job-name=llama3b-instruct-grpo-gsm-script-diversity  # Name of the job
#SBATCH --output=/om/user/tiffany8/llama3b_instruct_grpo_gsm_script_diversity.out
#SBATCH --error=/om/user/tiffany8/llama3b_instruct_grpo_gsm_script_diversity.err
#SBATCH -n 4                     # 4 CPU core
#SBATCH --gres=gpu:a100:4             # 4 A100 GPU 
#SBATCH --mem=512G
#SBATCH --time=48:00:00               # Time limit (HH:MM:SS)

set -x

whoami

source /om/user/tiffany8/miniconda/etc/profile.d/conda.sh
conda activate zero 

echo "check gpu status before running job"
nvidia-smi

cd /om/user/tiffany8/grpo-urop/TinyZero

export N_GPUS=4
export BASE_MODEL=/om/user/tiffany8/grpo-urop/TinyZero/model/Llama-3.2-3B-Instruct
export DATA_DIR=/om/user/tiffany8/grpo-urop/TinyZero/dataset/gsm
export ROLLOUT_TP_SIZE=1
export EXPERIMENT_NAME=llama-3b-instruct-gsm-script-diversity
export VLLM_ATTENTION_BACKEND=XFORMERS
export CKPT_DIR=/om/user/tiffany8/grpo-urop/TinyZero/checkpoints/TinyZero/llama-3b-instruct-grpo-gsm-script-diversity
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

bash ./scripts/train_llama_instruct_grpo.sh
