#!/bin/bash                     
#SBATCH --job-name=gpt2-xl-grpo  # Name of the job
#SBATCH --output=/om/user/tiffany8/gpt2_xl_grpo.out
#SBATCH --error=/om/user/tiffany8/gpt2_xl_grpo.err
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
export BASE_MODEL=/om/user/tiffany8/grpo-urop/TinyZero/model/gpt2-xl-sft/global_step_116
export DATA_DIR=/om/user/tiffany8/grpo-urop/TinyZero/dataset
export ROLLOUT_TP_SIZE=1
export EXPERIMENT_NAME=gpt2-xl-grpo
export VLLM_ATTENTION_BACKEND=XFORMERS
export CKPT_DIR=/om/user/tiffany8/grpo-urop/TinyZero/checkpoints/TinyZero/gpt2-xl-grpo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_USE_CUDA_DSA=1
export HYDRA_FULL_ERROR=1

bash ./scripts/train_gpt2_grpo.sh
