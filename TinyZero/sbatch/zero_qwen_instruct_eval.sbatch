#!/bin/bash                     
#SBATCH --job-name=zero_qwen_instruct_eval   # Name of the job
#SBATCH --output=/om/user/tiffany8/zero_qwen_instruct_eval_output.log
#SBATCH --error=/om/user/tiffany8/zero_qwen_instruct_eval_error.log
#SBATCH -n 4                     # 2 CPU core
#SBATCH --gres=gpu:4             # 2 A100 GPU 
#SBATCH --mem=64G
#SBATCH --time=48:00:00               # Time limit (HH:MM:SS)

set -x

whoami

source /om/user/tiffany8/miniconda/etc/profile.d/conda.sh
conda activate zero 

echo "check gpu status before running job"
nvidia-smi

cd /om/user/tiffany8/grpo-urop/TinyZero

export N_GPUS=4
export ACTOR=/om/user/tiffany8/TinyZero/backup/20250401/TinyZero/countdown-qwen2.5-3b-instruct/actor/global_step_3600
export CRITIC=/om/user/tiffany8/TinyZero/backup/20250401/TinyZero/countdown-qwen2.5-3b-instruct/critic/global_step_3600
export DATA_DIR=/om/user/tiffany8/TinyZero/qwen_dataset
export ROLLOUT_TP_SIZE=4
export EXPERIMENT_NAME=countdown-qwen2.5-3b-instruct-eval
export VLLM_ATTENTION_BACKEND=XFORMERS

bash ./scripts/eval_tiff.sh
