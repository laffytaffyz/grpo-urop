torchrun -m verl.trainer.fsdp_sft_trainer \
data.prompt_key=prompts \
data.response_key=target \
data.micro_batch_size=8 \
model.partial_pretrain=$BASE_MODEL \
trainer.default_local_dir=$CKPT_DIR \
trainer.project_name=urop \
trainer.experiment_name=gpt2-sft \
trainer.total_epochs=4 \
trainer.logger=['console','wandb']