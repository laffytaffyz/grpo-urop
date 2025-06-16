import hydra
import numpy as np
import json

import torch
import torch.distributed
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoConfig

from verl import DataProto
from verl.utils import hf_tokenizer
from verl.utils.model import get_generation_config
from verl.utils.fs import copy_local_path_from_hdfs
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.workers.reward_manager import NaiveRewardManager
from verl.workers.rollout.hf_rollout import HFRollout

# change this
model_path = '/om/user/tiffany8/grpo-urop/TinyZero/model/Qwen2.5-3B-Instruct' 
data_dir = "/om/user/tiffany8/TinyZero/qwen_dataset/train.parquet"
if ('nstruct' in model_path or 'chat' in model_path) and 'qwen' not in data_dir: raise ValueError("Expected instruct dataset for instruct model")
if ('nstruct' not in model_path and 'chat' not in model_path) and 'qwen' in data_dir: raise ValueError("Expected noninstruct dataset for noninstruct model")

local_path = copy_local_path_from_hdfs(model_path)

trust_remote_code = True
tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

torch_dtype = torch.bfloat16

actor_model_config = AutoConfig.from_pretrained(
    local_path, trust_remote_code=trust_remote_code
)

actor_module = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=local_path,
    torch_dtype=torch_dtype,
    config=actor_model_config,
    attn_implementation="flash_attention_2",
    trust_remote_code=trust_remote_code,
)

actor_module.to("cuda:0")

generation_config = get_generation_config(
    local_path, trust_remote_code=trust_remote_code
)
generation_config.n = 2 

# reward_fn = NaiveRewardManager(
#     tokenizer=tokenizer, num_examine=1, compute_score=None
# )

@hydra.main()
def main(config):

    dataset = RLHFDataset(
        parquet_files=data_dir, 
        tokenizer=tokenizer,
        prompt_key="prompt",
        max_prompt_length=512,
        filter_prompts=True,
        return_raw_chat=False,
        truncation="error",
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=3,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    assert len(dataloader) >= 1
    sample_inputs = []
    sample_outputs = []
    # sample_scores = []
    # reward_tensor_lst = []
    # data_source_lst = []

    hfrollout = HFRollout(module=actor_module, config=config)
    for data in dataloader:
        batch = DataProto.from_single_dict(data)
        batch = batch.to("cuda")
        input_ids = batch.batch["input_ids"]
        input_texts = [
            tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids
        ]
        # sample_inputs.extend(input_texts)

        gen_batch = batch.pop(["input_ids", "attention_mask", "position_ids"])
        gen_batch.meta_info = {
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": True,
            "validate": False,
            "n": 2,
        }

        # pad to be divisible by dp_size
        gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, 1)
        output_gen_batch_padded = hfrollout.generate_sequences(
            gen_batch_padded
        )
        # unpad
        output_gen_batch = unpad_dataproto(
            output_gen_batch_padded, pad_size=pad_size
        )
        # print("Generation end")

        # Store generated outputs
        output_ids = output_gen_batch.batch["responses"]
        output_texts = [
            tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids
        ]
        # sample_outputs.extend(output_texts)
        batch = batch.union(output_gen_batch)

        for prompt, responses in zip(input_texts, output_texts):
            print("PROMPT:")
            print(prompt)
            print("RESPONSE:")
            print(response)
            print(20 * "~")
            samples.append({
                "prompt": prompt,
                "responses": responses  # list of generated outputs
            })

        # data_source_lst.append(
        #     batch.non_tensor_batch.get(
        #         "data_source", ["unknown"] * reward_tensor.shape[0]
        #     )
        # )

    # # evaluate score based on data source
    # data_source_reward = {}
    # for i in range(reward_tensor.shape[0]):
    #     data_source = data_sources[i]
    #     if data_source not in data_source_reward:
    #         data_source_reward[data_source] = []
    #     data_source_reward[data_source].append(reward_tensor[i].item())

    # metric_dict = {}
    # for data_source, rewards in data_source_reward.items():
    #     metric_dict[f"val/score/{data_source}"] = np.mean(rewards)
    
    
    with open("cand.jsonl", "w") as f:
        for entry in samples:
            f.write(json.dumps(entry) + "\n")
    
    # print(metric_dict)


if __name__ == "__main__":
    main()
