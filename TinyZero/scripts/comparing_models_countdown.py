import hydra
import numpy as np
from collections import defaultdict
import pandas as pd
import re
import os

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

### ~~PICK MODEL PATH~~
model_paths = ["/om/user/tiffany8/grpo-urop/TinyZero/model/Llama-3.1-8B-Instruct",
                "/om2/user/tiffany8/checkpoints/TinyZero/llama-8b-instruct-grpo-taco/actor/global_step_50",

                "/om/user/tiffany8/grpo-urop/TinyZero/model/deepseek-math-7b-instruct",
                "/om2/user/tiffany8/checkpoints/TinyZero/deepmath7b-instruct-taco/actor/global_step_100", 

                "/om/user/tiffany8/grpo-urop/TinyZero/model/Mistral-7B-Instruct-v0.3",
                "/om2/user/tiffany8/checkpoints/TinyZero/mistral-7b-instruct-grpo-taco/actor/global_step_50",
                
                "/om/user/tiffany8/grpo-urop/TinyZero/model/Qwen2.5-3B-Instruct",
                "/om2/user/tiffany8/checkpoints/TinyZero/qwen-3b-instruct-grpo-taco/actor/global_step_200",
                "/om2/user/tiffany8/checkpoints/TinyZero/qwen-3b-instruct-grpo/actor/global_step_500",

                "/om/user/tiffany8/grpo-urop/TinyZero/model/Llama-3.2-3B-Instruct",
                "/om2/user/tiffany8/checkpoints/TinyZero/llama-3b-instruct-grpo-taco/actor/global_step_200",
                "/om2/user/tiffany8/checkpoints/TinyZero/llama-3b-instruct-grpo/actor/global_step_100",
                "/om2/user/tiffany8/checkpoints/TinyZero/llama-3b-instruct-grpo/actor/global_step_300",
                "/om2/user/tiffany8/checkpoints/TinyZero/llama-3b-instruct-grpo/actor/global_step_700"]

model_names = ["llama8b base", "llama8b taco", "deepmath base", "deepmath taco", "mistral base", "mistral taco",
                "qwen base", "qwen taco", "qwen 500",
                "llama base", "llama taco", "llama 100", "llama 300", "llama 700"]

model_paths = ["/om/user/tiffany8/grpo-urop/TinyZero/checkpoints/TinyZero/llama-3b-instruct-grpo-countdown-answer-only/actor/global_step_500",
                "/om/user/tiffany8/grpo-urop/TinyZero/checkpoints/TinyZero/llama-3b-instruct-grpo-countdown-coherence/actor/global_step_110",
                "/om/user/tiffany8/grpo-urop/TinyZero/checkpoints/TinyZero/llama-3b-instruct-grpo-countdown-lang-consistency/actor/global_step_500",
                "/om/user/tiffany8/grpo-urop/TinyZero/checkpoints/TinyZero/llama-3b-instruct-grpo-countdown-lang-diversity/actor/global_step_100",
                "/om2/user/tiffany8/checkpoints/TinyZero/qwen-3b-instruct-grpo-countdown-answer-only/actor/global_step_600",
                "/om2/user/tiffany8/checkpoints/TinyZero/qwen-3b-instruct-grpo-coherence/actor/global_step_100",
                "/om2/user/tiffany8/checkpoints/TinyZero/qwen-3b-instruct-grpo-lang-consistency/actor/global_step_400",
                "/om2/user/tiffany8/checkpoints/TinyZero/qwen-3b-instruct-grpo-lang-diversity/actor/global_step_400"
                ]

model_names = ['llama answer only','llama coherence','llama consistency','llama diversity',
                'qwen answer only','qwen coherence','qwen consistency','qwen diversity']

# model_paths = ["/om/user/tiffany8/grpo-urop/TinyZero/model/Qwen2.5-3B-Instruct",
#                 "/om/user/tiffany8/grpo-urop/TinyZero/checkpoints/TinyZero/qwen-3b-instruct-ppo/actor/global_step_1300",
#                 "/om2/user/tiffany8/checkpoints/TinyZero/qwen-3b-instruct-grpo/actor/global_step_500",
                
#                 "/om/user/tiffany8/grpo-urop/TinyZero/model/Llama-3.2-3B-Instruct",
#                 "/om2/user/tiffany8/checkpoints/TinyZero/llama-3b-instruct-ppo/actor/global_step_800",
#                 "/om2/user/tiffany8/checkpoints/TinyZero/llama-3b-instruct-grpo/actor/global_step_700"]

# model_names = ["qwen base", "qwen ppo", "qwen grpo", "llama base", "llama ppo", "llama grpo"]

### ~~PICK DATA~~
# data_path = "/om/user/tiffany8/grpo-urop/TinyZero/dataset/test.parquet"
data_path = "/om/user/tiffany8/grpo-urop/TinyZero/qwen_dataset/test.parquet"

assert len(model_paths) == len(model_names)

@hydra.main()
def main(config):
    all_model_outputs = defaultdict(list)
    metric_dicts = []

    for i in range(len(model_paths)):
        model_path = model_paths[i]
        model_name = model_names[i]
        print("EVALUATING:", model_names[i])
        local_path = copy_local_path_from_hdfs(model_path)
        data_counter = 0
        verbose = False

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

        # ckpt_new_version = "checkpoints/TinyZero/grpo-countdown-qwen2.5-3b-v2/global_step_100/actor/model_world_size_2_rank_0.pt"
        # model_state = torch.load(ckpt_new_version, map_location="cpu")
        # actor_module.load_state_dict(model_state)

        # actor_module.to(torch_dtype) # change by tiffany
        actor_module.to("cuda:0")

        generation_config = get_generation_config(
            local_path, trust_remote_code=trust_remote_code
        )

        val_reward_fn = NaiveRewardManager(
            tokenizer=tokenizer, num_examine=1, compute_score=None
        )

        val_dataset = RLHFDataset(
            parquet_files=data_path, 
            tokenizer=tokenizer,
            prompt_key="prompt",
            max_prompt_length=512,
            filter_prompts=True,
            return_raw_chat=False,
            truncation="error",
        )
        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=3,
            shuffle=False,
            drop_last=True,
            collate_fn=collate_fn,
        )

        assert len(val_dataloader) >= 1
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        reward_tensor_lst = []
        data_source_lst = []

        hfrollout = HFRollout(module=actor_module, config=config)
        for data in val_dataloader:
            test_batch = DataProto.from_single_dict(data)
            test_batch = test_batch.to("cuda")
            input_ids = test_batch.batch["input_ids"]
            input_texts = [
                tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids
            ]
            sample_inputs.extend(input_texts)

            test_gen_batch = test_batch.pop(["input_ids", "attention_mask", "position_ids"])
            test_gen_batch.meta_info = {
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": True,
                "validate": True,
            }

            # pad to be divisible by dp_size
            # test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, hfrollout.world_size)
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, 1)
            test_output_gen_batch_padded = hfrollout.generate_sequences(
                test_gen_batch_padded
            )
            # unpad
            test_output_gen_batch = unpad_dataproto(
                test_output_gen_batch_padded, pad_size=pad_size
            )
            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [
                tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids
            ]
            sample_outputs.extend(output_texts)
            test_batch = test_batch.union(test_output_gen_batch)

            # store into dict
            for prompt, response in zip(input_texts, output_texts):
                all_model_outputs[prompt].append(response)

            # evaluate using reward_function
            reward_tensor = val_reward_fn(test_batch)
            # Store scores
            scores = reward_tensor.sum(-1).cpu().tolist()
            print('scores',scores)
            sample_scores.extend(scores)
            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(
                test_batch.non_tensor_batch.get(
                    "data_source", ["unknown"] * reward_tensor.shape[0]
                )
            )

            data_counter += 1
            if data_counter >= 120: break
            break

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)

        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f"val/test_score/{data_source}"] = np.mean(rewards)
        
        # debugging
        # with open("eval_outputs.txt", "w") as f:
        #     for p, r, s in zip(sample_inputs, sample_outputs, sample_scores):
        #         f.write(f"Prompt: {p}\nResponse: {r}\nReward: {s:.4f}\n\n")

        print('metric dictionary:', metric_dict)

        metric_dicts.append((model_name, metric_dict))

    rows = []
    print("\n\n==== Prompt-wise Model Outputs ====")
    for prompt, responses in all_model_outputs.items():
        row = {"prompt": prompt}
        print(f"\nPrompt: {prompt}\n")
        for model_name, response in zip(model_names, responses):
            row[model_name] = response
            row[f"{model_name}_len"] = len(response)
            print(f"[{model_name}]: {response}")
            print()
        rows.append(row)

    df = pd.DataFrame(rows)
    print(df)
    df.to_json("/om/user/tiffany8/grpo-urop/TinyZero/response_analysis/countdown/responses_countdown.jsonl", orient="records", lines=True)

    print('all metric dictionaries:', metric_dicts)

if __name__ == "__main__":
    main()
