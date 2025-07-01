import hydra
import numpy as np
from collections import defaultdict

import torch
import torch.distributed
import tensordict
from tensordict import TensorDict
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
from collections import defaultdict

### ~~PICK MODEL PATH~~
# gpt2 xl
# TODO: change all these paths to xl
model_paths = ["/om/user/tiffany8/grpo-urop/TinyZero/model/gpt2-xl-sft/global_step_116",
                "/om/user/tiffany8/grpo-urop/TinyZero/checkpoints/TinyZero/gpt2-xl-ppo/actor/global_step_500",
                "/om/user/tiffany8/grpo-urop/TinyZero/checkpoints/TinyZero/gpt2-xl-grpo/actor/global_step_300,
                "/om/user/tiffany8/grpo-urop/TinyZero/checkpoints/TinyZero/gpt2-xl-reinforce/actor/global_step_700",
                "/om/user/tiffany8/grpo-urop/TinyZero/checkpoints/TinyZero/gpt2-xl-dpo/actor/global_step_800"]

### ~~PICK DATA~~
data_path = "/om/user/tiffany8/grpo-urop/TinyZero/dataset/test.parquet"
# data_path = "/om/user/tiffany8/grpo-urop/TinyZero/qwen_dataset/test.parquet"

model_names = [path.split("/")[-3] for path in model_paths] 
model_names[0] = 'base'

assert len(model_paths) == len(model_names)

@hydra.main()
def main(config):
    all_model_outputs = defaultdict(list)
    metric_dicts = []

    for i in range(len(model_paths)):
        model_path = model_paths[i]
        print("EVALUATING:", model_names[i])
        local_path = copy_local_path_from_hdfs(model_path)

        trust_remote_code = True
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        tokenizer.padding_side = 'right'

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
            device = input_ids.device
            input_texts = [
                tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids
            ]
            sample_inputs.extend(input_texts)

            test_gen_batch = test_batch.pop(["input_ids", "attention_mask"])
            test_gen_batch.meta_info = {
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": False,
                "validate": True,
            }

            max_length = actor_module.config.n_positions
            pos_ids = torch.arange(input_ids.shape[1], device=device)
            pos_ids = pos_ids.clamp(max=max_length - 1)
            test_gen_batch.batch["position_ids"] = pos_ids.unsqueeze(0).expand_as(input_ids)

            # pad to be divisible by dp_size
            # test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, hfrollout.world_size)
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, 1)

            # set token generation limit 
            prompt_len = input_ids.shape[1]
            safe_new_tokens = max_length - prompt_len
            safe_new_tokens = max(safe_new_tokens, 1)

            # test_gen_batch_padded.meta_info["max_new_tokens"] = safe_new_tokens
            # test_gen_batch_padded.meta_info["max_length"] = max_length

            # generate
            input_ids = test_gen_batch.batch["input_ids"]
            a_mask = test_gen_batch.batch["attention_mask"]
            with torch.inference_mode():
                out_ids = actor_module.generate(
                    input_ids,                                   # (3, 512)
                    attention_mask = a_mask,
                    max_new_tokens  = safe_new_tokens,
                    eos_token_id    = tokenizer.eos_token_id,
                    pad_token_id    = tokenizer.pad_token_id,
                    do_sample       = False,
                )
            test_output_gen_batch = DataProto(batch=TensorDict({"prompts" : input_ids, "attention_mask" : a_mask, "responses": out_ids}, batch_size = (out_ids.size(0),)))

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

            # pad reward tensor
            batch_size, seq_len = reward_tensor.shape
            if seq_len < max_length:
                pad_len = max_length - seq_len
                pad = torch.zeros(batch_size, pad_len,
                                dtype=reward_tensor.dtype,
                                device=reward_tensor.device)
                reward_tensor = torch.cat([reward_tensor, pad], dim=1)

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

        print('metric dictionary:', metric_dict)

        metric_dicts.append(metric_dict)
    
    # final print for comparison
    print("\n\n==== Prompt-wise Model Outputs ====")
    for prompt, responses in all_model_outputs.items():
        print(f"\nPrompt: {prompt}\n")
        for model_name, response in zip(model_names, responses):
            print(f"[{model_name}]: {response}")

    print('all metric dictionaries:', metric_dicts)

if __name__ == "__main__":
    main()
