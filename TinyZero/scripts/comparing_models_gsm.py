import hydra
import numpy as np
from collections import defaultdict
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
# qwen
model_paths = ["/om/user/tiffany8/grpo-urop/TinyZero/model/Qwen2.5-3B-Instruct",
                "/om/user/tiffany8/grpo-urop/TinyZero/model/Llama-3.2-3B-Instruct",
                "/om2/user/tiffany8/checkpoints/TinyZero/qwen-3b-instruct-grpo/actor/global_step_500",
                "/om2/user/tiffany8/checkpoints/TinyZero/llama-3b-instruct-grpo/actor/global_step_100",
                "/om2/user/tiffany8/checkpoints/TinyZero/llama-3b-instruct-grpo/actor/global_step_300",
                "/om2/user/tiffany8/checkpoints/TinyZero/llama-3b-instruct-grpo/actor/global_step_700"]
model_names=["qwen base", "llama base","qwen 500", "llama 100", "llama 300", "llama 700"]

### ~~PICK DATA~~
# data_path = "/om/user/tiffany8/grpo-urop/TinyZero/dataset/test.parquet"
# data_path = "/om/user/tiffany8/grpo-urop/TinyZero/qwen_dataset/test.parquet"
data_path = "/om/user/tiffany8/grpo-urop/TinyZero/dataset/gsm8k_sft_test.parquet"

assert len(model_paths) == len(model_names)

class GSM8KDataset(torch.utils.data.Dataset):
    def __init__(self, parquet_path, tokenizer, max_len=512):
        import pandas as pd, pyarrow.parquet as pq
        df = pq.read_table(parquet_path).to_pandas()
        self.questions = df["question"].tolist()
        self.answers   = df["answer"].astype(str).str.strip().tolist()
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        q = self.questions[idx]
        enc = self.tok(q, truncation=True, max_length=self.max_len,
                       return_tensors="pt")
        enc = {"input_ids" : enc["input_ids"].squeeze(0), 
                "attention_mask": enc["attention_mask"].squeeze(0)}
        return q, enc, self.answers[idx] 

def extract_numeric_answer(text):
    # everything after '####' if present
    if "####" in text:
        text = text.split("####")[-1]
    numbers = re.findall("-?\\d+\\.?\\d*", text)
    return numbers[-1].lstrip("0") if numbers else ""

def main():
    all_model_outputs = defaultdict(list)
    metric_dicts = []

    for i in range(len(model_paths)):
        model_path = model_paths[i]
        print("EVALUATING:", model_names[i])
        local_path = copy_local_path_from_hdfs(model_path)
        data_counter = 0

        trust_remote_code = True
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        tokenizer.padding_side = "left"   

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

        # generation_config = get_generation_config(
        #     local_path, trust_remote_code=trust_remote_code
        # )
        # generation_config.temperature       = 0.0   # deterministic
        # generation_config.max_new_tokens    = 256
        # generation_config.do_sample         = False
        # generation_config.pad_token_id      = tokenizer.eos_token_id
        # generation_config.eos_token_id      = tokenizer.eos_token_id

        config = {
            "micro_batch_size": 4,

            "do_sample":        False,
            "max_new_tokens":   256,
            "pad_token_id":     tokenizer.pad_token_id,
            "eos_token_id":     tokenizer.eos_token_id,
            "top_p":            1.0,
            "top_k":            0,
            "temperature":      1.0, 
        }

        val_dataset   = GSM8KDataset(data_path, tokenizer)

        def pad_and_collate(batch):
            # batch = list of (prompt_text, enc_dict, answer)
            prompts, encs, answers = zip(*batch)
            # find max len this batch
            max_len = max(enc["input_ids"].shape[0] for enc in encs)

            def pad(t, pad_val):
                pad_len = max_len - t.shape[0]
                if pad_len:
                    t = torch.cat([t, t.new_full((pad_len,), pad_val)])
                return t

            input_ids      = torch.stack([pad(e["input_ids"], tokenizer.pad_token_id) for e in encs])
            attention_mask = torch.stack([pad(e["attention_mask"], 0)                 for e in encs])

            enc_batch = {"input_ids": input_ids, "attention_mask": attention_mask}
            return prompts, enc_batch, list(answers)

        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=3, 
            shuffle=False, 
            drop_last=True,
            collate_fn=pad_and_collate)

        assert len(val_dataloader) >= 1

        correct, total = 0, 0

        with torch.no_grad():
            for prompt_texts, enc_batch, answers in val_dataloader:
                enc_batch = {k: v.to("cuda") for k, v in enc_batch.items()}
                print(enc_batch.keys())
                dp = DataProto.from_single_dict(enc_batch)
                dp.meta_info = {
                    "eos_token_id": tokenizer.eos_token_id,
                    "pad_token_id": tokenizer.pad_token_id,
                    "do_sample": False,
                    }

                input_ids = dp.batch["input_ids"]
                attention_mask = dp.batch["attention_mask"]
                seq_length = input_ids.size(1)
                base = torch.arange(seq_length, device=input_ids.device)         # (L,)
                position_ids = base.unsqueeze(0).expand_as(input_ids)            # (B, L)
                # optional: zero-out pad positions so the model ignores them
                position_ids = position_ids * attention_mask + (1 - attention_mask) * 0
                dp.batch["position_ids"] = position_ids

                hfrollout = HFRollout(module=actor_module, config=config)
                out_dp = hfrollout.generate_sequences(dp)

                outputs = tokenizer.batch_decode(out_dp.batch["responses"], 
                                                skip_special_tokens=True)

                for prompt, pred, ans in zip(prompt_texts, outputs, answers): 
                    all_model_outputs[prompt].append(ans)
                    total += 1
                    if extract_numeric_answer(pred) == ans.lstrip("0"):
                        correct += 1
                
                data_counter += 1
                if data_counter >= 20 and total > 0: break

        accuracy = 100.0 * correct/total
        metric_dict = {f"val/accuracy": accuracy}

        print('metric dictionary:', metric_dict)

        metric_dicts.append(metric_dict)

    print("\n\n==== Prompt-wise Model Outputs ====")
    for prompt, responses in all_model_outputs.items():
        print(f"\nPrompt: {prompt}\n")
        for model_name, response in zip(model_names, responses):
            print(f"[{model_name}]: {response}")

    print('all metric dictionaries:', metric_dicts)

if __name__ == "__main__":
    main()
