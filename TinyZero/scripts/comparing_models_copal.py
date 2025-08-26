import hydra
import numpy as np
from collections import defaultdict
import re
import os
import pandas as pd

import torch
import torch.distributed
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoConfig, DataCollatorWithPadding

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
### ~~PICK DATA~~
data_path = "/om/user/tiffany8/grpo-urop/TinyZero/dataset/copal_test_modified.parquet"

assert len(model_paths) == len(model_names)

class COPALDataset(torch.utils.data.Dataset):
    def __init__(self, parquet_path, tokenizer, max_len=512):
        import pandas as pd, pyarrow.parquet as pq
        df = pq.read_table(parquet_path).to_pandas()
        self.questions = df["prompt"].tolist()
        self.answers   = df["label"].astype(str).str.strip().tolist()
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

ANSWER_TAG_RE = re.compile(
    r"<\s*answer\s*>(.*?)<\s*/\s*answer\s*>",
    flags=re.IGNORECASE | re.DOTALL
)
BOXED_RE = re.compile(r"oxed\s*\{\s*([^{}]+?)\s*\}", flags=re.DOTALL)

# takes returns letter answer in <answer></answer>, 
def extract_answer_tag(text: str) -> str:
    t = text.strip().strip("`")

    answer_chunks = ANSWER_TAG_RE.findall(t)
    print('answer chunks', answer_chunks)
    if answer_chunks:
        return answer_chunks[-1].strip().strip("`").lower()

    boxed_chunks = BOXED_RE.findall(t)
    print('boxed chunks', boxed_chunks)
    if boxed_chunks:
        return boxed_chunks[-1].strip().strip("`").lower()
        
    return ""

def main():
    all_model_outputs = defaultdict(list)
    metric_dicts = []

    for i in range(len(model_paths)):
        model_path = model_paths[i]
        print("EVALUATING:", model_names[i])
        local_path = copy_local_path_from_hdfs(model_path)
        data_counter = 0
        verbose = True

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
            "temperature":      1.0, 
        }

        val_dataset   = COPALDataset(data_path, tokenizer)
        
        collator = DataCollatorWithPadding(tokenizer, padding="longest", return_tensors="pt")

        def collate(batch):
            prompts, encs, answers = zip(*batch)
            padded = collator(list(encs))             
            return prompts, padded, list(answers)

        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=3, 
            shuffle=False, 
            drop_last=True,
            collate_fn=collate)

        assert len(val_dataloader) >= 1

        correct, total = 0, 0

        print('evaluation start')
        with torch.no_grad():
            for prompt_texts, enc_batch, answers in val_dataloader:
                enc_batch = {k: v.to("cuda") for k, v in enc_batch.items()}

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
                    
                    # only responses of first 20 responses will appear in out file
                    if verbose:
                        all_model_outputs[prompt].append(pred)
                        print("prompt:", prompt)
                        print("response:", pred)
                        print("response answer:", extract_answer_tag(pred))
                    
                    ans = "a" if ans == 0 else "b"

                    total += 1
                    if extract_answer_tag(pred) == ans:
                        correct += 1
                        if verbose: print('correct:', ans)
                    else:
                        if verbose: print('wrong:', ans)
                
                data_counter += 1
                print('evaluated batch #', data_counter)
                print('accuracy so far', correct, '/', total)
                if data_counter >= 20 and total > 0: verbose = False
        
        metric_dict = {100.0 * correct/total}

        print('metric dictionary:', metric_dict)

        metric_dicts.append((model_names[i], metric_dict))

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
    df.to_json("/om/user/tiffany8/grpo-urop/TinyZero/response_analysis/copal/responses_copal.jsonl", orient="records", lines=True)

    print('all metric dictionaries:', metric_dicts)

if __name__ == "__main__":
    main()
