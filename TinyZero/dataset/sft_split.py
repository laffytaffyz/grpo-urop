# from datasets import load_dataset

# # download and save 
# gsm = load_dataset("gsm8k", "main")
# print(len(gsm["train"]), len(gsm["test"]))

# gsm["train"].to_parquet("/om/user/tiffany8/grpo-urop/TinyZero/dataset/gsm8k_sft_train.parquet")
# gsm["test"].to_parquet("/om/user/tiffany8/grpo-urop/TinyZero/dataset/gsm8k_sft_test.parquet")

from transformers import AutoTokenizer
import pandas as pd
from pathlib import Path
mpath = "/om/user/tiffany8/grpo-urop/TinyZero/model/Llama-3.2-3B-Instruct"
tok = AutoTokenizer.from_pretrained(mpath, use_fast=True)

df = pd.read_parquet("/om/user/tiffany8/grpo-urop/TinyZero/dataset/gsm/train.parquet")

df["prompt_text"] = df["prompt"].map(lambda p : p[0]['content'])
prompts = df["prompt_text"].tolist()
print(type(prompts))
print(prompts)
# Tokenize in batches for speed
enc = tok(prompts, add_special_tokens=True, truncation=False, return_length=True)
lens = enc["length"]  # per-row token lengths

df["prompt_len"] = lens
print(df["prompt_len"].describe(percentiles=[.5,.9,.95,.99]))
print(df["prompt_len"].max())