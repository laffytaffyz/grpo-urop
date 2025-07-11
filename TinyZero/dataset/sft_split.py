from datasets import load_dataset

# download and save 
gsm = load_dataset("gsm8k", "main")
print(len(gsm["train"]), len(gsm["test"]))

gsm["train"].to_parquet("/om/user/tiffany8/grpo-urop/TinyZero/dataset/gsm8k_sft_train.parquet")
gsm["test"].to_parquet("/om/user/tiffany8/grpo-urop/TinyZero/dataset/gsm8k_sft_test.parquet")

# check
import pandas as pd

train = pd.read_parquet("/om/user/tiffany8/grpo-urop/TinyZero/dataset/gsm8k_sft_train.parquet")
test = pd.read_parquet("/om/user/tiffany8/grpo-urop/TinyZero/dataset/gsm8k_sft_test.parquet")
