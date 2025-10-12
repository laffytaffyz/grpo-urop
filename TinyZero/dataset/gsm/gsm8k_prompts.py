import pandas as pd, re
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/om/user/tiffany8/grpo-urop/TinyZero/model/Llama-3.2-3B-Instruct", use_fast=True)

# def extract_numeric_answer(text):
#     # everything after '####' if present
#     if "####" in text:
#         text = text.split("####")[-1]
#     numbers = re.findall("-?\\d+\\.?\\d*", text)
#     return numbers[-1].lstrip("0") if numbers else ""

# template = """<|im_start|>system
# You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>
# <|im_start|>user
# {question}
# Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> 123 </answer>.<|im_end|>
# <|im_start|>assistant
# Let me solve this step by step.
# <think>
# """

# df = pd.read_parquet("/om/user/tiffany8/grpo-urop/TinyZero/dataset/gsm/test_old.parquet")
# df2 = pd.read_parquet("/om/user/tiffany8/grpo-urop/TinyZero/dataset/gsm/train_old.parquet")
# print(df.iloc[0])

# countdown = pd.read_parquet("/om/user/tiffany8/grpo-urop/TinyZero/qwen_dataset/test.parquet")
# print(countdown.iloc[0])
# print(countdown['prompt'].iloc[0])
# print(countdown['data_source'].iloc[0])
# print(countdown['reward_model'].iloc[0])
# print(countdown['extra_info'].iloc[0])

# df["prompt"] = df.apply(lambda row: [
#                                 {"content": template.format(
#                                             question=row["question"]), 
#                                 "role": "user"}], axis=1)
# df["data_source"] = 'gsm8k'
# df["reward_model"] = [{"ground_truth": extract_numeric_answer(answer)} for answer in df["answer"]]
# print(df.iloc[0])
# print(df['prompt'].iloc[0])
# print(df['data_source'].iloc[0])
# print(df['reward_model'].iloc[0])

# df2["prompt"] = df2.apply(lambda row: [
#                                 {"content": template.format(
#                                             question=row["question"]), 
#                                 "role": "user"}], axis=1)
# df2["data_source"] = 'gsm8k'
# df2["reward_model"] = [{"ground_truth": extract_numeric_answer(answer)} for answer in df2["answer"]]

# keep_rows = []
# MAX_PROMPT = 320

# for i, row in df.iterrows():
#     text = row["prompt"][0]['content']
#     length = len(tok(text, add_special_tokens=True, truncation=False)["input_ids"])
#     if length <= MAX_PROMPT:
#         keep_rows.append(row)

# kept_df = pd.DataFrame(keep_rows)
# print(f"train: kept {len(kept_df)}/{len(df)} rows ≤ {MAX_PROMPT} tokens")

# keep_rows = []

# for i, row in df2.iterrows():
#     text = row["prompt"][0]['content']
#     length = len(tok(text, add_special_tokens=True, truncation=False)["input_ids"])
#     if length <= MAX_PROMPT:
#         keep_rows.append(row)

# kept_df2 = pd.DataFrame(keep_rows)
# print(f"test: kept {len(kept_df2)}/{len(df2)} rows ≤ {MAX_PROMPT} tokens")

# kept_df.to_parquet("/om/user/tiffany8/grpo-urop/TinyZero/dataset/gsm/train.parquet")
# kept_df2.to_parquet("/om/user/tiffany8/grpo-urop/TinyZero/dataset/gsm/test.parquet")

# print(kept_df)
# print(kept_df2)

train_df = pd.read_parquet("/om/user/tiffany8/grpo-urop/TinyZero/dataset/gsm/train.parquet")
test_df = pd.read_parquet("/om/user/tiffany8/grpo-urop/TinyZero/dataset/gsm/test.parquet")

print(train_df)
print(train_df.iloc[0]['question'])

def filter_by_length(df, max_len=256):
    mask = df["question"].apply(
        lambda text: len(tok.encode(text, add_special_tokens=False)) <= max_len
    )
    return df[mask]

tiny_train_df = filter_by_length(train_df, max_len=256).head(500)
tiny_test_df = filter_by_length(test_df, max_len=256).head(100)

tiny_train_df.to_parquet("/om/user/tiffany8/grpo-urop/TinyZero/dataset/gsm/tiny_train.parquet")
tiny_test_df.to_parquet("/om/user/tiffany8/grpo-urop/TinyZero/dataset/gsm/tiny_test.parquet")

print(tiny_train_df.iloc[0]['question'])
print(tiny_test_df)
print(tiny_test_df.iloc[0]['question'])
print(tiny_test_df.iloc[0]['reward_model'])