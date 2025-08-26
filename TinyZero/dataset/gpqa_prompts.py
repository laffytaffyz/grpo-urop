from datasets import load_dataset
import pandas as pd

# LOADING DATA
# features "question", "answer"
# data = load_dataset("fingertap/GPQA-Diamond", "default")["test"]
# print(data)
# data.to_parquet("/om/user/tiffany8/grpo-urop/TinyZero/dataset/gpqa_test.parquet")

# PROMPT ENGINEERING
# head = """<|im_start|>system
# You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>
# <|im_start|>user
# """
# tail = """
# Show your work in <think> </think> tags. And return the letter associated with your final answer in <answer> </answer> tags, for example <answer> A </answer>.<|im_end|>
# <|im_start|>assistant
# Let me solve this step by step.
# <think>
# """
# test = pd.read_parquet("/om/user/tiffany8/grpo-urop/TinyZero/dataset/gpqa_test.parquet")

# needs_tail = ~test["question"].str.contains(repr(head[1:20]), regex=False)
# test.loc[needs_tail, "question"] = head + test.loc[needs_tail, "question"] + tail

# print(test.iloc[0])

# 198
test.to_parquet("/om/user/tiffany8/grpo-urop/TinyZero/dataset/gpqa_test_modified.parquet")