import pandas as pd

head = """<|im_start|>system
You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>
<|im_start|>user
"""
tail = """ Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> 123 </answer>.<|im_end|>
<|im_start|>assistant
Let me solve this step by step.
<think>
"""
test = pd.read_parquet("/om/user/tiffany8/grpo-urop/TinyZero/dataset/gsm8k_sft_test.parquet")
print(test['question'])

needs_tail = ~test["question"].str.contains(repr(head[1:20]), regex=False)
test.loc[needs_tail, "question"] = head + test.loc[needs_tail, "question"] + tail

print(test['question'].iloc[0])

# 1319
test.to_parquet("/om/user/tiffany8/grpo-urop/TinyZero/dataset/gsm8k_sft_test_modified.parquet")