from datasets import load_dataset
import pandas as pd

# LOADING DATA
# only contains training dataset
# features: 'id', 'problem', 'solution', 'answer', 'url'
# aime = load_dataset("AI-MO/aimo-validation-aime")['train']
# aime.to_parquet("/om/user/tiffany8/grpo-urop/TinyZero/dataset/aime_train.parquet")

# PROMPT ENGINEERING
head = """<|im_start|>system
You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>
<|im_start|>user
"""
tail = """ Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> 123 </answer>.<|im_end|>
<|im_start|>assistant
Let me solve this step by step.
<think>
"""
test = pd.read_parquet("/om/user/tiffany8/grpo-urop/TinyZero/dataset/aime_train.parquet")
print(test['problem'])

needs_tail = ~test["problem"].str.contains(repr(head[1:20]), regex=False)
test.loc[needs_tail, "problem"] = head + test.loc[needs_tail, "problem"] + tail

print(test['problem'].iloc[0])

test.to_parquet("/om/user/tiffany8/grpo-urop/TinyZero/dataset/aime_train_modified.parquet")