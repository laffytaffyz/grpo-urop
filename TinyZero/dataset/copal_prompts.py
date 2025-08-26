from datasets import load_dataset
import pandas as pd

# LOADING DATA
# features "sentence", "question", "answer", "label", "category"
# data = load_dataset("haryoaw/COPAL", "id")["test"]
# print(data.iloc[0])
# data.to_parquet("/om/user/tiffany8/grpo-urop/TinyZero/dataset/copal_test.parquet")
# 
# PROMPT ENGINEERING
template = """<|im_start|>system
You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>
<|im_start|>user
{premise}
a) {choice1}
b) {choice2}
Show your work in <think> </think> tags. And return the letter associated with your final answer in <answer> </answer> tags, for example <answer> a </answer>.<|im_end|>
<|im_start|>assistant
Let me solve this step by step.
<think>
"""

test = pd.read_parquet("/om/user/tiffany8/grpo-urop/TinyZero/dataset/copal_test.parquet")
print(test.iloc[0])

test["prompt"] = test.apply(lambda row: template.format(
                        premise=row["premise"],
                        choice1=row["choice1"],
                        choice2=row["choice2"]), 
                        axis = 1)
print(test['prompt'].iloc[0])

# 559 
test.to_parquet("/om/user/tiffany8/grpo-urop/TinyZero/dataset/copal_test_modified.parquet")