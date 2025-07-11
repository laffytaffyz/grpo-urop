from datasets import load_dataset
import pandas as pd

# LOADING DATA
# features "Problem", "Rationale", "options", "correct", "annotated_formula", "linear_formula", "category"
# mathqa = load_dataset("allenai/math_qa", "default")["test"]
# mathqa.to_parquet("/om/user/tiffany8/grpo-urop/TinyZero/dataset/mathqa_test.parquet")

# PROMPT ENGINEERING
# head = """<|im_start|>system
# You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>
# <|im_start|>user
# """
# middle = """ The following are your answer options: """
# tail = """ Show your work in <think> </think> tags. And return the letter associated with your final answer in <answer> </answer> tags, for example <answer> a </answer>.<|im_end|>
# <|im_start|>assistant
# Let me solve this step by step.
# <think>
# """
test = pd.read_parquet("/om/user/tiffany8/grpo-urop/TinyZero/dataset/mathqa_test.parquet")

# needs_tail = ~test["Problem"].str.contains(repr(head[1:20]), regex=False)
# test.loc[needs_tail, "Problem"] = head + test.loc[needs_tail, "Problem"] + middle + test.loc[needs_tail, "options"] + tail

test = (
        test.groupby("category", sort=False, group_keys=False)
            .head(10)
            .reset_index(drop=True)
)

categories = {test['category'].iloc[i] for i in range(len(test['category']))}
for cat in categories:
    print('category:', cat)
    for p, a in zip(test[test['category'] == cat]['Problem'], test[test['category'] == cat]['correct']):
        print(p)
        print(a)
    print()

# test.to_parquet("/om/user/tiffany8/grpo-urop/TinyZero/dataset/mathqa_test_modified.parquet")