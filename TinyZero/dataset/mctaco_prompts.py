from datasets import load_dataset
import pandas as pd

# # LOADING DATA
# # features "sentence", "question", "answer", "label", "category"
# # data = load_dataset("CogComp/mc_taco", "plain_text")["test"]
# # print(data.iloc[0])
# # data.to_parquet("/om/user/tiffany8/grpo-urop/TinyZero/dataset/mctaco_test.parquet")

# # # PROMPT ENGINEERING
# template = """<|im_start|>system
# You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>
# <|im_start|>user
# Passage: {sentence}
# Question: {question}
# Answer: {answer}
# Is this answer valid? Show your work in <think> </think> tags. And return your final answer <answer> </answer> tags, for example <answer> yes </answer>.<|im_end|>
# <|im_start|>assistant
# Let me solve this step by step.
# <think>
# """

# df = pd.read_parquet("/om/user/tiffany8/grpo-urop/TinyZero/dataset/mctaco_test.parquet")
# print(df.iloc[0])

# countdown = pd.read_parquet("/om/user/tiffany8/grpo-urop/TinyZero/qwen_dataset/test.parquet")
# print(countdown.iloc[0])
# print(countdown['prompt'].iloc[0])
# print(countdown['data_source'].iloc[0])
# print(countdown['reward_model'].iloc[0])
# print(countdown['extra_info'].iloc[0])

# df["prompt"] = df.apply(lambda row: [
#                                 {"content": template.format(
#                                             sentence=row["sentence"],
#                                             question=row["question"],
#                                             answer=row["answer"]), 
#                                 "role": "user"}], axis=1)

# df["data_source"] = 'mc_taco'
# df["reward_model"] = [{"ground_truth": 'yes' if label == 1 else 'no'} for label in df["label"]]
# print(df.iloc[0])

# test_frac = 0.2

# test_df  = df.sample(frac=test_frac, random_state=42)
# train_df = df.drop(test_df.index)

# # (Optional) reset the row indices
# train_df = train_df.reset_index(drop=True)
# test_df  = test_df.reset_index(drop=True)

# # 9442
# train_df.to_parquet("/om/user/tiffany8/grpo-urop/TinyZero/dataset/mctaco_train_modified.parquet")
# test_df.to_parquet("/om/user/tiffany8/grpo-urop/TinyZero/dataset/mctaco_test_modified.parquet")
# print(train_df)
# print(test_df)

train_df = pd.read_parquet("/om/user/tiffany8/grpo-urop/TinyZero/dataset/mctaco_train_modified.parquet")
test_df = pd.read_parquet("/om/user/tiffany8/grpo-urop/TinyZero/dataset/mctaco_test_modified.parquet")

print(train_df)
print(test_df)