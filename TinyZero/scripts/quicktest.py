# quick_vllm_test.py
import torch
from vllm import LLM, SamplingParams

# --- pick any small HF checkpoint so the test is light ---
MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

engine = LLM(
    model=MODEL,
    tensor_parallel_size=1,          # single-GPU test
    dtype="bfloat16",                # or "float16" if needed
    gpu_memory_utilization=0.4,
    max_model_len=2048,
)

params = SamplingParams(
    n=4,                 # fork into 4 branches
    max_tokens=8,
    temperature=0.7,
    top_p=0.9,
    logprobs=1,
)

prompt  = "Hello, how are you today? "
outputs = engine.generate([prompt], sampling_params=params)

print("Completions returned:", len(outputs[0].outputs))
for i, branch in enumerate(outputs[0].outputs):
    toks = branch.token_ids
    probs = branch.logprobs
    print(f"--- completion {i} ---")
    print(engine.tokenizer.decode(toks))
    print("token-count:", len(toks), "logp-len:", len(probs))

# internal correctness check: query_len must be 1 for every live sequence
assert all(q == 1 for q in engine.debug_trace[-1].query_lens), \
       f"query_lens were {engine.debug_trace[-1].query_lens}"
print("✅  quick check passed (all query_len == 1)")
