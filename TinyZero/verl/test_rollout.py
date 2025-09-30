# CPU-only smoke test for vLLMRollout.generate_sequences
# Run from TinyZero repo root:
#   python -m verl.test_rollout

import os, sys
import torch
from types import SimpleNamespace

from tensordict import TensorDict
from verl import DataProto
from verl.workers.rollout.vllm_rollout.vllm_rollout import vLLMRollout

# ---------- helpers ----------

class DummySamplingParams:
    def __init__(self, max_tokens, n):
        self.max_tokens = max_tokens
        self.n = n
    # your rollout’s update_sampling_params uses hasattr/setattr,
    # so it’s fine if we add fields dynamically.

def make_batch(token_ids, pad_token_id, eos_token_id, device="cpu"):
    """
    token_ids: list[list[int]]  (un-padded)
    Returns a DataProto with left-padded tensors.
    """
    L = max(len(t) for t in token_ids)
    B = len(token_ids)
    ids = torch.full((B, L), pad_token_id, dtype=torch.long, device=device)
    attn = torch.zeros((B, L), dtype=torch.long, device=device)
    pos  = torch.zeros((B, L), dtype=torch.long, device=device)
    for i, t in enumerate(token_ids):
        ids[i, -len(t):]  = torch.tensor(t, dtype=torch.long, device=device)
        attn[i, -len(t):] = 1
        pos[i, -len(t):]  = torch.arange(len(t), dtype=torch.long, device=device)
    batch = TensorDict({
        "input_ids": ids,
        "attention_mask": attn,
        "position_ids": pos,
    }, batch_size=B)
    meta = {"eos_token_id": eos_token_id, "do_sample": True}
    return DataProto(batch=batch, meta_info=meta)

class DummyEngine:
    """Mimics vLLM engine's .generate() contract in your fork."""
    def generate(self, prompts, sampling_params, prompt_token_ids, use_tqdm):
        B = len(prompt_token_ids)
        n = getattr(sampling_params, "n", 1) or 1
        T = sampling_params.max_tokens
        rows = B * n
        # Return tensors shaped like (responses, log_probs)
        # Use CPU tensors; caller moves to device if needed.
        responses = torch.randint(low=5, high=50, size=(rows, T), dtype=torch.long)
        log_probs = torch.zeros(rows, T, dtype=torch.long)
        return (responses, log_probs)

# ---------- build a rollout instance without calling __init__ ----------

def build_rollout_for_cpu_test():
    # Skip __init__ to avoid constructing real vLLM LLM(...)
    rollout = vLLMRollout.__new__(vLLMRollout)

    # Minimal config fields that generate_sequences uses
    # (match names accessed in your patched code)
    cfg = SimpleNamespace(
        free_cache_engine=False,     # we won't call engine.init/free_cache_engine
        n=4,                         # GRPO samples (we'll also test fast path)
        response_length=8,
        enable_chunked_prefill=True, # force serialized path in first test
        # fields below aren't used in generate_sequences but exist elsewhere
        prompt_length=64,
        path="dummy",
        adv_estimator="grpo",
    )
    rollout.config = cfg
    rollout.pad_token_id = 0
    rollout.inference_engine = DummyEngine()  # monkeypatch dummy engine

    rollout.sampling_params = DummySamplingParams(
        max_tokens=cfg.response_length, n=cfg.n
    )
    return rollout

# ---------- tests ----------

def test_serialized_chunked_path():
    print("[TEST] serialized path with chunked prefill ON, B=2, n=4")
    rollout = build_rollout_for_cpu_test()

    dp = make_batch([[1,2,3,4], [5,6,7]], pad_token_id=0, eos_token_id=2, device="cpu")
    out = rollout.generate_sequences(dp)

    tb = out.batch
    B, n, T = 2, rollout.config.n, rollout.config.response_length
    assert tb["responses"].shape == (B * n, T)
    assert tb["input_ids"].shape[0] == B * n
    assert tb["attention_mask"].shape[0] == B * n
    assert tb["position_ids"].shape[0] == B * n
    print("  OK shapes:", {k: v.shape for k, v in tb.items()})

def test_fast_batched_path():
    print("[TEST] fast batched path with chunked prefill OFF, B=2, n=4")
    rollout = build_rollout_for_cpu_test()
    rollout.config.enable_chunked_prefill = False  # take the original batched path

    dp = make_batch([[10,11], [12,13,14]], pad_token_id=0, eos_token_id=2, device="cpu")
    out = rollout.generate_sequences(dp)

    tb = out.batch
    B, n, T = 2, rollout.config.n, rollout.config.response_length
    assert tb["responses"].shape == (B * n, T)
    print("  OK shapes:", {k: v.shape for k, v in tb.items()})

if __name__ == "__main__":
    torch.set_num_threads(1)
    test_serialized_chunked_path()
    test_fast_batched_path()
    print("All good ✅")
