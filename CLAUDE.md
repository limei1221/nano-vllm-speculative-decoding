# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Nano-vLLM is a lightweight vLLM-style inference engine for Qwen3 models (~1,200 lines of Python). This fork adds **speculative decoding** (a small draft model proposes K tokens, the target model verifies them in one forward pass) on top of the paged KV cache + prefix caching + CUDA graph stack. The public API (`from nanovllm import LLM, SamplingParams`) mirrors vLLM.

Runtime assumes a CUDA GPU with `flash-attn` + `triton`. There is no CPU fallback.

## Common commands

```bash
# Install (editable) — pyproject pins torch, triton, transformers, flash-attn, xxhash
pip install -e .

# Download a model locally (paths below assume ~/huggingface/…)
huggingface-cli download --resume-download Qwen/Qwen3-0.6B --local-dir ~/huggingface/Qwen3-0.6B/ --local-dir-use-symlinks False

# Offline smoke test (argparse-driven; defaults to ~/huggingface/Qwen3-0.6B/)
python example.py --model ~/huggingface/Qwen3-0.6B --enforce-eager

# Smoke test with speculative decoding (target + draft)
python example.py --model ~/huggingface/Qwen3-8B \
  --speculative-model ~/huggingface/Qwen3-0.6B --num-speculative-tokens 5

# Throughput benchmark (flags: --num-seqs --max-input-len --max-output-len --max-model-len)
python bench.py --model ~/huggingface/Qwen3-8B \
  --speculative-model ~/huggingface/Qwen3-0.6B --num-speculative-tokens 5 --num-seqs 256
```

There is **no** test suite, linter, or server/client script — `example.py` and `bench.py` are the only practical smoke tests. (Despite what older docs may say, there is no `server.py`/`client.py` and no FastAPI dependency.)

Per the README benchmark (RTX 4090), speculative decoding is a **net win only at small batch sizes** (B≈1–8) and *hurts* throughput at large batch (B≈64–256), because verification cost scales with `B*(K+1)`.

## Architecture

### Entry points
- `nanovllm/llm.py` — `LLM` is just a subclass alias of `LLMEngine`.
- `nanovllm/engine/llm_engine.py` — orchestrates `Scheduler` + `ModelRunner` and exposes `generate()`. `step()` = `scheduler.schedule()` → `model_runner.call("run", seqs, is_prefill)` → `scheduler.postprocess()`. For `tensor_parallel_size > 1`, rank 0 runs in-process; ranks 1..N-1 are spawned via `torch.multiprocessing` (`spawn`) and driven over POSIX shared memory (`SharedMemory(name="nanovllm", size=2**20)`) plus per-rank `mp.Event`s. `ModelRunner.call` on rank 0 pickles `(method_name, *args)` into the shm and sets the events; worker ranks sit in `loop()` → `read_shm()` and dispatch the same method locally.

### Scheduling and KV cache
- `engine/scheduler.py` — single `schedule()` method over two deques (`waiting`, `running`), returning `(seqs, is_prefill)`:
  - **Prefill** is preferred: packs waiting seqs up to `max_num_batched_tokens`. Chunked prefill is allowed **only for the first seq in a batch** (`remaining < num_tokens and scheduled_seqs` breaks the loop); a seq stays in `waiting` until fully prefilled (`num_scheduled_tokens == num_tokens`). There is **no** `enable_chunked_prefill` config flag — this behavior is always on.
  - **Decode**: one token per running seq. Preemption: when `can_append` fails, pop the tail of `running`, `deallocate` it, and push it to the front of `waiting`.
  - `postprocess()` dispatches on the shape of `token_ids`: a `list[list[int]]` → `_postprocess_speculative`; a `list[int]` → `_postprocess_normal`. This is the single place sequences advance, get EOS/`max_tokens`-terminated, and (for speculative) get trimmed.
- `engine/block_manager.py` — paged KV allocator. The **target** pool uses xxhash-based **prefix caching** (`hash_to_block_id`, ref-counted block reuse for shared prefixes). When speculative decoding is on, a **parallel draft pool** (`draft_blocks` / `free_draft_block_ids` / `used_draft_block_ids`) is sized independently and allocated **in lockstep** via `allocate`/`may_append`/`deallocate` — but the draft pool is **not** prefix-cached (linear allocation, no hashing). `can_append`/`may_append` reserve `num_speculative_tokens` extra slots ahead (`future_len = len(seq) + num_speculative_tokens`) so a propose round never runs out mid-flight.
- `engine/sequence.py` — `Sequence` holds `block_table` (target) and `draft_block_table` (draft), plus `draft_kv_len` (how far the draft KV cache is caught up). `append_token` / `pop_last_n_tokens` are the mutation points; `num_scheduled_tokens` / `num_cached_tokens` are scheduler bookkeeping. `__getstate__`/`__setstate__` keep pickling small when shipping seqs to TP worker ranks.

### Model execution
- `engine/model_runner.py` — one instance per GPU rank.
  - `allocate_kv_cache()` computes the post-warmup GPU budget and splits it between target and draft caches by **block-byte ratio** (`main_block_bytes / (main + draft)`) — there is no fixed floor for the target share. `num_kvcache_blocks` / `num_draft_kvcache_blocks` are written back into `Config` here.
  - Three CUDA-graph families (captured when `enforce_eager=False`): `graphs` (target decode), `draft_graphs` (draft decode), `verify_graphs` (target verify, captured at `n_tokens = B*(K+1)`). Prefill, `enforce_eager`, and any batch >512 always run eagerly (`run_model` / `run_draft_model` early-return to `compute_logits`).
  - `prepare_prefill` / `prepare_decode` / `prepare_verify` (+ `prepare_draft_prefill` / `prepare_draft_decode`) build `input_ids`/`positions`/`slot_mapping`/`block_tables`/`cu_seqlens` on the host and ship them to GPU, then call `set_context` (module-global `Context` in `utils/context.py`) which `layers/attention.py` reads. **Always pair `set_context` with `reset_context`**, and CUDA-graph replays must *fill* the pre-allocated `graph_vars` tensors (`[:bs]`), not allocate fresh ones.
  - `run()` dispatch: prefill → `run_model`; plain decode → `run_model`; decode + draft model + rank 0 → `run_speculative_decode` takes over.
- `layers/attention.py` — flash-attn wrapper. Prefill uses `flash_attn_varlen_func` (with `block_table` for prefix-cache hits); decode uses `flash_attn_with_kvcache`. In verify mode (`context.num_spec_tokens > 0`) it reshapes `q` to `(B, K+1, H, D)` so all K+1 positions per sequence run in one call. `store_kvcache` is a Triton kernel writing into the paged cache at `slot_mapping` (slots of `-1` are skipped).
- `layers/sampler.py` — `forward` is Gumbel-max argmax sampling (`@torch.compile`); `compute_probs` / `sample_with_probs` (multinomial) are used by the speculative path. Temperature is applied by dividing logits.
- `models/qwen3.py` + `layers/{linear,rotary_embedding,layernorm,activation,embed_head}.py` — Qwen3 transformer with column/row-parallel linears for tensor parallelism. Only Qwen3 is supported; `ModelRunner` hard-codes `Qwen3ForCausalLM` for both target and draft.

### Speculative decoding flow (`run_speculative_decode`, all inline — no separate propose/verify methods)
1. **Propose**: loop K times over the draft model, calling `sample_with_probs` each step and retaining probs. Step 0 runs a draft *prefill* if any seq's `draft_kv_len < len(seq) - 1` (catching the draft cache up to newly accepted tokens); later steps are decode. Each sampled token is `append_token`-ed so it lives in both block tables and gets KV-cached on both sides; `draft_kv_len` is advanced to `len(seq)`.
2. **Verify**: single target forward over `B*(K+1)` positions → `compute_probs` reshaped to `(B, K+1, V)`. Standard speculative sampling: `ratio = clamp(p_target/p_draft, max=1)`, accept via `rand < ratio` with a `cumprod` mask (first rejection stops acceptance for that seq) → `num_accepted`. The correction token at the first rejection is sampled from the normalized `clamp(p_target - p_draft, 0)`; if all K were accepted, it's sampled from the bonus (K+1)-th target distribution.
3. **Rollback (in `model_runner`, not the scheduler)**: for each seq, `pop_last_n_tokens(K - num_accepted)` then `append_token(correction)`. Returns one `accepted_ids + [correction]` list per seq.
4. **Reconcile (`scheduler._postprocess_speculative`)**: scans the returned tokens for the earliest EOS / `max_tokens` stop *within the accepted span*, trims the overshoot with `pop_last_n_tokens` (clamping `draft_kv_len`), refreshes `num_cached_tokens`, and finalizes/deallocates finished seqs.

### Configuration
`nanovllm/config.py` (`Config`, a `slots=True` dataclass) centralizes knobs: `max_num_batched_tokens` (16384), `max_num_seqs` (512), `max_model_len` (4096, clamped to the HF `max_position_embeddings`), `gpu_memory_utilization` (0.9), `tensor_parallel_size` (1–8), `enforce_eager`, `kvcache_block_size` (256; must be a multiple of 256), `speculative_model`, `num_speculative_tokens`. `num_kvcache_blocks` / `num_draft_kvcache_blocks` are `-1` until computed at runtime in `allocate_kv_cache`. `LLMEngine.__init__` filters `**kwargs` to `Config` field names, so extra kwargs are silently dropped.

### Invariants to preserve when editing
- Draft and target models must share the same vocab — the verify step allocates `draft_probs` with the **target's** `vocab_size` and gathers both distributions by the same token ids (this is assumed, not asserted; mismatched vocabs break silently/crash).
- `block_table` and `draft_block_table` must stay length-synced per sequence; `BlockManager` allocates/frees them in lockstep.
- CUDA-graph batch sizes are `graph_bs = [1, 2, 4, 8] + range(16, max_bs+1, 16)`; decode/verify dispatch rounds the live batch up to the next `graph_bs` element, so any batching change must keep that rounding valid.
- `draft_kv_len` tracks draft-cache progress and is read by both propose (to decide prefill vs decode and which tail to prefill) and the reconcile step — keep them consistent when changing rollback logic.
