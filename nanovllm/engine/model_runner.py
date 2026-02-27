import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory
from transformers import AutoConfig

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.num_spec_tokens = config.num_speculative_tokens

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")

        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()

        self.draft_model = None
        self.draft_hf_config = None
        if config.speculative_model is not None:
            self.draft_hf_config = AutoConfig.from_pretrained(config.speculative_model)
            self.draft_model = Qwen3ForCausalLM(self.draft_hf_config)
            load_model(self.draft_model, config.speculative_model)

        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
            if self.draft_model is not None:
                self.capture_draft_cudagraph()
                self.capture_verify_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
            if self.draft_model is not None:
                del self.draft_graphs, self.draft_graph_pool
                del self.verify_graphs, self.verify_graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    # ── KV cache allocation ─────────────────────────────────────────────

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        available = int(total * config.gpu_memory_utilization - used - peak + current)

        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        main_block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize

        if self.draft_model is not None:
            dc = self.draft_hf_config
            d_num_kv_heads = dc.num_key_value_heads // self.world_size
            d_head_dim = getattr(dc, "head_dim", dc.hidden_size // dc.num_attention_heads)
            draft_block_bytes = 2 * dc.num_hidden_layers * self.block_size * d_num_kv_heads * d_head_dim * dc.dtype.itemsize
            ratio = main_block_bytes / (main_block_bytes + draft_block_bytes)
            main_budget = int(available * ratio)
            draft_budget = available - main_budget
        else:
            main_budget = available
            draft_budget = 0

        config.num_kvcache_blocks = main_budget // main_block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        self._bind_kv_cache(self.model, self.kv_cache)

        config.num_draft_kvcache_blocks = 0
        if self.draft_model is not None and draft_budget > 0:
            config.num_draft_kvcache_blocks = draft_budget // draft_block_bytes
            assert config.num_draft_kvcache_blocks > 0
            self.draft_kv_cache = torch.empty(2, dc.num_hidden_layers, config.num_draft_kvcache_blocks, self.block_size, d_num_kv_heads, d_head_dim)
            self._bind_kv_cache(self.draft_model, self.draft_kv_cache)

    @staticmethod
    def _bind_kv_cache(model, kv_cache):
        layer_id = 0
        for module in model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = kv_cache[0, layer_id]
                module.v_cache = kv_cache[1, layer_id]
                layer_id += 1

    # ── Input preparation ───────────────────────────────────────────────

    def _prepare_block_tables(self, seqs: list[Sequence], use_draft: bool = False):
        tables = [seq.draft_block_table if use_draft else seq.block_table for seq in seqs]
        max_len = max(len(t) for t in tables)
        padded = [t + [-1] * (max_len - len(t)) for t in tables]
        return torch.tensor(padded, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            start = min(seq.num_cached_tokens, seqlen - 1)
            seqlen_q = seq.num_scheduled_tokens
            seqlen_k = seqlen
            end = start + seqlen_q
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self._prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self._prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        return torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)

    # ── Draft model input preparation ───────────────────────────────────

    def _slot_for(self, block_table: list[int], pos: int) -> int:
        block_idx = pos // self.block_size
        offset = pos % self.block_size
        return block_table[block_idx] * self.block_size + offset

    def prepare_draft_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        for seq in seqs:
            start = seq.draft_kv_len
            end = len(seq)
            ids = seq[start:end]
            input_ids.extend(ids)
            positions.extend(list(range(start, end)))
            seqlen_q = end - start
            seqlen_k = end
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            for pos in range(start, end):
                slot_mapping.append(self._slot_for(seq.draft_block_table, pos))
        block_tables = None
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self._prepare_block_tables(seqs, use_draft=True)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_draft_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            pos = len(seq) - 1
            positions.append(pos)
            context_lens.append(len(seq))
            slot_mapping.append(self._slot_for(seq.draft_block_table, pos))
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self._prepare_block_tables(seqs, use_draft=True)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_verify(self, seqs: list[Sequence]):
        K = self.num_spec_tokens
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            start = len(seq) - K - 1
            for pos in range(start, len(seq)):
                input_ids.append(seq[pos])
                positions.append(pos)
                slot_mapping.append(self._slot_for(seq.block_table, pos))
            context_lens.append(len(seq))
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self._prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens,
                    block_tables=block_tables, num_spec_tokens=K)
        return input_ids, positions

    # ── Model execution ─────────────────────────────────────────────────

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        context = get_context()
        if context.num_spec_tokens > 0:
            return self._run_verify_graph(input_ids, positions, context)
        return self._run_decode_graph(input_ids, positions, context,
                                      self.graphs, self.graph_vars, self.model)

    @torch.inference_mode()
    def run_draft_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.draft_model.compute_logits(self.draft_model(input_ids, positions))
        context = get_context()
        return self._run_decode_graph(input_ids, positions, context,
                                      self.draft_graphs, self.draft_graph_vars, self.draft_model)

    def _run_decode_graph(self, input_ids, positions, context, graphs, graph_vars, model):
        bs = input_ids.size(0)
        graph = graphs[next(x for x in self.graph_bs if x >= bs)]
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
        graph.replay()
        return model.compute_logits(graph_vars["outputs"][:bs])

    def _run_verify_graph(self, input_ids, positions, context):
        K1 = self.num_spec_tokens + 1
        n_tokens = input_ids.size(0)
        B = n_tokens // K1
        graph = self.verify_graphs[next(x for x in self.graph_bs if x >= B)]
        gv = self.verify_graph_vars
        gv["input_ids"][:n_tokens] = input_ids
        gv["positions"][:n_tokens] = positions
        gv["slot_mapping"].fill_(-1)
        gv["slot_mapping"][:n_tokens] = context.slot_mapping
        gv["context_lens"].zero_()
        gv["context_lens"][:B] = context.context_lens
        gv["block_tables"][:B, :context.block_tables.size(1)] = context.block_tables
        graph.replay()
        return self.model.compute_logits(gv["outputs"][:n_tokens])

    # ── Main run entrypoint ─────────────────────────────────────────────

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int] | list[list[int]]:
        if not is_prefill and self.draft_model is not None and self.rank == 0:
            temperatures = self.prepare_sample(seqs)
            return self.run_speculative_decode(seqs, temperatures)

        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    # ── Speculative decoding ────────────────────────────────────────────

    @torch.inference_mode()
    def run_speculative_decode(self, seqs: list[Sequence], temperatures: torch.Tensor) -> list[list[int]]:
        K = self.num_spec_tokens
        B = len(seqs)
        device = temperatures.device
        vocab_size = self.config.hf_config.vocab_size

        draft_tokens = torch.empty(B, K, dtype=torch.int64, device=device)
        draft_probs = torch.empty(B, K, vocab_size, dtype=torch.float32, device=device)

        # ── Draft phase: generate K tokens from draft model ──
        needs_prefill = any(seq.draft_kv_len < len(seq) - 1 for seq in seqs)
        for t in range(K):
            if t == 0 and needs_prefill:
                input_ids, positions = self.prepare_draft_prefill(seqs)
                logits = self.run_draft_model(input_ids, positions, True)
            else:
                input_ids, positions = self.prepare_draft_decode(seqs)
                logits = self.run_draft_model(input_ids, positions, False)
            reset_context()

            tokens, probs = self.sampler.sample_with_probs(logits, temperatures)
            draft_tokens[:, t] = tokens
            draft_probs[:, t] = probs

            for seq in seqs:
                seq.draft_kv_len = len(seq)
            for i, seq in enumerate(seqs):
                seq.append_token(tokens[i].item())

        # ── Verify phase: run target model on K+1 tokens per seq ──
        input_ids, positions = self.prepare_verify(seqs)
        logits = self.run_model(input_ids, positions, False)
        reset_context()

        # Compute target probabilities for all K+1 positions
        temps_expanded = temperatures.repeat_interleave(K + 1)
        target_probs = self.sampler.compute_probs(logits, temps_expanded)
        target_probs = target_probs.view(B, K + 1, -1)

        # ── Rejection sampling ──
        tp = target_probs[:, :K, :]    # (B, K, V)
        indices = draft_tokens.unsqueeze(-1)
        p_x = torch.gather(tp, 2, indices).squeeze(-1)         # (B, K)
        q_x = torch.gather(draft_probs, 2, indices).squeeze(-1) # (B, K)

        ratio = (p_x / q_x.clamp(min=1e-10)).clamp(max=1.0)
        accepted = torch.rand_like(ratio) < ratio
        valid = torch.cumprod(accepted.int(), dim=1).bool()
        num_accepted = valid.sum(dim=1)                          # (B,)

        # Compute correction distribution for first rejected position
        rej_pos = num_accepted.clamp(max=K - 1)
        batch_idx = torch.arange(B, device=device)
        p_rej = tp[batch_idx, rej_pos]
        q_rej = draft_probs[batch_idx, rej_pos]
        adjusted = (p_rej - q_rej).clamp(min=0)
        adjusted_sum = adjusted.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        adjusted = adjusted / adjusted_sum

        all_accepted = (num_accepted == K)
        bonus_probs = target_probs[:, -1, :]
        next_probs = torch.where(all_accepted.unsqueeze(-1), bonus_probs, adjusted)
        next_tokens = torch.multinomial(next_probs, num_samples=1).squeeze(1)

        # ── Build per-sequence results, roll back rejected tokens ──
        results = []
        for i, seq in enumerate(seqs):
            n_acc = num_accepted[i].item()
            accepted_ids = draft_tokens[i, :n_acc].tolist()
            correction = next_tokens[i].item()

            n_to_pop = K - n_acc
            if n_to_pop > 0:
                seq.pop_last_n_tokens(n_to_pop)
            seq.draft_kv_len = min(seq.draft_kv_len, len(seq))
            seq.append_token(correction)

            results.append(accepted_ids + [correction])

        return results

    # ── CUDA graph capture ──────────────────────────────────────────────

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )

    @torch.inference_mode()
    def capture_draft_cudagraph(self):
        hf_config = self.draft_hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (self.config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.draft_graphs = {}
        self.draft_graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.draft_model(input_ids[:bs], positions[:bs])
            with torch.cuda.graph(graph, self.draft_graph_pool):
                outputs[:bs] = self.draft_model(input_ids[:bs], positions[:bs])
            if self.draft_graph_pool is None:
                self.draft_graph_pool = graph.pool()
            self.draft_graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.draft_graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )

    @torch.inference_mode()
    def capture_verify_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        K1 = self.num_spec_tokens + 1
        max_B = min(config.max_num_seqs, 512)
        max_tokens = max_B * K1
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_tokens, dtype=torch.int64)
        positions = torch.zeros(max_tokens, dtype=torch.int64)
        slot_mapping = torch.zeros(max_tokens, dtype=torch.int32)
        context_lens = torch.zeros(max_B, dtype=torch.int32)
        block_tables = torch.zeros(max_B, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_tokens, hf_config.hidden_size)
        self.verify_graphs = {}
        self.verify_graph_pool = None

        for B in reversed(self.graph_bs):
            if B > max_B:
                continue
            n = B * K1
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:n], context_lens=context_lens[:B],
                        block_tables=block_tables[:B], num_spec_tokens=self.num_spec_tokens)
            outputs[:n] = self.model(input_ids[:n], positions[:n])
            with torch.cuda.graph(graph, self.verify_graph_pool):
                outputs[:n] = self.model(input_ids[:n], positions[:n])
            if self.verify_graph_pool is None:
                self.verify_graph_pool = graph.pool()
            self.verify_graphs[B] = graph
            torch.cuda.synchronize()
            reset_context()

        self.verify_graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
