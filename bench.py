import os
import time
import argparse
from random import randint, seed
from nanovllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="~/huggingface/Qwen3-0.6B/")
    parser.add_argument("--speculative-model", type=str, default=None)
    parser.add_argument("--num-speculative-tokens", type=int, default=5)
    parser.add_argument("--num-seqs", type=int, default=256)
    parser.add_argument("--max-input-len", type=int, default=1024)
    parser.add_argument("--max-output-len", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()

    seed(0)
    path = os.path.expanduser(args.model)

    llm_kwargs = dict(enforce_eager=args.enforce_eager, max_model_len=args.max_model_len)
    if args.speculative_model:
        llm_kwargs["speculative_model"] = os.path.expanduser(args.speculative_model)
        llm_kwargs["num_speculative_tokens"] = args.num_speculative_tokens
    llm = LLM(path, **llm_kwargs)

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, args.max_input_len))] for _ in range(args.num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, args.max_output_len)) for _ in range(args.num_seqs)]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = (time.time() - t)
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
