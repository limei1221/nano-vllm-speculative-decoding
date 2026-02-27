import os
import argparse
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="~/huggingface/Qwen3-0.6B/")
    parser.add_argument("--speculative-model", type=str, default=None)
    parser.add_argument("--num-speculative-tokens", type=int, default=5)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()

    path = os.path.expanduser(args.model)
    tokenizer = AutoTokenizer.from_pretrained(path)

    llm_kwargs = dict(enforce_eager=args.enforce_eager, tensor_parallel_size=1)
    if args.speculative_model:
        llm_kwargs["speculative_model"] = os.path.expanduser(args.speculative_model)
        llm_kwargs["num_speculative_tokens"] = args.num_speculative_tokens
    llm = LLM(path, **llm_kwargs)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
