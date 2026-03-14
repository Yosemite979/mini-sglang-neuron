import os
import time
from random import randint, seed

from vllm import LLM, SamplingParams, TokensPrompt


def main():
    seed(0)

    tp_size = 2
    os.environ["NEURON_RT_NUM_CORES"] = f"{tp_size}"

    num_seqs = 256
    max_input_len = 1024
    max_output_len = 1024

    # vllm-neurion (no prefix caching)
    llm = LLM(
        model="Qwen/Qwen3-0.6B",
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=tp_size,
        max_num_seqs=6,
        max_model_len=2048,
        max_num_batched_tokens=8192,
        block_size=128,
        num_gpu_blocks_override=9,
        enable_prefix_caching=False,
    )

    # vllm-neurion (prefix caching)
    # llm = LLM(
    #     model="Qwen/Qwen3-0.6B",
    #     trust_remote_code=True,
    #     dtype="bfloat16",
    #     tensor_parallel_size=tp_size,
    #     max_num_seqs=6,
    #     max_model_len=2048,
    #     max_num_batched_tokens=8192,
    #     block_size=128,
    #     num_gpu_blocks_override=128,
    #     enable_prefix_caching=True,
    # )

    prompts = [
        TokensPrompt(
            prompt_token_ids=[
                randint(0, 10000) for _ in range(randint(100, max_input_len))
            ]
        )
        for _ in range(num_seqs)
    ]

    sampling_params = [
        SamplingParams(
            temperature=0.6,
            ignore_eos=True,
            max_tokens=randint(100, max_output_len),
        )
        for _ in range(num_seqs)
    ]

    llm.generate(
        [TokensPrompt(prompt_token_ids=[1, 2, 3])],
        SamplingParams(temperature=0.1, max_tokens=1),
    )

    start = time.time()
    llm.generate(prompts, sampling_params)
    elapsed = time.time() - start

    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / elapsed
    print(f"Total: {total_tokens} tok, Time: {elapsed:.2f}s, Throughput: {throughput:.2f} tok/s")


if __name__ == "__main__":
    main()