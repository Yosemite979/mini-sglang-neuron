# Adapted from: https://github.com/GeeeekExplorer/nano-vllm/blob/main/bench.py

import time
import os
from random import randint, seed

from minisgl.core import SamplingParams
from minisgl.llm import LLM


def main():
    seed(0)
    tp_size = 2
    os.environ["NEURON_RT_NUM_CORES"] = str(tp_size)

    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024

    # align the hyperparameters
    # mini-sglang-neurion (no radix)
    llm = LLM(
        "Qwen/Qwen3-0.6B",
        max_seq_len_override=2048, 
        max_running_req=6,
        tp_size=tp_size, 
        num_page_override=16384,
        max_extend_tokens=8192,
        cache_type="naive",
    )
    
    # mini-sglang-neurion (radix)
    # llm = LLM(
    #     "Qwen/Qwen3-0.6B",
    #     max_seq_len_override=2048, 
    #     max_running_req=6,
    #     tp_size=tp_size, 
    #     num_page_override=16384,
    #     max_extend_tokens=8192,
    #     cache_type="radix",
    # )

    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)
    ]
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len))
        for _ in range(num_seqs)
    ]
    llm.generate(["Benchmark: "], SamplingParams(temperature=0.1)) 
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params)
    t = time.time() - t
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
