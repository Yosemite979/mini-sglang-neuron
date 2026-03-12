from __future__ import annotations

import asyncio
import os
import random
from pathlib import Path

from minisgl.benchmark.client import (
    benchmark_trace,
    get_model_name,
    process_benchmark_results,
    read_qwen_trace,
    scale_traces,
)
from minisgl.utils import init_logger
from openai import AsyncOpenAI as OpenAI
from transformers import AutoTokenizer

logger = init_logger(__name__)

URL = "https://media.githubusercontent.com/media/alibaba-edu/qwen-bailian-usagetraces-anon/refs/heads/main/qwen_traceA_blksz_16.jsonl"


def download_qwen_trace(url: str) -> str:
    dir = Path(os.path.dirname(__file__))
    # download the file if not exists
    file_path = dir / "qwen_traceA_blksz_16.jsonl"
    if not file_path.exists():
        import urllib.request

        logger.info(f"Downloading trace from {url} to {file_path}...")
        urllib.request.urlretrieve(url, file_path)
        logger.info("Download completed.")
    return str(file_path)


async def main():
    random.seed(42)  # reproducibility
    PORT = 1919
    N = 500
    MAX_INPUT_LEN = 1024
    SCALES = [0.5, 1.0, 1.5, 2.0, 2.5]  # from fast to slow
    async with OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="") as client:
        MODEL = await get_model_name(client)
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        TRACES = read_qwen_trace(download_qwen_trace(URL), tokenizer, n=N, dummy=True, max_input_len=MAX_INPUT_LEN)
        logger.info(f"Start benchmarking with {N} requests using model {MODEL}...")
        for scale in SCALES:
            traces = scale_traces(TRACES, scale)
            results = await benchmark_trace(client, traces, MODEL)
            process_benchmark_results(results)
        logger.info("Benchmarking completed.")


if __name__ == "__main__":
    asyncio.run(main())
