# Features of Mini-SGLang-Neuron

## Online Serving

Mini-SGLang supports online serving with an OpenAI-compatible API server. It provides the standard `/v1/chat/completions` endpoint, allowing seamless integration with existing tools and clients. For detailed command-line arguments and configuration options, run `python -m minisgl --help`.

## Interactive Shell Mode

For demonstration and testing purposes, an interactive shell mode is available. In this mode, users can input prompts directly, and the LLM will generate responses in real-time. The shell automatically caches chat history to maintain context. To clear the conversation history and start a new session, use the `/reset` command.

Example:

```bash
python -m minisgl --model "Qwen/Qwen3-0.6B" --shell
```

## Distributed Serving

To scale performance across multiple GPUs, Mini-SGLang supports Tensor Parallelism (TP). You can enable distributed serving by specifying the number of GPUs with the `--tp n` argument, where `n` is the degree of parallelism.

## Supported Models

Our framework currently supports the following dense model architectures:

- [`Qwen-3`](https://huggingface.co/collections/Qwen/qwen3) series

## Chunked Prefill

Chunked Prefill, a technique introduced by [Sarathi-Serve](https://arxiv.org/abs/2403.02310), is enabled by default. This feature splits long prompts into smaller chunks during the prefill phase, significantly reducing peak memory usage and preventing Out-Of-Memory (OOM) errors in long-context serving. The chunk size can be configured using `--max-prefill-length n`. Note that setting `n` to a very small value (e.g., 128) is not recommended as it may significantly degrade performance.

## Runtime Backend

The current Mini-SGLang-Neuron runtime in this repository is Neuron/XLA-focused and does not expose CUDA attention backend selection or CUDA graph controls in the CLI.

## Radix Cache

Adopting the original design from [SGLang](https://github.com/sgl-project/sglang.git), Mini-SGLang-Neuron integrates Radix Cache with the NxDI block KV cache manager to manage the Key-Value (KV) cache. This allows the reuse of KV cache for shared prefixes across requests, reducing redundant computation. This feature is enabled by default but can be switched to a naive cache management strategy using `--cache naive`.

![radix](https://lmsys.org/images/blog/sglang/radix_attn.jpg)
*Illustration of Radix Attention from [LMSYS Blog](https://lmsys.org/blog/2024-01-17-sglang/).*

