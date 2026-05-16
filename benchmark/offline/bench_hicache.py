"""
HiCache Benchmark: Multi-turn agent scenario with shared system prompt.

Usage:
  1. Start server WITHOUT HiCache:
     python -m minisgl --model "Qwen/Qwen3-0.6B" --num-pages 2000 --cache-type radix

  2. Run this benchmark:
     python benchmark/offline/bench_hicache.py --label "no_hicache"

  3. Restart server WITH HiCache:
     python -m minisgl --model "Qwen/Qwen3-0.6B" --num-pages 2000 --cache-type hiradix --hicache-ratio 2.0

  4. Run again:
     python benchmark/offline/bench_hicache.py --label "with_hicache"

  Compare the results between step 2 and step 4.
"""

import argparse
import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import List

from openai import AsyncOpenAI


# ─── Scenario: Agent with long system prompt ──────────────────────────────

SYSTEM_PROMPT = """You are a helpful AI assistant with deep knowledge across multiple domains including computer science, mathematics, physics, and literature.
Your role is to answer user questions accurately and concisely. You should follow these guidelines:
1. Always respond in the same language as the user's question.
2. Be concise but thorough in your explanations.
3. When discussing technical topics, provide code examples when appropriate.
4. If you are unsure about something, acknowledge that rather than making things up.
5. Use structured formatting (lists, headers) for complex answers.
6. When comparing multiple options, provide a balanced analysis of pros and cons.
7. For mathematical or logical problems, show your step-by-step reasoning.
8. When summarizing documents, focus on key points and actionable insights.
9. Maintain a professional and friendly tone throughout the conversation.
10. If a question is ambiguous, ask clarifying questions before providing a detailed answer.
Additional context: The user is a software engineer working on large language model inference systems.
They are particularly interested in KV cache management, GPU memory optimization, and serving throughput.
The technical stack includes PyTorch, CUDA, and distributed training frameworks.
Background knowledge: KV cache stores key and value vectors from the attention mechanism to avoid redundant computation during autoregressive decoding.
The cache grows linearly with sequence length and batch size, becoming the dominant memory consumer in long-context scenarios.
Hierarchical caching offloads cold KV cache entries to CPU host memory, expanding effective cache capacity.
Prefix caching with radix tree enables sharing common prefixes across multiple requests, reducing prefill computation.
The system uses paged KV cache with configurable page sizes for efficient memory management.
Asynchronous transfers using CUDA streams overlap data movement with model computation.
NUMA-aware memory allocation ensures optimal PCIe bandwidth for host-device transfers.
Tensor parallelism splits KV cache across multiple GPUs requiring synchronization.
Continuous batching allows requests of different lengths to be processed together efficiently.
CUDA graphs eliminate Python and kernel launch overhead for decode steps.
Speculative decoding uses a draft model to generate multiple tokens in parallel.
This is the end of the system instruction. Please acknowledge that you understand these guidelines."""

USER_QUESTIONS = [
    "What is KV cache and why is it important in LLM inference?",
    "How does prefix caching work with radix tree? Explain the mechanism.",
    "What are the trade-offs between page size = 1 and page_size > 1?",
    "Explain how CUDA graph works in LLM serving systems.",
    "What is tensor parallelism and how does it affect KV cache layout?",
    "How do you measure and optimize KV cache memory bandwidth?",
    "Compare continuous batching with static batching for LLM inference.",
    "What are the challenges of multi-GPU KV cache synchronization?",
    "How does speculative decoding interact with KV cache management?",
    "What is the impact of different data types (fp16 vs int8) on KV cache?",
    "How do you handle KV cache fragmentation in long-running serving?",
    "Explain the difference between layer-first and page-first memory layouts.",
    "What is NUMA and why does it matter for host-side KV cache?",
    "How would you design a distributed KV cache across multiple nodes?",
    "What are the pros and cons of offloading KV cache to NVMe storage?",
    "How does sliding window attention affect KV cache requirements?",
    "What is the relationship between context length and serving throughput?",
    "How do you implement efficient eviction policies for prefix cache?",
    "What role does the scheduler play in LLM inference performance?",
    "How would you benchmark a KV cache system for production use?",
    "What is the difference between MHA, MQA, and GQA attention?",
    "How does FlashAttention improve upon standard attention computation?",
    "What is PagedAttention and how does it solve memory fragmentation?",
    "Explain the concept of chunked prefill and when it is useful.",
    "How do you handle request preemption in an LLM serving system?",
]


@dataclass
class RequestResult:
    uid: int
    prompt_len: int
    output_len: int
    ttft: float
    total_time: float
    tokens_per_second: float


@dataclass
class BenchmarkResult:
    label: str
    total_requests: int
    total_prompt_tokens: int
    total_output_tokens: int
    avg_ttft: float
    p50_ttft: float
    p99_ttft: float
    avg_total_time: float
    throughput: float
    wall_time: float
    results: List[RequestResult] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"  Label:              {self.label}",
            f"  Total requests:     {self.total_requests}",
            f"  Total prompt tokens: {self.total_prompt_tokens}",
            f"  Total output tokens: {self.total_output_tokens}",
            f"  Avg TTFT:           {self.avg_ttft:.3f}s",
            f"  P50 TTFT:           {self.p50_ttft:.3f}s",
            f"  P99 TTFT:           {self.p99_ttft:.3f}s",
            f"  Avg total time:     {self.avg_total_time:.3f}s",
            f"  Throughput:         {self.throughput:.2f} tok/s",
            f"  Wall time:          {self.wall_time:.2f}s",
        ]
        return "\n".join(lines)


async def run_single_request(
    client: AsyncOpenAI,
    model: str,
    conversation: List[dict],
    max_tokens: int,
    uid: int,
) -> RequestResult:
    prompt_len = 0
    for msg in conversation:
        prompt_len += len(msg["content"]) // 4

    start = time.time()
    first_token_time = None

    stream = await client.chat.completions.create(
        model=model,
        messages=conversation,
        max_tokens=max_tokens,
        temperature=0.7,
        stream=True,
    )

    output_tokens = 0
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            output_tokens += 1
            if first_token_time is None:
                first_token_time = time.time() - start

    total_time = time.time() - start
    ttft = first_token_time if first_token_time is not None else total_time

    return RequestResult(
        uid=uid,
        prompt_len=prompt_len,
        output_len=output_tokens,
        ttft=ttft,
        total_time=total_time,
        tokens_per_second=output_tokens / total_time if total_time > 0 else 0,
    )


async def run_benchmark(
    base_url: str,
    label: str,
    num_agents: int = 16,
    turns_per_agent: int = 3,
    max_tokens: int = 64,
    concurrency: int = 8,
) -> BenchmarkResult:
    client = AsyncOpenAI(base_url=f"{base_url}/v1", api_key="not-needed")

    models = await client.models.list()
    model = models.data[0].id
    print(f"  Connected to model: {model}")

    conversations: List[List[dict]] = []
    random.seed(42)

    for i in range(num_agents * turns_per_agent):
        sys_msg = {"role": "system", "content": SYSTEM_PROMPT}
        user_msg = {"role": "user", "content": USER_QUESTIONS[i % len(USER_QUESTIONS)]}
        conversations.append([sys_msg, user_msg])

    random.shuffle(conversations)
    print(f"  Launching {len(conversations)} requests with concurrency={concurrency}...")

    all_results: List[RequestResult] = []
    wall_start = time.time()

    semaphore = asyncio.Semaphore(concurrency)
    completed = 0
    lock = asyncio.Lock()

    async def bounded_run(convo, idx):
        async with semaphore:
            result = await run_single_request(client, model, convo, max_tokens, idx)
            nonlocal completed
            completed += 1
            async with lock:
                print(f"  [{completed}/{len(tasks)}] Request {idx} done in {result.total_time:.2f}s, TTFT={result.ttft:.3f}s", flush=True)
            return result

    tasks = [bounded_run(convo, i) for i, convo in enumerate(conversations)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for r in results:
        if isinstance(r, Exception):
            print(f"  Warning: request failed: {r}")
        else:
            all_results.append(r) 

    wall_time = time.time() - wall_start

    if not all_results:
        print("  No successful requests.")
        return BenchmarkResult(
            label=label,
            total_requests=0,
            total_prompt_tokens=0,
            total_output_tokens=0,
            avg_ttft=0,
            p50_ttft=0,
            p99_ttft=0,
            avg_total_time=0,
            throughput=0,
            wall_time=0,
        )

    ttfts = sorted([r.ttft for r in all_results])
    total_output = sum(r.output_len for r in all_results)
    total_prompt = sum(r.prompt_len for r in all_results)

    return BenchmarkResult(
        label=label,
        total_requests=len(all_results),
        total_prompt_tokens=total_prompt,
        total_output_tokens=total_output,
        avg_ttft=sum(ttfts) / len(ttfts),
        p50_ttft=ttfts[len(ttfts) // 2],
        p99_ttft=ttfts[int(len(ttfts) * 0.99)],
        avg_total_time=sum(r.total_time for r in all_results) / len(all_results),
        throughput=total_output / wall_time if wall_time > 0 else 0,
        wall_time=wall_time,
        results=all_results,
    )


def main():
    parser = argparse.ArgumentParser(description="HiCache Benchmark")
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://127.0.0.1:1919",
        help="Server base URL",
    )
    parser.add_argument("--num-agents", type=int, default=16, help="Number of simulated agents")
    parser.add_argument(
        "--turns",
        type=int,
        default=3,
        help="Conversation rounds per agent",
    )
    parser.add_argument("--max-tokens", type=int, default=64, help="Max output tokens per request")
    parser.add_argument(
        "--concurrency", type=int, default=8, help="Max concurrent requests"
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Label for this run",
    )
    args = parser.parse_args()

    label = args.label or "benchmark"

    print("=" * 70)
    print(f"HiCache Benchmark: {label}")
    print(f"Server: {args.base_url}")
    print(f"Agents: {args.num_agents}, Turns: {args.turns}, "
          f"Concurrency: {args.concurrency}, Max tokens: {args.max_tokens}")
    print("=" * 70)

    result = asyncio.run(
        run_benchmark(
            base_url=args.base_url,
            label=label,
            num_agents=args.num_agents,
            turns_per_agent=args.turns,
            max_tokens=args.max_tokens,
            concurrency=args.concurrency,
        )
    )

    print(f"\nResults ({result.label}):")
    print(result.summary())
    return result


if __name__ == "__main__":
    main()
