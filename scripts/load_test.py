"""Concurrent load probe for POST /ask.

Fires N requests at the endpoint in parallel and reports latency percentiles,
throughput, SSE integrity and error counts.

    python scripts/load_test.py --url http://127.0.0.1:8000/ask --requests 300 --concurrency 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time

import httpx

QUERIES = [
    "what is 12 * (4 + 3)",
    "15% of 240",
    "tell me a joke about latency",
    "square root of 144",
    "summarise the idea of backpressure",
    "48 * 12",
]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((pct / 100) * (len(ordered) - 1)))
    return ordered[index]


async def one_request(
    client: httpx.AsyncClient, sem: asyncio.Semaphore, stats: dict, index: int, url: str
) -> None:
    query = QUERIES[index % len(QUERIES)]
    tokens = 0
    terminal = None
    async with sem:
        started = time.perf_counter()  # latency = time inside the gate only
        try:
            async with client.stream(
                "POST", url, json={"query": query}, timeout=60.0
            ) as response:
                if response.status_code != 200:
                    stats["bad_status"] += 1
                    return
                async for line in response.aiter_lines():
                    if line.startswith("event:"):
                        terminal = line.split(":", 1)[1].strip()
                    elif line.startswith("data:") and terminal == "token":
                        json.loads(line.split(":", 1)[1].strip())
                        tokens += 1
        except Exception as exc:  # noqa: BLE001
            stats["errors"] += 1
            stats["error_samples"].append(f"{type(exc).__name__}: {exc}")
            return
    elapsed = (time.perf_counter() - started) * 1000
    if terminal != "done":
        stats["bad_terminal"] += 1
        return
    if tokens == 0:
        stats["no_tokens"] += 1
    stats["latencies"].append(elapsed)
    stats["tokens"].append(tokens)
    stats["ok"] += 1


async def run(url: str, requests: int, concurrency: int) -> int:
    stats: dict = {
        "latencies": [],
        "tokens": [],
        "ok": 0,
        "errors": 0,
        "bad_status": 0,
        "bad_terminal": 0,
        "no_tokens": 0,
        "error_samples": [],
    }
    sem = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency + 10)
    wall = time.perf_counter()
    async with httpx.AsyncClient(
        transport=httpx.AsyncHTTPTransport(), limits=limits
    ) as client:
        # warm-up: make sure the lazy orchestrator is built before timing
        warm = await client.post(url, json={"query": "warmup"})
        if warm.status_code != 200:
            print(f"warmup failed: HTTP {warm.status_code}")
            return 1
        await asyncio.gather(
            *[one_request(client, sem, stats, i, url) for i in range(requests)]
        )
    wall = time.perf_counter() - wall

    lat = stats["latencies"]
    print(f"target            : {url}")
    print(f"requests          : {requests} (concurrency {concurrency})")
    print(f"wall time         : {wall:.2f}s  ({requests / wall:.1f} req/s)")
    print(f"successful        : {stats['ok']}")
    print(
        f"errors            : {stats['errors']}  bad_status={stats['bad_status']} "
        f"bad_terminal={stats['bad_terminal']} no_tokens={stats['no_tokens']}"
    )
    if lat:
        print(
            "latency ms        : "
            f"p50={percentile(lat, 50):.1f}  p90={percentile(lat, 90):.1f}  "
            f"p95={percentile(lat, 95):.1f}  p99={percentile(lat, 99):.1f}  "
            f"max={max(lat):.1f}  mean={statistics.fmean(lat):.1f}"
        )
        print(
            f"tokens per stream : mean={statistics.fmean(stats['tokens']):.1f} min={min(stats['tokens'])} max={max(stats['tokens'])}"
        )
    for sample in stats["error_samples"][:5]:
        print(f"  ! {sample}")
    return 0 if stats["ok"] == requests and not stats["errors"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/ask")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=40)
    args = parser.parse_args()
    return asyncio.run(run(args.url, args.requests, args.concurrency))


if __name__ == "__main__":
    raise SystemExit(main())
