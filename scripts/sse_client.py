"""Tiny SSE client used for demos (and the Loom walkthrough).

Prints every event the moment it arrives, with a millisecond offset, which
makes the incremental nature of the stream visible:

    python scripts/sse_client.py --query "tell me a joke about latency"

    +    0.9ms event: meta
    +    1.6ms event: route
    +    4.1ms event: token   (delta="Mock ")
    +   43.7ms event: token   (delta="model ")
    ...
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import httpx


async def stream_query(url: str, query: str, timeout: float) -> int:
    started = time.perf_counter()
    payload = {"query": query}
    try:
        async with (
            httpx.AsyncClient(timeout=timeout) as client,
            client.stream("POST", url, json=payload) as response,
        ):
            print(f"HTTP {response.status_code} {response.headers.get('content-type')}")
            print(f"request-id: {response.headers.get('x-request-id')}")
            print("-" * 72)
            current_event = ""
            async for line in response.aiter_lines():
                elapsed = (time.perf_counter() - started) * 1000
                if line.startswith("event:"):
                    current_event = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    data = json.loads(line.split(":", 1)[1].strip())
                    if current_event == "token":
                        print(
                            f"+{elapsed:8.1f}ms  token   {data['index']:>3}  {data['delta']!r}"
                        )
                    elif current_event == "route":
                        print(
                            f"+{elapsed:8.1f}ms  route   {data['route']} "
                            f"(source={data['source']}, routing_ms={data['routing_ms']})"
                        )
                    elif current_event == "tool":
                        print(f"+{elapsed:8.1f}ms  tool    {data}")
                    else:
                        print(f"+{elapsed:8.1f}ms  {current_event:<7} {data}")
                elif line == "":
                    current_event = ""
    except httpx.HTTPError as exc:  # pragma: no cover - demo helper
        print(f"request failed: {exc}", flush=True)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/ask")
    parser.add_argument("--query", default="tell me a joke about streaming")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    return asyncio.run(stream_query(args.url, args.query, args.timeout))


if __name__ == "__main__":
    raise SystemExit(main())
