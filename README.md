# High-Speed Streaming LangChain Orchestrator

> Repository description (upstream): *A highly optimized langchain orchestrator
> that provides a FastAPI streaming endpoint.*

A production-shaped FastAPI service that takes a user query, **routes it with
LangChain**, executes the right branch (a sandboxed calculator tool or a
general LLM chain) and **streams the answer back as Server-Sent Events**,
chunk by chunk, as soon as tokens exist. Supports **OpenAI and Google Gemini**
as interchangeable providers.

```
POST /ask  {"query": "what is 15% of 240"}
      │
      ▼
┌────────────────────────────  SSE stream  ────────────────────────────┐
│ event: meta   {"request_id": "..."}            ← instant TTFB        │
│ event: route  {"route":"math","source":"heuristic","routing_ms":1.7} │
│ event: tool   {"expression":"(15 / 100) * 240","result":"36"}        │
│ event: token  {"delta":"36","index":0}          ← streamed deltas     │
│ ...                                                                  │
│ event: done   {"chunks":12,"first_token_ms":41.2,"duration_ms":96.4} │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # put your OPENAI_API_KEY in there (git-ignored)
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

No key handy? The service still runs: both branches **degrade gracefully to
deterministic, LLM-free behaviour** (see *Degraded modes* below).

```bash
curl -sN -X POST localhost:8000/ask \
  -H 'content-type: application/json' \
  -d '{"query":"what is 12 * (4 + 3)"}'
```

Watch the stream land token-by-token with timestamps:

```bash
python scripts/sse_client.py --query "tell me a joke about latency"
```

### Full local demo without any provider (mock OpenAI)

`scripts/mock_openai.py` implements just enough of the OpenAI wire protocol
(non-streaming JSON-schema structured output + chunked
`chat.completion.chunk` SSE) that the **real `langchain-openai` client** runs
end-to-end offline:

```bash
python scripts/mock_openai.py --port 8124 &
OPENAI_API_KEY=mock-key OPENAI_BASE_URL=http://127.0.0.1:8124/v1 \
  uvicorn app.main:app --port 8000
python scripts/sse_client.py --query "tell me a joke about backpressure"
```

---

## The endpoint

| | |
|---|---|
| Route | `POST /ask` |
| Request | `{"query": "..."}` (1..`MAX_QUERY_CHARS`, no extra fields) |
| Response | `200`, `Content-Type: text/event-stream; charset=utf-8` |
| Errors | `422` for malformed/oversized payloads, `event: error` frame for runtime failures |

Headers that matter for streaming through proxies:

```
Cache-Control: no-cache, no-transform      # never buffer/compress
X-Accel-Buffering: no                     # disable nginx buffering
Connection: keep-alive
X-Request-ID: <id>                        # propagated request id
```

### Event protocol

| `event` | `data` | Meaning |
|---|---|---|
| `meta` | `request_id, server, protocol, ts` | Stream opened (emitted **before** any model call → tiny TTFB) |
| `route` | `route, reason, source, routing_ms` | Router decision. `source` ∈ `heuristic \| llm \| fallback` |
| `tool` | `name, ok, expression?, result?, error?` | Calculator tool result (math branch) |
| `token` | `delta, source, index` | One incremental chunk of model/tool output |
| `done` | `route, chunks, first_token_ms, duration_ms` | Successful terminal frame |
| `error` | `code, message` | Failure terminal frame (message is redacted) |

Every frame also carries `id: <n>` (monotonic) so clients can resume with
`Last-Event-ID` semantics.

Browser client:

```js
const es = new EventSource("/ask");          // GET; use fetch+ReadableStream for POST
```

`scripts/sse_client.py` shows the equivalent with `httpx` and prints each
event's millisecond offset.

---

## Orchestration & routing (LangChain)

Routing is a single `Runnable[str, RouteDecision]` composed from two stages
(`app/orchestrator/router.py`):

1. **Heuristic fast path** — a pure function that short-circuits unambiguous
   math queries (an extractable arithmetic expression, or math vocabulary).
   Zero network cost (~1 ms observed).
2. **Semantic classifier** — an LLM constrained with **structured output**
   (`llm.with_structured_output(RouteDecision)`, `json_schema`/strict) that
   decides everything else. It runs with `streaming=False` (one buffered round
   trip is cheapest) and is wrapped in `asyncio.wait_for(ROUTER_TIMEOUT_S)`:

   * timeout  → `source=fallback`, branch `general`
   * any exception → same, with a redacted reason
   * routing **never fails a request**

Both stages are LangChain Runnables, so the whole route emits callbacks and is
traced end-to-end in LangSmith (`run_name` is set on every stage:
`route_heuristic`, `semantic_classifier`, `query_router`, `general_chain`,
`math_narrator`).

### Branches

* **math** (`app/orchestrator/math_tool.py`) — extracts an expression from
  the query (`2 + 2`, `15% of 240`, `square root of 144`, `12 times 4`, …),
  evaluates it in a **sandboxed AST evaluator** (allow-listed nodes only — no
  `eval`, no attribute access, no imports, exponent/size/magnitude caps), emits
  a `tool` frame, then streams an explanation from the LLM when one is
  configured. If the narrator stalls, the already-correct tool answer is
  streamed instead of failing the request.
* **general** (`app/orchestrator/chains.py`) — LCEL chain
  `prompt | llm | StrOutputParser()` streamed with `astream`.

---

## Streaming engineering notes

* **Async generator pipeline** — `Orchestrator.stream()` yields `StreamEvent`
  objects; the endpoint maps them to SSE frames. No intermediate buffering of
  the full response.
* **Per-chunk stall timeout** — every `__anext__` await is wrapped in
  `asyncio.timeout(STREAM_STALL_TIMEOUT_S)`; a hung provider becomes a clean
  `error` frame instead of a socket held open forever.
* **Cancellation-safe** — client disconnects surface as
  `GeneratorExit`/`CancelledError`, which is re-raised so uvicorn cancels the
  upstream task instead of generating tokens nobody will read
  (`test_stream_cancels_cleanly_when_client_disconnects`).
* **Lazy singleton orchestrator** — models are built once per process, on the
  first request, and closed on shutdown (`lifespan` → `aclose()`).
* **Backpressure** — handled by Starlette's `StreamingResponse`; each frame is
  written to the socket only when the client is ready to read it.
* **Observability** — request id, route, source, routing ms, first-token ms and
  total duration are logged and returned in-band (`done` frame).

Measured with the bundled mock provider (no artificial delay on our side):

```
+  104.9ms  meta
+  116.3ms  route   general (source=llm, routing_ms=11.74)
+  121.2ms  token     0  'Mock '
+  141.2ms  token     1  'model '
+  161.7ms  token     2  'answer '
```

### Load characteristics

`scripts/load_test.py` fires N concurrent streams and checks every frame
(status, `event: done` present, deltas parse, ≥1 token):

```bash
python scripts/load_test.py --requests 200 --concurrency 100
```

Offline deployment (no provider — routes, tool and framing only), single
uvicorn worker, 200 requests per run:

| concurrency | p50 latency | throughput | failures |
|---:|---:|---:|---:|
| 1 | 4.6 ms | 189 req/s | 0 |
| 5 | 20.6 ms | 219 req/s | 0 |
| 20 | 78.5 ms | 218 req/s | 0 |
| 50 | 200.6 ms | 216 req/s | 0 |
| 100 | 399.3 ms | 217 req/s | 0 |

Zero errors across every run; latency above ~5 concurrent is pure queueing
behind the single worker's ~4.6 ms/request budget, and scales linearly with
`--workers` (800 requests @ 100 concurrent: **237 req/s** with 1 worker →
**435 req/s** with 4, p50 399 ms → 184 ms).

End-to-end through the real `langchain-openai` client against the bundled
mock provider (500 requests @ 100 concurrent): **500/500 successful**, ~20
req/s — throughput there is bounded by the two provider round trips per
request (routing + generation), not by the streaming layer.


---

## Security

* **Zero hardcoded credentials.** The credential enters the process at exactly
  one point: `os.getenv("OPENAI_API_KEY")` in `app/config.py`, optionally
  seeded from a git-ignored `.env` (`python-dotenv`, never overriding real
  process env vars, so Docker/K8s secrets win).
* `tests/test_security.py` fails the build if any file under `app/` contains a
  key-shaped literal, and asserts `.env` is git-ignored and `.env.example`
  holds only a placeholder.
* Every outbound error message passes through `app/security.py::redact()`
  (OpenAI/Anthropic/HF/Google/Slack token shapes + `Bearer …`), so a provider
  error echoing a key can never reach a client or a log line.
* Payload hardening: `extra="forbid"`, length limits, stripped non-empty
  query, and a sandboxed calculator (no `eval`, no introspection).

---

## Degraded modes (no API key configured)

| Capability | Behaviour |
|---|---|
| Router | heuristic fast path, otherwise `general` with `source=fallback` |
| `general` branch | streams an explanatory offline notice |
| `math` branch | calculator still works (it's local); a missing expression or a stalled narrator falls back to deterministic text |

This is what lets the test-suite — including the SSE contract tests — run with
**no network and no credentials**.

---

## Testing

```bash
pytest                        # 92 tests, ~5s, no network needed
pytest --cov=app              # 92% line coverage
ruff check app tests scripts  # lint clean
```

| Suite | Covers |
|---|---|
| `test_endpoint_sse.py` | SSE contract, headers, chunking, ids, validation, error frames, concurrency |
| `test_router.py` | heuristic vs semantic routing, timeout + failure degradation |
| `test_math_tool.py` | expression evaluation, NL extraction, sandbox escape attempts, resource bombs |
| `test_security.py` | no hardcoded keys, env loading, redaction |
| `test_lifecycle.py` | cancellation, offline modes, stall timeout |
| `test_integration_provider.py` | **real `ChatOpenAI` client** against an in-process OpenAI-compatible mock (structured routing + streamed tokens) |

---

## Project layout

```
app/
├── main.py                     # FastAPI app, POST /ask, /health, middleware
├── config.py                   # os.getenv + .env -> frozen Settings
├── sse.py                      # SSE framing (id/event/data, headers)
├── security.py                 # redaction + secret detection
├── schemas.py                  # AskRequest, RouteDecision, StreamEvent
├── llm.py                      # provider factory (returns None w/o key)
└── orchestrator/
    ├── router.py               # heuristic + LLM structured-output router
    ├── chains.py               # LCEL general / math-narrator chains
    ├── math_tool.py            # sandboxed calculator + NL extraction
    └── pipeline.py             # Orchestrator.stream() async generator
scripts/
├── mock_openai.py              # offline OpenAI-compatible provider
├── sse_client.py               # timestamped SSE demo client
└── load_test.py                # concurrent SSE load probe (latency percentiles)
tests/                          # 92 tests, pytest-asyncio
```

---

## Running in production

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
# or
docker build -t orchestrator . && docker run -p 8000:8000 --env-file .env orchestrator
```

Configuration (all optional except `OPENAI_API_KEY` for live model output):
`ROUTE_MODEL`, `GENERAL_MODEL`, `MATH_MODEL`, `MODEL_TEMPERATURE`,
`ROUTER_TIMEOUT_S`, `STREAM_STALL_TIMEOUT_S`, `MAX_QUERY_CHARS`, `LOG_LEVEL`.

> Python 3.11+ is required (`asyncio.timeout`). Verified on 3.12.

---

## Demo video (Loom)

The Loom walkthrough covers the architecture, the two routing stages, a
live streaming demo against the bundled mock provider, the sandbox test
suite, and the security review.

**Loom URL:** _\<paste the recording link here\>_
