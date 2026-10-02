# streaming-langchain-orchestrator

A minimal FastAPI + LangChain orchestrator with SSE streaming.

## Endpoint

- `POST /ask`
- JSON payload: `{"query": "..."}`
- Returns Server-Sent Events (`text/event-stream`) with chunked `data:` events.

## Routing behavior

LangChain routing is handled with `RunnableBranch`:

- Queries containing `Math` route to a safe arithmetic chain.
- Queries containing `General` route to a standard LLM chain.
- Other queries default to the general chain.

## Security

No API keys are hardcoded. The general LLM chain reads from environment variables:

- `OPENAI_API_KEY` (required for OpenAI-backed responses)
- `OPENAI_MODEL` (optional, defaults to `gpt-4o-mini`)

## Run

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```
