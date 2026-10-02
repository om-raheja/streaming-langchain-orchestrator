.PHONY: install run run-gemini run-demo demo mock demo-gemini test cov lint check docker

install:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

run:
	.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

# Gemini requires GEMINI_API_KEY in .env (or the environment) — never in code
run-gemini:
	LLM_PROVIDER=gemini .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000

# Offline demo: mock OpenAI-compatible provider + orchestrator + timestamped client
mock:
	.venv/bin/python scripts/mock_openai.py --port 8124 &

run-demo:
	OPENAI_API_KEY=mock-key OPENAI_BASE_URL=http://127.0.0.1:8124/v1 \
		.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000

demo:
	.venv/bin/python scripts/sse_client.py --url http://127.0.0.1:8000/ask --query "tell me a joke about backpressure"
	.venv/bin/python scripts/sse_client.py --url http://127.0.0.1:8000/ask --query "what is 15% of 240"

# Requires `make run-gemini` in another shell (uses free-tier quota: 20 req/day)
demo-gemini:
	.venv/bin/python scripts/sse_client.py --url http://127.0.0.1:8000/ask --query "In one sentence, what is backpressure?"
	.venv/bin/python scripts/sse_client.py --url http://127.0.0.1:8000/ask --query "what is 15% of 240"

test:
	.venv/bin/python -m pytest

cov:
	.venv/bin/python -m pytest --cov=app --cov-report=term-missing

# Requires a running server, e.g. `make run` first
load:
	.venv/bin/python scripts/load_test.py --requests 200 --concurrency 100

lint:
	.venv/bin/ruff check app tests scripts

check: lint test

docker:
	docker build -t streaming-orchestrator .
