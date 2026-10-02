import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.orchestrator import stream_query

app = FastAPI()


class AskRequest(BaseModel):
    query: str


@app.post("/ask")
async def ask(request: AskRequest) -> StreamingResponse:
    async def event_stream():
        async for chunk in stream_query(request.query):
            yield f"data: {json.dumps({'chunk': chunk})}\\n\\n"
        yield "data: [DONE]\\n\\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
