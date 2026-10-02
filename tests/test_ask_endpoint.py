from fastapi.testclient import TestClient

from app.main import app


def test_math_query_streams_result():
    client = TestClient(app)

    with client.stream("POST", "/ask", json={"query": "Math: 2 + 3 * 4"}) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "data:" in body
    assert "Math result: 14" in body
    assert "[DONE]" in body


def test_general_query_uses_general_chain_without_key_message():
    client = TestClient(app)

    with client.stream("POST", "/ask", json={"query": "General: explain orchestration"}) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "General response unavailable: OPENAI_API_KEY is not configured." in body
