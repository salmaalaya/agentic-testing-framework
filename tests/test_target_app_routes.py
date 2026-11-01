"""HTTP routes of the target app, exercised in-process (no server, no Ollama call on the fast path)."""

import pytest
from fastapi.testclient import TestClient

from target_app.app import app

client = TestClient(app)


@pytest.mark.parametrize("path", ["/", "/studio", "/dashboard"])
def test_html_pages_render(path):
    response = client.get(path)
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_products_lists_the_fictional_catalogue():
    products = client.get("/products").json()["products"]
    assert "HeadsetPro ANC" in {p["name"] for p in products.values()}


def test_chat_fast_path_answers_from_the_catalogue():
    response = client.post("/chat", json={"message": "Où en est ma commande #1234 ?", "session_id": "t-routes"})
    assert response.status_code == 200
    body = response.json()
    assert body["engine"] == "keyword"
    assert body["intent"] == "order_tracking"
    assert "1234" in body["response"]


def test_chat_empty_message_does_not_crash():
    response = client.post("/chat", json={"message": "", "session_id": "t-routes-empty"})
    assert response.status_code == 200
    assert response.json()["intent"] == "empty_input"
