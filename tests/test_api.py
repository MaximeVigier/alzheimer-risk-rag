"""
Tests unitaires — API FastAPI. Mocke answer_question pour ne dépendre ni de ChromaDB ni
d'Ollama : teste seulement le contrat HTTP (statuts, schéma de réponse, gestion d'erreurs).
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    import api as api_module
    api_module._retrievers.clear()
    return TestClient(api_module.app)


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "generation_model" in body


def test_ask_rejects_short_query(client):
    response = client.post("/ask", json={"query": "ab"})
    assert response.status_code == 422


def test_ask_rejects_unknown_strategy(client):
    response = client.post("/ask", json={"query": "does APOE increase risk?", "strategy": "bogus"})
    assert response.status_code == 422


def test_ask_returns_answer_on_success(client):
    fake_result = {
        "answer": "APOE increases risk [PMID: 123].",
        "sources_cited": ["123"],
        "sources_retrieved": [
            {"pmid": "123", "title": "APOE study", "year": "2020", "url": "https://x"}
        ],
        "refused": False,
    }
    with patch("api.HybridRetriever") as mock_retriever_cls, \
         patch("api.answer_question", return_value=fake_result) as mock_answer:
        mock_retriever_cls.return_value = MagicMock()

        response = client.post("/ask", json={"query": "does APOE increase risk?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == fake_result["answer"]
    assert body["sources_cited"] == ["123"]
    assert body["sources"][0]["pmid"] == "123"
    assert body["refused"] is False
    mock_answer.assert_called_once()


def test_ask_returns_503_when_retriever_fails_to_load(client):
    with patch("api.HybridRetriever", side_effect=RuntimeError("index missing")):
        response = client.post("/ask", json={"query": "does APOE increase risk?"})

    assert response.status_code == 503


def test_ask_returns_502_when_generation_fails(client):
    with patch("api.HybridRetriever") as mock_retriever_cls, \
         patch("api.answer_question", side_effect=RuntimeError("ollama down")):
        mock_retriever_cls.return_value = MagicMock()

        response = client.post("/ask", json={"query": "does APOE increase risk?"})

    assert response.status_code == 502
