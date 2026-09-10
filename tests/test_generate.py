"""
Tests unitaires — génération : garde-fou de refus sur retrieval faible, extraction des PMID
cités. Le LLM (Ollama) est mocké — on ne teste pas la qualité de génération ici (voir eval/
pour les métriques RAGAS), seulement la logique du garde-fou et le parsing.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from generate import MIN_RRF_SCORE, _extract_cited_pmids, answer_question


def test_extract_cited_pmids_dedupes_and_sorts():
    answer = "Risk increases [PMID: 999] with age [PMID: 111], see also [PMID: 999]."
    assert _extract_cited_pmids(answer) == ["111", "999"]


def test_extract_cited_pmids_empty_when_no_citation():
    assert _extract_cited_pmids("No sources cited here.") == []


def test_answer_question_refuses_on_empty_candidates():
    retriever = MagicMock()
    retriever.search.return_value = []

    result = answer_question("unrelated question", retriever=retriever)

    assert result["refused"] is True
    assert result["sources_cited"] == []
    retriever.search.assert_called_once()


def test_answer_question_refuses_below_score_threshold():
    retriever = MagicMock()
    retriever.search.return_value = [
        {"pmid": "1", "title": "T", "year": "2020", "url": "u", "text": "irrelevant",
         "score": MIN_RRF_SCORE / 2}
    ]

    result = answer_question("borderline question", retriever=retriever)

    assert result["refused"] is True


def test_answer_question_calls_ollama_when_retrieval_strong():
    retriever = MagicMock()
    retriever.search.return_value = [
        {"pmid": "1", "title": "T", "year": "2020", "url": "u", "text": "relevant chunk",
         "score": MIN_RRF_SCORE * 10}
    ]

    fake_response = {"message": {"content": "APOE increases risk [PMID: 1]."}}

    with patch("generate.rerank") as mock_rerank, patch("generate.ollama") as mock_ollama:
        mock_rerank.return_value = retriever.search.return_value
        mock_ollama.chat.return_value = fake_response

        result = answer_question("does APOE increase risk?", retriever=retriever)

    assert result["refused"] is False
    assert result["sources_cited"] == ["1"]
    assert "APOE" in result["answer"]
    mock_ollama.chat.assert_called_once()
