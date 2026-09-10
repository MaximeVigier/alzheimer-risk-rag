"""
Tests unitaires — retrieval (fusion RRF), avec un HybridRetriever minimal sans I/O
(pas de ChromaDB ni de modèle d'embedding réel, seulement la logique de fusion testée).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retrieval import HybridRetriever


def _make_retriever(dense_order, sparse_order, chunk_by_id):
    """Construit un HybridRetriever sans passer par __init__ (pas de ChromaDB/BM25 réels),
    puis monkeypatche les deux méthodes de recherche brutes."""
    r = HybridRetriever.__new__(HybridRetriever)
    r.rrf_k = 60
    r.chunk_by_id = chunk_by_id
    r._dense_search = lambda query, top_n: dense_order[:top_n]
    r._sparse_search = lambda query, top_n: sparse_order[:top_n]
    return r


def test_hybrid_search_favors_docs_ranked_high_in_both():
    chunk_by_id = {cid: {"chunk_id": cid, "text": cid} for cid in ["a", "b", "c", "d"]}
    # "a" est bien classé dans les deux listes -> doit sortir en tête du RRF
    dense_order = ["a", "b", "c", "d"]
    sparse_order = ["a", "d", "c", "b"]
    r = _make_retriever(dense_order, sparse_order, chunk_by_id)

    results = r.search("query", k=2, mode="hybrid", candidate_pool=4)

    assert results[0]["chunk_id"] == "a"
    assert len(results) == 2


def test_dense_mode_returns_dense_order_only():
    chunk_by_id = {cid: {"chunk_id": cid, "text": cid} for cid in ["x", "y", "z"]}
    r = _make_retriever(["x", "y", "z"], ["z", "y", "x"], chunk_by_id)

    results = r.search("query", k=2, mode="dense")

    assert [c["chunk_id"] for c in results] == ["x", "y"]
    assert all(c["score"] is None for c in results)


def test_sparse_mode_returns_sparse_order_only():
    chunk_by_id = {cid: {"chunk_id": cid, "text": cid} for cid in ["x", "y", "z"]}
    r = _make_retriever(["x", "y", "z"], ["z", "y", "x"], chunk_by_id)

    results = r.search("query", k=2, mode="sparse")

    assert [c["chunk_id"] for c in results] == ["z", "y"]


def test_hybrid_search_score_present_and_ordered_desc():
    chunk_by_id = {cid: {"chunk_id": cid, "text": cid} for cid in ["a", "b", "c"]}
    r = _make_retriever(["a", "b", "c"], ["a", "b", "c"], chunk_by_id)

    results = r.search("query", k=3, mode="hybrid", candidate_pool=3)

    scores = [c["score"] for c in results]
    assert scores == sorted(scores, reverse=True)
