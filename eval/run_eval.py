"""
Évaluation quantitative du RAG : pour chaque configuration (stratégie de chunking x mode
de retrieval), calcule recall@k (le bon document source apparaît-il dans le top-k retrieved ?)
et faithfulness (RAGAS — la réponse générée est-elle vraiment supportée par le contexte cité ?).

Modèle générateur : qwen3:14b (celui utilisé en production dans generate.py).
Modèle juge (faithfulness) : gpt-oss:20b — DIFFÉRENT du générateur, pour éviter le biais
d'auto-évaluation (un modèle qui juge ses propres réponses tend à se sur-noter).

Usage:
    python eval/run_eval.py --strategy fixed --modes dense,hybrid,hybrid_rerank
    python eval/run_eval.py --strategy fixed --modes hybrid_rerank --skip-faithfulness  # rapide, recall@k seul
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import openai
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from generate import answer_question  # noqa: E402
from rerank import rerank  # noqa: E402
from retrieval import HybridRetriever  # noqa: E402

GENERATOR_MODEL = "qwen3:14b"
JUDGE_MODEL = "gpt-oss:20b"
K_VALUES = [3, 5, 10]
RERANK_CANDIDATE_POOL = 20

_faithfulness_metric = None


def _get_faithfulness_metric():
    global _faithfulness_metric
    if _faithfulness_metric is None:
        from ragas.llms import llm_factory
        from ragas.metrics.collections import Faithfulness
        client = openai.AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        judge_llm = llm_factory(JUDGE_MODEL, provider="openai", client=client, max_tokens=4096)
        _faithfulness_metric = Faithfulness(llm=judge_llm)
    return _faithfulness_metric


def load_testset(path: str) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


# ---------------------------------------------------------------------------
# Recall@k
# ---------------------------------------------------------------------------

def compute_recall_at_k(retriever: HybridRetriever, testset: list[dict], mode: str) -> dict:
    """Pour chaque k, proportion de questions où le PMID source attendu apparaît
    dans le top-k des chunks retrieved (avant reranking, sauf si mode='hybrid_rerank')."""
    max_k = max(K_VALUES)
    hits_per_k = {k: 0 for k in K_VALUES}
    n = 0

    for item in tqdm(testset, desc=f"recall@k (mode={mode})"):
        query = item["question"]
        expected_pmid = item["source_pmid"]

        if mode == "hybrid_rerank":
            candidates = retriever.search(query, k=RERANK_CANDIDATE_POOL, mode="hybrid")
            ranked = rerank(query, candidates, k=max_k)
        else:
            ranked = retriever.search(query, k=max_k, mode=mode)

        retrieved_pmids = [c["pmid"] for c in ranked]
        n += 1
        for k in K_VALUES:
            if expected_pmid in retrieved_pmids[:k]:
                hits_per_k[k] += 1

    return {f"recall@{k}": round(hits_per_k[k] / n, 4) for k in K_VALUES} | {"n_questions": n}


# ---------------------------------------------------------------------------
# Faithfulness (RAGAS, generation via generate.py -> mode hybrid_rerank de facto)
# ---------------------------------------------------------------------------

async def compute_faithfulness(retriever: HybridRetriever, testset: list[dict], strategy: str) -> dict:
    metric = _get_faithfulness_metric()
    scores = []
    per_category = {}
    per_edge_case = {"edge_case": [], "normal": []}

    for item in tqdm(testset, desc="faithfulness (RAGAS, generation qwen3:14b)"):
        result = answer_question(item["question"], strategy=strategy, retriever=retriever, model=GENERATOR_MODEL)
        if result["refused"] or not result["sources_retrieved"]:
            continue  # pas de contexte -> pas de faithfulness calculable

        context = "\n\n".join(
            f"[{s['pmid']}] {s['title']}\n{s['text']}" for s in result["sources_retrieved"]
        )
        try:
            score = await metric.ascore(
                user_input=item["question"],
                response=result["answer"],
                retrieved_contexts=[context],
            )
            value = float(score.value)
        except Exception as e:
            print(f"[warn] faithfulness échouée pour '{item['question'][:60]}...': {e}")
            continue

        scores.append(value)
        cat = item["category"]
        per_category.setdefault(cat, []).append(value)
        bucket = "edge_case" if item.get("edge_case") else "normal"
        per_edge_case[bucket].append(value)

    def _avg(values):
        return round(sum(values) / len(values), 4) if values else None

    return {
        "faithfulness_mean": _avg(scores),
        "faithfulness_n": len(scores),
        "faithfulness_by_category": {cat: _avg(v) for cat, v in per_category.items()},
        "faithfulness_edge_case_vs_normal": {b: _avg(v) for b, v in per_edge_case.items()},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Évaluation quantitative du RAG")
    parser.add_argument("--strategy", choices=["fixed", "semantic"], default="fixed")
    parser.add_argument("--testset", default="eval/testset.jsonl")
    parser.add_argument("--modes", default="dense,sparse,hybrid,hybrid_rerank",
                         help="Modes de retrieval à évaluer pour recall@k, séparés par des virgules")
    parser.add_argument("--skip-faithfulness", action="store_true",
                         help="Ne calcule que recall@k (rapide, pas besoin du LLM générateur+juge)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    testset = load_testset(args.testset)
    print(f"[eval] {len(testset)} questions, stratégie={args.strategy}")

    retriever = HybridRetriever(strategy=args.strategy)

    results = {"strategy": args.strategy, "n_questions": len(testset), "recall_at_k": {}}

    for mode in args.modes.split(","):
        results["recall_at_k"][mode] = compute_recall_at_k(retriever, testset, mode)

    if not args.skip_faithfulness:
        results["faithfulness"] = asyncio.run(compute_faithfulness(retriever, testset, args.strategy))
        results["faithfulness"]["judge_model"] = JUDGE_MODEL
        results["faithfulness"]["generator_model"] = GENERATOR_MODEL

    out_path = Path(args.out or f"eval/results/eval_{args.strategy}_{int(time.time())}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n-> Résultats écrits dans {out_path}\n")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
