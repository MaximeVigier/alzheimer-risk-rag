"""
Compare le même LLM générateur (qwen3:14b) AVEC le pipeline RAG (retrieval + contexte
sourcé) vs EN ZERO-SHOT (connaissances internes seules, pas de contexte), sur le même
jeu de test (eval/testset.jsonl). Montre le gain apporté par le RAG lui-même.

Métrique : AnswerCorrectness (RAGAS) comparée à la réponse de référence (reference_answer
du testset), poids [1.0, 0.0] = facticité pure (pas de similarité embeddings nécessaire).
Juge : gpt-oss:20b, différent du modèle générateur (qwen3:14b) — même logique que run_eval.py.

Usage:
    python eval/compare_rag_vs_zeroshot.py --strategy fixed
    python eval/compare_rag_vs_zeroshot.py --strategy fixed --limit 5  # test rapide
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import ollama
import openai
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from generate import answer_question  # noqa: E402
from retrieval import HybridRetriever  # noqa: E402

GENERATOR_MODEL = "qwen3:14b"
JUDGE_MODEL = "gpt-oss:20b"

ZERO_SHOT_SYSTEM_PROMPT = """You are a scientific assistant answering questions about Alzheimer's disease \
risk factors, using your own general knowledge (no external documents are provided). \
Be concise and factual."""

_correctness_metric = None


def _get_correctness_metric():
    global _correctness_metric
    if _correctness_metric is None:
        from ragas.llms import llm_factory
        from ragas.metrics.collections import AnswerCorrectness
        client = openai.AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        judge_llm = llm_factory(JUDGE_MODEL, provider="openai", client=client, max_tokens=16384)
        # weights=[1.0, 0.0] : facticité pure jugée par LLM, pas de composante similarité
        # sémantique (qui nécessiterait un modèle d'embeddings ragas séparé).
        _correctness_metric = AnswerCorrectness(llm=judge_llm, weights=[1.0, 0.0])
    return _correctness_metric


def load_testset(path: str) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def zero_shot_answer(query: str, model: str = GENERATOR_MODEL) -> str:
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": ZERO_SHOT_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        options={"temperature": 0.1},
    )
    return response["message"]["content"]


async def score_correctness(question: str, response: str, reference: str) -> float | None:
    metric = _get_correctness_metric()
    try:
        result = await metric.ascore(user_input=question, response=response, reference=reference)
        return float(result.value)
    except Exception as e:
        print(f"[warn] scoring échoué pour '{question[:60]}...': {e}")
        return None


async def run(testset: list[dict], strategy: str) -> dict:
    retriever = HybridRetriever(strategy=strategy)

    rag_scores, zs_scores = [], []
    per_question = []

    for item in tqdm(testset, desc="RAG vs zero-shot"):
        question = item["question"]
        reference = item["reference_answer"]

        rag_result = answer_question(question, strategy=strategy, retriever=retriever, model=GENERATOR_MODEL)
        rag_answer = rag_result["answer"]
        zs_answer = zero_shot_answer(question, model=GENERATOR_MODEL)

        rag_score = await score_correctness(question, rag_answer, reference)
        zs_score = await score_correctness(question, zs_answer, reference)

        if rag_score is not None:
            rag_scores.append(rag_score)
        if zs_score is not None:
            zs_scores.append(zs_score)

        per_question.append({
            "question": question,
            "category": item["category"],
            "rag_correctness": rag_score,
            "zero_shot_correctness": zs_score,
            "rag_refused": rag_result["refused"],
        })

    def _avg(values):
        return round(sum(values) / len(values), 4) if values else None

    return {
        "strategy": strategy,
        "generator_model": GENERATOR_MODEL,
        "judge_model": JUDGE_MODEL,
        "n_questions": len(testset),
        "rag_correctness_mean": _avg(rag_scores),
        "zero_shot_correctness_mean": _avg(zs_scores),
        "rag_correctness_n": len(rag_scores),
        "zero_shot_correctness_n": len(zs_scores),
        "per_question": per_question,
    }


def main():
    parser = argparse.ArgumentParser(description="RAG vs zero-shot — comparaison AnswerCorrectness")
    parser.add_argument("--strategy", choices=["fixed", "semantic"], default="fixed")
    parser.add_argument("--testset", default="eval/testset.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="Limiter à N questions (test rapide)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    testset = load_testset(args.testset)
    if args.limit:
        testset = testset[: args.limit]
    print(f"[compare] {len(testset)} questions, stratégie={args.strategy}")

    results = asyncio.run(run(testset, args.strategy))

    out_path = Path(args.out or f"eval/results/rag_vs_zeroshot_{args.strategy}_{int(time.time())}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n-> Résultats écrits dans {out_path}")
    print(f"RAG correctness moyenne     : {results['rag_correctness_mean']}")
    print(f"Zero-shot correctness moyenne: {results['zero_shot_correctness_mean']}")


if __name__ == "__main__":
    main()
