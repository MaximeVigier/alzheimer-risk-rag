"""
Génération — appelle un LLM local (Ollama) avec le contexte reranké pour produire
une réponse sourcée, avec garde-fous anti-hallucination.

Règles imposées au modèle par le prompt :
  1. Répondre UNIQUEMENT à partir du contexte fourni (pas de connaissances générales du modèle).
  2. Citer les PMID des sources effectivement utilisées.
  3. Dire explicitement qu'il ne sait pas si le contexte est insuffisant.

Garde-fou supplémentaire côté code (pas seulement prompt) :
  - si le meilleur score de retrieval est trop faible, on refuse de générer et on le dit.

Usage (import) :
    from generate import answer_question
    result = answer_question("does sleep quality affect Alzheimer's risk?", strategy="fixed")

Usage (CLI) :
    python src/generate.py --query "..." --strategy fixed
"""
from __future__ import annotations

import argparse
import re

import ollama

from rerank import rerank
from retrieval import HybridRetriever

OLLAMA_MODEL = "qwen3:14b"  # modèle local dispo (ollama list) ; changer ici si besoin
CANDIDATE_POOL = 20
TOP_K_CONTEXT = 5

# Seuil de score RRF en dessous duquel on considère le retrieval "faible" et on refuse de répondre
# franchement plutôt que de laisser le LLM halluciner sur un contexte non pertinent.
MIN_RRF_SCORE = 0.01

SYSTEM_PROMPT = """You are a scientific assistant answering questions about Alzheimer's disease risk factors, \
strictly based on the provided excerpts from PubMed abstracts.

Rules you MUST follow:
1. Answer ONLY using information contained in the provided context. Do not use prior knowledge.
2. Every claim in your answer must be traceable to a specific source. Cite sources inline using \
their PMID like this: [PMID: 12345678].
3. If the context does not contain enough information to answer the question, say so explicitly \
("The provided sources do not contain enough information to answer this question.") — do NOT guess \
or fill gaps with general knowledge.
4. Be concise and factual. Avoid hedging language beyond what the evidence supports.
"""


def _build_context_block(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        parts.append(f"[PMID: {c['pmid']}] ({c['year']}) {c['title']}\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def _extract_cited_pmids(answer: str) -> list[str]:
    return sorted(set(re.findall(r"PMID:\s*(\d+)", answer)))


def answer_question(query: str, strategy: str = "fixed", retriever: HybridRetriever | None = None,
                     model: str = OLLAMA_MODEL) -> dict:
    """Retourne un dict : answer, sources_cited (PMIDs extraits de la réponse),
    sources_retrieved (tous les chunks utilisés comme contexte), refused (bool)."""
    if retriever is None:
        retriever = HybridRetriever(strategy=strategy)

    candidates = retriever.search(query, k=CANDIDATE_POOL, mode="hybrid")

    if not candidates or candidates[0]["score"] is None or candidates[0]["score"] < MIN_RRF_SCORE:
        return {
            "answer": "The retrieval system did not find sufficiently relevant sources in the corpus "
                      "to answer this question reliably. This question may be outside the scope of the "
                      "indexed literature (Alzheimer's disease risk factors).",
            "sources_cited": [],
            "sources_retrieved": [],
            "refused": True,
        }

    top_chunks = rerank(query, candidates, k=TOP_K_CONTEXT)
    context = _build_context_block(top_chunks)

    user_prompt = f"Context:\n\n{context}\n\n---\n\nQuestion: {query}\n\nAnswer (with PMID citations):"

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0.1},
    )
    answer = response["message"]["content"]

    return {
        "answer": answer,
        "sources_cited": _extract_cited_pmids(answer),
        "sources_retrieved": [
            {"chunk_id": c["chunk_id"], "pmid": c["pmid"], "title": c["title"], "year": c["year"],
             "url": c["url"], "text": c["text"]}
            for c in top_chunks
        ],
        "refused": False,
    }


def main():
    parser = argparse.ArgumentParser(description="Test manuel de la génération RAG")
    parser.add_argument("--query", required=True)
    parser.add_argument("--strategy", choices=["fixed", "semantic"], default="fixed")
    parser.add_argument("--model", default=OLLAMA_MODEL)
    args = parser.parse_args()

    result = answer_question(args.query, strategy=args.strategy, model=args.model)

    print(f"\nQuestion: {args.query}\n")
    print(f"Réponse:\n{result['answer']}\n")
    print(f"Sources citées (PMID extraits de la réponse) : {result['sources_cited']}")
    print(f"Refusé (retrieval trop faible) : {result['refused']}")


if __name__ == "__main__":
    main()
