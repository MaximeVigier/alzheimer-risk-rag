"""
Génère un jeu de test (questions + réponse de référence + PMID source attendu) à partir
du corpus, pour servir de gold standard à l'évaluation (recall@k, faithfulness, etc.).

Stratégie : pour chaque question, on part d'UN abstract précis (échantillonné pour couvrir
les 6 catégories de facteurs de risque), on demande au LLM de générer une question dont la
réponse se trouve dans cet abstract, avec la réponse attendue. Le PMID source est donc connu
par construction (utile pour calculer recall@k : le chunk de ce PMID doit apparaître dans le
top-k retrieved).

⚠️ Ce jeu de test est un point de départ semi-automatique — à relire et corriger à la main
avant de l'utiliser comme gold standard définitif (voir eval/testset.jsonl après relecture).

Usage:
    python eval/generate_testset.py --n-per-category 6 --out eval/testset_draft.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import ollama

OLLAMA_MODEL = "qwen3:14b"

RISK_CATEGORIES = [
    "genetic", "diet_nutrition", "sleep", "physical_activity",
    "cardiovascular_metabolic", "lifestyle_environment",
]

GEN_PROMPT_TEMPLATE = """Based ONLY on the following scientific abstract, generate ONE clear, \
specific factual question about Alzheimer's disease risk factors that this abstract directly \
answers, along with a concise reference answer (2-3 sentences) based strictly on the abstract's content.

Abstract (PMID {pmid}):
{text}

Respond in this exact format, nothing else:
QUESTION: <the question>
ANSWER: <the reference answer>
"""


def parse_generated(raw: str) -> tuple[str, str] | None:
    q_match = re.search(r"QUESTION:\s*(.+?)(?=\nANSWER:|\Z)", raw, re.DOTALL)
    a_match = re.search(r"ANSWER:\s*(.+)", raw, re.DOTALL)
    if not q_match or not a_match:
        return None
    return q_match.group(1).strip(), a_match.group(1).strip()


def generate_testset(records: list[dict], n_per_category: int, model: str, seed: int = 42) -> list[dict]:
    random.seed(seed)
    testset = []

    for category in RISK_CATEGORIES:
        candidates = [r for r in records if category in r["risk_categories"]
                      and 150 <= len(r["abstract"].split()) <= 350]  # abstracts ni trop courts ni trop longs
        if not candidates:
            print(f"[warn] aucun candidat exploitable pour la catégorie '{category}'")
            continue
        sample = random.sample(candidates, min(n_per_category, len(candidates)))

        for record in sample:
            prompt = GEN_PROMPT_TEMPLATE.format(pmid=record["pmid"], text=record["abstract"])
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3},
            )
            parsed = parse_generated(response["message"]["content"])
            if parsed is None:
                print(f"[warn] parsing échoué pour PMID {record['pmid']}, skip")
                continue
            question, reference_answer = parsed
            testset.append({
                "question": question,
                "reference_answer": reference_answer,
                "source_pmid": record["pmid"],
                "category": category,
                "reviewed": False,  # à passer à true après relecture manuelle
            })
            print(f"  [{category}] {question[:90]}")

    return testset


def main():
    parser = argparse.ArgumentParser(description="Génère un jeu de test semi-automatique pour l'éval")
    parser.add_argument("--records", default="data/raw/pubmed_alzheimer.json")
    parser.add_argument("--n-per-category", type=int, default=6, help="~6 x 6 catégories = 36 questions")
    parser.add_argument("--out", default="eval/testset_draft.jsonl")
    parser.add_argument("--model", default=OLLAMA_MODEL)
    args = parser.parse_args()

    with open(args.records, encoding="utf-8") as f:
        records = json.load(f)

    print(f"[testset] génération à partir de {len(records)} documents, "
          f"~{args.n_per_category} questions/catégorie...")
    testset = generate_testset(records, args.n_per_category, args.model)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for item in testset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n-> {len(testset)} questions générées dans {out_path}")
    print("⚠️  À RELIRE ET CORRIGER avant utilisation comme gold standard (voir champ 'reviewed').")


if __name__ == "__main__":
    main()
