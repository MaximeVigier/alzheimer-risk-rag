# Alzheimer Risk Factors RAG

🚧 **Projet en cours de construction — commits progressifs, voir l'historique git pour la démarche.**

Assistant de question-réponse sourcé sur la littérature scientifique des facteurs de risque
de la maladie d'Alzheimer (génétique, mode de vie, métabolique, cardiovasculaire, sommeil,
activité physique), construit à partir d'un corpus PubMed réel — avec évaluation quantitative
du retrieval et de la fidélité des réponses, pas juste une démo qui a l'air de marcher.

## Pourquoi ce projet
Voir [`PLAN.md`](PLAN.md) pour la démarche complète (architecture, choix méthodologiques, étapes).

## État d'avancement
- [x] Ingestion PubMed (1198 abstracts, API E-utilities)
- [x] Chunking — 2 stratégies comparées (taille fixe vs découpage sémantique)
- [x] Embeddings + indexation ChromaDB (local, gratuit)
- [x] Retrieval hybride (dense + BM25, fusion RRF)
- [x] Reranking (cross-encoder)
- [x] Génération sourcée avec garde-fous anti-hallucination (Ollama, 100% local)
- [x] Jeu de test d'évaluation (36 questions, généré semi-automatiquement puis relu)
- [ ] Évaluation quantitative (recall@k, faithfulness via RAGAS, LLM-as-judge)
- [ ] Figures scientifiques (comparaison chunking/retrieval, RAG vs zero-shot, UMAP embeddings)
- [ ] API FastAPI
- [ ] Tests + CI GitHub Actions
- [ ] Dockerfile + docker-compose
- [ ] README final avec résultats chiffrés

## Stack
Python · Ollama (LLM + embeddings, local) · ChromaDB · rank_bm25 · sentence-transformers
(reranking) · RAGAS (évaluation) · FastAPI · Streamlit (à venir)

## Lancer en local

```bash
python -m venv .venv
./.venv/Scripts/activate  # ou source .venv/bin/activate sur Linux/Mac
pip install -r requirements.txt

# 1. Ingestion du corpus PubMed
python src/ingest.py --max-records 1200

# 2. Chunking (2 stratégies)
python src/chunking.py --strategy fixed
python src/chunking.py --strategy semantic

# 3. Indexation vectorielle
python src/embed.py --strategy both

# 4. Chat interactif
python src/chat.py --strategy fixed
```

Nécessite [Ollama](https://ollama.com) avec un modèle de génération (`qwen3:14b` par défaut,
configurable dans `src/generate.py`) et le modèle juge pour l'évaluation (`gpt-oss:20b`).

## Structure

```
├── src/
│   ├── ingest.py       # récupération du corpus PubMed
│   ├── chunking.py     # 2 stratégies de découpage en chunks
│   ├── embed.py        # embeddings + indexation ChromaDB
│   ├── retrieval.py    # retrieval hybride dense + BM25
│   ├── rerank.py       # reranking cross-encoder
│   ├── generate.py     # génération sourcée + garde-fous
│   └── chat.py         # mode interactif de test
├── eval/
│   ├── generate_testset.py  # génération semi-auto du jeu de test
│   ├── testset.jsonl         # jeu de test relu et validé (36 questions)
│   └── run_eval.py           # (à venir) métriques recall@k, faithfulness
└── PLAN.md              # plan détaillé et journal de décisions
```

## Limites (mises à jour au fil du projet)
- Le tagging automatique des catégories de facteurs de risque est fait par mots-clés, non
  mutuellement exclusif — la catégorie "genetic" (APOE) domine le corpus.
- Corpus limité aux abstracts (pas de full-text), donc contexte parfois incomplet.
