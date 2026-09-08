# Projet 2 du portfolio — RAG "Facteurs de risque d'Alzheimer"

> Statut : plan validé le 08/09/2026, avant écriture de code.
> Objectif recruteur : prouver NLP/LLM/RAG (noté 5/5 dans PROFIL.md mais sans preuve publique)
> ET l'avantage domaine santé/neurodégénérescence (thèse Alzheimer/lipides).

## Pitch en une phrase (pour le README et l'entretien)
« Un assistant qui répond à des questions sur les facteurs de risque de la maladie d'Alzheimer
en citant ses sources dans la littérature scientifique (PubMed), avec une évaluation quantitative
du retrieval et de la fidélité des réponses — pas juste une démo qui a l'air de marcher. »

## Pourquoi ce sujet (et pas un RAG générique sur PDF quelconques)
- Recoupe directement la thèse (Alzheimer, neurodégénérescence) → légitimité domaine, discutable en entretien
- Scope volontairement élargi vs la thèse (pas seulement lipides) : génétique (APOE), mode de vie
  (sommeil, exercice, diète), métabolique (diabète, cholestérol), cardiovasculaire, environnemental
  → montre une capacité de synthèse au-delà de la niche exacte de la thèse
- Corpus PubMed = gratuit, structuré, API stable (E-utilities), pas de scraping fragile

## Corpus
- Source : PubMed via API E-utilities (esearch + efetch), champ abstract (+ titre, auteurs, année, MeSH terms)
- Requête de recherche multi-catégories, ex. :
  `("Alzheimer disease"[MeSH]) AND (risk factor[tiab] OR APOE[tiab] OR diet[tiab] OR sleep[tiab]
  OR cardiovascular[tiab] OR diabetes[tiab] OR lipid[tiab] OR lifestyle[tiab])`
- Cible : 800–1500 abstracts, filtrés sur review + études (2015–2026) pour avoir du signal récent
- Métadonnées conservées : PMID (lien direct), année, catégorie de facteur de risque (taggée par mots-clés MeSH)
  → permet un filtre par catégorie dans la démo, feature différenciante facile

## Architecture pipeline

```
1. Ingestion     : script Python, API PubMed → JSON brut (data/raw/)
2. Nettoyage     : dédup, filtrage langue EN, retrait abstracts vides
3. Chunking      : comparaison de 2 stratégies (à documenter avec impact mesuré)
                   a) taille fixe (500 tokens, overlap 50)
                   b) découpage sémantique (par phrase, regroupement par similarité)
4. Embeddings    : modèle local via Ollama (nomic-embed-text ou bge-small) — gratuit, pas d'API
5. Vector store  : ChromaDB (local, Docker) — simple, bien documenté
6. Retrieval     : hybride — dense (cosine) + BM25 (rank_bm25), fusion RRF
7. Reranking     : cross-encoder léger (ex. bge-reranker-base) sur le top-20 → top-5
8. Génération    : LLM local (Ollama, ex. llama3.1:8b ou qwen2.5) avec prompt strict :
                   - répondre UNIQUEMENT à partir du contexte fourni
                   - citer les PMID des sources utilisées
                   - dire explicitement "je ne sais pas" si le contexte est insuffisant
9. Garde-fous    : détection de question hors-sujet, refus si retrieval score < seuil
```

## Évaluation (le vrai différenciant)
- Constituer un jeu de test de ~30-50 questions/réponses de référence :
  - manuel (toi, en t'appuyant sur ton expertise du domaine) pour 15-20 questions "dures"
  - généré semi-automatiquement (LLM à partir d'abstracts) pour le reste, relu et corrigé
- Métriques à calculer et publier dans le README :
  - **Recall@k** (le bon abstract est-il dans le top-k retrieved ?) pour k=3,5,10
  - **Faithfulness / groundedness** : la réponse est-elle supportée par le contexte cité (RAGAS ou implémentation maison à base de NLI)
  - **Answer relevancy** : la réponse répond-elle à la question posée
  - **Comparaison chunking a) vs b)** sur ces métriques → tableau chiffré comme celui du projet COVID
  - **Latence et coût par requête** (même si local = coût CPU/temps, à mesurer et présenter)
- Ablation attendue à montrer : retrieval dense seul vs hybride vs hybride+reranking → gain chiffré

## Stack technique
| Composant | Choix | Pourquoi |
|---|---|---|
| Langage | Python 3.11+ | standard |
| LLM + embeddings (dev) | Ollama (local) | gratuit, illimité en dev, cohérent avec low-overhead |
| Vector store | ChromaDB | léger, Docker-friendly, pas de service cloud à payer |
| Retrieval sparse | rank_bm25 | simple, pas de dépendance lourde |
| Reranking | bge-reranker-base (via sentence-transformers) | tourne en local CPU |
| Orchestration | code Python direct (pas de LangChain) | plus impressionnant en entretien : montre que tu comprends le pipeline, pas juste l'assemblage d'une boîte noire. LangChain seulement si le temps presse. |
| Éval | RAGAS ou métriques maison si RAGAS trop lourd à faire tourner en local | |
| API | FastAPI | réutilisable pour le projet 3 MLOps (même pattern) |
| Front démo | Streamlit | cohérent avec le projet COVID, tu connais déjà |
| Conteneurisation | Docker + docker-compose (app + ChromaDB) | comble le gap Docker identifié dans PROFIL.md |
| CI | GitHub Actions : lint + tests + build Docker | |
| Déploiement démo publique | HF Spaces (Streamlit), LLM basculé sur une API légère à coût plafonné (ex. petit modèle Groq/Mistral gratuit) car Ollama local n'est pas déployable gratuitement en ligne | |

## Structure du repo
```
alzheimer-risk-rag/
├── README.md                  # problème → données → approche → résultats chiffrés → limites
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
├── .github/workflows/ci.yml
├── data/
│   ├── raw/                   # abstracts bruts (ou script pour les régénérer, pas commité si volumineux)
│   └── processed/
├── src/
│   ├── ingest.py              # appel API PubMed
│   ├── chunking.py            # les 2 stratégies
│   ├── embed.py
│   ├── retrieval.py           # dense + BM25 + fusion
│   ├── rerank.py
│   ├── generate.py            # prompt + garde-fous
│   └── api.py                 # FastAPI
├── eval/
│   ├── testset.jsonl          # questions/réponses de référence
│   ├── run_eval.py
│   └── results/                # tableaux de métriques versionnés
├── app/
│   └── streamlit_app.py
└── tests/
    ├── test_chunking.py
    ├── test_retrieval.py
    └── test_api.py
```

## Étapes (2 week-ends, comme les autres projets du portfolio)

**Week-end 1 — pipeline qui tourne**
1. ✅ Script ingestion PubMed → JSON (data/raw) — 1198 abstracts
2. ✅ Chunking (les 2 stratégies) — fixed: 1212 chunks, semantic: 7301 chunks (sur-fragmentation à documenter)
3. ✅ Embeddings + ChromaDB, indexation — 2 collections indexées
4. ✅ Retrieval hybride (dense + BM25 + RRF) + reranking (cross-encoder) — testé et validé
5. ✅ Génération avec citations + garde-fous (Ollama qwen3:14b local) — testé, citations correctes,
   refus correct sur question hors-sujet
6. Notebook/CLI de test manuel élargi (5-10 questions) — partiellement fait via CLI, à formaliser


**Week-end 2 — ce qui fait la différence**
7. Jeu de test éval (30-50 Q/R) + script d'éval automatisé
8. Tableau comparatif chunking a) vs b), dense vs hybride vs hybride+rerank
9. FastAPI + Streamlit (réutilise le pattern du projet COVID)
10. Dockerfile + docker-compose, vérifier que ça tourne clean depuis zéro
11. Tests (pytest) + CI GitHub Actions
12. README complet avec les résultats chiffrés (calqué sur le style du README covid-radiography-benchmark)
13. Déploiement démo sur HF Spaces
14. Repo créé sous MaximeVigier, commits atomiques, pas de co-auteur IA visible

## Squelette de README (à remplir avec les vrais chiffres une fois l'éval faite)
```markdown
# Alzheimer Risk Factors RAG

Assistant de question-réponse sourcé sur la littérature scientifique des facteurs de risque
de la maladie d'Alzheimer (génétique, mode de vie, métabolique, cardiovasculaire).

## Problème
[X lignes — pourquoi ce sujet, pourquoi le RAG plutôt qu'un simple moteur de recherche]

## Données
[N abstracts PubMed, requête utilisée, période, catégories]

## Approche
[Schéma pipeline : ingestion → chunking → retrieval hybride → reranking → génération sourcée]

## Résultats
| Config | Recall@5 | Faithfulness | Answer relevancy |
|---|---|---|---|
| Dense seul | ... | ... | ... |
| Hybride | ... | ... | ... |
| Hybride + rerank | ... | ... | ... |

Chunking sémantique vs fixe : [tableau]

## Limites
[corpus abstracts uniquement pas full-text, LLM local peut halluciner malgré garde-fous, taille du jeu de test éval, etc.]

## Démo
[lien HF Spaces]

## Lancer en local
docker-compose up
```

## Documentation scientifique (figures)
Ne pas se limiter à un tableau de métriques dans le README — produire de vraies figures (matplotlib/seaborn),
comme dans un papier ou un rapport de bootcamp (cf. style du projet COVID) :
- Barplot recall@k (k=3,5,10) par configuration (dense / hybride / hybride+rerank)
- Comparaison chunking fixed vs semantic (distribution taille des chunks, métriques retrieval)
- Courbe faithfulness / answer relevancy par catégorie de facteur de risque (radar chart ou barplot groupé)
- **[RETENU] RAG vs zero-shot** : même LLM (qwen3:14b), avec RAG vs sans (connaissances internes seules),
  comparé sur faithfulness/exactitude — montre le gain apporté par le RAG lui-même, argument fort en entretien
- **[RETENU] UMAP des embeddings** coloré par catégorie de risque — vérifie visuellement que les clusters
  sémantiques correspondent aux catégories taguées automatiquement
- Toutes les figures dans `eval/results/figures/`, générées par un script versionné (pas de figures à la main)

## Décisions de peaufinage (08/09/2026)
- Jeu de test éval : généré semi-automatiquement par LLM à partir du corpus (questions + réponses de
  référence), puis relu et corrigé par Maxime avant utilisation comme gold standard.
- Démo finale : reste 100% locale (Ollama), pas de déploiement public HF Spaces avec bascule API.
  Le README montrera un GIF/vidéo de démo à la place d'un lien cliquable en ligne.
- Figures prioritaires ajoutées : comparaison RAG vs zero-shot, visualisation UMAP des embeddings.
- Faithfulness : calculée via **RAGAS** (librairie standard du marché, meilleur signal recruteur
  "connaît les outils") plutôt qu'une métrique maison.
- **Modèle évaluateur/juge différent du modèle générateur** : qwen3:14b génère les réponses,
  gpt-oss:20b (autre modèle local dispo) sert de juge pour faithfulness et comparaison RAG vs zero-shot
  → évite le biais d'auto-évaluation (un modèle qui juge ses propres réponses tend à se sur-noter).
  Point méthodologique à expliciter dans le README, c'est un vrai argument de rigueur en entretien.

## Ce que ce projet coche dans PROFIL.md
- NLP/LLM/RAG : preuve publique concrète (trou identifié en priorité)
- Docker : usage réel, pas un `docker run hello-world`
- Positionnement santé : cohérent avec cible n°1 (nutrition/pharma/biotech) et n°3 (imagerie médicale)
- Rigueur expérimentale : éval chiffrée, ablation chunking et retrieval — même logique que le tableau
  13 modèles du projet COVID, donc discours cohérent entre les deux projets en entretien
