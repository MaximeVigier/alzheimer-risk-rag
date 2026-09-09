"""
Visualisation UMAP des embeddings du corpus, colorée par catégorie de facteur de risque.
Vérifie visuellement que les clusters sémantiques (calculés uniquement à partir du texte)
correspondent aux catégories taguées automatiquement par mots-clés — un signal de cohérence
entre l'espace d'embedding et le tagging, sans supervision.

Usage:
    python eval/make_umap.py --strategy fixed
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")

RESULTS_DIR = Path("eval/results")
FIG_DIR = RESULTS_DIR / "figures"
CHUNKS_DIR = Path("data/processed")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# Une catégorie primaire par chunk pour la couleur (un chunk peut être multi-tag ;
# on garde la première catégorie détectée dans cet ordre de priorité — du plus spécifique
# au plus générique — pour éviter que "genetic" domine tout par sur-représentation).
CATEGORY_PRIORITY = [
    "sleep", "physical_activity", "diet_nutrition",
    "cardiovascular_metabolic", "lifestyle_environment", "genetic",
]


def primary_category(risk_categories: list[str]) -> str:
    for cat in CATEGORY_PRIORITY:
        if cat in risk_categories:
            return cat
    return "other"


def main():
    parser = argparse.ArgumentParser(description="UMAP des embeddings du corpus, coloré par catégorie")
    parser.add_argument("--strategy", choices=["fixed", "semantic"], default="fixed")
    parser.add_argument("--sample", type=int, default=None, help="Sous-échantillonner N chunks (par défaut: tous)")
    args = parser.parse_args()

    chunks_path = CHUNKS_DIR / f"chunks_{args.strategy}.json"
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)

    if args.sample:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(chunks), size=min(args.sample, len(chunks)), replace=False)
        chunks = [chunks[i] for i in idx]

    print(f"[umap] {len(chunks)} chunks (stratégie={args.strategy}), calcul des embeddings...")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL_NAME)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True, batch_size=128)

    print("[umap] réduction de dimension (384 -> 2)...")
    import umap
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
    coords = reducer.fit_transform(embeddings)

    categories = [primary_category(c["risk_categories"]) for c in chunks]
    cat_order = [c for c in CATEGORY_PRIORITY if c in categories] + (["other"] if "other" in categories else [])
    palette = dict(zip(cat_order, sns.color_palette("tab10", len(cat_order))))

    fig, ax = plt.subplots(figsize=(11, 9))
    for cat in cat_order:
        mask = np.array([c == cat for c in categories])
        pts = coords[mask]
        ax.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.5, label=f"{cat} (n={int(mask.sum())})",
                   color=palette[cat])

    ax.set_title(f"UMAP des embeddings du corpus ({len(chunks)} chunks, stratégie {args.strategy})\n"
                 "Couleur = catégorie de facteur de risque taguée par mots-clés (non supervisée pour l'UMAP)",
                 fontsize=13)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(loc="best", fontsize=9, markerscale=2)
    fig.tight_layout()

    out_path = FIG_DIR / f"umap_embeddings_{args.strategy}.png"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
