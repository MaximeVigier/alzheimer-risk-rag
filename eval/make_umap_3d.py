"""
UMAP 3D des embeddings du corpus + export JSON pour la visualisation Three.js du front.

Contrairement à make_umap.py (figure statique 2D pour le README), ce script réduit à 3
dimensions et exporte les positions + métadonnées de chaque chunk en JSON, consommé
directement par web/app.js pour afficher un nuage de points interactif navigable.

Usage:
    python eval/make_umap_3d.py --strategy fixed
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CHUNKS_DIR = Path("data/processed")
WEB_DATA_DIR = Path("web/data")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# Même logique de catégorie primaire que make_umap.py (une couleur par chunk).
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
    parser = argparse.ArgumentParser(description="UMAP 3D des embeddings, export JSON pour le front")
    parser.add_argument("--strategy", choices=["fixed", "semantic"], default="fixed")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    chunks_path = CHUNKS_DIR / f"chunks_{args.strategy}.json"
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"[umap3d] {len(chunks)} chunks (stratégie={args.strategy}), calcul des embeddings...")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL_NAME)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True, batch_size=128)

    print("[umap3d] réduction de dimension (384 -> 3)...")
    import umap
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=3, metric="cosine", random_state=42)
    coords = reducer.fit_transform(embeddings)

    points = []
    for chunk, xyz in zip(chunks, coords):
        points.append({
            "chunk_id": chunk["chunk_id"],
            "pmid": chunk["pmid"],
            "title": chunk["title"],
            "year": chunk["year"],
            "category": primary_category(chunk["risk_categories"]),
            "x": round(float(xyz[0]), 4),
            "y": round(float(xyz[1]), 4),
            "z": round(float(xyz[2]), 4),
        })

    out_path = Path(args.out) if args.out else WEB_DATA_DIR / f"umap3d_{args.strategy}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"strategy": args.strategy, "points": points}, f, ensure_ascii=False)

    print(f"-> {out_path} ({len(points)} points, {out_path.stat().st_size / 1024:.0f} Ko)")


if __name__ == "__main__":
    main()
