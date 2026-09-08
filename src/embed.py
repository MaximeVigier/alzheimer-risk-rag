"""
Embeddings + indexation vectorielle dans ChromaDB.

Utilise le même modèle d'embedding local que chunking.py (all-MiniLM-L6-v2,
sentence-transformers, CPU) — pas d'appel API, gratuit et reproductible.

Usage:
    python src/embed.py --strategy fixed
    python src/embed.py --strategy semantic
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import chromadb
from tqdm import tqdm

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_DIR = "data/chroma"
BATCH_SIZE = 128

_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def collection_name(strategy: str) -> str:
    return f"alzheimer_risk_{strategy}"


def build_index(chunks: list[dict], strategy: str, chroma_dir: str = CHROMA_DIR) -> None:
    client = chromadb.PersistentClient(path=chroma_dir)
    name = collection_name(strategy)

    # Repart de zéro à chaque run pour rester reproductible
    try:
        client.delete_collection(name)
    except Exception:
        pass
    collection = client.create_collection(name=name, metadata={"strategy": strategy})

    model = _get_embed_model()

    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc=f"indexation ({strategy})"):
        batch = chunks[i:i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()

        collection.add(
            ids=[c["chunk_id"] for c in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{
                "pmid": c["pmid"],
                "title": c["title"],
                "year": c["year"] or "",
                "journal": c["journal"] or "",
                "url": c["url"],
                # Chroma n'accepte pas les listes en metadata -> on joint en string
                "risk_categories": ",".join(c["risk_categories"]),
            } for c in batch],
        )

    print(f"  -> collection '{name}' : {collection.count()} vecteurs indexés dans {chroma_dir}")


def main():
    parser = argparse.ArgumentParser(description="Embedding + indexation ChromaDB des chunks")
    parser.add_argument("--strategy", choices=["fixed", "semantic", "both"], default="both")
    parser.add_argument("--chunks-dir", default="data/processed")
    parser.add_argument("--chroma-dir", default=CHROMA_DIR)
    args = parser.parse_args()

    strategies = ["fixed", "semantic"] if args.strategy == "both" else [args.strategy]

    for strategy in strategies:
        chunks_path = Path(args.chunks_dir) / f"chunks_{strategy}.json"
        with open(chunks_path, encoding="utf-8") as f:
            chunks = json.load(f)
        print(f"[embed] stratégie={strategy}, {len(chunks)} chunks à indexer")
        build_index(chunks, strategy, args.chroma_dir)


if __name__ == "__main__":
    main()
