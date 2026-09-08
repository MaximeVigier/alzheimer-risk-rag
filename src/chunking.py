"""
Chunking — découpe les abstracts PubMed en chunks pour l'indexation vectorielle.

Deux stratégies comparées (voir eval/ pour les résultats chiffrés) :
  - "fixed"    : taille fixe en mots, avec overlap. Simple, rapide, référence standard.
  - "semantic" : regroupement de phrases par similarité sémantique (embeddings),
                 coupe là où le sujet change plutôt qu'à une position arbitraire.

Usage:
    python src/chunking.py --strategy fixed --in data/raw/pubmed_alzheimer.json --out data/processed/chunks_fixed.json
    python src/chunking.py --strategy semantic --in data/raw/pubmed_alzheimer.json --out data/processed/chunks_semantic.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Modèle d'embedding léger, tourne en local CPU — cohérent avec le choix fait pour embed.py
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# Approximation tokens ~ mots * 1.3 (évite une dépendance tiktoken juste pour un ordre de grandeur).
WORDS_PER_TOKEN = 1 / 1.3


def word_count(text: str) -> int:
    return len(text.split())


def approx_tokens(text: str) -> int:
    return round(word_count(text) / WORDS_PER_TOKEN)


def split_sentences(text: str) -> list[str]:
    """Découpage en phrases par regex — évite une dépendance nltk (et son download de data)
    pour un cas d'usage simple sur de l'anglais scientifique standard."""
    # Protège les abréviations courantes en biomédical pour ne pas couper à tort (e.g., et al., Fig., vs.)
    protected = re.sub(r"\b(e\.g|i\.e|et al|Fig|vs|approx)\.", r"\1<DOT>", text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", protected)
    return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Stratégie A : taille fixe avec overlap
# ---------------------------------------------------------------------------

def chunk_fixed(text: str, chunk_size_tokens: int = 500, overlap_tokens: int = 50) -> list[str]:
    words = text.split()
    chunk_size_words = round(chunk_size_tokens * WORDS_PER_TOKEN)
    overlap_words = round(overlap_tokens * WORDS_PER_TOKEN)
    if len(words) <= chunk_size_words:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap_words
    return chunks


# ---------------------------------------------------------------------------
# Stratégie B : découpage sémantique par phrases
# ---------------------------------------------------------------------------

_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def chunk_semantic(text: str, max_tokens: int = 500, similarity_threshold: float = 0.55) -> list[str]:
    """Regroupe les phrases consécutives tant qu'elles restent sémantiquement proches
    (similarité cosine avec la phrase précédente >= seuil) et que la taille max n'est pas dépassée.
    Coupe un nouveau chunk quand le sujet change (chute de similarité) ou que la limite de taille est atteinte."""
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return [text]

    model = _get_embed_model()
    embeddings = model.encode(sentences, normalize_embeddings=True, show_progress_bar=False)

    chunks = []
    current_sentences = [sentences[0]]
    current_tokens = approx_tokens(sentences[0])

    for i in range(1, len(sentences)):
        sim = float(embeddings[i] @ embeddings[i - 1])  # cosine sim (vecteurs normalisés)
        sent_tokens = approx_tokens(sentences[i])

        if sim >= similarity_threshold and current_tokens + sent_tokens <= max_tokens:
            current_sentences.append(sentences[i])
            current_tokens += sent_tokens
        else:
            chunks.append(" ".join(current_sentences))
            current_sentences = [sentences[i]]
            current_tokens = sent_tokens

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


# ---------------------------------------------------------------------------
# Pipeline commun
# ---------------------------------------------------------------------------

def build_chunks(records: list[dict], strategy: str) -> list[dict]:
    chunk_fn = chunk_fixed if strategy == "fixed" else chunk_semantic
    all_chunks = []
    for record in records:
        text = f"{record['title']}. {record['abstract']}"
        pieces = chunk_fn(text)
        for idx, piece in enumerate(pieces):
            all_chunks.append({
                "chunk_id": f"{record['pmid']}_{idx}",
                "pmid": record["pmid"],
                "chunk_index": idx,
                "n_chunks_in_doc": len(pieces),
                "text": piece,
                "title": record["title"],
                "year": record["year"],
                "journal": record["journal"],
                "risk_categories": record["risk_categories"],
                "url": record["url"],
            })
    return all_chunks


def main():
    parser = argparse.ArgumentParser(description="Chunking des abstracts PubMed")
    parser.add_argument("--strategy", choices=["fixed", "semantic"], required=True)
    parser.add_argument("--in", dest="input_path", default="data/raw/pubmed_alzheimer.json")
    parser.add_argument("--out", dest="output_path", default=None)
    parser.add_argument("--chunk-size-tokens", type=int, default=500)
    args = parser.parse_args()

    output_path = Path(args.output_path or f"data/processed/chunks_{args.strategy}.json")

    with open(args.input_path, encoding="utf-8") as f:
        records = json.load(f)

    print(f"[chunking] stratégie={args.strategy}, {len(records)} documents source")
    chunks = build_chunks(records, args.strategy)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    n_tokens = [approx_tokens(c["text"]) for c in chunks]
    avg_chunks_per_doc = len(chunks) / len(records)
    print(f"  -> {len(chunks)} chunks écrits dans {output_path}")
    print(f"  -> moyenne {avg_chunks_per_doc:.2f} chunks/document")
    print(f"  -> taille chunk (tokens approx) : min={min(n_tokens)} "
          f"moy={sum(n_tokens)/len(n_tokens):.0f} max={max(n_tokens)}")


if __name__ == "__main__":
    main()
