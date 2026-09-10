"""
Tests unitaires — chunking pur (pas de dépendance modèle/réseau), rapide en CI.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chunking import approx_tokens, build_chunks, chunk_fixed, split_sentences, word_count


def test_word_count():
    assert word_count("one two three") == 3
    assert word_count("") == 0


def test_approx_tokens_roughly_proportional():
    short = approx_tokens("one two three four")
    long = approx_tokens(" ".join(["word"] * 40))
    assert 0 < short < long


def test_split_sentences_basic():
    text = "APOE increases risk. Sleep quality matters too."
    sentences = split_sentences(text)
    assert sentences == ["APOE increases risk.", "Sleep quality matters too."]


def test_split_sentences_protects_abbreviations():
    text = "Risk factors include diet, e.g. high sugar intake, and smoking (see Fig. 2)."
    sentences = split_sentences(text)
    # ne doit pas couper sur "e.g." ni "Fig." -> une seule phrase
    assert len(sentences) == 1
    assert "e.g." in sentences[0]
    assert "Fig." in sentences[0]


def test_chunk_fixed_short_text_returns_single_chunk():
    text = "short abstract about Alzheimer risk factors"
    chunks = chunk_fixed(text, chunk_size_tokens=500, overlap_tokens=50)
    assert chunks == [text]


def test_chunk_fixed_splits_long_text_with_overlap():
    words = [f"word{i}" for i in range(1200)]
    text = " ".join(words)
    chunks = chunk_fixed(text, chunk_size_tokens=100, overlap_tokens=10)
    assert len(chunks) > 1
    # chaque chunk (sauf le dernier) doit respecter la taille en mots (100 tokens ~ 130 mots)
    for c in chunks[:-1]:
        assert len(c.split()) <= 130 + 1
    # overlap : le dernier mot d'un chunk doit réapparaître en tête du suivant
    first_words = chunks[0].split()
    second_words = chunks[1].split()
    assert first_words[-1] in second_words[:20]


def test_build_chunks_preserves_metadata():
    records = [{
        "pmid": "12345",
        "title": "APOE and Alzheimer risk",
        "abstract": "Short abstract text about genetic risk factors.",
        "year": "2024",
        "journal": "Test Journal",
        "risk_categories": ["genetic"],
        "url": "https://pubmed.ncbi.nlm.nih.gov/12345",
    }]
    chunks = build_chunks(records, strategy="fixed")
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["chunk_id"] == "12345_0"
    assert chunk["pmid"] == "12345"
    assert chunk["risk_categories"] == ["genetic"]
    assert chunk["title"] == "APOE and Alzheimer risk"


def test_build_chunks_multiple_records_unique_ids():
    records = [
        {"pmid": "1", "title": "T1", "abstract": "abstract one", "year": "2020",
         "journal": "J", "risk_categories": [], "url": "u1"},
        {"pmid": "2", "title": "T2", "abstract": "abstract two", "year": "2021",
         "journal": "J", "risk_categories": [], "url": "u2"},
    ]
    chunks = build_chunks(records, strategy="fixed")
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))
