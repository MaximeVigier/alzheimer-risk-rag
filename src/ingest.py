"""
Ingestion PubMed — récupère les abstracts liés aux facteurs de risque
de la maladie d'Alzheimer via l'API E-utilities (NCBI).

Usage:
    python src/ingest.py --max-records 1200 --out data/raw/pubmed_alzheimer.json

Sans clé API NCBI : limite à 3 req/s (le script respecte ce quota).
Avec une clé API (variable d'env NCBI_API_KEY, gratuite sur https://www.ncbi.nlm.nih.gov/account/settings/) :
limite à 10 req/s, ingestion ~3x plus rapide.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from tqdm import tqdm

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ESEARCH_URL = f"{EUTILS_BASE}/esearch.fcgi"
EFETCH_URL = f"{EUTILS_BASE}/efetch.fcgi"

# Requête PubMed : Alzheimer + facteurs de risque multi-catégories.
# tiab = title/abstract, MeSH = vocabulaire contrôlé PubMed.
DEFAULT_QUERY = (
    '("Alzheimer Disease"[MeSH Terms] OR "alzheimer\'s disease"[tiab]) '
    'AND ('
    '"risk factor"[tiab] OR "risk factors"[tiab] OR APOE[tiab] OR genetic[tiab] '
    'OR diet[tiab] OR nutrition[tiab] OR sleep[tiab] OR exercise[tiab] OR "physical activity"[tiab] '
    'OR cardiovascular[tiab] OR diabetes[tiab] OR cholesterol[tiab] OR lipid[tiab] '
    'OR obesity[tiab] OR hypertension[tiab] OR lifestyle[tiab] OR environmental[tiab] OR smoking[tiab]'
    ') '
    'AND ("2015"[PDAT] : "2026"[PDAT]) '
    'AND (english[Language]) '
    'AND (Review[ptyp] OR "Journal Article"[ptyp])'
)

# Catégories utilisées pour taguer chaque abstract (retrieval filtrable dans la démo).
RISK_CATEGORIES = {
    "genetic": ["apoe", "genetic", "gene", "genotype", "hereditary", "mutation"],
    "diet_nutrition": ["diet", "nutrition", "nutrient", "vitamin", "omega", "dha", "lipid", "fatty acid"],
    "sleep": ["sleep", "insomnia", "circadian"],
    "physical_activity": ["exercise", "physical activity", "sedentary", "fitness"],
    "cardiovascular_metabolic": ["cardiovascular", "diabetes", "cholesterol", "hypertension", "obesity",
                                 "metabolic", "blood pressure", "stroke"],
    "lifestyle_environment": ["smoking", "alcohol", "environmental", "pollution", "education", "social"],
}


def tag_categories(text: str) -> list[str]:
    text_lower = text.lower()
    tags = [cat for cat, keywords in RISK_CATEGORIES.items()
            if any(kw in text_lower for kw in keywords)]
    return tags or ["other"]


def esearch_pmids(query: str, max_records: int, api_key: str | None, batch_size: int = 500) -> list[str]:
    """Récupère la liste des PMID matchant la requête, par pages."""
    pmids: list[str] = []
    retstart = 0
    with tqdm(total=max_records, desc="esearch (recherche PMIDs)") as pbar:
        while len(pmids) < max_records:
            params = {
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "retstart": retstart,
                "retmax": min(batch_size, max_records - len(pmids)),
                "sort": "relevance",
            }
            if api_key:
                params["api_key"] = api_key
            resp = requests.get(ESEARCH_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            id_list = data.get("esearchresult", {}).get("idlist", [])
            if not id_list:
                break
            pmids.extend(id_list)
            pbar.update(len(id_list))
            retstart += len(id_list)
            time.sleep(0.11 if api_key else 0.35)  # respecte le quota NCBI (10/s ou 3/s)
    return pmids[:max_records]


def parse_efetch_xml(xml_bytes: bytes) -> list[dict]:
    """Parse une réponse efetch (PubmedArticleSet) en liste de dicts."""
    root = ET.fromstring(xml_bytes)
    records = []
    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        pmid = pmid_el.text if pmid_el is not None else None
        if not pmid:
            continue

        title_el = article.find(".//ArticleTitle")
        title = "".join(title_el.itertext()).strip() if title_el is not None else ""

        abstract_parts = article.findall(".//Abstract/AbstractText")
        if abstract_parts:
            abstract = " ".join(
                (f"{p.get('Label')}: " if p.get("Label") else "") + "".join(p.itertext())
                for p in abstract_parts
            ).strip()
        else:
            abstract = ""

        if not abstract:
            continue  # on ne garde que les records avec un abstract exploitable

        year_el = article.find(".//PubDate/Year")
        if year_el is None:
            year_el = article.find(".//PubDate/MedlineDate")
        year = year_el.text[:4] if year_el is not None and year_el.text else None

        journal_el = article.find(".//Journal/Title")
        journal = journal_el.text if journal_el is not None else None

        authors = []
        for author in article.findall(".//AuthorList/Author"):
            last = author.find("LastName")
            fore = author.find("ForeName")
            if last is not None:
                name = last.text
                if fore is not None:
                    name = f"{fore.text} {name}"
                authors.append(name)

        mesh_terms = [
            m.text for m in article.findall(".//MeshHeadingList/MeshHeading/DescriptorName")
            if m.text
        ]

        full_text_for_tagging = f"{title} {abstract} {' '.join(mesh_terms)}"

        records.append({
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "year": year,
            "journal": journal,
            "authors": authors,
            "mesh_terms": mesh_terms,
            "risk_categories": tag_categories(full_text_for_tagging),
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        })
    return records


def efetch_records(pmids: list[str], api_key: str | None, batch_size: int = 200) -> list[dict]:
    """Récupère titre/abstract/métadonnées pour une liste de PMIDs, par lots."""
    records = []
    batches = [pmids[i:i + batch_size] for i in range(0, len(pmids), batch_size)]
    for batch in tqdm(batches, desc="efetch (abstracts)"):
        params = {
            "db": "pubmed",
            "id": ",".join(batch),
            "retmode": "xml",
            "rettype": "abstract",
        }
        if api_key:
            params["api_key"] = api_key
        resp = requests.get(EFETCH_URL, params=params, timeout=60)
        resp.raise_for_status()
        records.extend(parse_efetch_xml(resp.content))
        time.sleep(0.11 if api_key else 0.35)
    return records


def main():
    parser = argparse.ArgumentParser(description="Ingestion PubMed pour le RAG facteurs de risque Alzheimer")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Requête PubMed (syntaxe esearch)")
    parser.add_argument("--max-records", type=int, default=1200, help="Nombre max d'abstracts à récupérer")
    parser.add_argument("--out", default="data/raw/pubmed_alzheimer.json", help="Fichier de sortie JSON")
    args = parser.parse_args()

    api_key = os.getenv("NCBI_API_KEY")
    if not api_key:
        print("[info] Pas de NCBI_API_KEY définie — limite à 3 req/s. "
              "Clé gratuite : https://www.ncbi.nlm.nih.gov/account/settings/")

    print(f"[1/2] Recherche des PMIDs (cible: {args.max_records})...")
    pmids = esearch_pmids(args.query, args.max_records, api_key)
    print(f"  -> {len(pmids)} PMIDs trouvés")

    print("[2/2] Récupération des abstracts...")
    records = efetch_records(pmids, api_key)
    print(f"  -> {len(records)} abstracts exploitables (après filtrage des vides)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Écrit : {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")

    # Petit résumé par catégorie, utile pour vérifier que le corpus est équilibré
    from collections import Counter
    cat_counts = Counter(cat for r in records for cat in r["risk_categories"])
    print("\nRépartition par catégorie de facteur de risque :")
    for cat, count in cat_counts.most_common():
        print(f"  {cat:28s} {count}")


if __name__ == "__main__":
    main()
