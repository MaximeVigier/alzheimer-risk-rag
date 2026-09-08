"""
Mode interactif pour tester le RAG à la main — pose une question, obtiens une réponse
sourcée, sans relancer le script à chaque fois.

Usage:
    python src/chat.py --strategy fixed
    (puis tape tes questions, "quit" ou Ctrl+C pour sortir)
"""
from __future__ import annotations

import argparse

from generate import answer_question
from retrieval import HybridRetriever


def main():
    parser = argparse.ArgumentParser(description="Chat interactif avec le RAG Alzheimer")
    parser.add_argument("--strategy", choices=["fixed", "semantic"], default="fixed")
    args = parser.parse_args()

    print(f"Chargement du retriever (stratégie={args.strategy})...")
    retriever = HybridRetriever(strategy=args.strategy)
    print("Prêt. Pose tes questions sur les facteurs de risque de la maladie d'Alzheimer.")
    print("Tape 'quit' pour sortir.\n")

    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query or query.lower() in {"quit", "exit"}:
            break

        result = answer_question(query, strategy=args.strategy, retriever=retriever)
        print(f"\n{result['answer']}\n")
        if result["sources_retrieved"]:
            print("Sources :")
            for s in result["sources_retrieved"]:
                if s["pmid"] in result["sources_cited"]:
                    print(f"  - [{s['pmid']}] {s['title']} ({s['year']}) — {s['url']}")
        print()


if __name__ == "__main__":
    main()
