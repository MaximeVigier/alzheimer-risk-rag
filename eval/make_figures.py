"""
Génère les figures scientifiques du projet à partir des résultats d'éval (eval/results/*.json).

Usage:
    python eval/make_figures.py

Lit eval/results/eval_fixed_full.json et eval/results/eval_semantic_full.json (si présent),
écrit les figures dans eval/results/figures/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")
RESULTS_DIR = Path("eval/results")
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODE_LABELS = {
    "dense": "Dense seul",
    "sparse": "Sparse (BM25)",
    "hybrid": "Hybride",
    "hybrid_rerank": "Hybride + rerank",
}
MODE_ORDER = ["dense", "sparse", "hybrid", "hybrid_rerank"]
K_VALUES = [3, 5, 10]


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fig_recall_by_mode(results: dict, strategy_label: str, out_name: str):
    """Barplot groupé : recall@k pour chaque mode de retrieval."""
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.2
    x = range(len(K_VALUES))

    for i, mode in enumerate(MODE_ORDER):
        if mode not in results["recall_at_k"]:
            continue
        values = [results["recall_at_k"][mode][f"recall@{k}"] for k in K_VALUES]
        positions = [xi + (i - 1.5) * width for xi in x]
        bars = ax.bar(positions, values, width, label=MODE_LABELS[mode])
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"k={k}" for k in K_VALUES])
    ax.set_ylabel("Recall@k")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Recall@k par mode de retrieval — {strategy_label}\n"
                 f"({results['n_questions']} questions du jeu de test)")
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / out_name, dpi=150)
    plt.close(fig)
    print(f"  -> {out_name}")


def fig_chunking_comparison(fixed: dict, semantic: dict):
    """Barplot groupé : compare fixed vs semantic sur le meilleur mode (hybrid_rerank)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.3
    x = range(len(K_VALUES))

    for i, (label, data) in enumerate([("Chunking taille fixe", fixed), ("Chunking sémantique", semantic)]):
        values = [data["recall_at_k"]["hybrid_rerank"][f"recall@{k}"] for k in K_VALUES]
        positions = [xi + (i - 0.5) * width for xi in x]
        bars = ax.bar(positions, values, width, label=label)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"k={k}" for k in K_VALUES])
    ax.set_ylabel("Recall@k (mode hybride + rerank)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Comparaison des stratégies de chunking\n(retrieval hybride + reranking, même jeu de test)")
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "chunking_comparison.png", dpi=150)
    plt.close(fig)
    print("  -> chunking_comparison.png")


def fig_faithfulness_by_category(results: dict, strategy_label: str, out_name: str):
    """Barplot : faithfulness moyenne par catégorie de facteur de risque."""
    data = results.get("faithfulness", {}).get("faithfulness_by_category", {})
    if not data:
        print(f"  [skip] pas de données faithfulness pour {out_name}")
        return

    categories = list(data.keys())
    values = list(data.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(categories, values, color=sns.color_palette("viridis", len(categories)))
    for bar, v in zip(bars, values):
        ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2, f"{v:.2f}", va="center", fontsize=9)

    overall = results["faithfulness"]["faithfulness_mean"]
    ax.axvline(overall, color="red", linestyle="--", linewidth=1, label=f"Moyenne globale ({overall:.2f})")

    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Faithfulness (RAGAS, juge = gpt-oss:20b)")
    ax.set_title(f"Fidélité des réponses par catégorie de facteur de risque — {strategy_label}")
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / out_name, dpi=150)
    plt.close(fig)
    print(f"  -> {out_name}")


def fig_edge_case_comparison(results: dict, strategy_label: str, out_name: str):
    """Barplot : faithfulness questions normales vs edge cases (questions pièges hors-sujet)."""
    data = results.get("faithfulness", {}).get("faithfulness_edge_case_vs_normal", {})
    data = {k: v for k, v in data.items() if v is not None}
    if not data:
        print(f"  [skip] pas de données edge_case pour {out_name}")
        return

    labels = {"normal": "Questions normales", "edge_case": "Cas limites (hors-sujet)"}
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar([labels[k] for k in data], list(data.values()),
                  color=["#4C72B0", "#DD8452"])
    for bar, v in zip(bars, data.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=11)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Faithfulness")
    ax.set_title(f"Robustesse sur les cas limites — {strategy_label}\n"
                 "(questions dont le contexte source ne répond que partiellement)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / out_name, dpi=150)
    plt.close(fig)
    print(f"  -> {out_name}")


def main():
    fixed = load(RESULTS_DIR / "eval_fixed_full.json")
    semantic = load(RESULTS_DIR / "eval_semantic_full.json")

    print("[figures] génération...")

    if fixed:
        fig_recall_by_mode(fixed, "chunking taille fixe", "recall_by_mode_fixed.png")
        fig_faithfulness_by_category(fixed, "chunking taille fixe", "faithfulness_by_category_fixed.png")
        fig_edge_case_comparison(fixed, "chunking taille fixe", "edge_case_comparison_fixed.png")

    if semantic:
        fig_recall_by_mode(semantic, "chunking sémantique", "recall_by_mode_semantic.png")
        fig_faithfulness_by_category(semantic, "chunking sémantique", "faithfulness_by_category_semantic.png")

    if fixed and semantic:
        fig_chunking_comparison(fixed, semantic)

    print(f"\nFigures écrites dans {FIG_DIR}/")


if __name__ == "__main__":
    main()
