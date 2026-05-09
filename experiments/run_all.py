"""
experiments/run_all.py
======================

Lance Cora et Citeseer pour les deux variantes (GAT et GATv2), affiche les
trois tableaux récapitulatifs, et SAUVEGARDE tous les résultats sous un format
structuré pour qu'un notebook d'analyse puisse les charger sans avoir à
réentraîner les modèles.

Fichiers produits dans `results/` :
    - summary.txt    : tableau récapitulatif lisible
    - runs.json      : structure complète (5 runs × 4 configs) pour le notebook

Usage :
    python -m experiments.run_all
"""

import json
from pathlib import Path

import numpy as np

from experiments.run_cora import main as run_cora
from experiments.run_citeseer import main as run_citeseer


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _delta(ours_accs: list[float], theirs_accs: list[float]) -> str:
    """Calcule l'écart moyen GATv2 - GAT en points de pourcentage."""
    diff = (np.mean(ours_accs) - np.mean(theirs_accs)) * 100
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.2f} pts"


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    cora_results = run_cora()
    citeseer_results = run_citeseer()

    # ─── Préparation des chiffres pour l'affichage ──────────────────────────
    cora_gat = cora_results['gat']['mean_std']
    cora_gatv2 = cora_results['gatv2']['mean_std']
    cite_gat = citeseer_results['gat']['mean_std']
    cite_gatv2 = citeseer_results['gatv2']['mean_std']

    cora_delta = _delta(cora_results['gatv2']['accs'], cora_results['gat']['accs'])
    cite_delta = _delta(citeseer_results['gatv2']['accs'], citeseer_results['gat']['accs'])

    # ─── Construction des trois tableaux ────────────────────────────────────
    lines = []
    lines.append("")
    lines.append("=" * 78)
    lines.append("  RÉCAPITULATIF FINAL")
    lines.append("=" * 78)

    lines.append("")
    lines.append("  ┌───────────────────────────────────────────────────────────────────────┐")
    lines.append("  │  TABLEAU 1 — GAT (Veličković et al., ICLR 2018)                       │")
    lines.append("  └───────────────────────────────────────────────────────────────────────┘")
    lines.append(f"    {'Dataset':<12} | {'Notre implémentation':<24} | {'Papier 2018':<20}")
    lines.append("    " + "-" * 64)
    lines.append(f"    {'Cora':<12} | {cora_gat + ' %':<24} | {'83.0 ± 0.7 %':<20}")
    lines.append(f"    {'Citeseer':<12} | {cite_gat + ' %':<24} | {'72.5 ± 0.7 %':<20}")

    lines.append("")
    lines.append("  ┌───────────────────────────────────────────────────────────────────────┐")
    lines.append("  │  TABLEAU 2 — GATv2 (Brody et al., ICLR 2022)                          │")
    lines.append("  └───────────────────────────────────────────────────────────────────────┘")
    lines.append(f"    {'Dataset':<12} | {'Notre implémentation':<24} | {'Papier 2022':<20}")
    lines.append("    " + "-" * 64)
    lines.append(f"    {'Cora':<12} | {cora_gatv2 + ' %':<24} | {'non rapporté *':<20}")
    lines.append(f"    {'Citeseer':<12} | {cite_gatv2 + ' %':<24} | {'non rapporté *':<20}")
    lines.append("")
    lines.append("    * Le papier GATv2 n'évalue pas sur Cora/Citeseer, jugés trop simples")
    lines.append("      pour révéler la différence statique vs dynamique. Sur Pubmed, le")
    lines.append("      papier rapporte GAT 78.1% vs GATv2 78.5% (différence faible mais")
    lines.append("      significative). Voir Annexe D.3 du papier 2022.")

    lines.append("")
    lines.append("  ┌───────────────────────────────────────────────────────────────────────┐")
    lines.append("  │  TABLEAU 3 — Comparaison directe GAT vs GATv2 (NOTRE implémentation)  │")
    lines.append("  └───────────────────────────────────────────────────────────────────────┘")
    lines.append(f"    {'Dataset':<12} | {'GAT':<16} | {'GATv2':<16} | {'Δ (GATv2 - GAT)':<18}")
    lines.append("    " + "-" * 70)
    lines.append(f"    {'Cora':<12} | {cora_gat + ' %':<16} | {cora_gatv2 + ' %':<16} | {cora_delta:<18}")
    lines.append(f"    {'Citeseer':<12} | {cite_gat + ' %':<16} | {cite_gatv2 + ' %':<16} | {cite_delta:<18}")

    lines.append("")
    lines.append("=" * 78)

    summary = "\n".join(lines)
    print(summary)

    # ─── Sauvegarde texte (lisible) ─────────────────────────────────────────
    summary_file = RESULTS_DIR / "summary.txt"
    with summary_file.open("w", encoding="utf-8") as f:
        f.write(summary + "\n\n")
        f.write("Détail des runs :\n")
        f.write(f"  Cora     GAT   : {[f'{a*100:.2f}%' for a in cora_results['gat']['accs']]}\n")
        f.write(f"  Cora     GATv2 : {[f'{a*100:.2f}%' for a in cora_results['gatv2']['accs']]}\n")
        f.write(f"  Citeseer GAT   : {[f'{a*100:.2f}%' for a in citeseer_results['gat']['accs']]}\n")
        f.write(f"  Citeseer GATv2 : {[f'{a*100:.2f}%' for a in citeseer_results['gatv2']['accs']]}\n")
    print(f"\n  → Résumé sauvegardé dans : {summary_file}")

    # ─── Sauvegarde structurée (JSON, pour le notebook) ─────────────────────
    structured = {
        "Cora": {
            "gat": {
                "accs": [float(a) for a in cora_results['gat']['accs']],
                "mean_pct": float(np.mean(cora_results['gat']['accs']) * 100),
                "std_pct": float(np.std(cora_results['gat']['accs']) * 100),
            },
            "gatv2": {
                "accs": [float(a) for a in cora_results['gatv2']['accs']],
                "mean_pct": float(np.mean(cora_results['gatv2']['accs']) * 100),
                "std_pct": float(np.std(cora_results['gatv2']['accs']) * 100),
            },
        },
        "Citeseer": {
            "gat": {
                "accs": [float(a) for a in citeseer_results['gat']['accs']],
                "mean_pct": float(np.mean(citeseer_results['gat']['accs']) * 100),
                "std_pct": float(np.std(citeseer_results['gat']['accs']) * 100),
            },
            "gatv2": {
                "accs": [float(a) for a in citeseer_results['gatv2']['accs']],
                "mean_pct": float(np.mean(citeseer_results['gatv2']['accs']) * 100),
                "std_pct": float(np.std(citeseer_results['gatv2']['accs']) * 100),
            },
        },
        "paper_baselines": {
            "Cora": {"gat_mean": 83.0, "gat_std": 0.7},
            "Citeseer": {"gat_mean": 72.5, "gat_std": 0.7},
        },
    }

    json_file = RESULTS_DIR / "runs.json"
    with json_file.open("w", encoding="utf-8") as f:
        json.dump(structured, f, indent=2, ensure_ascii=False)
    print(f"  → Données structurées sauvegardées dans : {json_file}")
    print(f"     (utilisées par le notebook d'analyse)")


if __name__ == "__main__":
    main()
