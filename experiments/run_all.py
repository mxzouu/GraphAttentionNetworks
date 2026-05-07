"""
experiments/run_all.py
======================

Lance les expériences Cora et Citeseer, puis affiche un tableau récapitulatif
comparant nos résultats à ceux du papier. Sauvegarde aussi un fichier
`results/summary.txt`.

Usage :
    python -m experiments.run_all
"""

from pathlib import Path

from experiments.run_cora import main as run_cora
from experiments.run_citeseer import main as run_citeseer


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    # --- Cora ---
    cora_accs, cora_result = run_cora()

    # --- Citeseer ---
    citeseer_accs, citeseer_result = run_citeseer()

    # --- Tableau récapitulatif ---
    summary_lines = [
        "",
        "=" * 70,
        "  RÉCAPITULATIF — Comparaison avec le papier",
        "=" * 70,
        "",
        f"  {'Dataset':<12} | {'Notre implémentation':<22} | {'Papier (GAT)':<15}",
        "  " + "-" * 60,
        f"  {'Cora':<12} | {cora_result + ' %':<22} | {'83.0 ± 0.7 %':<15}",
        f"  {'Citeseer':<12} | {citeseer_result + ' %':<22} | {'72.5 ± 0.7 %':<15}",
        "",
        "=" * 70,
    ]

    summary = "\n".join(summary_lines)
    print(summary)

    # Sauvegarde
    output_file = RESULTS_DIR / "summary.txt"
    with output_file.open("w", encoding="utf-8") as f:
        f.write(summary + "\n\n")
        f.write("Détail des runs :\n")
        f.write(f"  Cora     : {[f'{a*100:.2f}%' for a in cora_accs]}\n")
        f.write(f"  Citeseer : {[f'{a*100:.2f}%' for a in citeseer_accs]}\n")
    print(f"\n  → Résumé sauvegardé dans : {output_file}")


if __name__ == "__main__":
    main()
