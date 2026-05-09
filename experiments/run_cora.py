"""
experiments/run_cora.py
=======================

Reproduit les résultats sur Cora pour les DEUX variantes :
    - GAT  (Veličković et al., ICLR 2018) → papier rapporte 83.0 ± 0.7%
    - GATv2 (Brody et al., ICLR 2022)     → non rapporté sur Cora

Usage :
    python -m experiments.run_cora
"""

import torch

from src.data import load_dataset
from src.train import TrainConfig, run_experiment
from src.utils import format_mean_std


# Résultat rapporté dans le papier 2018 (Tableau 2)
GAT_PAPER_RESULT = "83.0 ± 0.7"
# Le papier 2022 (GATv2) ne rapporte PAS de résultat sur Cora
GATV2_PAPER_RESULT = "non rapporté"


def main(num_runs: int = 5):
    print("=" * 60)
    print("  Reproduction Cora — GAT (2018) vs GATv2 (2022)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device : {device}")

    print("\n[1/3] Chargement du dataset Cora...")
    data = load_dataset("Cora")
    print(f"      N = {data.x.shape[0]} nodes, "
          f"F = {data.num_features}, C = {data.num_classes}")

    base_config = dict(
        hidden_features=8, num_heads=8, num_out_heads=1,
        dropout=0.6, alpha=0.2,
        learning_rate=0.005, weight_decay=5e-4,
        max_epochs=300, patience=100,
    )

    # --- GAT (papier 2018) ---
    print(f"\n[2/3] Entraînement GAT ({num_runs} runs)...")
    gat_config = TrainConfig(variant='gat', **base_config)
    gat_accs = run_experiment(data, gat_config, num_runs=num_runs, device=device, seed_base=0)
    gat_result = format_mean_std(gat_accs)

    # --- GATv2 (papier 2022) ---
    print(f"\n[3/3] Entraînement GATv2 ({num_runs} runs)...")
    gatv2_config = TrainConfig(variant='gatv2', **base_config)
    gatv2_accs = run_experiment(data, gatv2_config, num_runs=num_runs, device=device, seed_base=0)
    gatv2_result = format_mean_std(gatv2_accs)

    # --- Résumé ---
    print("\n" + "-" * 60)
    print(f"  CORA — Résultats")
    print("-" * 60)
    print(f"  GAT   :  notre implémentation = {gat_result} %  |  papier = {GAT_PAPER_RESULT} %")
    print(f"  GATv2 :  notre implémentation = {gatv2_result} %  |  papier = {GATV2_PAPER_RESULT}")
    print("-" * 60)

    return {
        'gat': {'accs': gat_accs, 'mean_std': gat_result},
        'gatv2': {'accs': gatv2_accs, 'mean_std': gatv2_result},
    }


if __name__ == "__main__":
    main()
