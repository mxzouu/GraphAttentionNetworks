"""
experiments/run_citeseer.py
===========================

Reproduit le résultat GAT sur Citeseer (papier : 72.5 ± 0.7%).

Usage :
    python -m experiments.run_citeseer
"""

import torch

from src.data import load_dataset
from src.train import TrainConfig, run_experiment
from src.utils import format_mean_std


# Résultat rapporté dans le papier (Tableau 2)
PAPER_RESULT = "72.5 ± 0.7"


def main():
    print("=" * 60)
    print("  GAT — Reproduction Citeseer")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device : {device}")

    print("\n[1/3] Chargement du dataset Citeseer...")
    data = load_dataset("Citeseer")
    print(f"      N = {data.x.shape[0]} nodes, "
          f"F = {data.num_features}, C = {data.num_classes}")

    # Mêmes hyperparamètres que pour Cora (le papier réutilise la config Cora pour Citeseer)
    config = TrainConfig(
        hidden_features=8,
        num_heads=8,
        num_out_heads=1,
        dropout=0.6,
        alpha=0.2,
        learning_rate=0.005,
        weight_decay=5e-4,
        max_epochs=1000,
        patience=100,
    )

    print("\n[2/3] Entraînement (5 runs avec des seeds différentes)...")
    test_accs = run_experiment(data, config, num_runs=5, device=device, seed_base=0)

    print("\n[3/3] Résultats finaux")
    our_result = format_mean_std(test_accs)
    print("-" * 60)
    print(f"  Notre implémentation :  {our_result} %")
    print(f"  Papier (GAT)         :  {PAPER_RESULT} %")
    print("-" * 60)

    return test_accs, our_result


if __name__ == "__main__":
    main()
