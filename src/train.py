"""
src/train.py
============

Boucle d'entraînement pour le modèle GAT (variant='gat' ou 'gatv2').

Suit le protocole du papier 2018 (Section 3.3) :
- Optimiseur : Adam (lr = 0.005 pour Cora/Citeseer)
- L2 weight decay : 5e-4
- Loss : NLLLoss (le modèle renvoie déjà du log_softmax)
- Early stopping sur la val accuracy (patience = 100)
- max_epochs : 300 par défaut (cap pour éviter les entraînements trop longs sur CPU)
"""

from copy import deepcopy
from dataclasses import dataclass
import time

import torch
import torch.nn as nn

from src.data import GraphData
from src.models import GAT
from src.utils import accuracy


@dataclass
class TrainConfig:
    """Hyperparamètres d'entraînement."""
    hidden_features: int = 8
    num_heads: int = 8
    num_out_heads: int = 1
    dropout: float = 0.6
    alpha: float = 0.2

    learning_rate: float = 0.005
    weight_decay: float = 5e-4
    max_epochs: int = 300
    patience: int = 100

    variant: str = 'gat'         # 'gat' (papier 2018) ou 'gatv2' (papier 2022)


def train_one_run(data: GraphData, config: TrainConfig,
                  device: torch.device,
                  verbose: bool = False) -> tuple[float, float, int]:
    """
    Effectue un entraînement complet avec early stopping et retourne :
        (test_acc, best_val_acc, best_epoch)
    """
    data = data.to(device)

    model = GAT(
        in_features=data.num_features,
        hidden_features=config.hidden_features,
        num_classes=data.num_classes,
        num_heads=config.num_heads,
        num_out_heads=config.num_out_heads,
        dropout=config.dropout,
        alpha=config.alpha,
        variant=config.variant,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.NLLLoss()

    best_val_acc = 0.0
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, config.max_epochs + 1):
        # ---- Train ----
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.adj)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # ---- Eval ----
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.adj)
            train_acc = accuracy(out[data.train_mask], data.y[data.train_mask])
            val_acc = accuracy(out[data.val_mask], data.y[data.val_mask])

        # ---- Early stopping (sur la val accuracy) ----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(f"  epoch {epoch:4d} | loss {loss.item():.4f} "
                  f"| train_acc {train_acc:.4f} | val_acc {val_acc:.4f} "
                  f"| best_val {best_val_acc:.4f}")

        if epochs_without_improvement >= config.patience:
            if verbose:
                print(f"  Early stopping à l'époque {epoch} "
                      f"(meilleure val à l'époque {best_epoch}).")
            break

    # ---- Test avec les meilleurs poids ----
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.adj)
        test_acc = accuracy(out[data.test_mask], data.y[data.test_mask])

    return test_acc, best_val_acc, best_epoch


def run_experiment(data: GraphData, config: TrainConfig,
                   num_runs: int = 5, device: torch.device | None = None,
                   seed_base: int = 0, verbose: bool = False) -> list[float]:
    """Lance `num_runs` entraînements et retourne la liste des test accuracies."""
    from src.utils import set_seed

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_accuracies = []
    for run_idx in range(num_runs):
        set_seed(seed_base + run_idx)
        t0 = time.time()
        test_acc, val_acc, best_epoch = train_one_run(
            data, config, device=device, verbose=verbose
        )
        elapsed = time.time() - t0
        test_accuracies.append(test_acc)
        print(f"  Run {run_idx + 1}/{num_runs} : "
              f"test_acc = {test_acc * 100:.2f}%  "
              f"(best_val = {val_acc * 100:.2f}% à l'epoch {best_epoch}, "
              f"{elapsed:.1f}s)")

    return test_accuracies
