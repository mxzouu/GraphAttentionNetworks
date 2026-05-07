"""
src/data.py
===========

Chargement des datasets Cora et Citeseer.

On utilise `torch_geometric.datasets.Planetoid` UNIQUEMENT pour télécharger les données
et accéder au split standard du papier (Yang et al., 2016 ; même split que Kipf & Welling 2017).

Les données sont ensuite converties dans le format simple attendu par notre modèle :
    - x   : (N, F) features des nœuds
    - y   : (N,)   labels
    - adj : (N, N) matrice d'adjacence dense, AVEC SELF-LOOPS
    - train_mask, val_mask, test_mask : masques booléens (N,)
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch_geometric.datasets import Planetoid


# Chemin où les datasets seront téléchargés/cachés
DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


@dataclass
class GraphData:
    """Conteneur léger pour les données d'un graphe."""
    x: torch.Tensor          # (N, F)
    y: torch.Tensor          # (N,)
    adj: torch.Tensor        # (N, N), dense, avec self-loops
    train_mask: torch.Tensor # (N,) bool
    val_mask: torch.Tensor   # (N,) bool
    test_mask: torch.Tensor  # (N,) bool
    num_features: int
    num_classes: int

    def to(self, device: torch.device) -> "GraphData":
        """Déplace tous les tenseurs sur le device demandé."""
        return GraphData(
            x=self.x.to(device),
            y=self.y.to(device),
            adj=self.adj.to(device),
            train_mask=self.train_mask.to(device),
            val_mask=self.val_mask.to(device),
            test_mask=self.test_mask.to(device),
            num_features=self.num_features,
            num_classes=self.num_classes,
        )


def _edge_index_to_dense_adj(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Convertit un edge_index (2, E) en matrice d'adjacence dense (N, N) avec self-loops."""
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    # edge_index est de shape (2, E) : [src; dst]
    adj[edge_index[0], edge_index[1]] = 1.0
    # Self-loops (un nœud doit pouvoir s'attendre à lui-même).
    adj.fill_diagonal_(1.0)
    return adj


def load_dataset(name: str) -> GraphData:
    """
    Charge un dataset Planetoid au format GraphData.

    Args:
        name: 'Cora' ou 'Citeseer' (ou 'Pubmed', non testé ici).

    Returns:
        GraphData prêt à l'emploi.
    """
    name = name.capitalize()
    if name not in {"Cora", "Citeseer", "Pubmed"}:
        raise ValueError(f"Dataset inconnu : {name}")

    dataset = Planetoid(root=str(DATA_ROOT / name), name=name)
    data = dataset[0]      # un seul graphe

    adj = _edge_index_to_dense_adj(data.edge_index, num_nodes=data.num_nodes)

    return GraphData(
        x=data.x.float(),
        y=data.y.long(),
        adj=adj,
        train_mask=data.train_mask.bool(),
        val_mask=data.val_mask.bool(),
        test_mask=data.test_mask.bool(),
        num_features=dataset.num_features,
        num_classes=dataset.num_classes,
    )


if __name__ == "__main__":
    # Petit sanity-check exécutable via : python -m src.data
    for name in ["Cora", "Citeseer"]:
        d = load_dataset(name)
        print(f"\n=== {name} ===")
        print(f"  N = {d.x.shape[0]} nodes")
        print(f"  F = {d.num_features} features par nœud")
        print(f"  C = {d.num_classes} classes")
        print(f"  |E| (avec self-loops) = {int(d.adj.sum().item())}")
        print(f"  train / val / test = "
              f"{int(d.train_mask.sum())} / {int(d.val_mask.sum())} / {int(d.test_mask.sum())}")
