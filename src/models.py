"""
src/models.py
=============

Modèle GAT à 2 couches utilisé pour les datasets transductifs (Cora, Citeseer).
Architecture exacte du papier (Section 3.3 du papier 2018) :

    Input H (N, F_in)
       │
       ▼   dropout(p=0.6)
    ┌──────────────────────────────────────────────────────────┐
    │ MultiHeadGATLayer(F_in -> F_hid, num_heads=8, concat)    │   sortie (N, 8 * F_hid)
    │   ELU intégrée                                           │
    └──────────────────────────────────────────────────────────┘
       │
       ▼   dropout(p=0.6)
    ┌──────────────────────────────────────────────────────────┐
    │ MultiHeadGATLayer(8*F_hid -> n_classes, num_heads=1, avg)│   sortie (N, n_classes)
    └──────────────────────────────────────────────────────────┘
       │
       ▼
    log_softmax(dim=1)  → utilisé avec NLLLoss

Le paramètre `variant` permet de choisir entre l'attention statique du papier
2018 ('gat') et l'attention dynamique du papier 2022 ('gatv2'). Le reste de
l'architecture (nombre de couches, têtes, dropout, etc.) est strictement
identique pour permettre une comparaison équitable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers import MultiHeadGATLayer


class GAT(nn.Module):
    """GAT à 2 couches pour la classification de nœuds en régime transductif."""

    def __init__(self, in_features: int, hidden_features: int, num_classes: int,
                 num_heads: int = 8, num_out_heads: int = 1,
                 dropout: float = 0.6, alpha: float = 0.2,
                 variant: str = 'gat'):
        """
        Args:
            in_features:     F (1433 pour Cora, 3703 pour Citeseer).
            hidden_features: F' par tête dans la couche cachée (8 dans le papier).
            num_classes:     C, nombre de classes.
            num_heads:       K, nombre de têtes en couche cachée (8 dans le papier).
            num_out_heads:   K' nombre de têtes en sortie (1 pour Cora/Citeseer).
            dropout:         probabilité de dropout (0.6 dans le papier).
            alpha:           pente négative LeakyReLU (0.2 dans le papier).
            variant:         'gat' (papier 2018) ou 'gatv2' (papier 2022).
        """
        super().__init__()
        self.dropout = dropout
        self.variant = variant

        self.layer1 = MultiHeadGATLayer(
            in_features=in_features,
            out_features=hidden_features,
            num_heads=num_heads,
            dropout=dropout,
            alpha=alpha,
            concat=True,
            variant=variant,
        )

        self.layer2 = MultiHeadGATLayer(
            in_features=num_heads * hidden_features,
            out_features=num_classes,
            num_heads=num_out_heads,
            dropout=dropout,
            alpha=alpha,
            concat=False,
            variant=variant,
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.layer1(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.layer2(x, adj)
        return F.log_softmax(x, dim=1)
