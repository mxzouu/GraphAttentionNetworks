"""
src/models.py
=============

Modèle GAT à 2 couches utilisé pour les datasets transductifs (Cora, Citeseer).
Architecture exacte du papier (Section 3.3) :

    Input H (N, F_in)
       │
       ▼   dropout(p=0.6)
    ┌──────────────────────────────────────────────────────────┐
    │ MultiHeadGATLayer(F_in -> F_hid, num_heads=8, concat)   │   sortie (N, 8 * F_hid)
    │   suivi d'ELU (intégré dans la couche)                  │
    └──────────────────────────────────────────────────────────┘
       │
       ▼   dropout(p=0.6)
    ┌──────────────────────────────────────────────────────────┐
    │ MultiHeadGATLayer(8*F_hid -> n_classes, num_heads=1, avg)│   sortie (N, n_classes)
    └──────────────────────────────────────────────────────────┘
       │
       ▼
    log_softmax(dim=1)  → utilisé avec NLLLoss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers import MultiHeadGATLayer


class GAT(nn.Module):
    """Modèle GAT à 2 couches pour la classification de nœuds en régime transductif."""

    def __init__(self, in_features: int, hidden_features: int, num_classes: int,
                 num_heads: int = 8, num_out_heads: int = 1,
                 dropout: float = 0.6, alpha: float = 0.2):
        """
        Args:
            in_features: F, dimension des features d'entrée (1433 pour Cora, 3703 pour Citeseer).
            hidden_features: F', features par tête dans la couche cachée (8 dans le papier).
            num_classes: nombre de classes pour la classification.
            num_heads: K, nombre de têtes dans la couche cachée (8 dans le papier).
            num_out_heads: K', nombre de têtes pour la couche de sortie
                           (1 pour Cora/Citeseer, 8 pour Pubmed).
            dropout: probabilité de dropout (0.6 dans le papier).
            alpha: pente négative LeakyReLU (0.2 dans le papier).
        """
        super().__init__()
        self.dropout = dropout

        # Couche 1 : multi-head, concaténation.
        # Sortie : (N, num_heads * hidden_features).
        self.layer1 = MultiHeadGATLayer(
            in_features=in_features,
            out_features=hidden_features,
            num_heads=num_heads,
            dropout=dropout,
            alpha=alpha,
            concat=True,
        )

        # Couche 2 : multi-head moyennée, produit directement les logits.
        # Entrée : num_heads * hidden_features. Sortie : num_classes.
        self.layer2 = MultiHeadGATLayer(
            in_features=num_heads * hidden_features,
            out_features=num_classes,
            num_heads=num_out_heads,
            dropout=dropout,
            alpha=alpha,
            concat=False,
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   (N, F) features d'entrée.
            adj: (N, N) matrice d'adjacence avec self-loops.

        Returns:
            log-probabilités (N, num_classes), prêtes pour NLLLoss.
        """
        # Dropout sur les inputs (papier, Section 3.3).
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.layer1(x, adj)

        # Dropout entre les deux couches.
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.layer2(x, adj)

        return F.log_softmax(x, dim=1)
