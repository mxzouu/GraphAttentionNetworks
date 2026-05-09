"""
src/layers.py
=============

Implémentations des couches d'attention :
  - GATLayer    : attention statique de Veličković et al. (ICLR 2018)
  - GATv2Layer  : attention dynamique de Brody et al. (ICLR 2022)
  - MultiHeadGATLayer : wrapper multi-têtes paramétrable (variant='gat' ou 'gatv2')

DIFFÉRENCE FONDAMENTALE entre les deux variantes :

  GAT   :  e_ij = LeakyReLU( a^T · [W h_i || W h_j] )
              ↑ a appliqué APRÈS la concaténation
              ↑ W et a sont consécutifs ⇒ peuvent fusionner ⇒ attention « statique »
              ↑ tous les nœuds partagent le même classement de leurs voisins

  GATv2 :  e_ij = a^T · LeakyReLU( W · [h_i || h_j] )
              ↑ a appliqué APRÈS le LeakyReLU
              ↑ MLP à 1 couche cachée ⇒ universal approximator ⇒ attention « dynamique »
              ↑ chaque nœud peut avoir un classement différent de ses voisins

Comme prouvé dans Brody et al. 2022 (Théorème 1), GAT ne peut PAS exprimer une attention
qui dépend vraiment de la paire (i, j). GATv2 corrige ça en changeant simplement
l'ordre des opérations : un seul changement de code, gros changement d'expressivité.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
#  GAT (Veličković et al., 2018)
# ─────────────────────────────────────────────────────────────────────────────

class GATLayer(nn.Module):
    """
    Une seule tête d'attention GAT (équations 1-4 du papier 2018).

        1. z_i  = W h_i
        2. e_ij = LeakyReLU( a^T [z_i || z_j] )
        3. α_ij = softmax_j(e_ij)  sur les voisins (masquée par adj)
        4. h'_i = σ( Σ_j α_ij · z_j )
    """

    def __init__(self, in_features: int, out_features: int,
                 dropout: float = 0.6, alpha: float = 0.2,
                 concat: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.concat = concat

        # W ∈ R^{F' × F}  — transformation linéaire partagée
        self.W = nn.Linear(in_features, out_features, bias=False)

        # a ∈ R^{2F'} décomposé en deux moitiés pour éviter une concaténation explicite :
        #   e_ij = a^T [z_i || z_j] = a_src · z_i + a_dst · z_j
        self.a_src = nn.Parameter(torch.empty(out_features, 1))
        self.a_dst = nn.Parameter(torch.empty(out_features, 1))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        z = self.W(h)                                        # (N, F')

        attn_src = (z @ self.a_src).squeeze(-1)              # (N,)
        attn_dst = (z @ self.a_dst).squeeze(-1)              # (N,)
        e = self.leakyrelu(attn_src.unsqueeze(1) + attn_dst.unsqueeze(0))   # (N, N)

        e = e.masked_fill(adj <= 0, float('-inf'))
        alpha = F.softmax(e, dim=1)
        alpha = F.dropout(alpha, self.dropout, training=self.training)

        h_prime = alpha @ z                                  # (N, F')
        return F.elu(h_prime) if self.concat else h_prime


# ─────────────────────────────────────────────────────────────────────────────
#  GATv2 (Brody et al., 2022)
# ─────────────────────────────────────────────────────────────────────────────

class GATv2Layer(nn.Module):
    """
    Une seule tête d'attention GATv2 (équation 7 du papier 2022).

        1. z_self_i     = W_self · h_i
           z_neighbor_j = W_neighbor · h_j
        2. pre_ij = LeakyReLU( z_self_i + z_neighbor_j )    # MLP à 1 couche cachée
        3. e_ij   = a^T · pre_ij                            # le « a » est appliqué APRÈS
        4. α_ij   = softmax_j(e_ij)                         # masquée par adj
        5. h'_i   = σ( Σ_j α_ij · z_neighbor_j )

    L'aggrégation utilise z_neighbor comme « valeur » (équation 4 du papier 2022,
    inchangée vs GAT) — c'est la convention la plus naturelle qui réutilise la
    même matrice qui agit sur h_j dans le calcul du score.
    """

    def __init__(self, in_features: int, out_features: int,
                 dropout: float = 0.6, alpha: float = 0.2,
                 concat: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.concat = concat

        # W = [W_self | W_neighbor] ∈ R^{F' × 2F}, séparée en deux pour l'efficacité :
        #   W · [h_i || h_j] = W_self · h_i + W_neighbor · h_j
        self.W_self = nn.Linear(in_features, out_features, bias=False)
        self.W_neighbor = nn.Linear(in_features, out_features, bias=False)

        # a ∈ R^{F'} — appliqué APRÈS le LeakyReLU. C'est LE changement clé vs GAT.
        self.a = nn.Parameter(torch.empty(out_features, 1))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W_self.weight)
        nn.init.xavier_uniform_(self.W_neighbor.weight)
        nn.init.xavier_uniform_(self.a)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        z_self = self.W_self(h)                              # (N, F')
        z_neighbor = self.W_neighbor(h)                      # (N, F')

        # Pré-activation : pre[i, j, :] = z_self[i] + z_neighbor[j], shape (N, N, F')
        pre = z_self.unsqueeze(1) + z_neighbor.unsqueeze(0)
        pre = self.leakyrelu(pre)

        # Score : e[i, j] = a^T · pre[i, j, :]
        # (N, N, F') @ (F', 1) → (N, N, 1) → squeeze → (N, N)
        e = (pre @ self.a).squeeze(-1)

        e = e.masked_fill(adj <= 0, float('-inf'))
        alpha = F.softmax(e, dim=1)
        alpha = F.dropout(alpha, self.dropout, training=self.training)

        # Aggrégation : on utilise z_neighbor comme valeur (W·h_j de l'équation 4)
        h_prime = alpha @ z_neighbor                         # (N, F')
        return F.elu(h_prime) if self.concat else h_prime


# ─────────────────────────────────────────────────────────────────────────────
#  Wrapper multi-têtes (paramétrable GAT / GATv2)
# ─────────────────────────────────────────────────────────────────────────────

class MultiHeadGATLayer(nn.Module):
    """
    Empile K têtes d'attention indépendantes (équations 5-6 du papier 2018).

    Mode `concat=True` (couches cachées) :
        h'_i = ‖_{k=1..K}  σ( Σ_j α_ij^k · W^k · h_j )
        → sortie (N, K * F').

    Mode `concat=False` (couche finale) :
        h'_i = (1/K) · Σ_k Σ_j α_ij^k · W^k · h_j
        → sortie (N, F').

    Le paramètre `variant` choisit la couche d'attention :
        - 'gat'   → GATLayer   (papier 2018)
        - 'gatv2' → GATv2Layer (papier 2022)
    """

    _VARIANTS = {'gat': GATLayer, 'gatv2': GATv2Layer}

    def __init__(self, in_features: int, out_features: int,
                 num_heads: int, dropout: float = 0.6, alpha: float = 0.2,
                 concat: bool = True, variant: str = 'gat'):
        super().__init__()
        if variant not in self._VARIANTS:
            raise ValueError(f"variant doit être dans {list(self._VARIANTS)}, reçu {variant!r}")
        self.variant = variant
        self.concat = concat
        self.num_heads = num_heads

        layer_cls = self._VARIANTS[variant]
        self.heads = nn.ModuleList([
            layer_cls(in_features, out_features,
                      dropout=dropout, alpha=alpha,
                      concat=False)            # ELU appliquée par le wrapper, pas par les têtes
            for _ in range(num_heads)
        ])

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        head_outputs = [head(h, adj) for head in self.heads]   # liste de K tenseurs (N, F')

        if self.concat:
            out = torch.cat(head_outputs, dim=1)               # (N, K * F')
            return F.elu(out)
        else:
            return torch.mean(torch.stack(head_outputs, dim=0), dim=0)   # (N, F')
