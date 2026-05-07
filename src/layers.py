"""
src/layers.py
=============

Implémentation de la couche Graph Attention (GAT) à partir du papier
« Graph Attention Networks » (Veličković et al., ICLR 2018).

On implémente :
- `GATLayer`        : une seule tête d'attention (équations 1, 2, 3, 4 du papier)
- `MultiHeadGATLayer` : plusieurs têtes en parallèle, fusionnées par
                        concaténation (couches cachées) ou moyenne (couche de sortie)

Les noms de variables suivent volontairement la notation du papier
pour faciliter la lecture croisée avec celui-ci.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """
    Une seule tête d'attention GAT.

    Étant donné des features de nœuds H ∈ R^{N×F} et une matrice d'adjacence A ∈ {0,1}^{N×N},
    cette couche calcule de nouvelles features H' ∈ R^{N×F'} de la manière suivante :

        1. z_i = W h_i                               (transformation linéaire partagée)
        2. e_ij = LeakyReLU( a^T [z_i || z_j] )      (score d'attention non normalisé)
        3. α_ij = softmax_j(e_ij) sur les voisins   (normalisation, masquée par A)
        4. h'_i = σ( Σ_j α_ij · z_j )                (agrégation pondérée)

    Pour intégrer la structure du graphe, on remplace les e_ij par -∞ pour les paires (i,j)
    non connectées, ce qui les annule après la softmax (« masked attention »).
    """

    def __init__(self, in_features: int, out_features: int,
                 dropout: float = 0.6, alpha: float = 0.2,
                 concat: bool = True):
        """
        Args:
            in_features: F, dimension des features d'entrée.
            out_features: F', dimension des features de sortie.
            dropout: probabilité de dropout (appliqué sur les features ET sur les α).
            alpha: pente négative du LeakyReLU (0.2 dans le papier).
            concat: si True, applique ELU à la sortie (cas couche cachée).
                    Si False, pas d'activation (cas couche finale, où on appliquera
                    la softmax au niveau du modèle complet).
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        # W ∈ R^{F' × F} : transformation linéaire partagée appliquée à chaque nœud.
        # On l'implémente comme un Linear sans biais pour coller au papier.
        self.W = nn.Linear(in_features, out_features, bias=False)

        # a ∈ R^{2F'} : vecteur d'attention.
        # Pour des raisons d'efficacité on le coupe en deux moitiés a_src et a_dst,
        # ce qui permet d'écrire e_ij = a_src · z_i + a_dst · z_j (sans concaténation explicite).
        self.a_src = nn.Parameter(torch.empty(out_features, 1))
        self.a_dst = nn.Parameter(torch.empty(out_features, 1))

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialisation Glorot (Xavier), comme dans le papier."""
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h:   tenseur (N, F) — features des nœuds.
            adj: tenseur (N, N) — matrice d'adjacence binaire (avec self-loops déjà ajoutées).

        Returns:
            h':  tenseur (N, F') — nouvelles features.
        """
        # Étape 1 : transformation linéaire — z_i = W h_i pour tout i. Shape : (N, F').
        z = self.W(h)
        N = z.size(0)

        # Étape 2 : scores d'attention bruts e_ij = LeakyReLU(a^T [z_i || z_j]).
        # Astuce d'efficacité : a^T [z_i || z_j] = a_src · z_i + a_dst · z_j.
        # On calcule donc deux vecteurs (N,) puis on les broadcaste pour obtenir (N, N).
        attn_src = (z @ self.a_src).squeeze(-1)        # shape (N,) : a_src · z_i
        attn_dst = (z @ self.a_dst).squeeze(-1)        # shape (N,) : a_dst · z_j

        # broadcast : e[i, j] = attn_src[i] + attn_dst[j]
        e = attn_src.unsqueeze(1) + attn_dst.unsqueeze(0)   # shape (N, N)
        e = self.leakyrelu(e)

        # Étape 3 : masquage — les paires non connectées reçoivent -∞.
        # Après softmax elles seront strictement à 0.
        mask = (adj > 0)
        e = e.masked_fill(~mask, float('-inf'))

        # softmax sur les voisins (c'est-à-dire sur l'axe j)
        alpha = F.softmax(e, dim=1)                         # shape (N, N)

        # Dropout sur les coefficients d'attention. Important : c'est explicitement
        # mentionné dans le papier (Section 3.3) — chaque itération voit donc un
        # voisinage stochastiquement échantillonné.
        alpha = F.dropout(alpha, self.dropout, training=self.training)

        # Étape 4 : agrégation pondérée — h'_i = Σ_j α_ij · z_j.
        # En forme matricielle : H' = α · Z.
        h_prime = alpha @ z                                  # shape (N, F')

        if self.concat:
            return F.elu(h_prime)
        return h_prime


class MultiHeadGATLayer(nn.Module):
    """
    Empile K têtes d'attention GAT indépendantes (équations 5 et 6 du papier).

    Mode `concat=True` (couches cachées) :
        h'_i = ‖_{k=1..K}  σ( Σ_j α_ij^k · W^k · h_j )
        → la sortie a K * out_features features par nœud.

    Mode `concat=False` (couche finale) :
        h'_i = (1/K) · Σ_k Σ_j α_ij^k · W^k · h_j
        → la sortie a out_features features par nœud,
          et l'activation finale (softmax) est appliquée plus loin par le modèle.
    """

    def __init__(self, in_features: int, out_features: int,
                 num_heads: int, dropout: float = 0.6, alpha: float = 0.2,
                 concat: bool = True):
        super().__init__()
        self.concat = concat
        self.num_heads = num_heads

        # Une GATLayer par tête. Chaque tête a ses propres W et a.
        # Le `concat` interne contrôle juste si on applique ELU ; ici, dans tous les cas,
        # on diffère l'activation à la sortie de la couche multi-head.
        self.heads = nn.ModuleList([
            GATLayer(in_features, out_features,
                     dropout=dropout, alpha=alpha,
                     concat=False)            # on ne veut PAS d'ELU à l'intérieur
            for _ in range(num_heads)
        ])

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        head_outputs = [head(h, adj) for head in self.heads]   # liste de K tenseurs (N, F')

        if self.concat:
            # Concaténation : (N, K * F'), puis ELU
            out = torch.cat(head_outputs, dim=1)
            return F.elu(out)
        else:
            # Moyenne : (N, F'), pas d'activation (softmax appliquée par le modèle)
            return torch.mean(torch.stack(head_outputs, dim=0), dim=0)
