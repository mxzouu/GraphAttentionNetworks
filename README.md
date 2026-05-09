# Graph Attention Networks (GAT & GATv2)

Implémentation **from scratch en PyTorch** de deux papiers :

> **[1]** *Graph Attention Networks* — Veličković, Cucurull, Casanova, Romero, Liò, Bengio (ICLR 2018)
> [arXiv:1710.10903](https://arxiv.org/abs/1710.10903)

> **[2]** *How Attentive Are Graph Attention Networks?* — Brody, Alon, Yahav (ICLR 2022)
> [arXiv:2105.14491](https://arxiv.org/abs/2105.14491)

L'objectif de ce projet est double :

1. **Reproduire les résultats** du papier original GAT [1] sur les datasets **Cora** et **Citeseer**, en réimplémentant la couche d'attention à la main.
2. **Implémenter et comparer GATv2** [2], qui identifie une limitation théorique de GAT (l'attention "statique") et propose un correctif minimal pour obtenir une attention "dynamique" strictement plus expressive.

Seul le chargement des datasets utilise une librairie externe (`torch_geometric.datasets.Planetoid`).

---

## Structure du projet

```
GRAPHATTENTIONNETWORKS/
├── README.md                  # ce fichier
├── requirements.txt           # dépendances Python
├── src/
│   ├── __init__.py
│   ├── layers.py              # GATLayer (2018) + GATv2Layer (2022) + MultiHeadGATLayer
│   ├── models.py              # modèle à 2 couches paramétrable (variant='gat' ou 'gatv2')
│   ├── data.py                # chargement Cora/Citeseer (avec normalisation des features)
│   ├── train.py               # boucle d'entraînement avec early stopping
│   └── utils.py               # accuracy, fixation des seeds, formatage
├── experiments/
│   ├── run_cora.py            # reproduit Cora pour GAT et GATv2
│   ├── run_citeseer.py        # reproduit Citeseer pour GAT et GATv2
│   └── run_all.py             # lance les deux datasets et produit 3 tableaux comparatifs
└── results/                   # logs et résultats sauvegardés (créé automatiquement)
```

---

## Installation

### 1. Créer un environnement virtuel (recommandé)

```bash
python3 -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## Comment lancer

### Lancer l'expérience Cora (GAT + GATv2)

```bash
python -m experiments.run_cora
```

### Lancer l'expérience Citeseer (GAT + GATv2)

```bash
python -m experiments.run_citeseer
```

### Lancer les deux datasets et obtenir les tableaux récapitulatifs

```bash
python -m experiments.run_all
```

À la fin, **trois tableaux** s'affichent dans le terminal :

1. **GAT** : nos résultats vs ceux du papier 2018
2. **GATv2** : nos résultats vs ceux du papier 2022
3. **GAT vs GATv2** : comparaison directe de nos résultats

Un fichier `results/summary.txt` est également sauvegardé.

---

## Résultats

Chaque script lance **5 runs avec des seeds différentes** et affiche **moyenne ± écart-type**, comme dans le papier (qui en fait 100, mais 5 suffit pour vérifier l'ordre de grandeur sur une machine étudiante).

### Tableau 1 — GAT (papier 2018)

| Dataset  | Notre implémentation | Papier (GAT) |
|----------|----------------------|--------------|
| Cora     | s'affiche après `run_cora` | 83.0 ± 0.7% |
| Citeseer | s'affiche après `run_citeseer` | 72.5 ± 0.7% |

### Tableau 2 — GATv2 (papier 2022)

| Dataset  | Notre implémentation | Papier (GATv2) |
|----------|----------------------|----------------|
| Cora     | s'affiche après `run_cora` | non rapporté * |
| Citeseer | s'affiche après `run_citeseer` | non rapporté * |

> *\* Le papier GATv2 n'évalue pas explicitement sur Cora/Citeseer. Les auteurs notent eux-mêmes (Section 4.7) que ces datasets sont jugés trop simples pour révéler la différence entre attention statique et dynamique. Sur Pubmed (Annexe D.3), le papier rapporte GAT 78.1% vs GATv2 78.5%.*

### Tableau 3 — Comparaison directe GAT vs GATv2

| Dataset  | GAT (notre impl.) | GATv2 (notre impl.) | Δ (GATv2 − GAT) |
|----------|-------------------|---------------------|------------------|
| Cora     | s'affiche après `run_cora` | s'affiche après `run_cora` | s'affiche |
| Citeseer | s'affiche après `run_citeseer` | s'affiche après `run_citeseer` | s'affiche |

---

---
