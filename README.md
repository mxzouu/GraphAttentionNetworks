# Graph Attention Networks (GAT)

Implémentation **from scratch en PyTorch** du papier :

> **Graph Attention Networks** — Veličković, Cucurull, Casanova, Romero, Liò, Bengio (ICLR 2018)
> [arXiv:1710.10903](https://arxiv.org/abs/1710.10903)

L'objectif de ce projet est de reproduire les résultats du papier sur les datasets **Cora** et **Citeseer**, en réimplémentant la couche GAT à la main. Seul le chargement des datasets utilise une librairie externe.

---

## Structure du projet

```
GRAPHATTENTIONNETWORKS/
├── README.md                  # ce fichier
├── requirements.txt           # dépendances Python
├── src/
│   ├── __init__.py
│   ├── layers.py              # GATLayer + MultiHeadGATLayer (le cœur du papier)
│   ├── models.py              # modèle GAT à 2 couches
│   ├── data.py                # chargement Cora/Citeseer
│   ├── train.py               # boucle d'entraînement avec early stopping
│   └── utils.py               # accuracy, fixation des seeds
├── experiments/
│   ├── run_cora.py            # reproduit le résultat Cora (83.0 ± 0.7%)
│   ├── run_citeseer.py        # reproduit le résultat Citeseer (72.5 ± 0.7%)
│   └── run_all.py             # lance les deux et affiche un tableau récapitulatif
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

### Lancer l'expérience Cora 

```bash
python -m experiments.run_cora
```

### Lancer l'expérience Citeseer

```bash
python -m experiments.run_citeseer
```

### Lancer les deux d'un coup avec tableau comparatif

```bash
python -m experiments.run_all
```

À la fin, un tableau s'affiche dans le terminal **en comparant nos résultats à ceux du papier**, et un fichier `results/summary.txt` est sauvegardé.

---

## Résultats attendus (résultats attendus du papier inscrits dans le tableau)

| Dataset  | Papier (GAT)     | Notre implémentation (5 runs) |
|----------|------------------|-------------------------------|
| Cora     | 83.0 ± 0.7%      | s'affiche après `run_cora`    |
| Citeseer | 72.5 ± 0.7%      | s'affiche après `run_citeseer`|

Chaque script lance **5 runs avec des seeds différentes** et affiche **moyenne ± écart-type**, comme dans le papier (qui en fait 100, mais 5 suffit pour vérifier l'ordre de grandeur sur une machine étudiante).

---

## Détails de l'implémentation

### Choix d'architecture (suivant le papier, Section 3.3)

**Cora / Citeseer** : 2 couches GAT
- Couche 1 : 8 têtes × 8 features = 64 features cachées, activation ELU
- Couche 2 : 1 tête × C features (C = nombre de classes), softmax pour la classification

**Hyperparamètres** :
- Optimizer : Adam, lr = 0.005
- L2 weight decay : 5e-4
- Dropout : p = 0.6 (sur les inputs ET sur les coefficients d'attention)
- Early stopping : patience = 100 époques sur la val accuracy
- Initialisation : Glorot (Xavier)

### Le cœur du modèle (`src/layers.py`)

La couche GAT implémente exactement l'équation (3) du papier :

α_ij = softmax_j ( LeakyReLU( aᵀ [W·h_i ‖ W·h_j] ) )

puis l'agrégation (équation 4) :

h'_i = σ ( Σ_j α_ij · W·h_j )

avec masquage explicite des paires (i, j) qui ne sont pas connectées dans le graphe (par mise à -∞ avant le softmax), et dropout sur les α normalisés.

---

