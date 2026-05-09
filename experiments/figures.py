"""
experiments/figures.py
======================

Génère toutes les figures du rapport dans le dossier `results/figures/`.

Usage :
    python -m experiments.figures

Produit cinq figures :
    1. barchart.png      : comparaison synthétique GAT/GATv2 vs papier
    2. boxplot.png       : distribution des 5 runs par configuration
    3. curves.png        : courbes d'apprentissage (val accuracy par époque)
    4. tsne.png          : t-SNE des embeddings cachés appris (Cora)
    5. attention.png     : heatmap des coefficients d'attention pour 5 noeuds

Sur Colab GPU le script tourne en environ 5 minutes.
"""

from copy import deepcopy
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE

from src.data import load_dataset, GraphData
from src.models import GAT
from src.utils import set_seed, accuracy


FIGDIR = Path(__file__).resolve().parent.parent / "results" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)


# Palette de couleurs cohérente pour tout le rapport
COLOR_GAT = "#E07A5F"        # orange
COLOR_GATV2 = "#3D5A80"      # bleu
COLOR_PAPER = "#81B29A"      # vert
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 120,
    "savefig.bbox": "tight",
})


# ─────────────────────────────────────────────────────────────────────────────
#  Entraînement avec historique des métriques par époque
# ─────────────────────────────────────────────────────────────────────────────

def train_with_history(data: GraphData, variant: str, device: torch.device,
                       max_epochs: int = 300, patience: int = 100,
                       seed: int = 0):
    """Comme train_one_run mais retourne aussi l'historique val_acc par époque."""
    set_seed(seed)
    data = data.to(device)

    model = GAT(
        in_features=data.num_features,
        hidden_features=8, num_classes=data.num_classes,
        num_heads=8, num_out_heads=1,
        dropout=0.6, alpha=0.2, variant=variant,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = nn.NLLLoss()

    history = {"train_loss": [], "train_acc": [], "val_acc": []}
    best_val = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.adj)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.adj)
            tr_acc = accuracy(out[data.train_mask], data.y[data.train_mask])
            v_acc = accuracy(out[data.val_mask], data.y[data.val_mask])

        history["train_loss"].append(loss.item())
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(v_acc)

        if v_acc > best_val:
            best_val = v_acc
            best_state = deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.adj)
        test_acc = accuracy(out[data.test_mask], data.y[data.test_mask])
    return model, history, test_acc


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 1 — Barchart comparatif
# ─────────────────────────────────────────────────────────────────────────────

def figure_barchart(results: dict):
    """Bar chart : Nos GAT, Nos GATv2, Papier GAT — sur Cora et Citeseer."""
    datasets = ["Cora", "Citeseer"]
    paper = [83.0, 72.5]
    paper_err = [0.7, 0.7]
    our_gat = [results[ds]["gat"]["mean"] for ds in datasets]
    our_gat_err = [results[ds]["gat"]["std"] for ds in datasets]
    our_gatv2 = [results[ds]["gatv2"]["mean"] for ds in datasets]
    our_gatv2_err = [results[ds]["gatv2"]["std"] for ds in datasets]

    x = np.arange(len(datasets))
    width = 0.27
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(x - width, our_gat, width, yerr=our_gat_err,
           label="Notre GAT", color=COLOR_GAT, capsize=4, edgecolor="black", linewidth=0.6)
    ax.bar(x, our_gatv2, width, yerr=our_gatv2_err,
           label="Notre GATv2", color=COLOR_GATV2, capsize=4, edgecolor="black", linewidth=0.6)
    ax.bar(x + width, paper, width, yerr=paper_err,
           label="Papier GAT (2018)", color=COLOR_PAPER, capsize=4, edgecolor="black", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Comparaison des accuracies sur Cora et Citeseer")
    ax.set_ylim(60, 90)
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.savefig(FIGDIR / "barchart.png", dpi=150)
    plt.close()
    print(f"  ✓ barchart.png")


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 2 — Boxplot des runs individuels
# ─────────────────────────────────────────────────────────────────────────────

def figure_boxplot(results: dict):
    """Boxplot : 5 runs pour chaque config, montre la stabilité."""
    data_to_plot = [
        results["Cora"]["gat"]["accs"],
        results["Cora"]["gatv2"]["accs"],
        results["Citeseer"]["gat"]["accs"],
        results["Citeseer"]["gatv2"]["accs"],
    ]
    labels = ["GAT\nCora", "GATv2\nCora", "GAT\nCiteseer", "GATv2\nCiteseer"]
    colors = [COLOR_GAT, COLOR_GATV2, COLOR_GAT, COLOR_GATV2]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bp = ax.boxplot(data_to_plot, patch_artist=True, labels=labels,
                    medianprops=dict(color="black", linewidth=1.5),
                    boxprops=dict(linewidth=0.8))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.65)

    # superposer les points individuels
    for i, run in enumerate(data_to_plot, start=1):
        ax.scatter([i] * len(run), run, color="black", alpha=0.6, s=18, zorder=3)

    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Distribution des 5 runs par configuration")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.savefig(FIGDIR / "boxplot.png", dpi=150)
    plt.close()
    print(f"  ✓ boxplot.png")


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 3 — Courbes d'apprentissage
# ─────────────────────────────────────────────────────────────────────────────

def figure_curves(history_gat, history_gatv2):
    """Courbes : val_acc par époque pour GAT et GATv2 sur Cora."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Train loss
    axes[0].plot(history_gat["train_loss"], label="GAT", color=COLOR_GAT, linewidth=1.4)
    axes[0].plot(history_gatv2["train_loss"], label="GATv2", color=COLOR_GATV2, linewidth=1.4)
    axes[0].set_xlabel("Époque")
    axes[0].set_ylabel("Train loss (NLL)")
    axes[0].set_title("Évolution de la loss d'entraînement")
    axes[0].legend()
    axes[0].grid(linestyle="--", alpha=0.5)

    # Val accuracy
    axes[1].plot(np.array(history_gat["val_acc"]) * 100, label="GAT",
                 color=COLOR_GAT, linewidth=1.4)
    axes[1].plot(np.array(history_gatv2["val_acc"]) * 100, label="GATv2",
                 color=COLOR_GATV2, linewidth=1.4)
    axes[1].set_xlabel("Époque")
    axes[1].set_ylabel("Validation accuracy (%)")
    axes[1].set_title("Évolution de la validation accuracy")
    axes[1].legend()
    axes[1].grid(linestyle="--", alpha=0.5)

    plt.suptitle("Cora — Dynamique d'entraînement (run avec seed = 0)", fontsize=12)
    plt.savefig(FIGDIR / "curves.png", dpi=150)
    plt.close()
    print(f"  ✓ curves.png")


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 4 — t-SNE des embeddings cachés
# ─────────────────────────────────────────────────────────────────────────────

def figure_tsne(model, data, device):
    """t-SNE 2D des embeddings de la couche 1 (Cora). Couleurs = classes."""
    data = data.to(device)
    model.eval()
    with torch.no_grad():
        # Forward partiel jusqu'à la sortie de la couche 1
        h = model.layer1(data.x, data.adj)
    embeddings = h.cpu().numpy()
    labels = data.y.cpu().numpy()

    print("    (calcul t-SNE en cours, environ 30 secondes...)")
    proj = TSNE(n_components=2, perplexity=30, init="pca",
                learning_rate="auto", random_state=42).fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    cmap = plt.get_cmap("tab10")
    class_names = [f"Classe {c}" for c in range(int(labels.max()) + 1)]
    for c in range(int(labels.max()) + 1):
        mask = labels == c
        ax.scatter(proj[mask, 0], proj[mask, 1],
                   color=cmap(c), s=10, alpha=0.7, label=class_names[c])
    ax.set_title("Projection t-SNE des embeddings cachés (GAT, Cora)")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.legend(loc="best", markerscale=2, ncol=2, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(FIGDIR / "tsne.png", dpi=150)
    plt.close()
    print(f"  ✓ tsne.png")


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 5 — Heatmap des coefficients d'attention (GAT vs GATv2)
# ─────────────────────────────────────────────────────────────────────────────

def compute_attention_gat(model, data):
    """Récupère les α de la première tête de la première couche d'un modèle GAT."""
    head = model.layer1.heads[0]
    with torch.no_grad():
        z = head.W(data.x)
        attn_src = (z @ head.a_src).squeeze(-1)
        attn_dst = (z @ head.a_dst).squeeze(-1)
        e = head.leakyrelu(attn_src.unsqueeze(1) + attn_dst.unsqueeze(0))
        e = e.masked_fill(data.adj <= 0, float("-inf"))
        alpha = F.softmax(e, dim=1)
    return alpha.cpu().numpy()


def compute_attention_gatv2(model, data):
    """Récupère les α de la première tête de la première couche d'un modèle GATv2."""
    head = model.layer1.heads[0]
    with torch.no_grad():
        z_self = head.W_self(data.x)
        z_nbr = head.W_neighbor(data.x)
        pre = z_self.unsqueeze(1) + z_nbr.unsqueeze(0)
        pre = head.leakyrelu(pre)
        e = (pre @ head.a).squeeze(-1)
        e = e.masked_fill(data.adj <= 0, float("-inf"))
        alpha = F.softmax(e, dim=1)
    return alpha.cpu().numpy()


def figure_attention(model_gat, model_gatv2, data, device):
    """Heatmap : pour 5 noeuds choisis, les poids α sur leurs voisins."""
    data = data.to(device)
    alpha_gat = compute_attention_gat(model_gat, data)
    alpha_gatv2 = compute_attention_gatv2(model_gatv2, data)

    # Choisir 5 noeuds avec un voisinage de taille raisonnable (entre 8 et 15)
    adj_np = data.adj.cpu().numpy()
    degrees = adj_np.sum(axis=1)
    candidates = np.where((degrees >= 8) & (degrees <= 15))[0]
    rng = np.random.default_rng(seed=42)
    selected = rng.choice(candidates, size=5, replace=False)

    # Pour chaque noeud, récupérer ses voisins et les α associés
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for ax, alpha, title, _color in [
        (axes[0], alpha_gat, "GAT — attention statique", COLOR_GAT),
        (axes[1], alpha_gatv2, "GATv2 — attention dynamique", COLOR_GATV2),
    ]:
        # Construire une matrice (5, max_deg) avec les α sur les voisins
        neighbor_lists = []
        for n in selected:
            nbrs = np.where(adj_np[n] > 0)[0]
            neighbor_lists.append(nbrs)
        max_deg = max(len(n) for n in neighbor_lists)

        mat = np.full((5, max_deg), np.nan)
        for i, n in enumerate(selected):
            nbrs = neighbor_lists[i]
            mat[i, :len(nbrs)] = alpha[n, nbrs]

        im = ax.imshow(mat, cmap="viridis", aspect="auto", vmin=0, vmax=mat[~np.isnan(mat)].max())
        ax.set_yticks(range(5))
        ax.set_yticklabels([f"Noeud {n}" for n in selected])
        ax.set_xlabel("Voisins (indexés localement)")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label="α (poids d'attention)")

    plt.suptitle("Distribution des coefficients d'attention sur Cora", fontsize=12)
    plt.savefig(FIGDIR / "attention.png", dpi=150)
    plt.close()
    print(f"  ✓ attention.png")


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Sortie : {FIGDIR}\n")

    # ---- 1. Charger les datasets et entraîner les modèles ----
    print("[1/3] Chargement des datasets et entraînements...")
    cora = load_dataset("Cora")
    citeseer = load_dataset("Citeseer")

    # Pour les figures 1 et 2, on a besoin des 5 runs de chaque config.
    # On les relance ici (5 minutes sur GPU) pour ne pas dépendre d'un fichier externe.
    print("  Entraînement Cora GAT (5 runs)...")
    cora_gat_accs = []
    for s in range(5):
        _, _, acc = train_with_history(cora, "gat", device, seed=s)
        cora_gat_accs.append(acc * 100)
    print(f"    {[f'{a:.1f}' for a in cora_gat_accs]}")

    print("  Entraînement Cora GATv2 (5 runs)...")
    cora_gatv2_accs = []
    for s in range(5):
        _, _, acc = train_with_history(cora, "gatv2", device, seed=s)
        cora_gatv2_accs.append(acc * 100)
    print(f"    {[f'{a:.1f}' for a in cora_gatv2_accs]}")

    print("  Entraînement Citeseer GAT (5 runs)...")
    cite_gat_accs = []
    for s in range(5):
        _, _, acc = train_with_history(citeseer, "gat", device, seed=s)
        cite_gat_accs.append(acc * 100)
    print(f"    {[f'{a:.1f}' for a in cite_gat_accs]}")

    print("  Entraînement Citeseer GATv2 (5 runs)...")
    cite_gatv2_accs = []
    for s in range(5):
        _, _, acc = train_with_history(citeseer, "gatv2", device, seed=s)
        cite_gatv2_accs.append(acc * 100)
    print(f"    {[f'{a:.1f}' for a in cite_gatv2_accs]}")

    results = {
        "Cora": {
            "gat": {"accs": cora_gat_accs, "mean": np.mean(cora_gat_accs), "std": np.std(cora_gat_accs)},
            "gatv2": {"accs": cora_gatv2_accs, "mean": np.mean(cora_gatv2_accs), "std": np.std(cora_gatv2_accs)},
        },
        "Citeseer": {
            "gat": {"accs": cite_gat_accs, "mean": np.mean(cite_gat_accs), "std": np.std(cite_gat_accs)},
            "gatv2": {"accs": cite_gatv2_accs, "mean": np.mean(cite_gatv2_accs), "std": np.std(cite_gatv2_accs)},
        },
    }

    # ---- 2. Modèles entraînés sur Cora (un seul run avec historique) pour les figures 3, 4, 5 ----
    print("\n[2/3] Entraînement final pour figures détaillées (Cora)...")
    model_gat, hist_gat, _ = train_with_history(cora, "gat", device, seed=0)
    model_gatv2, hist_gatv2, _ = train_with_history(cora, "gatv2", device, seed=0)

    # ---- 3. Génération des figures ----
    print("\n[3/3] Génération des figures...")
    figure_barchart(results)
    figure_boxplot(results)
    figure_curves(hist_gat, hist_gatv2)
    figure_tsne(model_gat, cora, device)
    figure_attention(model_gat, model_gatv2, cora, device)

    print(f"\n✅ Toutes les figures sont dans : {FIGDIR}")


if __name__ == "__main__":
    main()
