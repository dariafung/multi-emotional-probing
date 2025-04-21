#!/usr/bin/env python
"""
Visualize emotion vectors via PCA or UMAP.
Each vector set comes from one BERT layer, shape (28, hidden_dim).

Saves:
    - visualization/pca_layer_{L}.png
"""

import torch
import matplotlib.pyplot as plt
import os
from pathlib import Path
from sklearn.decomposition import PCA
import json

VECTOR_DIR = Path("probe_model/vectors")
OUT_DIR = Path("visualization")
OUT_DIR.mkdir(exist_ok=True)

with open(VECTOR_DIR / "summary.json") as f:
    summary = json.load(f)

for name, path in summary.items():
    mat = torch.load(path)  # shape: (28, hidden_dim)
    mat = mat.numpy()
    pca = PCA(n_components=2)
    coords = pca.fit_transform(mat)

    plt.figure(figsize=(6, 5))
    plt.scatter(coords[:, 0], coords[:, 1])
    for i, label in enumerate(range(10)):
        plt.annotate(str(label), coords[i], fontsize=9)

    plt.title(f"Emotion Vectors - {name.upper()}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.tight_layout()

    out_path = OUT_DIR / f"pca_{name}.png"
    plt.savefig(out_path)
    print(f"Saved PCA plot to {out_path}")
