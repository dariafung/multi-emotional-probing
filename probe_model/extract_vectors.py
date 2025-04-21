# extract_vectors.py

#!/usr/bin/env python
"""
Visualize emotion vectors via PCA or UMAP.
Each vector set comes from one BERT layer's probe weight matrix, shape (10, hidden_dim).

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

# Ideally get labels from a shared config or define here
target_10_labels = [
    "neutral", "admiration", "gratitude", "curiosity", "love",
    "anger", "joy", "sadness", "amusement", "confusion"
]

summary_path = VECTOR_DIR / "summary.json"
if not summary_path.exists():
     raise FileNotFoundError(f"summary.json not found in {VECTOR_DIR}. Run extract_vectors.py first.")

with open(summary_path) as f:
    summary = json.load(f)

if not summary:
    print("Warning: summary.json is empty. No vectors found.")

for name, path in summary.items():
    if not Path(path).exists():
        print(f"Warning: Vector file {path} for {name} not found. Skipping.")
        continue
    try:
        mat = torch.load(path)  # shape: (10, hidden_dim)
    except Exception as e:
        print(f"Error loading {path}: {e}. Skipping.")
        continue

    if mat.shape[0] != 10:
        print(f"Warning: Expected 10 rows (emotions) in {path}, found {mat.shape[0]}. Skipping plot for {name}.")
        continue

    mat = mat.cpu().numpy() # Ensure it's on CPU and NumPy
    pca = PCA(n_components=2)
    coords = pca.fit_transform(mat)

    plt.figure(figsize=(8, 7)) # Adjusted size for labels
    plt.scatter(coords[:, 0], coords[:, 1])
    for i, label in enumerate(target_10_labels): # Use actual labels
        plt.annotate(label, (coords[i, 0], coords[i, 1]), fontsize=9, textcoords="offset points", xytext=(0,5), ha='center')

    plt.title(f"Emotion Vectors PCA - {name.upper()}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.tight_layout()

    out_path = OUT_DIR / f"pca_{name}.png"
    plt.savefig(out_path)
    plt.close() # Close figure to avoid display/memory issues in loops
    print(f"Saved PCA plot to {out_path}")

print("PCA plotting complete.")
