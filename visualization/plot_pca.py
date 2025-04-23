#!/usr/bin/env python
"""
Visualize emotion vectors via PCA.
Each vector set comes from one BERT layer, shape (10, hidden_dim).
Assumes vectors correspond to the TARGET_10_LABELS order.

Saves:
    - visualization/pca_layer_{L}.png
"""

import torch
import matplotlib.pyplot as plt
import os
from pathlib import Path
from sklearn.decomposition import PCA
import json

# Define the target labels consistently
# (Must match the order used during probe training/extraction)
TARGET_10_LABELS = [
    "neutral", "admiration", "gratitude", "curiosity", "love",
    "anger", "joy", "sadness", "amusement", "confusion"
]

VECTOR_DIR = Path("probe_model/vectors")
OUT_DIR = Path("visualization")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Check if summary.json exists
summary_path = VECTOR_DIR / "summary.json"
if not summary_path.exists():
    print(f"Error: Vector summary file not found at {summary_path}")
    print("Please run extract_vectors.py first.")
    exit()

try:
    with open(summary_path) as f:
        summary = json.load(f)
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {summary_path}")
    exit()
except Exception as e:
    print(f"Error reading {summary_path}: {e}")
    exit()

if not summary:
    print("Vector summary file is empty. No vectors to plot.")
    exit()

print(f"Found {len(summary)} vector files to process.")

for name, path_str in summary.items():
    layer_id = name.replace('layer_', '') # Extract layer ID for title
    vec_path = Path(path_str)

    if not vec_path.exists():
        print(f"Warning: Vector file not found for {name} at {vec_path}, skipping.")
        continue

    try:
        # Load the matrix W (10, hidden_dim_probe)
        mat = torch.load(vec_path, map_location="cpu")
        print(f"Processing {name}: Loaded vectors with shape {mat.shape}")

        if mat.shape[0] != len(TARGET_10_LABELS):
            print(f"Warning: Vector matrix {vec_path} has {mat.shape[0]} rows, expected {len(TARGET_10_LABELS)}. Skipping PCA plot.")
            continue

        # Perform PCA
        mat_np = mat.numpy()
        pca = PCA(n_components=2)
        coords = pca.fit_transform(mat_np) # shape: (10, 2)

        # Plotting
        plt.figure(figsize=(8, 7)) # Adjusted size for labels
        scatter = plt.scatter(coords[:, 0], coords[:, 1], alpha=0.8)

        # Annotate with emotion names
        for i, label in enumerate(TARGET_10_LABELS):
            plt.annotate(label, (coords[i, 0], coords[i, 1]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=9)

        plt.title(f"PCA of Emotion Vectors - Layer {layer_id}")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        out_path = OUT_DIR / f"pca_layer_{layer_id}.png"
        plt.savefig(out_path)
        plt.close() # Close the figure to prevent display in non-interactive environments
        print(f"  Saved PCA plot to {out_path}")

    except Exception as e:
        print(f"Error processing {name}: {e}")

print("\nFinished generating PCA plots.")
