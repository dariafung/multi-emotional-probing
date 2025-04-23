#!/usr/bin/env python
"""
Visualize emotion vector similarity within each layer using cosine similarity heatmaps.
Reads vector files listed in probe_model/vectors/summary.json.

Saves:
    - visualization/cosine_heatmap_layer_{L}.png
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Define the target labels consistently
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
            print(f"Warning: Vector matrix {vec_path} has {mat.shape[0]} rows, expected {len(TARGET_10_LABELS)}. Skipping heatmap.")
            continue

        # Calculate cosine similarity matrix
        mat_np = mat.numpy()
        # Normalize vectors before cosine similarity (optional but good practice)
        # norms = np.linalg.norm(mat_np, axis=1, keepdims=True)
        # mat_normalized = mat_np / norms
        # sim_matrix = cosine_similarity(mat_normalized) # shape: (10, 10)
        # Or directly compute:
        sim_matrix = cosine_similarity(mat_np) # shape: (10, 10)


        # Plotting the heatmap
        plt.figure(figsize=(10, 8)) # Adjusted size for labels
        sns.heatmap(
            sim_matrix,
            annot=True,           # Show similarity values
            fmt=".2f",            # Format values to 2 decimal places
            cmap="viridis",       # Color map (others: 'coolwarm', 'YlGnBu', etc.)
            xticklabels=TARGET_10_LABELS,
            yticklabels=TARGET_10_LABELS,
            linewidths=.5,        # Add lines between cells
            cbar=True             # Show color bar
        )
        plt.title(f"Cosine Similarity Between Emotion Vectors - Layer {layer_id}", fontsize=14)
        plt.xticks(rotation=45, ha='right') # Rotate x-axis labels for better readability
        plt.yticks(rotation=0)
        plt.tight_layout() # Adjust layout to prevent labels overlapping

        out_path = OUT_DIR / f"cosine_heatmap_layer_{layer_id}.png"
        plt.savefig(out_path)
        plt.close() # Close the figure
        print(f"  Saved cosine similarity heatmap to {out_path}")

    except Exception as e:
        print(f"Error processing {name}: {e}")

print("\nFinished generating cosine similarity heatmaps.")
