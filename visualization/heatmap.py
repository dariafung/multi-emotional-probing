#!/usr/bin/env python
"""
Visualize emotion intensity across BERT layers by injecting a fixed emotion vector.
This script assumes you have the emotion vectors and a decoder/generator.

Since the generator is not implemented, this is a placeholder for your later use.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

# Mock: Load vector (e.g., anger vector from several layers)
# This assumes you already extracted vectors into probe_model/vectors/layer_*.pt

layers = [0, 6, 12, 18, 23]
emotion_idx = 0  # e.g., anger

intensities = []
for layer in layers:
    vecs = torch.load(f"probe_model/vectors/layer_{layer}.pt")
    v = vecs[emotion_idx]  # (hidden_dim,)
    score = torch.norm(v).item()  # Just a placeholder intensity
    intensities.append(score)

# Plot
plt.figure(figsize=(6, 4))
plt.bar([str(l) for l in layers], intensities, color="salmon")
plt.title(f"Mock Emotion Intensity: Index {emotion_idx}")
plt.xlabel("Layer")
plt.ylabel("Vector Norm (placeholder)")
plt.tight_layout()
plt.savefig("visualization/heatmap_mock.png")
plt.show()
print("Saved placeholder heatmap to visualization/heatmap_mock.png")
