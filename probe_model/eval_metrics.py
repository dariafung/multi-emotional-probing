#!/usr/bin/env python
"""
Plot F1 scores of trained probe layers.
Run after probe_model/run_probe.py has finished.

This script reads:
    probe_model/metrics.csv

And displays:
    - Bar plot of F1 scores by BERT layer
    - Optional save to file
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

METRICS_PATH = "probe_model/metrics.csv"
SAVE_PLOT = False  # Set to True to save as PNG

# Load the metrics table
if not os.path.exists(METRICS_PATH):
    raise FileNotFoundError("No metrics.csv found. Run run_probe.py first.")

df = pd.read_csv(METRICS_PATH).sort_values("layer")

# Plot
plt.figure(figsize=(8, 5))
plt.bar(df["layer"].astype(str), df["f1"], color="skyblue")
plt.title("F1 Score by BERT Layer")
plt.xlabel("Layer")
plt.ylabel("F1 Score")
plt.ylim(0, 1.0)
plt.grid(axis="y")
plt.tight_layout()

if SAVE_PLOT:
    plt.savefig("probe_model/f1_plot.png")
    print("Saved plot to probe_model/f1_plot.png")
else:
    plt.show()
