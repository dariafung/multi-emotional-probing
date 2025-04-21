#!/usr/bin/env python
"""
Train a linear probe on top of frozen BERT‑large hidden states to predict GoEmotions labels.
Saves the trained probe weights for each layer and reports metrics.

Usage:
    python probe_model/run_probe.py \
        --data_dir data/processed \
        --bert_name bert-large-uncased \
        --layers 0 6 12 18 23 \
        --epochs 3

Outputs:
    probe_model/checkpoints/layer_{L}.pt        # linear weight, bias, F1 on val
    probe_model/metrics.csv                     # per‑layer metrics table
"""

import argparse
import os
from pathlib import Path
import json

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_data(data_dir: str, tokenizer, max_len: int = 128, split: str = "train"):
    df = pd.read_parquet(Path(data_dir) / f"{split}.parquet")
    enc = tokenizer(
        df["text"].tolist(),
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    labels = torch.tensor(df["labels"].tolist(), dtype=torch.float)
    ds = TensorDataset(enc["input_ids"], enc["attention_mask"], labels)
    return ds


def train_probe(hidden_size: int, num_labels: int):
    model = nn.Linear(hidden_size, num_labels)
    return model.to(DEVICE)


def evaluate(probe, feats, labels):
    with torch.no_grad():
        logits = probe(feats)
        preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
        y_true = labels.cpu().numpy()
        p, r, f1, _ = precision_recall_fscore_support(y_true, preds, average="micro", zero_division=0)
    return p, r, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--bert_name", type=str, default="bert-large-uncased")
    parser.add_argument("--layers", type=int, nargs="*", default=[-1])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.bert_name)
    bert = AutoModel.from_pretrained(args.bert_name, output_hidden_states=True).to(DEVICE)
    bert.eval()  # freeze weights

    num_labels = 10
    train_ds = load_data(args.data_dir, tokenizer, split="train")
    val_ds = load_data(args.data_dir, tokenizer, split="validation")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    metrics = []
    ckpt_dir = Path("probe_model/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for layer_idx in args.layers:
        probe = train_probe(bert.config.hidden_size, num_labels)
        optim = torch.optim.AdamW(probe.parameters(), lr=args.lr)
        loss_fn = nn.BCEWithLogitsLoss()

        for epoch in range(args.epochs):
            probe.train()
            for input_ids, attn_mask, labels in train_loader:
                input_ids, attn_mask, labels = input_ids.to(DEVICE), attn_mask.to(DEVICE), labels.to(DEVICE)
                with torch.no_grad():
                    outputs = bert(input_ids, attention_mask=attn_mask)
                    hidden = outputs.hidden_states[layer_idx]  # (B, seq, dim)
                    feats = hidden[:, 0, :]  # [CLS] token
                logits = probe(feats)
                loss = loss_fn(logits, labels)
                loss.backward()
                optim.step()
                optim.zero_grad()

        # validation
        probe.eval()
        all_feats, all_labels = [], []
        with torch.no_grad():
            for input_ids, attn_mask, labels in val_loader:
                input_ids, attn_mask = input_ids.to(DEVICE), attn_mask.to(DEVICE)
                outputs = bert(input_ids, attention_mask=attn_mask)
                hidden = outputs.hidden_states[layer_idx]
                all_feats.append(hidden[:, 0, :])
                all_labels.append(labels)
        feats = torch.cat(all_feats)
        labels = torch.cat(all_labels)
        p, r, f1 = evaluate(probe, feats.to(DEVICE), labels)
        print(f"Layer {layer_idx:2d} | F1: {f1:.4f}")

        # save checkpoint
        torch.save({"state_dict": probe.state_dict(), "f1": f1}, ckpt_dir / f"layer_{layer_idx}.pt")
        metrics.append({"layer": layer_idx, "precision": p, "recall": r, "f1": f1})

    # write metrics table
    pd.DataFrame(metrics).to_csv("probe_model/metrics.csv", index=False)
    print("Finished training probes.")


if __name__ == "__main__":
    main()

