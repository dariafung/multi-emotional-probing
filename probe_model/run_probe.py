#!/usr/bin/env python
"""
Train an MLP probe on top of frozen BERT‑large hidden states (using mean pooling)
to predict GoEmotions labels. Saves the trained probe weights for each layer
and reports metrics.

Usage:
    python probe_model/run_probe.py \
        --data_dir data/processed \
        --bert_name bert-large-uncased \
        --layers 12 18 23 \
        --epochs 5 \
        --lr 3e-4 \
        --use_mlp \
        --pooling_strategy mean

Outputs:
    probe_model/checkpoints/layer_{L}.pt        # MLP state_dict, F1 on val
    probe_model/metrics.csv                     # per‑layer metrics table
"""

import argparse
import os
from pathlib import Path
import json
import numpy as np # For loss averaging

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_data(data_dir: str, tokenizer, max_len: int = 128, split: str = "train"):
    """Loads data, tokenizes, and creates TensorDataset."""
    df = pd.read_parquet(Path(data_dir) / f"{split}.parquet")
    print(f"Loading {split} data from {Path(data_dir) / f'{split}.parquet'}, {len(df)} samples.")
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


def create_probe_model(input_size: int, num_labels: int, use_mlp: bool = False,
                      probe_hidden_dim: int = 512, probe_dropout: float = 0.1):
    """Creates either a Linear or an MLP probe."""
    if use_mlp:
        print(f"Creating MLP probe: Linear({input_size}, {probe_hidden_dim}) -> ReLU -> Dropout({probe_dropout}) -> Linear({probe_hidden_dim}, {num_labels})")
        model = nn.Sequential(
            nn.Linear(input_size, probe_hidden_dim),
            nn.ReLU(),
            nn.Dropout(probe_dropout),
            nn.Linear(probe_hidden_dim, num_labels)
        )
    else:
        print(f"Creating Linear probe: Linear({input_size}, {num_labels})")
        model = nn.Linear(input_size, num_labels)
    return model.to(DEVICE)


def extract_features(hidden_states, attention_mask, strategy='cls'):
    """Extracts features using the specified pooling strategy."""
    if strategy == 'cls':
        # Use the [CLS] token embedding
        return hidden_states[:, 0, :]
    elif strategy == 'mean':
        # Mean pooling: average embeddings of non-padding tokens
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9) # Avoid division by zero
        return sum_embeddings / sum_mask
    # Add other strategies like 'max' if needed
    else:
        raise ValueError(f"Unknown pooling strategy: {strategy}. Choose 'cls' or 'mean'.")


def evaluate(probe, data_loader, bert_model, layer_idx, pooling_strategy):
    """Evaluates the probe on the given data loader."""
    probe.eval() # Set probe to evaluation mode
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for input_ids, attn_mask, labels in data_loader:
            input_ids, attn_mask = input_ids.to(DEVICE), attn_mask.to(DEVICE)

            # Get hidden states from BERT
            bert_outputs = bert_model(input_ids, attention_mask=attn_mask)
            hidden = bert_outputs.hidden_states[layer_idx] # (B, seq, dim)

            # Extract features using the specified strategy
            feats = extract_features(hidden, attn_mask, strategy=pooling_strategy).to(DEVICE)

            # Get logits from probe
            logits = probe(feats)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu()) # Keep labels on CPU

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Calculate metrics
    preds = (torch.sigmoid(all_logits) > 0.5).numpy()
    y_true = all_labels.numpy()
    p, r, f1, _ = precision_recall_fscore_support(y_true, preds, average="micro", zero_division=0)

    return p, r, f1


def main():
    parser = argparse.ArgumentParser(description="Train Emotion Probe on BERT Hidden States")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Directory containing processed parquet files.")
    parser.add_argument("--bert_name", type=str, default="bert-large-uncased", help="Name of the BERT model.")
    parser.add_argument("--layers", type=int, nargs="+", required=True, help="List of BERT layer indices to probe (0-based).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and evaluation.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for the probe.")
    parser.add_argument("--max_len", type=int, default=128, help="Maximum sequence length for tokenizer.")
    parser.add_argument("--pooling_strategy", type=str, default="mean", choices=['cls', 'mean'], help="Pooling strategy ('cls' or 'mean').")
    parser.add_argument("--use_mlp", action='store_true', help="Use an MLP probe instead of a linear one.")
    parser.add_argument("--probe_hidden_dim", type=int, default=512, help="Hidden dimension for MLP probe.")
    parser.add_argument("--probe_dropout", type=float, default=0.1, help="Dropout probability for MLP probe.")
    parser.add_argument("--log_steps", type=int, default=50, help="Log training loss every N steps.")
    args = parser.parse_args()

    print("Arguments:", args)
    print(f"Using device: {DEVICE}")

    print(f"Loading tokenizer: {args.bert_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.bert_name)

    print(f"Loading BERT model: {args.bert_name}")
    # Ensure output_hidden_states=True is set
    bert_config = AutoModel.from_pretrained(args.bert_name).config # Load config first
    bert_config.output_hidden_states = True
    bert = AutoModel.from_pretrained(args.bert_name, config=bert_config).to(DEVICE)
    bert.eval()  # Freeze BERT weights
    for param in bert.parameters():
        param.requires_grad = False
    print("BERT model loaded and frozen.")

    num_labels = 10 # Defined by GoEmotions subset
    bert_hidden_size = bert.config.hidden_size

    print("Loading datasets...")
    train_ds = load_data(args.data_dir, tokenizer, max_len=args.max_len, split="train")
    val_ds = load_data(args.data_dir, tokenizer, max_len=args.max_len, split="validation")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    print("Dataloaders created.")

    metrics_log = []
    ckpt_dir = Path("probe_model/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for layer_idx in args.layers:
        print(f"\n--- Training Probe for Layer {layer_idx} ---")

        # Create the probe model (Linear or MLP)
        probe = create_probe_model(
            input_size=bert_hidden_size,
            num_labels=num_labels,
            use_mlp=args.use_mlp,
            probe_hidden_dim=args.probe_hidden_dim,
            probe_dropout=args.probe_dropout
        )

        optim = torch.optim.AdamW(probe.parameters(), lr=args.lr)
        loss_fn = nn.BCEWithLogitsLoss() # Suitable for multi-label

        best_val_f1 = -1.0 # Track best F1 on validation set for this layer

        for epoch in range(args.epochs):
            probe.train() # Set probe to training mode
            total_loss = 0.0
            batch_count = 0

            for step, (input_ids, attn_mask, labels) in enumerate(train_loader):
                input_ids, attn_mask, labels = input_ids.to(DEVICE), attn_mask.to(DEVICE), labels.to(DEVICE)

                # Get hidden states from frozen BERT
                with torch.no_grad():
                    outputs = bert(input_ids, attention_mask=attn_mask)
                    hidden = outputs.hidden_states[layer_idx]  # (B, seq, dim)

                    # Extract features using the specified strategy
                    feats = extract_features(hidden, attn_mask, strategy=args.pooling_strategy).to(DEVICE)

                # Forward pass through probe
                logits = probe(feats)
                loss = loss_fn(logits, labels)

                # Backward pass and optimization
                optim.zero_grad()
                loss.backward()
                optim.step()

                total_loss += loss.item()
                batch_count += 1

                # Log training loss periodically
                if (step + 1) % args.log_steps == 0:
                    print(f"  Epoch [{epoch+1}/{args.epochs}], Step [{step+1}/{len(train_loader)}], Avg Batch Loss: {total_loss/batch_count:.4f}")

            avg_epoch_loss = total_loss / batch_count
            print(f"--- Epoch {epoch+1} Completed ---")
            print(f"  Average Training Loss: {avg_epoch_loss:.4f}")

            # --- Validation after each epoch ---
            val_p, val_r, val_f1 = evaluate(probe, val_loader, bert, layer_idx, args.pooling_strategy)
            print(f"  Validation F1: {val_f1:.4f} (P: {val_p:.4f}, R: {val_r:.4f})")

            # Save the best model checkpoint based on validation F1
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                save_path = ckpt_dir / f"layer_{layer_idx}.pt"
                # Save the probe state dict and the best F1 score
                torch.save({
                    "state_dict": probe.state_dict(),
                    "f1": best_val_f1,
                    "epoch": epoch + 1,
                    "pooling": args.pooling_strategy,
                    "use_mlp": args.use_mlp
                }, save_path)
                print(f"  * New best F1! Checkpoint saved to {save_path}")

        # Log metrics for the best model of this layer
        print(f"--- Layer {layer_idx} Finished Training ---")
        print(f"  Best Validation F1 achieved: {best_val_f1:.4f}")
        # Reload best model to ensure metrics are from the best epoch
        best_ckpt = torch.load(ckpt_dir / f"layer_{layer_idx}.pt")
        probe.load_state_dict(best_ckpt['state_dict'])
        # Re-evaluate the best model on the validation set to get its precise P, R, F1
        final_p, final_r, final_f1 = evaluate(probe, val_loader, bert, layer_idx, args.pooling_strategy)
        metrics_log.append({
            "layer": layer_idx,
            "precision": final_p,
            "recall": final_r,
            "f1": final_f1,
            "pooling": args.pooling_strategy,
             "use_mlp": args.use_mlp
        })


    # Write metrics table
    metrics_df = pd.DataFrame(metrics_log)
    metrics_filename = f"probe_model/metrics_{args.pooling_strategy}{'_mlp' if args.use_mlp else ''}.csv"
    metrics_df.to_csv(metrics_filename, index=False)
    print(f"\nFinished training probes for all specified layers.")
    print(f"Metrics saved to {metrics_filename}")
    print(metrics_df)

    # --- IMPORTANT NOTE FOR MLP ---
    if args.use_mlp:
        print("\n*** NOTE: MLP probe was used. ***")
        print("The saved checkpoints contain the state_dict of the entire MLP.")
        print("The 'extract_vectors.py' script needs modification if you want to extract ")
        print("meaningful 'direction vectors' (e.g., using only the weights of the *last* linear layer of the MLP).")
        print("Directly using the full MLP state_dict is not standard for vector injection.")
    # --- END NOTE ---

if __name__ == "__main__":
    main()
