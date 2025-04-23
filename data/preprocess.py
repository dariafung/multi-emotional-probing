#!/usr/bin/env python
"""
Filtered GoEmotions Preprocessing Script
----------------------------------------
Keeps examples with at least one of the 10 selected emotions.
For training: up to 2500 examples per emotion.
For validation & test: up to 250 examples per emotion.
"""

import argparse
import os
import re
import string
import random
from typing import List, Optional

import emoji
import pandas as pd
from datasets import load_dataset, Dataset
from tqdm import tqdm

# Cleaning regex
URL_PATTERN   = re.compile(r"http\S+")
USER_PATTERN  = re.compile(r"u\/\w+|@\w+")
SUB_PATTERN   = re.compile(r"r\/\w+")
MULTI_SPACE   = re.compile(r"\s{2,}")

# Target 10 emotions
TARGET_10_LABELS = [
    "neutral", "admiration", "gratitude", "curiosity", "love",
    "anger", "joy", "sadness", "amusement", "confusion"
]

TRAIN_LIMIT = 2500   # max per emotion for training
EVAL_LIMIT  = 250    # max per emotion for validation & test


def clean_text(text: str, max_len: Optional[int] = None) -> str:
    text = URL_PATTERN.sub("", text)
    text = USER_PATTERN.sub("", text)
    text = SUB_PATTERN.sub("", text)
    text = emoji.replace_emoji(text, replace="")
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = MULTI_SPACE.sub(" ", text).strip().lower()
    if max_len is not None:
        text = text[:max_len]
    return text


def to_one_hot(indices: List[int], target_indices: List[int]) -> Optional[List[float]]: # Hint changed
    """Converts label indices to a multi-hot vector of floats."""
    filtered = [target_indices.index(i) for i in indices if i in target_indices]
    if not filtered:
        return None
    vec = [0.0] * len(target_indices) # Initialize with float 0.0
    for idx in filtered:
        vec[idx] = 1.0 # Assign float 1.0
    return vec # Return list of floats

def filter_and_sample(dataset: Dataset,
                      label_names: List[str],
                      limit: int) -> Dataset:
    """Keep only examples with at least one target emotion;
    sample up to `limit` per emotion."""
    target_indices = [label_names.index(em) for em in TARGET_10_LABELS]
    buckets = {i: [] for i in range(len(TARGET_10_LABELS))}

    for ex in dataset:
        one_hot = to_one_hot(ex["labels"], target_indices)
        if one_hot:
            ex["text"] = clean_text(ex["text"])
            ex["labels"] = one_hot
            for i, bit in enumerate(one_hot):
                if bit == 1:
                    buckets[i].append(ex)

    final_data = []
    for i, samples in buckets.items():
        count = len(samples)
        if count <= limit:
            chosen = samples
        else:
            chosen = random.sample(samples, limit)
        final_data.extend(chosen)
        print(f"Emotion {TARGET_10_LABELS[i]:<12} → {len(chosen):4d} samples (out of {count})")

    return Dataset.from_list(final_data)


def preprocess_and_save(dataset: Dataset, name: str, out_dir: str):
    df = pd.DataFrame({
        "text": dataset["text"],
        "labels": dataset["labels"],
    })
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.parquet")
    df.to_parquet(path, index=False)
    print(f"Saved {name:11s} | {len(df):5d} rows -> {path}")


def main():
    parser = argparse.ArgumentParser(description="Filtered GoEmotions Preprocessing")
    parser.add_argument("--out_dir", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--max_len", type=int, default=None, help="Max comment length")
    args = parser.parse_args()

    print("Loading GoEmotions dataset …")
    dataset = load_dataset("go_emotions")
    label_names = dataset["train"].features["labels"].feature.names

    print("Filtering and sampling …")
    for split in ["train", "validation", "test"]:
        print(f"\n--- Processing split: {split} ---")
        limit = TRAIN_LIMIT if split == "train" else EVAL_LIMIT
        filtered = filter_and_sample(dataset[split], label_names, limit)
        preprocess_and_save(filtered, split, args.out_dir)

    print("Finished preprocessing all splits.")


if __name__ == "__main__":
    main()
