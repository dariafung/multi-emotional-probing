#!/usr/bin/env python
"""
Filtered GoEmotions Preprocessing Script
----------------------------------------
Only keeps examples with at least one of the 10 selected emotions.
For each emotion, keeps up to 2500 examples (random sample if more).
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
URL_PATTERN = re.compile(r"http\S+")
USER_PATTERN = re.compile(r"u\/\w+|@\w+")
SUB_PATTERN = re.compile(r"r\/\w+")
MULTI_SPACE = re.compile(r"\s{2,}")

# Target 10 emotions
target_10_labels = [
    "neutral", "admiration", "gratitude", "curiosity", "love",
    "anger", "joy", "sadness", "amusement", "confusion"
]

EMOTION_LIMIT = 2500  # Max examples per emotion


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


def to_one_hot(indices: List[int], target_indices: List[int]) -> Optional[List[int]]:
    filtered = [target_indices.index(i) for i in indices if i in target_indices]
    if not filtered:
        return None
    vec = [0] * len(target_indices)
    for idx in filtered:
        vec[idx] = 1
    return vec


def filter_and_sample(dataset: Dataset, label_names: List[str]) -> Dataset:
    """Keep only examples with at least one of the target emotions.
    Sample max 2500 per class (based on multi-label inclusion)."""

    target_indices = [label_names.index(em) for em in target_10_labels]
    buckets = {i: [] for i in range(len(target_10_labels))}

    for ex in dataset:
        one_hot = to_one_hot(ex["labels"], target_indices)
        if one_hot:
            ex["text"] = clean_text(ex["text"])
            ex["labels"] = one_hot
            for i, bit in enumerate(one_hot):
                if bit == 1:
                    buckets[i].append(ex)

    # Subsample
    final_data = []
    for i, samples in buckets.items():
        if len(samples) <= EMOTION_LIMIT:
            final_data.extend(samples)
        else:
            final_data.extend(random.sample(samples, EMOTION_LIMIT))
        print(f"Emotion {target_10_labels[i]:<12} → {min(len(samples), EMOTION_LIMIT):4d} samples")

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
        filtered = filter_and_sample(dataset[split], label_names)
        preprocess_and_save(filtered, split, args.out_dir)

    print("Finished preprocessing all splits.")


if __name__ == "__main__":
    main()
