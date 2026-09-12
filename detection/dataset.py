"""
Shared dataset class and Jigsaw data preparation utilities.

Used by both train_classifier.py and test_cross_dataset.py.
"""

import json
import logging
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# PyTorch Dataset
# ═══════════════════════════════════════════════════════════════════════════

class ToxicityDataset(Dataset):
    """Text + label dataset that tokenizes on-the-fly."""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Jigsaw Data Loading
# ═══════════════════════════════════════════════════════════════════════════

def download_jigsaw():
    """Download Jigsaw Toxic Comment dataset and return as DataFrame."""
    log.info("Loading Jigsaw Toxic Comment dataset from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset("Arsive/toxicity_classification_jigsaw", split="train")
    df = pd.DataFrame(ds)
    log.info("Loaded %d samples.", len(df))
    return df


def prepare_binary_labels(df):
    """Convert multi-label Jigsaw data to binary: toxic (1) vs non-toxic (0)."""
    toxic_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    toxic_cols = [c for c in toxic_cols if c in df.columns]

    df = df.copy()
    df["label"] = (df[toxic_cols].sum(axis=1) > 0).astype(int)
    df["text"] = df["comment_text"].astype(str)
    df = df[df["text"].str.strip().str.len() > 0].reset_index(drop=True)

    dist = df["label"].value_counts().to_dict()
    log.info("Binary distribution — Non-toxic: %d | Toxic: %d (%.1f%%)",
             dist.get(0, 0), dist.get(1, 0), 100 * dist.get(1, 0) / len(df))
    return df[["text", "label"]]


def stratified_split(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """Split DataFrame into train/val/test with stratification."""
    from sklearn.model_selection import train_test_split

    train_df, temp_df = train_test_split(
        df, test_size=(val_ratio + test_ratio), stratify=df["label"], random_state=seed,
    )
    relative_test = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test, stratify=temp_df["label"], random_state=seed,
    )

    log.info("Split — train: %d, val: %d, test: %d", len(train_df), len(val_df), len(test_df))
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def save_splits(train_df, val_df, test_df, path):
    """Cache train/val/test splits to disk so they're identical across runs."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(p / "train.parquet", index=False)
    val_df.to_parquet(p / "val.parquet", index=False)
    test_df.to_parquet(p / "test.parquet", index=False)
    log.info("Splits cached to %s/", p)


def load_splits(path):
    """Load cached splits. Returns (train, val, test) or None if not found."""
    p = Path(path)
    files = [p / "train.parquet", p / "val.parquet", p / "test.parquet"]
    if all(f.exists() for f in files):
        train_df = pd.read_parquet(files[0])
        val_df = pd.read_parquet(files[1])
        test_df = pd.read_parquet(files[2])
        log.info("Loaded cached splits from %s/ (train=%d, val=%d, test=%d)",
                 p, len(train_df), len(val_df), len(test_df))
        return train_df, val_df, test_df
    return None


# ═══════════════════════════════════════════════════════════════════════════
# External Datasets (for cross-dataset evaluation)
# ═══════════════════════════════════════════════════════════════════════════

def load_hatexplain():
    """Load HateXplain → binary. hatespeech/offensive → toxic(1), normal → non-toxic(0)."""
    log.info("Loading HateXplain...")
    from datasets import load_dataset
    from collections import Counter

    ds = load_dataset("hatexplain", trust_remote_code=True)
    texts, labels = [], []

    for split_name in ["train", "validation", "test"]:
        if split_name not in ds:
            continue
        for ex in ds[split_name]:
            if "post_tokens" not in ex:
                continue
            text = " ".join(ex["post_tokens"])
            if not text.strip():
                continue

            if "annotators" in ex:
                annot_labels = [a["label"] for a in ex["annotators"]]
                majority = Counter(annot_labels).most_common(1)[0][0]
            elif "label" in ex:
                majority = ex["label"]
            else:
                continue

            labels.append(0 if majority == 0 else 1)
            texts.append(text)

    dist = Counter(labels)
    log.info("HateXplain: %d samples — Non-toxic: %d, Toxic: %d", len(texts), dist[0], dist[1])
    return texts, labels


def load_davidson():
    """Load Davidson → binary. hate/offensive → toxic(1), neither → non-toxic(0)."""
    log.info("Loading Davidson...")
    from datasets import load_dataset
    from collections import Counter

    ds = load_dataset("hate_speech_offensive", trust_remote_code=True, split="train")
    texts, labels = [], []

    for ex in ds:
        text = ex.get("tweet", "")
        label = ex.get("class", -1)
        if not text.strip() or label == -1:
            continue
        labels.append(0 if label == 2 else 1)
        texts.append(text)

    dist = Counter(labels)
    log.info("Davidson: %d samples — Non-toxic: %d, Toxic: %d", len(texts), dist[0], dist[1])
    return texts, labels


EXTERNAL_DATASETS = {
    "hatexplain": load_hatexplain,
    "davidson": load_davidson,
}
