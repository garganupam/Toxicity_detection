#!/usr/bin/env python3
"""
ToxiGAN Toxicity Detection Classifier
======================================

Trains baseline (original Jigsaw only) and augmented (Jigsaw + ToxiGAN)
DistilBERT classifiers, evaluates on held-out test set.

Improvements over v1:
  - Class-weighted loss (handles 90/10 imbalance)
  - tqdm progress bars
  - Split caching (same splits every run without re-downloading)
  - Shared model/dataset/metrics (no code duplication)
  - --quick_test flag for fast iteration

Usage:
    python train_classifier.py
    python train_classifier.py --generated_data ../outputs/data_gen_toxigan.json
    python train_classifier.py --quick_test
    python train_classifier.py --epochs 5 --use_class_weights
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── Ensure detection/ is importable ─────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import DistilBertClassifier
from dataset import (
    ToxicityDataset, download_jigsaw, prepare_binary_labels,
    stratified_split, save_splits, load_splits,
)
from data_cleaning import load_and_clean_generated
from metrics import compute_metrics, print_comparison, save_report

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("toxicity-classifier")

# Optional: tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Train toxicity detection classifier.")
    p.add_argument("--generated_data", type=str, default="data_gen_toxigan.json")
    p.add_argument("--output_dir", type=str, default="outputs/classifier_outputs")
    p.add_argument("--splits_dir", type=str, default="outputs/jigsaw_splits",
                   help="Cache dir for train/val/test splits.")
    p.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    # ── Improvements ────────────────────────────────────────────────────
    p.add_argument("--use_class_weights", action="store_true", default=True,
                   help="Use class-weighted loss to handle imbalance.")
    p.add_argument("--no_class_weights", action="store_true",
                   help="Disable class weights.")
    p.add_argument("--quick_test", action="store_true",
                   help="Cap dataset at 5000 samples for fast iteration.")
    return p.parse_args()


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def get_device(s):
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def compute_class_weights(labels, device):
    """Compute inverse-frequency class weights for CrossEntropyLoss."""
    from collections import Counter
    counts = Counter(labels)
    total = len(labels)
    n_classes = max(counts.keys()) + 1
    weights = torch.zeros(n_classes)
    for cls, count in counts.items():
        weights[cls] = total / (n_classes * count)
    log.info("Class weights: %s", {i: f"{w:.3f}" for i, w in enumerate(weights.tolist())})
    return weights.to(device)


def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    iterator = tqdm(dataloader, desc="  Training", leave=False) if HAS_TQDM else dataloader
    for batch in iterator:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []

    iterator = tqdm(dataloader, desc="  Evaluating", leave=False) if HAS_TQDM else dataloader
    with torch.no_grad():
        for batch in iterator:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    return total_loss / len(all_labels), np.array(all_preds), np.array(all_labels), np.array(all_probs)


def train_model(model, train_loader, val_loader, criterion, args, device, run_name="model"):
    """Full training loop with best-val-loss checkpointing."""
    from transformers import get_linear_schedule_with_warmup

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps,
    )

    best_val_loss = float("inf")
    best_state = None

    log.info("Training '%s' — %d epochs, %d batches/epoch", run_name, args.epochs, len(train_loader))
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        val_loss, val_preds, val_labels, _ = evaluate(model, val_loader, criterion, device)
        val_acc = (val_preds == val_labels).mean()
        val_f1 = compute_metrics(val_labels, val_preds, np.zeros_like(val_preds, dtype=float))["f1"]
        elapsed = time.time() - t0

        log.info(
            "  [%s] Epoch %d/%d — train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f val_f1=%.4f (%.1fs)",
            run_name, epoch + 1, args.epochs, train_loss, train_acc, val_loss, val_acc, val_f1, elapsed,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    model.to(device)
    return model


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    if args.no_class_weights:
        args.use_class_weights = False

    set_seed(args.seed)
    device = get_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Device: %s", device)

    # ── Step 1: Prepare Jigsaw data ─────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 1: Preparing Jigsaw data")
    log.info("=" * 60)

    cached = load_splits(args.splits_dir)
    if cached:
        train_df, val_df, test_df = cached
    else:
        df = download_jigsaw()
        df = prepare_binary_labels(df)
        train_df, val_df, test_df = stratified_split(df, seed=args.seed)
        save_splits(train_df, val_df, test_df, args.splits_dir)

    if args.quick_test:
        log.info("QUICK TEST MODE — capping to 5000 train, 500 val, 500 test")
        train_df = train_df.head(5000)
        val_df = val_df.head(500)
        test_df = test_df.head(500)

    # ── Step 2: Clean generated data ────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 2: Cleaning generated data")
    log.info("=" * 60)
    generated_texts = load_and_clean_generated(args.generated_data)

    aug_rows = pd.DataFrame({"text": generated_texts, "label": [1] * len(generated_texts)})
    augmented_train_df = pd.concat([train_df, aug_rows], ignore_index=True)

    log.info("Training set sizes:")
    log.info("  Baseline:  %d (toxic: %d)", len(train_df), int(train_df["label"].sum()))
    log.info("  Augmented: %d (toxic: %d)", len(augmented_train_df), int(augmented_train_df["label"].sum()))

    # ── Step 3: Build datasets ──────────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 3: Tokenizing")
    log.info("=" * 60)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    baseline_train_ds = ToxicityDataset(train_df["text"], train_df["label"], tokenizer, args.max_length)
    augmented_train_ds = ToxicityDataset(augmented_train_df["text"], augmented_train_df["label"], tokenizer, args.max_length)
    val_ds = ToxicityDataset(val_df["text"], val_df["label"], tokenizer, args.max_length)
    test_ds = ToxicityDataset(test_df["text"], test_df["label"], tokenizer, args.max_length)

    baseline_loader = DataLoader(baseline_train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    augmented_loader = DataLoader(augmented_train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=0)

    # ── Criterion (with optional class weights) ─────────────────────────
    if args.use_class_weights:
        weights = compute_class_weights(train_df["label"].tolist(), device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # ── Step 4: Train BASELINE ──────────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 4: Training BASELINE")
    log.info("=" * 60)
    baseline_model = DistilBertClassifier(args.model_name, dropout=args.dropout).to(device)
    baseline_model = train_model(baseline_model, baseline_loader, val_loader, criterion, args, device, "baseline")
    torch.save(baseline_model.state_dict(), output_dir / "baseline_model.pt")

    # ── Step 5: Train AUGMENTED ─────────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 5: Training AUGMENTED")
    log.info("=" * 60)
    augmented_model = DistilBertClassifier(args.model_name, dropout=args.dropout).to(device)
    augmented_model = train_model(augmented_model, augmented_loader, val_loader, criterion, args, device, "augmented")
    torch.save(augmented_model.state_dict(), output_dir / "augmented_model.pt")

    # ── Step 6: Evaluate ────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 6: Evaluating on test set")
    log.info("=" * 60)
    eval_criterion = nn.CrossEntropyLoss()

    _, b_preds, b_labels, b_probs = evaluate(baseline_model, test_loader, eval_criterion, device)
    baseline_metrics = compute_metrics(b_labels, b_preds, b_probs)

    _, a_preds, a_labels, a_probs = evaluate(augmented_model, test_loader, eval_criterion, device)
    augmented_metrics = compute_metrics(a_labels, a_preds, a_probs)

    # ── Step 7: Report ──────────────────────────────────────────────────
    print_comparison(baseline_metrics, augmented_metrics, "IN-DOMAIN: Jigsaw (Baseline vs Augmented)")

    report = {
        "experiment": "ToxiGAN In-Domain — Baseline vs Augmented",
        "model": args.model_name,
        "epochs": args.epochs,
        "class_weights": args.use_class_weights,
        "generated_samples_used": len(generated_texts),
        "train_size_baseline": len(train_df),
        "train_size_augmented": len(augmented_train_df),
        "test_size": len(test_df),
        "baseline": baseline_metrics,
        "augmented": augmented_metrics,
        "improvement": {
            k: round(augmented_metrics.get(k, 0) - baseline_metrics.get(k, 0), 6)
            for k in ["accuracy", "precision", "recall", "f1", "auc_roc"]
            if baseline_metrics.get(k) is not None and augmented_metrics.get(k) is not None
        },
    }
    save_report(report, str(output_dir / "comparison_report.json"))

    # Save predictions
    preds = {
        "true_labels": b_labels.tolist(),
        "baseline_preds": b_preds.tolist(),
        "baseline_probs": b_probs.tolist(),
        "augmented_preds": a_preds.tolist(),
        "augmented_probs": a_probs.tolist(),
    }
    with open(output_dir / "test_predictions.json", "w") as f:
        json.dump(preds, f)

    log.info("All outputs → %s/", output_dir)
    log.info("Done.")


if __name__ == "__main__":
    main()
