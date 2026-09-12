#!/usr/bin/env python3
"""
Cross-Dataset Evaluation: Jigsaw-trained model → HateXplain / Davidson
=======================================================================

Loads saved baseline + augmented models from train_classifier.py and
evaluates on an external dataset. No retraining required.

Usage:
    python test_cross_dataset.py
    python test_cross_dataset.py --dataset davidson
    python test_cross_dataset.py --model_dir ../outputs/classifier_outputs
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import DistilBertClassifier
from dataset import ToxicityDataset, EXTERNAL_DATASETS
from metrics import compute_metrics, print_comparison, save_report

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("cross-dataset-eval")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable


def parse_args():
    p = argparse.ArgumentParser(description="Cross-dataset evaluation.")
    p.add_argument("--model_dir", type=str, default="../outputs/classifier_outputs")
    p.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--dataset", type=str, default="hatexplain",
                   choices=list(EXTERNAL_DATASETS.keys()))
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--output_dir", type=str, default="../outputs/classifier_outputs")
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    iterator = tqdm(dataloader, desc="  Evaluating", leave=False) if HAS_TQDM else dataloader
    with torch.no_grad():
        for batch in iterator:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def main():
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load models ─────────────────────────────────────────────────────
    baseline_path = model_dir / "baseline_model.pt"
    augmented_path = model_dir / "augmented_model.pt"

    for p in [baseline_path, augmented_path]:
        if not p.exists():
            log.error("Checkpoint not found: %s. Run train_classifier.py first.", p)
            sys.exit(1)

    log.info("Loading models from %s", model_dir)
    baseline_model = DistilBertClassifier(args.model_name, dropout=args.dropout).to(device)
    baseline_model.load_state_dict(torch.load(baseline_path, map_location=device, weights_only=True))

    augmented_model = DistilBertClassifier(args.model_name, dropout=args.dropout).to(device)
    augmented_model.load_state_dict(torch.load(augmented_path, map_location=device, weights_only=True))

    # ── Load external dataset ───────────────────────────────────────────
    log.info("Loading dataset: %s", args.dataset)
    texts, labels = EXTERNAL_DATASETS[args.dataset]()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    ds = ToxicityDataset(texts, labels, tokenizer, args.max_length)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=0)

    # ── Evaluate ────────────────────────────────────────────────────────
    log.info("Evaluating baseline on %s...", args.dataset)
    b_preds, b_labels, b_probs = evaluate_model(baseline_model, loader, device)
    baseline_metrics = compute_metrics(b_labels, b_preds, b_probs)

    log.info("Evaluating augmented on %s...", args.dataset)
    a_preds, a_labels, a_probs = evaluate_model(augmented_model, loader, device)
    augmented_metrics = compute_metrics(a_labels, a_preds, a_probs)

    # ── Report ──────────────────────────────────────────────────────────
    title = f"CROSS-DATASET: Jigsaw-trained → {args.dataset.upper()}"
    print_comparison(baseline_metrics, augmented_metrics, title)

    report = {
        "experiment": f"Cross-dataset: Jigsaw → {args.dataset}",
        "evaluation_data": args.dataset,
        "evaluation_samples": len(texts),
        "baseline": baseline_metrics,
        "augmented": augmented_metrics,
        "improvement": {
            k: round(augmented_metrics.get(k, 0) - baseline_metrics.get(k, 0), 6)
            for k in ["accuracy", "precision", "recall", "f1", "auc_roc"]
            if baseline_metrics.get(k) is not None and augmented_metrics.get(k) is not None
        },
    }
    save_report(report, str(output_dir / f"cross_dataset_{args.dataset}_report.json"))

    f1_delta = report["improvement"].get("f1", 0)
    verdict = "OUTPERFORMS" if f1_delta > 0 else "UNDERPERFORMS" if f1_delta < 0 else "MATCHES"
    log.info("Verdict: Augmented %s baseline on %s (F1 %+.4f)", verdict, args.dataset, f1_delta)


if __name__ == "__main__":
    main()
