#!/usr/bin/env python3
"""
Download Jigsaw Toxic Comment dataset → per-class .txt files.

Auto-detects project root. Saves to <root>/data/raw/.

Usage:
    python download_data.py
    python download_data.py --max_neutral 5000
"""

import argparse
import os
import sys
from pathlib import Path


def find_project_root():
    """Walk up from this file to find root (contains generation/ or configs/)."""
    p = Path(__file__).resolve().parent
    for parent in [p] + list(p.parents):
        if (parent / "generation").is_dir() or (parent / "configs").is_dir():
            return parent
    return p.parent


def main():
    parser = argparse.ArgumentParser(description="Download Jigsaw dataset.")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_neutral", type=int, default=5000)
    args = parser.parse_args()

    root = find_project_root()
    out_dir = Path(args.output_dir) if args.output_dir else root / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project root: {root}")
    print(f"Saving to:    {out_dir}")
    print("Downloading Jigsaw dataset...")

    try:
        from datasets import load_dataset
        import pandas as pd
    except ImportError:
        print("ERROR: pip install datasets pandas")
        sys.exit(1)

    ds = load_dataset("Arsive/toxicity_classification_jigsaw", split="train")
    df = pd.DataFrame(ds)
    print(f"Total rows: {len(df)}")

    # Toxic classes
    for col in ["toxic", "obscene", "insult", "identity_hate"]:
        subset = df[df[col] == 1]["comment_text"].dropna()
        path = out_dir / f"{col}.txt"
        subset.to_csv(path, index=False, header=False)
        print(f"  Saved {len(subset):>6} rows → {col}.txt")

    # Neutral
    all_toxic_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    avail = [c for c in all_toxic_cols if c in df.columns]
    neutral = df[df[avail].sum(axis=1) == 0]["comment_text"].dropna()
    if args.max_neutral and len(neutral) > args.max_neutral:
        neutral = neutral.sample(n=args.max_neutral, random_state=42)
    path = out_dir / "nor.txt"
    neutral.to_csv(path, index=False, header=False)
    print(f"  Saved {len(neutral):>6} rows → nor.txt")

    print(f"\nDone! Files in {out_dir}/:")
    for f in sorted(out_dir.glob("*.txt")):
        lines = sum(1 for _ in open(f, encoding="utf-8", errors="ignore"))
        print(f"  {f.name}: {lines} lines")


if __name__ == "__main__":
    main()
