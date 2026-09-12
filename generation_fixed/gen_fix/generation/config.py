# config.py — ToxiGAN generation config
# Auto-detects project root so it works from any working directory.

import argparse
import os
import sys


def _find_project_root():
    """Walk up from this file to find the project root."""
    here = os.path.dirname(os.path.abspath(__file__))
    # If we're in generation/, go up one level
    parent = os.path.abspath(os.path.join(here, ".."))
    if os.path.isdir(os.path.join(parent, "data")):
        return parent
    # If we're already at root
    if os.path.isdir(os.path.join(here, "data")):
        return here
    # Fallback
    return parent


def parse_opt():
    parser = argparse.ArgumentParser()

    # ── Paths ──────────────────────────────────────────────────────
    ROOT = _find_project_root()

    # Data: look for data/raw/ first, then data/
    if os.path.isdir(os.path.join(ROOT, "data", "raw")):
        data_dir = os.path.join(ROOT, "data", "raw")
    else:
        data_dir = os.path.join(ROOT, "data")

    # Artifacts: vocab, .id files, checkpoints, generated JSON
    save_dir = os.path.join(ROOT, "artifacts")

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    parser.add_argument('--project_root', type=str, default=ROOT)
    parser.add_argument('--data_dir', type=str, default=data_dir)
    parser.add_argument('--save_path', type=str, default=save_dir)

    # ── Generator hyperparameters ──────────────────────────────────
    parser.add_argument('--PRETRAIN_EPOCHS_G',  type=int,   default=350)
    parser.add_argument('--EMB_DIM',            type=int,   default=300)
    parser.add_argument('--HIDDEN_DIM',         type=int,   default=1024)
    parser.add_argument('--MAX_SEQ_LENGTH',     type=int,   default=20)
    parser.add_argument('--BATCH_SIZE',         type=int,   default=128)
    parser.add_argument('--TOTAL_BATCH',        type=int,   default=40)
    parser.add_argument('--START_TOKEN',        type=int,   default=1)
    parser.add_argument('--ROLL_OUT_NUM',       type=int,   default=16)
    parser.add_argument('--learning_rate_g',    type=float, default=0.0001)

    # ── Discriminator hyperparameters ──────────────────────────────
    parser.add_argument('--PRETRAIN_EPOCHS_D',      type=int,   default=30)
    parser.add_argument('--dis_embedding_dim',      type=int,   default=300)
    parser.add_argument('--dis_filter_sizes',       type=list,  default=[1,2,3,4,5,6,7,8,9,10,15])
    parser.add_argument('--dis_num_filters',        type=list,  default=[100,200,200,200,200,100,100,100,100,100,160])
    parser.add_argument('--dis_dropout_keep_prob',  type=float, default=0.5)
    parser.add_argument('--dis_l2_reg_lambda',      type=float, default=0.2)
    parser.add_argument('--dis_batch_size',         type=int,   default=128)
    parser.add_argument('--learning_rate_d',        type=float, default=0.0001)
    parser.add_argument('--DIS_UPDATES_PER_ROUND',  type=int,   default=4)

    # ── TOXIC_CLASSES ─────────────────────────────────────────────
    parser.add_argument('--TOXIC_CLASSES', type=dict, default={
        'nor':           os.path.join(data_dir, 'nor.txt'),
        'toxic':         os.path.join(data_dir, 'toxic.txt'),
        'obscene':       os.path.join(data_dir, 'obscene.txt'),
        'insult':        os.path.join(data_dir, 'insult.txt'),
        'identity_hate': os.path.join(data_dir, 'identity_hate.txt'),
    })

    # ── SENTIMENT_CLASSES (same as TOXIC_CLASSES) ─────────────────
    parser.add_argument('--SENTIMENT_CLASSES', type=dict, default={
        'nor':           os.path.join(data_dir, 'nor.txt'),
        'toxic':         os.path.join(data_dir, 'toxic.txt'),
        'obscene':       os.path.join(data_dir, 'obscene.txt'),
        'insult':        os.path.join(data_dir, 'insult.txt'),
        'identity_hate': os.path.join(data_dir, 'identity_hate.txt'),
    })

    # ── Generation ────────────────────────────────────────────────
    parser.add_argument('--gen_num', type=int, default=10000)

    # ── Penalty ───────────────────────────────────────────────────
    parser.add_argument('--use_penalty', type=bool, default=True)

    # ── Seed ──────────────────────────────────────────────────────
    parser.add_argument('--SEED', type=int, default=42)

    opt = parser.parse_args([])

    # ── Validate data files exist ─────────────────────────────────
    missing = []
    for tag, path in opt.TOXIC_CLASSES.items():
        if not os.path.isfile(path):
            missing.append(f"  {tag}: {path}")
    if missing:
        print(f"[config] WARNING: These data files are missing:")
        for m in missing:
            print(m)
        print(f"[config] Run: cd scripts && python download_data.py")
        print(f"[config] Data dir: {data_dir}")
        print(f"[config] Save dir: {save_dir}")

    return opt


if __name__ == "__main__":
    opt = parse_opt()
    print(f"Project root : {opt.project_root}")
    print(f"Data dir     : {opt.data_dir}")
    print(f"Save path    : {opt.save_path}")
    print(f"Classes      : {list(opt.TOXIC_CLASSES.keys())}")
    for tag, path in opt.TOXIC_CLASSES.items():
        exists = "✓" if os.path.isfile(path) else "✗ MISSING"
        print(f"  {tag}: {exists} → {path}")
