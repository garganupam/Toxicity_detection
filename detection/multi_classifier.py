#!/usr/bin/env python3
"""
Multi-Classifier Toxicity Detection — Final Version
=====================================================

Philosophy: Use real data AS-IS. No resampling, no capping, no tricks.
If the dataset has natural class imbalance (and Jigsaw does: ~90% non-toxic),
then ToxiGAN-generated toxic samples help fill that gap.

Pipeline:
  1. Load full Jigsaw (~160K rows, ~13% toxic) — untouched
  2. Clean ToxiGAN generated samples
  3. Augmented set = original + cleaned generated (all labeled toxic)
  4. Train 3 classifiers × 2 settings (baseline / augmented)
  5. Evaluate in-domain + cross-dataset

Usage:
    python multi_classifier.py --generated_data ../artifacts/data_gen_toxigan.json
    python multi_classifier.py --quick_test
    python multi_classifier.py --skip_cross_dataset
"""

import argparse, json, logging, os, sys, time, warnings, re
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("multi-clf")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kw): return iterable


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--generated_data", default="data_gen_toxigan.json")
    p.add_argument("--output_dir", default="outputs/multi_classifier")
    p.add_argument("--splits_dir", default="outputs/jigsaw_full_splits")
    p.add_argument("--classifiers", nargs="+", default=["distilbert", "bilstm", "tfidf_lr"])
    p.add_argument("--bert_model", default="distilbert-base-uncased")
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lstm_epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--quick_test", action="store_true")
    p.add_argument("--skip_cross_dataset", action="store_true")
    return p.parse_args()

def set_seed(s):
    import random; random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def get_device(s):
    return torch.device("cuda" if s == "auto" and torch.cuda.is_available() else "cpu" if s == "auto" else s)


# ═══════════════════════════════════════════════════════════════════════
# DATA — REAL DATASET, NO MODIFICATIONS
# ═══════════════════════════════════════════════════════════════════════

def download_jigsaw():
    """
    Load the FULL Jigsaw dataset as-is. No resampling. No capping.
    
    Tries OxAISH-AL-LLM/wiki_toxic (~160K, binary labels, ~13% toxic).
    Falls back to Arsive version (~26K) if unavailable.
    """
    from datasets import load_dataset

    # Primary: full Jigsaw with natural imbalance
    try:
        log.info("Downloading full Jigsaw (OxAISH-AL-LLM/wiki_toxic)...")
        ds = load_dataset("OxAISH-AL-LLM/wiki_toxic", split="train")
        df = pd.DataFrame(ds)
        if "label" in df.columns and "comment_text" in df.columns:
            df["text"] = df["comment_text"].astype(str)
            df = df[df["text"].str.strip().str.len() > 0].reset_index(drop=True)
            _log_distribution(df, "Full Jigsaw")
            return df[["text", "label"]]
    except Exception as e:
        log.warning("Primary source failed: %s", e)

    # Fallback
    log.info("Downloading Arsive/toxicity_classification_jigsaw (fallback)...")
    ds = load_dataset("Arsive/toxicity_classification_jigsaw", split="train")
    df = pd.DataFrame(ds)
    toxic_cols = [c for c in ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"] if c in df.columns]
    df["label"] = (df[toxic_cols].sum(axis=1) > 0).astype(int)
    df["text"] = df["comment_text"].astype(str)
    df = df[df["text"].str.strip().str.len() > 0].reset_index(drop=True)
    _log_distribution(df, "Arsive Jigsaw")
    return df[["text", "label"]]


def _log_distribution(df, name):
    d = df["label"].value_counts().to_dict()
    total = len(df)
    log.info("  %s: %d total", name, total)
    log.info("    Non-toxic (0): %6d  (%.1f%%)", d.get(0, 0), 100 * d.get(0, 0) / total)
    log.info("    Toxic     (1): %6d  (%.1f%%)", d.get(1, 0), 100 * d.get(1, 0) / total)
    ratio = d.get(0, 0) / max(d.get(1, 0), 1)
    if ratio > 3:
        log.info("    → Imbalanced (%.1f:1 non-toxic:toxic). ToxiGAN augmentation will help.", ratio)
    else:
        log.info("    → Roughly balanced. Augmentation may have limited impact.")


def stratified_split(df, seed=42):
    from sklearn.model_selection import train_test_split
    train_df, temp = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=seed)
    val_df, test_df = train_test_split(temp, test_size=0.5, stratify=temp["label"], random_state=seed)
    log.info("  Splits: train=%d  val=%d  test=%d", len(train_df), len(val_df), len(test_df))
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

def save_splits(tr, va, te, p):
    Path(p).mkdir(parents=True, exist_ok=True)
    tr.to_parquet(f"{p}/train.parquet", index=False)
    va.to_parquet(f"{p}/val.parquet", index=False)
    te.to_parquet(f"{p}/test.parquet", index=False)

def load_splits(p):
    fs = [f"{p}/{s}.parquet" for s in ["train", "val", "test"]]
    if all(os.path.exists(f) for f in fs):
        log.info("Loading cached splits from %s/", p)
        return [pd.read_parquet(f) for f in fs]
    return None


def load_and_clean_generated(json_path):
    """Clean ToxiGAN output: remove short, repetitive, UNK-heavy, gibberish."""
    if not Path(json_path).exists():
        log.warning("Generated data not found: %s", json_path)
        return []
    with open(json_path) as f:
        data = json.load(f)
    tweets = data.get("tweet", [])
    log.info("Raw generated samples: %d", len(tweets))

    cleaned = []
    reasons = Counter()
    for text in tweets:
        text = text.strip()
        if not text: reasons["empty"] += 1; continue
        words = text.split()
        if len(words) <= 2: reasons["too_short"] += 1; continue
        if sum(1 for w in words if w.lower() in ("<unk>", "unk")) / len(words) > 0.3:
            reasons["high_unk"] += 1; continue
        wc = Counter(w.lower() for w in words)
        if wc.most_common(1)[0][1] / len(words) > 0.5: reasons["repetitive"] += 1; continue
        if sum(1 for w in words if re.search(r"[a-zA-Z]", w)) / len(words) < 0.5:
            reasons["non_alpha"] += 1; continue
        cw = [w for w in words if w.lower() not in ("<unk>", "unk")]
        if len(cw) <= 2: reasons["short_after"] += 1; continue
        cleaned.append(" ".join(cw))

    log.info("After cleaning: %d kept (%.1f%%). Removed: %s",
             len(cleaned), 100 * len(cleaned) / max(len(tweets), 1), dict(reasons))
    return cleaned


# ═══════════════════════════════════════════════════════════════════════
# EXTERNAL DATASETS (cross-dataset eval)
# ═══════════════════════════════════════════════════════════════════════

def load_hatexplain():
    log.info("Loading HateXplain...")
    from datasets import load_dataset
    ds = load_dataset("hatexplain", trust_remote_code=True)
    texts, labels = [], []
    for split in ["train", "validation", "test"]:
        if split not in ds: continue
        for ex in ds[split]:
            if "post_tokens" not in ex: continue
            text = " ".join(ex["post_tokens"])
            if not text.strip(): continue
            if "annotators" in ex:
                majority = Counter(a["label"] for a in ex["annotators"]).most_common(1)[0][0]
            elif "label" in ex: majority = ex["label"]
            else: continue
            labels.append(0 if majority == 0 else 1)
            texts.append(text)
    log.info("  HateXplain: %d samples", len(texts))
    return texts, labels

def load_davidson():
    log.info("Loading Davidson...")
    from datasets import load_dataset
    ds = load_dataset("hate_speech_offensive", trust_remote_code=True, split="train")
    texts, labels = [], []
    for ex in ds:
        text = ex.get("tweet", ""); label = ex.get("class", -1)
        if not text.strip() or label == -1: continue
        labels.append(0 if label == 2 else 1)
        texts.append(text)
    log.info("  Davidson: %d samples", len(texts))
    return texts, labels


# ═══════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, y_prob=None):
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score, confusion_matrix)
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_prob is not None:
        try: m["auc_roc"] = float(roc_auc_score(y_true, y_prob))
        except: m["auc_roc"] = None
    else: m["auc_roc"] = None
    return m


# ═══════════════════════════════════════════════════════════════════════
# CLASSIFIER 1: DistilBERT
# ═══════════════════════════════════════════════════════════════════════

class BertDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = list(texts); self.labels = list(labels)
        self.tokenizer = tokenizer; self.max_length = max_length
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, padding="max_length",
                             max_length=self.max_length, return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "label": torch.tensor(self.labels[idx], dtype=torch.long)}

class DistilBertClassifier(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", dropout=0.3):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(model_name)
        h = self.bert.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(h, h // 2),
                                  nn.ReLU(), nn.Dropout(dropout), nn.Linear(h // 2, 2))
    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.head(out.last_hidden_state[:, 0, :])

def train_distilbert(train_texts, train_labels, val_texts, val_labels, args, device, name="bert"):
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    train_loader = DataLoader(BertDataset(train_texts, train_labels, tokenizer, args.max_length),
                              batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(BertDataset(val_texts, val_labels, tokenizer, args.max_length),
                            batch_size=args.batch_size, num_workers=0)

    counts = Counter(train_labels); total = len(train_labels)
    weights = torch.tensor([total / (2 * counts[0]), total / (2 * counts[1])], dtype=torch.float).to(device)
    log.info("  [%s] Class weights: non-toxic=%.3f, toxic=%.3f", name, weights[0].item(), weights[1].item())
    criterion = nn.CrossEntropyLoss(weight=weights)

    model = DistilBertClassifier(args.bert_model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    best_vl = float("inf"); best_st = None
    for epoch in range(args.epochs):
        model.train(); t0 = time.time()
        for batch in tqdm(train_loader, desc=f"  [{name}] Ep {epoch+1}", leave=False):
            ids, mask, labs = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["label"].to(device)
            optimizer.zero_grad(); loss = criterion(model(ids, mask), labs); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); scheduler.step()

        model.eval(); vl = 0; vp = []; vl_l = []
        with torch.no_grad():
            for batch in val_loader:
                ids, mask, labs = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["label"].to(device)
                logits = model(ids, mask); vl += criterion(logits, labs).item() * labs.size(0)
                vp.extend(logits.argmax(1).cpu().numpy()); vl_l.extend(labs.cpu().numpy())
        vl /= len(vl_l)
        from sklearn.metrics import f1_score
        log.info("  [%s] Ep %d/%d — val_loss=%.4f f1=%.4f (%.1fs)",
                 name, epoch+1, args.epochs, vl, f1_score(vl_l, vp, zero_division=0), time.time() - t0)
        if vl < best_vl: best_vl = vl; best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_st: model.load_state_dict(best_st); model.to(device)
    return model, tokenizer

def predict_distilbert(model, tokenizer, texts, labels, args, device):
    loader = DataLoader(BertDataset(texts, labels, tokenizer, args.max_length),
                        batch_size=args.batch_size, num_workers=0)
    model.eval(); preds = []; probs = []; all_l = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            p = torch.softmax(logits, dim=1)
            preds.extend(logits.argmax(1).cpu().numpy())
            probs.extend(p[:, 1].cpu().numpy())
            all_l.extend(batch["label"].numpy())
    return np.array(preds), np.array(all_l), np.array(probs)


# ═══════════════════════════════════════════════════════════════════════
# CLASSIFIER 2: BiLSTM
# ═══════════════════════════════════════════════════════════════════════

class LSTMDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=200):
        self.labels = list(labels); self.max_len = max_len
        self.encoded = [[vocab.get(w, 1) for w in t.lower().split()[:max_len]] for t in texts]
    def __len__(self): return len(self.encoded)
    def __getitem__(self, idx):
        ids = self.encoded[idx]; padded = ids + [0] * (self.max_len - len(ids))
        return (torch.tensor(padded, dtype=torch.long), torch.tensor(len(ids), dtype=torch.long),
                torch.tensor(self.labels[idx], dtype=torch.long))

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=2, dropout=dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim),
                                  nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 2))
    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(self.embedding(x), lengths.cpu().clamp(min=1),
                                                    batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.head(torch.cat([h[-2], h[-1]], dim=1))

def build_vocab(texts, min_freq=2):
    counter = Counter()
    for t in texts: counter.update(t.lower().split())
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for w, f in counter.most_common(50000):
        if f >= min_freq: vocab[w] = len(vocab)
    return vocab

def train_bilstm(train_texts, train_labels, val_texts, val_labels, args, device, name="bilstm"):
    vocab = build_vocab(train_texts)
    log.info("  [%s] Vocab: %d tokens", name, len(vocab))
    train_loader = DataLoader(LSTMDataset(train_texts, train_labels, vocab), batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(LSTMDataset(val_texts, val_labels, vocab), batch_size=64, num_workers=0)

    counts = Counter(train_labels); total = len(train_labels)
    weights = torch.tensor([total / (2 * counts[0]), total / (2 * counts[1])], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    model = BiLSTMClassifier(len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_vl = float("inf"); best_st = None
    for epoch in range(args.lstm_epochs):
        model.train()
        for ids, lengths, labels in tqdm(train_loader, desc=f"  [{name}] Ep {epoch+1}", leave=False):
            ids, lengths, labels = ids.to(device), lengths.to(device), labels.to(device)
            optimizer.zero_grad(); criterion(model(ids, lengths), labels).backward(); optimizer.step()
        model.eval(); vl = 0; n = 0; vp = []; vl_l = []
        with torch.no_grad():
            for ids, lengths, labels in val_loader:
                ids, lengths, labels = ids.to(device), lengths.to(device), labels.to(device)
                logits = model(ids, lengths); vl += criterion(logits, labels).item() * labels.size(0); n += labels.size(0)
                vp.extend(logits.argmax(1).cpu().numpy()); vl_l.extend(labels.cpu().numpy())
        vl /= n
        if (epoch + 1) % 2 == 0 or epoch == 0:
            log.info("  [%s] Ep %d/%d — val_loss=%.4f acc=%.4f", name, epoch+1, args.lstm_epochs, vl,
                     np.mean(np.array(vp) == np.array(vl_l)))
        if vl < best_vl: best_vl = vl; best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_st: model.load_state_dict(best_st); model.to(device)
    return model, vocab

def predict_bilstm(model, vocab, texts, labels, device):
    loader = DataLoader(LSTMDataset(texts, labels, vocab), batch_size=64, num_workers=0)
    model.eval(); preds = []; probs = []; all_l = []
    with torch.no_grad():
        for ids, lengths, labs in loader:
            logits = model(ids.to(device), lengths.to(device))
            preds.extend(logits.argmax(1).cpu().numpy())
            probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            all_l.extend(labs.numpy())
    return np.array(preds), np.array(all_l), np.array(probs)


# ═══════════════════════════════════════════════════════════════════════
# CLASSIFIER 3: TF-IDF + Logistic Regression
# ═══════════════════════════════════════════════════════════════════════

def train_tfidf_lr(train_texts, train_labels, name="tfidf"):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    log.info("  [%s] Training TF-IDF + LR...", name)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", solver="liblinear")),
    ])
    pipe.fit(train_texts, train_labels)
    return pipe

def predict_tfidf_lr(pipe, texts, labels):
    return np.array(pipe.predict(texts)), np.array(labels), np.array(pipe.predict_proba(texts)[:, 1])


# ═══════════════════════════════════════════════════════════════════════
# PRETTY PRINTING
# ═══════════════════════════════════════════════════════════════════════

def print_table(results, title):
    w = 90
    print(f"\n{'=' * w}\n  {title}\n{'=' * w}")
    print(f"  {'Classifier':<15} {'Setting':<12} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'AUC':>8}")
    print(f"{'-' * w}")
    for cn, data in results.items():
        for s in ["baseline", "augmented"]:
            if s not in data: continue
            m = data[s]
            auc = f"{m['auc_roc']:.4f}" if m.get('auc_roc') else "  N/A"
            print(f"  {cn:<15} {s:<12} {m['accuracy']:>8.4f} {m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} {auc:>8}")
        if "baseline" in data and "augmented" in data:
            b, a = data["baseline"], data["augmented"]
            for k in ["f1", "recall"]:
                d = a[k] - b[k]
            df1 = a["f1"] - b["f1"]; dr = a["recall"] - b["recall"]
            print(f"  {'':<15} {'Δ':<12} {'':>8} {'':>8} {'+' if dr >= 0 else ''}{dr:>7.4f} {'+' if df1 >= 0 else ''}{df1:>7.4f} {'':>8}")
        print(f"{'-' * w}")
    print(f"{'=' * w}")

def print_cross(results, title):
    w = 90
    print(f"\n{'=' * w}\n  {title}\n{'=' * w}")
    print(f"  {'Classifier':<15} {'Dataset':<15} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'AUC':>8}")
    print(f"{'-' * w}")
    for key, m in results.items():
        auc = f"{m['auc_roc']:.4f}" if m.get('auc_roc') else "  N/A"
        c, d = key.split("_on_")
        print(f"  {c:<15} {d:<15} {m['accuracy']:>8.4f} {m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} {auc:>8}")
    print(f"{'=' * w}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Device: %s", device)

    # ── Step 1: Load real data (NO modifications) ─────────────────────
    log.info("=" * 60)
    log.info("STEP 1: Loading real dataset (no modifications)")
    log.info("=" * 60)

    cached = load_splits(args.splits_dir)
    if cached:
        train_df, val_df, test_df = cached
    else:
        df = download_jigsaw()
        train_df, val_df, test_df = stratified_split(df, args.seed)
        save_splits(train_df, val_df, test_df, args.splits_dir)

    if args.quick_test:
        log.info("QUICK TEST — capping data")
        train_df = train_df.head(5000); val_df = val_df.head(500); test_df = test_df.head(500)

    # ── Step 2: Clean ToxiGAN generated data ──────────────────────────
    log.info("=" * 60)
    log.info("STEP 2: Cleaning ToxiGAN generated data")
    log.info("=" * 60)

    gen_texts = load_and_clean_generated(args.generated_data)
    aug_df = pd.concat([train_df, pd.DataFrame({"text": gen_texts, "label": [1] * len(gen_texts)})], ignore_index=True)

    # Show augmentation impact
    base_toxic = int(train_df["label"].sum())
    base_nontoxic = len(train_df) - base_toxic
    aug_toxic = int(aug_df["label"].sum())

    log.info("")
    log.info("  ┌──────────────────────────────────────────────────────┐")
    log.info("  │  AUGMENTATION IMPACT                                 │")
    log.info("  ├──────────────┬────────────┬────────────┬─────────────┤")
    log.info("  │              │ Non-toxic  │ Toxic      │ Toxic %%     │")
    log.info("  ├──────────────┼────────────┼────────────┼─────────────┤")
    log.info("  │ Original     │ %10d │ %10d │ %9.1f%%  │", base_nontoxic, base_toxic, 100 * base_toxic / len(train_df))
    log.info("  │ + ToxiGAN    │ %10d │ %10d │ %9.1f%%  │", base_nontoxic, aug_toxic, 100 * aug_toxic / len(aug_df))
    log.info("  │ Δ            │          0 │ +%9d │ %+8.1f%%  │", len(gen_texts), 100 * aug_toxic / len(aug_df) - 100 * base_toxic / len(train_df))
    log.info("  └──────────────┴────────────┴────────────┴─────────────┘")
    log.info("")

    # Extract lists
    tr_t, tr_l = train_df["text"].tolist(), train_df["label"].tolist()
    au_t, au_l = aug_df["text"].tolist(), aug_df["label"].tolist()
    va_t, va_l = val_df["text"].tolist(), val_df["label"].tolist()
    te_t, te_l = test_df["text"].tolist(), test_df["label"].tolist()

    all_r = {}; models = {}

    # ── Step 3: Train classifiers ─────────────────────────────────────
    if "distilbert" in args.classifiers:
        log.info("\n" + "=" * 60 + "\n  DistilBERT\n" + "=" * 60)
        log.info("Baseline:")
        mb, tok = train_distilbert(tr_t, tr_l, va_t, va_l, args, device, "bert-base")
        p, l, pr = predict_distilbert(mb, tok, te_t, te_l, args, device); bm = compute_metrics(l, p, pr)
        log.info("Augmented:")
        ma, _ = train_distilbert(au_t, au_l, va_t, va_l, args, device, "bert-aug")
        p, l, pr = predict_distilbert(ma, tok, te_t, te_l, args, device); am = compute_metrics(l, p, pr)
        all_r["DistilBERT"] = {"baseline": bm, "augmented": am}
        models["DistilBERT"] = {"model": ma, "tokenizer": tok, "type": "bert"}
        torch.save(mb.state_dict(), output_dir / "distilbert_baseline.pt")
        torch.save(ma.state_dict(), output_dir / "distilbert_augmented.pt")

    if "bilstm" in args.classifiers:
        log.info("\n" + "=" * 60 + "\n  BiLSTM\n" + "=" * 60)
        log.info("Baseline:")
        mb, vb = train_bilstm(tr_t, tr_l, va_t, va_l, args, device, "lstm-base")
        p, l, pr = predict_bilstm(mb, vb, te_t, te_l, device); bm = compute_metrics(l, p, pr)
        log.info("Augmented:")
        ma, va = train_bilstm(au_t, au_l, va_t, va_l, args, device, "lstm-aug")
        p, l, pr = predict_bilstm(ma, va, te_t, te_l, device); am = compute_metrics(l, p, pr)
        all_r["BiLSTM"] = {"baseline": bm, "augmented": am}
        models["BiLSTM"] = {"model": ma, "vocab": va, "type": "lstm"}

    if "tfidf_lr" in args.classifiers:
        log.info("\n" + "=" * 60 + "\n  TF-IDF + Logistic Regression\n" + "=" * 60)
        log.info("Baseline:")
        pb = train_tfidf_lr(tr_t, tr_l, "tfidf-base")
        p, l, pr = predict_tfidf_lr(pb, te_t, te_l); bm = compute_metrics(l, p, pr)
        log.info("Augmented:")
        pa = train_tfidf_lr(au_t, au_l, "tfidf-aug")
        p, l, pr = predict_tfidf_lr(pa, te_t, te_l); am = compute_metrics(l, p, pr)
        all_r["TF-IDF+LR"] = {"baseline": bm, "augmented": am}
        models["TF-IDF+LR"] = {"model": pa, "type": "sklearn"}

    # ── Step 4: Print results ─────────────────────────────────────────
    print_table(all_r, "IN-DOMAIN: Baseline vs Augmented (Jigsaw Test Set)")

    # ── Step 5: Cross-dataset ─────────────────────────────────────────
    cross_r = {}
    if not args.skip_cross_dataset:
        for dn, fn in [("HateXplain", load_hatexplain), ("Davidson", load_davidson)]:
            try: et, el = fn()
            except Exception as e: log.warning("Skip %s: %s", dn, e); continue
            for cn, info in models.items():
                key = f"{cn}_on_{dn}"
                if info["type"] == "bert":
                    p, l, pr = predict_distilbert(info["model"], info["tokenizer"], et, el, args, device)
                elif info["type"] == "lstm":
                    p, l, pr = predict_bilstm(info["model"], info["vocab"], et, el, device)
                elif info["type"] == "sklearn":
                    p, l, pr = predict_tfidf_lr(info["model"], et, el)
                else: continue
                cross_r[key] = compute_metrics(l, p, pr)
        if cross_r:
            print_cross(cross_r, "CROSS-DATASET: Augmented Models on External Datasets")

    # ── Save report ───────────────────────────────────────────────────
    report = {
        "experiment": "ToxiGAN Multi-Classifier (Real Imbalanced Data)",
        "dataset": "Full Jigsaw — no modifications",
        "train_original": len(train_df),
        "train_augmented": len(aug_df),
        "original_toxic": base_toxic,
        "original_nontoxic": base_nontoxic,
        "augmented_toxic": aug_toxic,
        "generated_cleaned": len(gen_texts),
        "test_size": len(test_df),
        "in_domain": all_r,
        "cross_dataset": cross_r,
    }
    rp = output_dir / "full_analysis_report.json"
    with open(rp, "w") as f: json.dump(report, f, indent=2, default=str)
    log.info("\nReport → %s", rp)
    log.info("Done.")


if __name__ == "__main__":
    main()