#!/usr/bin/env python3
"""
Resume ToxiGAN adversarial training from saved checkpoints.

Skips pretraining — loads generators + discriminator from artifacts/
and continues adversarial training from --start_batch.

Usage:
    python resume_training.py                        # resume from batch 0
    python resume_training.py --start_batch 5        # resume from batch 5
    python resume_training.py --total_batches 30     # run 30 more batches
"""

import os
import sys
import random
import re
import argparse
from collections import Counter, defaultdict

import nltk
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from dataloader import GenDataset, DisDataset, gen_collate_fn, dis_collate_fn
from discriminator import BertDiscriminator
from generator import Generator
from llm_neutral_provider import FewShotNeutralGenerator
from rollout import Rollout
from config import parse_opt


# ── Reuse helpers from train.py ───────────────────────────────────────────

def ensure_nltk_resources():
    resources = [
        ("punkt_tab", "tokenizers/punkt_tab/english"),
        ("punkt", "tokenizers/punkt"),
    ]
    for name, path in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(name, quiet=True)
            except Exception:
                pass


def clean_line(line):
    line = re.sub(r"[^a-zA-Z ]+", " ", line)
    return re.sub(r"\s+", " ", line).strip().lower()


def tokenize_text(text):
    text = clean_line(text)
    if not text:
        return []
    try:
        return nltk.word_tokenize(text)
    except Exception:
        return nltk.tokenize.wordpunct_tokenize(text)


def read_text_lines(file_path):
    with open(file_path, "rb") as f:
        raw = f.read()
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc).splitlines()
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="surrogateescape").splitlines()


class PretrainDisDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


def main():
    parser = argparse.ArgumentParser(description="Resume ToxiGAN adversarial training.")
    parser.add_argument("--start_batch", type=int, default=0,
                        help="Batch number to resume from (for logging only — all batches run).")
    parser.add_argument("--total_batches", type=int, default=None,
                        help="Total adversarial batches to run. Default: use config TOTAL_BATCH.")
    args, _ = parser.parse_known_args()

    ensure_nltk_resources()
    opt = parse_opt()

    seed = getattr(opt, "SEED", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load vocab ────────────────────────────────────────────────────────
    vocab_path = os.path.join(opt.save_path, "vocab.txt")
    if not os.path.exists(vocab_path):
        print(f"ERROR: vocab.txt not found at {vocab_path}")
        print("Run train.py first to generate vocab and pretrain models.")
        sys.exit(1)

    # Read vocab
    idx2word = {}
    with open(vocab_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            idx2word[i] = line.strip()
    vocab_dict = {v: k for k, v in idx2word.items()}
    vocab_size = len(vocab_dict)

    pad_id = vocab_dict["<PAD>"]
    start_id = vocab_dict["<START>"]
    eos_id = vocab_dict["<EOS>"]
    unk_id = vocab_dict.get("<UNK>", 0)

    print(f"Vocab loaded: {vocab_size} tokens")

    # ── Determine class structure ─────────────────────────────────────────
    original_tags = list(opt.TOXIC_CLASSES.keys())
    neutral_tag = "nor" if "nor" in opt.TOXIC_CLASSES else original_tags[0]
    toxic_tags = [tag for tag in original_tags if tag != neutral_tag]
    tags = [neutral_tag] + toxic_tags

    k = len(toxic_tags)
    num_classes = k + 2
    output_ids = {key: key + ".id" for key in tags}

    print(f"Classes: {tags} | k={k} toxic | num_classes={num_classes}")

    # ── Load generators from checkpoints ──────────────────────────────────
    generators = []
    for tag in toxic_tags:
        g = Generator(vocab_size, opt.EMB_DIM, opt.HIDDEN_DIM, opt.MAX_SEQ_LENGTH).to(device)
        ckpt = os.path.join(opt.save_path, f"generator_toxic_{tag}.pt")
        if os.path.exists(ckpt):
            g.load_state_dict(torch.load(ckpt, map_location=device))
            print(f"  Loaded generator: {tag}")
        else:
            print(f"  WARNING: No checkpoint for {tag}, using random init")
        g.eos_token = eos_id
        generators.append(g)

    rollouts = [Rollout(g) for g in generators]
    optimizers_g = [torch.optim.Adam(g.parameters(), lr=opt.learning_rate_g) for g in generators]

    # ── Load discriminator from checkpoint ────────────────────────────────
    dis_model = BertDiscriminator(vocab_dict, idx2word, num_classes=num_classes).to(device)
    dis_ckpt = os.path.join(opt.save_path, "discriminator.pt")
    if os.path.exists(dis_ckpt):
        dis_model.load_state_dict(torch.load(dis_ckpt, map_location=device))
        print(f"  Loaded discriminator")
    else:
        print(f"  WARNING: No discriminator checkpoint, using random init")
    optimizer_d = torch.optim.Adam(dis_model.parameters(), lr=opt.learning_rate_d)

    # ── Initialize LLM ballast ────────────────────────────────────────────
    print("Initializing LLM neutral provider...")
    fewshot_gen = FewShotNeutralGenerator()
    fewshot_gen.init_scored_pool(dis_model, vocab_dict, max_len=opt.MAX_SEQ_LENGTH)

    # ── Helpers ───────────────────────────────────────────────────────────
    def make_one_hot_label(class_idx):
        label = torch.zeros(num_classes, dtype=torch.float32)
        label[class_idx] = 1.0
        return label

    def build_real_discriminator_samples():
        real_samples = []
        for class_idx, tag in enumerate(tags):
            real_data = DisDataset([os.path.join(opt.save_path, output_ids[tag])], [], opt.MAX_SEQ_LENGTH)
            for tokens, _ in real_data:
                real_samples.append((tokens, make_one_hot_label(class_idx)))
        return real_samples

    def add_neutral_samples(dis_data, fewshot_generator, word2idx, max_len, num_samples):
        neutral_sentences = fewshot_generator.generate_neutral_sentences(num_sentences=num_samples)
        added = []
        for sentence in neutral_sentences:
            tokens = tokenize_text(sentence)
            if 3 <= len(tokens) < max_len:
                token_ids = [start_id] + [word2idx.get(t, unk_id) for t in tokens] + [eos_id]
                dis_data.append((torch.tensor(token_ids, dtype=torch.long), make_one_hot_label(0)))
                added.append(sentence)
        return added

    def add_fake_toxic_samples(dis_data, num_per_gen=100):
        for g in generators:
            fakes = g.sample(num_per_gen, start_id, device)
            for s in fakes:
                dis_data.append((s.detach().cpu(), make_one_hot_label(num_classes - 1)))

    # ── Adversarial Training ──────────────────────────────────────────────
    total = args.total_batches if args.total_batches else opt.TOTAL_BATCH
    total_iters = total * 2  # alternating tox/dis

    print(f"\n{'='*60}")
    print(f"RESUMING adversarial training: {total_iters} batches")
    print(f"{'='*60}\n")

    for batch_num in range(total_iters):
        display_num = args.start_batch + batch_num + 1

        # Update LLM fewshot
        fewshot_gen.update_scored_pool(dis_model, vocab_dict, max_len=opt.MAX_SEQ_LENGTH, evolve_rate=0.5)
        fewshot_gen.update_examples_from_pool(top_n=100)

        # Fresh neutral sentences
        neutral_sentences = fewshot_gen.generate_neutral_sentences(num_sentences=100)

        # ── Update generators ─────────────────────────────────────────────
        for gen_idx in range(k):
            G = generators[gen_idx]
            rollout = rollouts[gen_idx]
            optimizer = optimizers_g[gen_idx]

            G.train()
            samples = G.sample(opt.BATCH_SIZE, start_id, device)

            padded = []
            for s in samples:
                s = s.detach()
                if len(s) < opt.MAX_SEQ_LENGTH:
                    s = F.pad(s, (0, opt.MAX_SEQ_LENGTH - len(s)), value=pad_id)
                else:
                    s = s[:opt.MAX_SEQ_LENGTH]
                padded.append(s)

            samples_tensor = torch.stack(padded).to(device)
            input_seq = samples_tensor[:, :-1]
            target_seq = samples_tensor[:, 1:]
            lengths = (samples_tensor != pad_id).sum(dim=1).clamp(max=opt.MAX_SEQ_LENGTH).tolist()
            logits = G(input_seq, [max(l - 1, 1) for l in lengths])

            rollout.set_penalty_context(neutral_sentences, idx2word)
            ptype = "tox" if batch_num % 2 == 0 else "dis"
            raw_reward = rollout.get_penalty(
                samples_tensor, opt.ROLL_OUT_NUM, dis_model, start_id,
                device, current_class=gen_idx + 1, penalty_type=ptype,
            )

            rewards = raw_reward.detach().to(device)
            ml = min(logits.size(1), target_seq.size(1), rewards.size(1))
            logits = logits[:, :ml, :]
            target_seq = target_seq[:, :ml]
            rewards = rewards[:, :ml]

            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs.gather(dim=-1, index=target_seq.unsqueeze(-1)).squeeze(-1)
            mask = (target_seq != pad_id).float()
            pg_loss = -torch.sum(selected * rewards * mask) / mask.sum().clamp(min=1.0)

            optimizer.zero_grad()
            pg_loss.backward()
            optimizer.step()

            print(f"[Batch {display_num}] G_{gen_idx+1} Reward: {rewards.mean().item():.4f} | PG Loss: {pg_loss.item():.4f}")

        # Save generators
        for tag, G in zip(toxic_tags, generators):
            torch.save(G.state_dict(), os.path.join(opt.save_path, f"generator_toxic_{tag}.pt"))

        # ── Update discriminator ──────────────────────────────────────────
        dis_data = build_real_discriminator_samples()
        add_fake_toxic_samples(dis_data, num_per_gen=100)
        neutral_sentences = add_neutral_samples(dis_data, fewshot_gen, vocab_dict, opt.MAX_SEQ_LENGTH, 100)

        dis_loader = DataLoader(
            PretrainDisDataset(dis_data),
            batch_size=opt.dis_batch_size, shuffle=True, collate_fn=dis_collate_fn,
        )

        last_d = None
        for _ in range(opt.DIS_UPDATES_PER_ROUND):
            for x_batch, y_batch in dis_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                scores, _, _ = dis_model(x_batch)
                l2 = sum(torch.norm(p) for p in dis_model.parameters())
                d_loss = F.cross_entropy(scores, y_batch.argmax(dim=1)) + opt.dis_l2_reg_lambda * l2
                optimizer_d.zero_grad()
                d_loss.backward()
                optimizer_d.step()
                last_d = d_loss.item()

        if last_d:
            print(f"[Adv Batch {display_num}] D Loss: {last_d:.4f}")

        torch.save(dis_model.state_dict(), os.path.join(opt.save_path, "discriminator.pt"))
        print("[ADV] Models saved.\n")

    print("Adversarial training complete.")
    print(f"Checkpoints: {opt.save_path}/")
    print(f"Next step: python generate_samples.py")


if __name__ == "__main__":
    main()