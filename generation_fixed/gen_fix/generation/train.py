# Main training script of ToxiGAN.
# This code referred to the implementation of SeqGAN and SentiGAN:
#       SeqGAN: https://github.com/LantaoYu/SeqGAN
#       SentiGAN: https://github.com/Nrgeup/SentiGAN

import os
import random
import re
from collections import Counter, defaultdict

import nltk
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from dataloader import GenDataset, DisDataset, gen_collate_fn, dis_collate_fn  # Load data in training
from discriminator import BertDiscriminator  # Discriminator based on BERT
from generator import Generator  # Generator based on LSTM
from llm_neutral_provider import FewShotNeutralGenerator  # LLM-based Neutral Text Provider
from rollout import Rollout  # Using Monte Carlo Search with rollout, compute loss via policy gradient (REINFORCE).
from config import parse_opt  # configuration


"""
Preprocess raw texts and build vocabulary for generators
"""


def ensure_nltk_resources():
    """Best-effort setup for NLTK tokenizers.
    Falls back to wordpunct_tokenize if downloads are unavailable.
    """
    resources = [
        ("punkt_tab", "tokenizers/punkt_tab/english"),
        ("punkt", "tokenizers/punkt"),
    ]
    for resource_name, resource_path in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            try:
                nltk.download(resource_name, quiet=True)
            except Exception:
                pass


def clean_line(line: str) -> str:
    line = re.sub(r"[^a-zA-Z ]+", " ", line)
    line = re.sub(r"\s+", " ", line).strip()
    return line.lower()


def tokenize_text(text: str):
    text = clean_line(text)
    if not text:
        return []
    try:
        return nltk.word_tokenize(text)
    except LookupError:
        ensure_nltk_resources()
        try:
            return nltk.word_tokenize(text)
        except Exception:
            return nltk.tokenize.wordpunct_tokenize(text)


def read_text_lines(file_path: str):
    """Read text files robustly across UTF-8 / Windows encodings."""
    with open(file_path, "rb") as f:
        raw = f.read()

    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding).splitlines()
        except UnicodeDecodeError:
            continue

    return raw.decode("utf-8", errors="surrogateescape").splitlines()


def build_vocab_and_tokenize(files, max_seq_length):
    counter = Counter()

    for file_path in files.values():
        for line in read_text_lines(file_path):
            tokens = tokenize_text(line)
            if tokens:
                counter.update(tokens)

    vocab = defaultdict(lambda: len(vocab))
    vocab["<PAD>"]     # 0
    vocab["<START>"]   # 1
    vocab["<EOS>"]     # 2
    vocab["<UNK>"]     # 3

    for word, freq in counter.items():
        if freq >= 3:
            vocab[word]

    tokenized = {}
    unk_id = vocab["<UNK>"]
    eos_id = vocab["<EOS>"]

    for tag, file_path in files.items():
        tokenized_lines = []
        for line in read_text_lines(file_path):
            tokens = tokenize_text(line)
            if 1 < len(tokens) <= max_seq_length - 1:
                token_ids = [vocab.get(word, unk_id) for word in tokens] + [eos_id]
                tokenized_lines.append(token_ids)
        tokenized[tag] = tokenized_lines

    word2idx = dict(vocab)
    idx2word = {v: k for k, v in word2idx.items()}
    return word2idx, idx2word, tokenized


def save_token_ids(tokenized_data, output_paths, save_path):
    os.makedirs(save_path, exist_ok=True)
    for tag, lines in tokenized_data.items():
        with open(os.path.join(save_path, output_paths[tag]), "w", encoding="utf-8") as f:
            for line in lines:
                f.write(" ".join(map(str, line)) + "\n")


def save_vocab(idx2word, save_path, path="vocab.txt"):
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, path), "w", encoding="utf-8") as f:
        for i in range(len(idx2word)):
            f.write(idx2word[i] + "\n")


def decode_sentence(token_ids, id2word, start_id, eos_id):
    words = []
    for idx in token_ids:
        if idx == eos_id:
            break
        if idx == start_id:
            continue
        words.append(id2word.get(idx, "<UNK>"))
    return " ".join(words)


class PretrainDisDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


"""
ToxiGAN training:
    (1) pretrain generators
    (2) pretrain discriminator
    (3) update LLM-ballast, generators, discriminator by ToxiGAN
"""


def main():
    ensure_nltk_resources()

    # preprocess and initialize
    opt = parse_opt()

    seed = getattr(opt, "SEED", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(opt.save_path, exist_ok=True)

    # Make sure neutral class is index 0 and toxic classes follow after it.
    original_tags = list(opt.TOXIC_CLASSES.keys())
    neutral_tag = "nor" if "nor" in opt.TOXIC_CLASSES else original_tags[0]
    toxic_tags = [tag for tag in original_tags if tag != neutral_tag]
    tags = [neutral_tag] + toxic_tags

    ordered_files = {tag: opt.TOXIC_CLASSES[tag] for tag in tags}
    output_ids = {k: k + ".id" for k in tags}

    vocab_dict, id2word, tokenized = build_vocab_and_tokenize(ordered_files, opt.MAX_SEQ_LENGTH)
    save_token_ids(tokenized, output_ids, opt.save_path)
    save_vocab(id2word, opt.save_path)

    vocab_size = len(vocab_dict)
    k = len(toxic_tags)  # number of toxic generators / toxic classes
    num_classes = k + 2  # neutral + toxic classes + fake
    pad_id = vocab_dict["<PAD>"]
    start_id = vocab_dict["<START>"]
    eos_id = vocab_dict["<EOS>"]
    unk_id = vocab_dict["<UNK>"]

    fewshot_gen = FewShotNeutralGenerator()  # initialize LLM-based ballast

    def make_one_hot_label(class_idx):
        label = torch.zeros(num_classes, dtype=torch.float32)
        label[class_idx] = 1.0
        return label

    def add_neutral_samples(dis_data, fewshot_generator, word2idx, max_len, num_samples):
        neutral_sentences = fewshot_generator.generate_neutral_sentences(num_sentences=num_samples)
        added_sentences = []

        for sentence in neutral_sentences:
            tokens = tokenize_text(sentence)
            if 3 <= len(tokens) < max_len:
                token_ids = [start_id] + [word2idx.get(token, unk_id) for token in tokens] + [eos_id]
                dis_data.append((torch.tensor(token_ids, dtype=torch.long), make_one_hot_label(0)))
                added_sentences.append(sentence)

        return added_sentences

    def build_real_discriminator_samples():
        real_samples = []
        for class_idx, tag in enumerate(tags):
            real_data = DisDataset([os.path.join(opt.save_path, output_ids[tag])], [], opt.MAX_SEQ_LENGTH)
            for tokens, _ in real_data:
                real_samples.append((tokens, make_one_hot_label(class_idx)))
        return real_samples

    def add_fake_toxic_samples(dis_data, num_samples_per_generator=100):
        for generator in generators:
            fake_samples = generator.sample(num_samples_per_generator, start_id, device)
            for sample in fake_samples:
                dis_data.append((sample.detach().cpu(), make_one_hot_label(num_classes - 1)))

    # initialize settings for toxic generators
    generators = [
        Generator(vocab_size, opt.EMB_DIM, opt.HIDDEN_DIM, opt.MAX_SEQ_LENGTH).to(device)
        for _ in range(k)
    ]
    for generator in generators:
        generator.eos_token = eos_id

    rollouts = [Rollout(generator) for generator in generators]
    optimizers_g = [torch.optim.Adam(generator.parameters(), lr=opt.learning_rate_g) for generator in generators]

    # initialize settings for discriminator
    dis_model = BertDiscriminator(vocab_dict, id2word, num_classes=num_classes).to(device)
    optimizer_d = torch.optim.Adam(dis_model.parameters(), lr=opt.learning_rate_d)

    # pretrain toxic generators
    for gen_idx, tag in enumerate(toxic_tags):
        print(f"Pretraining Generator G_{gen_idx + 1} on '{tag}'...")
        gen_data = GenDataset([os.path.join(opt.save_path, output_ids[tag])])

        if len(gen_data) == 0:
            print(f"[WARN] No generator pretraining data found for class '{tag}'. Skipping.")
            continue

        gen_loader = DataLoader(
            gen_data,
            batch_size=opt.BATCH_SIZE,
            shuffle=True,
            collate_fn=gen_collate_fn,
        )

        last_loss = None
        for epoch in range(opt.PRETRAIN_EPOCHS_G):
            for batch in gen_loader:
                batch = batch.to(device)
                input_seq = batch[:, :-1]
                target_seq = batch[:, 1:].long()
                lengths = (batch != pad_id).sum(dim=1).clamp(max=opt.MAX_SEQ_LENGTH) - 1
                lengths = lengths.tolist()

                logits = generators[gen_idx](input_seq, lengths)
                min_len = min(logits.size(1), target_seq.size(1))
                logits = logits[:, :min_len, :]
                target_seq = target_seq[:, :min_len]

                loss = F.cross_entropy(
                    logits.reshape(-1, vocab_size),
                    target_seq.reshape(-1),
                    ignore_index=pad_id,
                )

                optimizers_g[gen_idx].zero_grad()
                loss.backward()
                optimizers_g[gen_idx].step()
                last_loss = loss.item()

            if last_loss is not None:
                print(f"[G_{gen_idx + 1} Pretrain Epoch {epoch + 1}] Loss: {last_loss:.4f}")

    print("Pretraining Discriminator with neutral + toxic + fake samples...")

    # prepare dataset for pretraining discriminator
    dis_data = build_real_discriminator_samples()
    neutral_sentences = add_neutral_samples(dis_data, fewshot_gen, vocab_dict, opt.MAX_SEQ_LENGTH, 100)
    add_fake_toxic_samples(dis_data, num_samples_per_generator=100)

    if len(dis_data) == 0:
        raise RuntimeError("Discriminator pretraining dataset is empty. Check your input files and tokenization.")

    dis_loader = DataLoader(
        PretrainDisDataset(dis_data),
        batch_size=opt.dis_batch_size,
        shuffle=True,
        collate_fn=dis_collate_fn,
    )

    # pretrain discriminator
    for epoch in range(opt.PRETRAIN_EPOCHS_D):
        last_d_loss = None
        for x_batch, y_batch in dis_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            scores, _, _ = dis_model(x_batch)
            l2_loss = sum(torch.norm(parameter) for parameter in dis_model.parameters())
            d_loss = F.cross_entropy(scores, y_batch.argmax(dim=1)) + opt.dis_l2_reg_lambda * l2_loss

            optimizer_d.zero_grad()
            d_loss.backward()
            optimizer_d.step()
            last_d_loss = d_loss.item()

        if last_d_loss is not None:
            print(f"[Discriminator Pretrain Epoch {epoch + 1}] Loss: {last_d_loss:.4f}")

    # initial LLM-ballast
    fewshot_gen.init_scored_pool(dis_model, vocab_dict, max_len=opt.MAX_SEQ_LENGTH)

    # Adversarial Training
    print("Starting Adversarial Training...")
    for batch_num in range(opt.TOTAL_BATCH * 2):  # Alternating between tox and dis penalty
        # update LLM-fewshot
        fewshot_gen.update_scored_pool(dis_model, vocab_dict, max_len=opt.MAX_SEQ_LENGTH, evolve_rate=0.5)
        fewshot_gen.update_examples_from_pool(top_n=100)
        print("LLM-fewshot updated.")

        # refresh neutral samples for rollout penalty context
        neutral_sentences = fewshot_gen.generate_neutral_sentences(num_sentences=100)

        # Update generator: use REINFORCE to train each toxic generator
        for gen_idx in range(k):
            generator = generators[gen_idx]
            rollout = rollouts[gen_idx]
            optimizer = optimizers_g[gen_idx]

            generator.train()
            samples = generator.sample(opt.BATCH_SIZE, start_id, device)

            padded_samples = []
            for sample in samples:
                sample = sample.detach()
                if len(sample) < opt.MAX_SEQ_LENGTH:
                    sample = F.pad(sample, (0, opt.MAX_SEQ_LENGTH - len(sample)), value=pad_id)
                else:
                    sample = sample[:opt.MAX_SEQ_LENGTH]
                padded_samples.append(sample)

            samples_tensor = torch.stack(padded_samples).to(device)
            input_seq = samples_tensor[:, :-1]
            target_seq = samples_tensor[:, 1:]
            lengths = (samples_tensor != pad_id).sum(dim=1).clamp(max=opt.MAX_SEQ_LENGTH).tolist()
            logits = generator(input_seq, [max(length - 1, 1) for length in lengths])

            rollout.set_penalty_context(neutral_sentences, id2word)
            if batch_num % 2 == 0:
                # reinforce toxicity by penalizing neutral generation
                raw_reward = rollout.get_penalty(
                    samples_tensor,
                    opt.ROLL_OUT_NUM,
                    dis_model,
                    start_id,
                    device,
                    current_class=gen_idx + 1,
                    penalty_type="tox",
                )
            else:
                # reinforce authenticity by penalizing generation identified fake of target class
                raw_reward = rollout.get_penalty(
                    samples_tensor,
                    opt.ROLL_OUT_NUM,
                    dis_model,
                    start_id,
                    device,
                    current_class=gen_idx + 1,
                    penalty_type="dis",
                )

            rewards = raw_reward.detach().to(device)
            min_len = min(logits.size(1), target_seq.size(1), rewards.size(1))
            logits = logits[:, :min_len, :]
            target_seq = target_seq[:, :min_len]
            rewards = rewards[:, :min_len]

            # Policy gradient should maximize reward-weighted log-probability.
            log_probs = F.log_softmax(logits, dim=-1)
            selected_log_probs = log_probs.gather(dim=-1, index=target_seq.unsqueeze(-1)).squeeze(-1)

            # mask padding
            mask = (target_seq != pad_id).float()
            denom = mask.sum().clamp(min=1.0)

            # negative sign is important: optimizer minimizes loss, but policy gradient should maximize reward.
            pg_loss = -torch.sum(selected_log_probs * rewards * mask) / denom

            optimizer.zero_grad()
            pg_loss.backward()
            optimizer.step()

            print(
                f"[Batch {batch_num + 1}] G_{gen_idx + 1} "
                f"Reward mean: {rewards.mean().item():.4f} | PG Loss: {pg_loss.item():.4f}"
            )

        # save toxic generators
        for tag, generator in zip(toxic_tags, generators):
            torch.save(generator.state_dict(), os.path.join(opt.save_path, f"generator_toxic_{tag}.pt"))

        # Update discriminator
        dis_data = build_real_discriminator_samples()
        add_fake_toxic_samples(dis_data, num_samples_per_generator=100)
        neutral_sentences = add_neutral_samples(dis_data, fewshot_gen, vocab_dict, opt.MAX_SEQ_LENGTH, 100)

        dis_loader = DataLoader(
            PretrainDisDataset(dis_data),
            batch_size=opt.dis_batch_size,
            shuffle=True,
            collate_fn=dis_collate_fn,
        )

        last_d_loss = None
        for _ in range(opt.DIS_UPDATES_PER_ROUND):
            for x_batch, y_batch in dis_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                scores, _, _ = dis_model(x_batch)
                l2_loss = sum(torch.norm(parameter) for parameter in dis_model.parameters())
                d_loss = F.cross_entropy(scores, y_batch.argmax(dim=1)) + opt.dis_l2_reg_lambda * l2_loss

                optimizer_d.zero_grad()
                d_loss.backward()
                optimizer_d.step()
                last_d_loss = d_loss.item()

        if last_d_loss is not None:
            print(f"[Adv Batch {batch_num + 1}] Discriminator Loss: {last_d_loss:.4f}")

        # save discriminator
        torch.save(dis_model.state_dict(), os.path.join(opt.save_path, "discriminator.pt"))
        print("[ADV] Models saved.")


if __name__ == "__main__":
    main()
