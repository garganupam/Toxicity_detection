# llm_neutral_provider.py
# Corrected Ollama-based neutral example provider for ToxiGAN

import os
import random
import re
from typing import List, Optional

import nltk
import requests
import torch
import torch.nn.functional as F
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import parse_opt

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b-instruct")


def ensure_nltk_resources():
    """Best-effort setup for tokenizer resources."""
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


def clean_text(text: str) -> str:
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_text(text: str) -> List[str]:
    text = clean_text(text).lower()
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


def read_text_lines(path: str) -> List[str]:
    """Robust reader for UTF-8 / UTF-8-SIG / Windows-encoded text files."""
    with open(path, "rb") as f:
        raw = f.read()

    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return [line.strip() for line in raw.decode(encoding).splitlines() if line.strip()]
        except UnicodeDecodeError:
            continue

    return [line.strip() for line in raw.decode("utf-8", errors="surrogateescape").splitlines() if line.strip()]


class FewShotNeutralGenerator:
    def __init__(self):
        self.opt = parse_opt()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.k_shots = 5
        self.ollama_host = OLLAMA_HOST
        self.ollama_model = OLLAMA_MODEL

        neutral_file_path = self._get_neutral_file_path()
        self.all_neutral_data = self._load_all_neutral_examples(neutral_file_path)
        if not self.all_neutral_data:
            raise ValueError(f"No neutral examples found in: {neutral_file_path}")

        self.examples = self._sample_examples(self.all_neutral_data, self.k_shots)

        # Unified, synchronized pool representation
        # each item: {"text": str, "tensor": Optional[Tensor], "score": float}
        self.pool = []
        self.scored_pool = []
        self.tensor_scored_pool = []

        self.session = self._build_session()

    def _build_session(self) -> requests.Session:
        """Use one persistent session to avoid repeated short-lived sockets on Windows."""
        session = requests.Session()
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["POST"]),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=1, pool_maxsize=4)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update({"Content-Type": "application/json"})
        return session

    def close(self):
        if hasattr(self, "session") and self.session is not None:
            self.session.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _get_neutral_file_path(self) -> str:
        if hasattr(self.opt, "SENTIMENT_CLASSES") and "nor" in self.opt.SENTIMENT_CLASSES:
            return self.opt.SENTIMENT_CLASSES["nor"]
        if hasattr(self.opt, "TOXIC_CLASSES") and "nor" in self.opt.TOXIC_CLASSES:
            return self.opt.TOXIC_CLASSES["nor"]
        raise KeyError("Could not find neutral class path under SENTIMENT_CLASSES['nor'] or TOXIC_CLASSES['nor'].")

    def _load_all_neutral_examples(self, path: str) -> List[str]:
        return read_text_lines(path)

    def _sample_examples(self, items: List[str], k: int) -> List[str]:
        if not items:
            return []
        if len(items) <= k:
            return list(items)
        return random.sample(items, k)

    def _build_prompt(self) -> str:
        lines = [
            "You generate exactly one neutral, non-toxic online message.",
            "Requirements:",
            "- Output exactly one single-line sentence.",
            "- Keep it natural, plain, and conversational.",
            "- Do not use labels, bullets, quotes, markdown, wiki markup, redirects, usernames, or explanations.",
            "- Do not mention toxicity, hate speech, or safety.",
            "- Keep it under 20 words.",
            "",
            "Examples:",
        ]
        for i, ex in enumerate(self.examples, 1):
            lines.append(f"{i}. {clean_text(ex)}")
        lines.extend(["", "Output one new neutral sentence only:"])
        return "\n".join(lines)

    def _sanitize_generated_line(self, text: str) -> str:
        text = clean_text(text)

        # Keep only first line
        text = text.split("\n")[0].strip()

        # Strip common prefixes from model continuation
        text = re.sub(r"^(example\s*\d+\s*[:\-]\s*)", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^(output\s*[:\-]\s*)", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^[-*>\s]+", "", text)
        text = text.strip(" \"'`")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _is_good_neutral_candidate(self, text: str, max_length: int) -> bool:
        if not text or len(text) < 5:
            return False
        if len(text.split()) < 3 or len(text.split()) > max_length:
            return False

        lower = text.lower()

        bad_patterns = [
            r"^redirect\b",
            r"\|",
            r"\{\{",
            r"\}\}",
            r"\[\[",
            r"\]\]",
            r"http[s]?://",
            r"www\.",
            r"^talk:",
            r"^category:",
            r"^template:",
            r"^file:",
            r"^user:",
            r"^name\s*=",
        ]
        for pattern in bad_patterns:
            if re.search(pattern, lower):
                return False

        alpha_chars = sum(ch.isalpha() for ch in text)
        total_chars = max(len(text), 1)
        if alpha_chars / total_chars < 0.55:
            return False

        if text in self.examples:
            return False

        return True

    def _fallback_neutral_sentences(self, num_sentences: int) -> List[str]:
        """Fallback to real neutral examples if Ollama call fails or returns junk."""
        candidates = [s for s in self.all_neutral_data if s not in self.examples]
        if not candidates:
            candidates = list(self.all_neutral_data)
        if not candidates:
            return []
        if len(candidates) <= num_sentences:
            return list(candidates)
        return random.sample(candidates, num_sentences)

    def _ollama_generate_once(self, prompt: str, max_length: int) -> Optional[str]:
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "10m",
            "options": {
                "temperature": 0.8,
                "num_predict": max_length,
                "top_p": 0.95,
            },
        }

        try:
            response = self.session.post(
                f"{self.ollama_host}/api/generate",
                json=payload,
                timeout=(10, 90),
            )
            response.raise_for_status()
            result = response.json().get("response", "").strip()
            line = self._sanitize_generated_line(result)
            if self._is_good_neutral_candidate(line, max_length=max_length):
                return line
            return None
        except Exception as e:
            print(f"[Ollama error] {e}")
            return None

    def generate_neutral_sentences(self, num_sentences: int = 10, max_length: int = 20) -> List[str]:
        """Generate neutral sentences with retries and safe fallback."""
        sentences = []
        seen = set()
        max_attempts = max(num_sentences * 4, 10)

        for _ in range(max_attempts):
            if len(sentences) >= num_sentences:
                break
            prompt = self._build_prompt()
            line = self._ollama_generate_once(prompt, max_length=max_length)
            if line and line not in seen:
                seen.add(line)
                sentences.append(line)

        if len(sentences) < num_sentences:
            needed = num_sentences - len(sentences)
            for line in self._fallback_neutral_sentences(needed):
                line = self._sanitize_generated_line(line)
                if line and line not in seen and self._is_good_neutral_candidate(line, max_length=max_length):
                    seen.add(line)
                    sentences.append(line)
                if len(sentences) >= num_sentences:
                    break

        return sentences

    def _sentence_to_tensor(self, sentence: str, vocab_dict, max_len: int) -> Optional[torch.Tensor]:
        tokens = tokenize_text(sentence)
        if not (3 <= len(tokens) < max_len):
            return None

        start_id = vocab_dict.get("<START>", 1)
        eos_id = vocab_dict.get("<EOS>", 2)
        pad_id = vocab_dict.get("<PAD>", 0)
        unk_id = vocab_dict.get("<UNK>", 0)

        token_ids = [start_id] + [vocab_dict.get(token, unk_id) for token in tokens] + [eos_id]
        token_tensor = torch.tensor(token_ids, dtype=torch.long)

        if len(token_tensor) > max_len:
            token_tensor = token_tensor[:max_len]
            if token_tensor[-1].item() != eos_id:
                token_tensor[-1] = eos_id

        if len(token_tensor) < max_len:
            token_tensor = F.pad(token_tensor, (0, max_len - len(token_tensor)), value=pad_id)

        return token_tensor

    def _score_tensor(self, dis_model, tensor_1d: torch.Tensor) -> float:
        input_tensor = tensor_1d.unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, probs, _ = dis_model(input_tensor)
            return float(probs[0][0].item())

    def _sync_compat_views(self):
        """Maintain backward-compatible attributes."""
        self.scored_pool = [(item["text"], item["score"]) for item in self.pool]
        self.tensor_scored_pool = [
            (item["tensor"].unsqueeze(0).to(self.device), item["score"])
            for item in self.pool
            if item["tensor"] is not None
        ]

    def init_scored_pool(self, dis_model, vocab_dict, max_len: int = 20):
        dis_model.eval()
        pool = []

        for sentence in self.all_neutral_data:
            tensor = self._sentence_to_tensor(sentence, vocab_dict, self.opt.MAX_SEQ_LENGTH)
            if tensor is None:
                score = 0.0
            else:
                score = self._score_tensor(dis_model, tensor)

            pool.append({
                "text": sentence,
                "tensor": tensor,
                "score": score,
            })

        self.pool = pool
        self._sync_compat_views()

    def update_scored_pool(self, dis_model, vocab_dict, max_len: int = 20, evolve_rate: float = 0.5):
        """Refresh scores and keep the pool synchronized with the latest discriminator."""
        dis_model.eval()

        if not self.pool:
            self.init_scored_pool(dis_model, vocab_dict, max_len=max_len)
            return

        current_pool = sorted(self.pool, key=lambda item: item["score"], reverse=True)
        keep_n = max(self.k_shots * 4, min(len(current_pool), int(len(current_pool) * evolve_rate)))
        current_pool = current_pool[:keep_n]

        refreshed_pool = []
        for item in current_pool:
            sentence = item["text"]
            tensor = self._sentence_to_tensor(sentence, vocab_dict, self.opt.MAX_SEQ_LENGTH)
            if tensor is None:
                score = 0.0
            else:
                score = self._score_tensor(dis_model, tensor)

            refreshed_pool.append({
                "text": sentence,
                "tensor": tensor,
                "score": score,
            })

        self.pool = refreshed_pool
        self._sync_compat_views()

    def update_examples_from_pool(self, top_n: int = 100):
        """Read from the refreshed pool instead of stale scored_pool."""
        if not self.pool:
            print("[Warning] pool is empty.")
            return

        sorted_pool = sorted(self.pool, key=lambda item: item["score"], reverse=True)
        top_candidates = [item["text"] for item in sorted_pool[:top_n] if item["text"]]

        if len(top_candidates) < self.k_shots:
            supplement_candidates = [s for s in self.all_neutral_data if s not in top_candidates]
            supplement = self._sample_examples(supplement_candidates, self.k_shots - len(top_candidates))
            topk = top_candidates + supplement
        else:
            topk = random.sample(top_candidates, self.k_shots)

        self.examples = topk
        print("Current Top k:", self.examples)


if __name__ == "__main__":
    gen = FewShotNeutralGenerator()
    print("Generating 5 neutral sentences via Ollama...")
    results = gen.generate_neutral_sentences(num_sentences=5)
    for i, sentence in enumerate(results, 1):
        print(f"{i}. {sentence}")
