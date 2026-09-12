"""
Aggressive filtering of ToxiGAN-generated samples.

Removes low-quality outputs (too short, high UNK, repetitive, gibberish)
and returns cleaned texts ready for augmentation.
"""

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import List

log = logging.getLogger(__name__)


def load_and_clean_generated(json_path: str) -> List[str]:
    """
    Load ToxiGAN generated samples and aggressively filter low-quality ones.

    Filters applied:
      1. Remove empty samples
      2. Remove samples with ≤ 2 words
      3. Remove samples where > 30% of tokens are <UNK>
      4. Remove samples with excessive repetition (any token > 50% of text)
      5. Remove samples with > 50% non-alphabetic words
      6. Strip <UNK> tokens from survivors
      7. Re-check length after stripping

    Returns list of cleaned text strings.
    """
    path = Path(json_path)
    if not path.exists():
        log.warning("Generated data not found at '%s'. Returning empty list.", json_path)
        return []

    with open(path) as f:
        data = json.load(f)

    tweets = data.get("tweet", [])
    log.info("Loaded %d generated samples from %s", len(tweets), json_path)

    cleaned = []
    reasons: Counter = Counter()

    for text in tweets:
        text = text.strip()

        if not text:
            reasons["empty"] += 1
            continue

        words = text.split()

        if len(words) <= 2:
            reasons["too_short"] += 1
            continue

        # High <UNK> ratio
        unk_count = sum(1 for w in words if w.lower() in ("<unk>", "unk"))
        if unk_count / len(words) > 0.3:
            reasons["high_unk"] += 1
            continue

        # Excessive repetition
        word_counts = Counter(w.lower() for w in words)
        if word_counts.most_common(1)[0][1] / len(words) > 0.5:
            reasons["repetitive"] += 1
            continue

        # Mostly non-alpha gibberish
        alpha_words = sum(1 for w in words if re.search(r"[a-zA-Z]", w))
        if alpha_words / len(words) < 0.5:
            reasons["non_alpha"] += 1
            continue

        # Strip UNK tokens
        cleaned_words = [w for w in words if w.lower() not in ("<unk>", "unk")]
        if len(cleaned_words) <= 2:
            reasons["short_after_clean"] += 1
            continue

        cleaned.append(" ".join(cleaned_words))

    log.info("Cleaning results:")
    log.info("  Input:    %d", len(tweets))
    log.info("  Kept:     %d (%.1f%%)", len(cleaned), 100 * len(cleaned) / max(len(tweets), 1))
    log.info("  Filtered: %s", dict(reasons))

    if cleaned:
        log.info("  Samples: %s", cleaned[:3])

    return cleaned
