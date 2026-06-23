from collections import Counter
from typing import Callable, Dict, List

import numpy as np
import regex as re


def strip_language_tag(text: str) -> str:
    return re.sub(r"^<[^>]+>\s*", "", text)


def simple_words(text: str) -> List[str]:
    """
    Extract words using Unicode letters.
    This is better than English-only tokenization.
    """
    text = strip_language_tag(text)
    return re.findall(r"\p{L}+(?:['’\-]\p{L}+)*", text)


def tokenization_fertility(texts: List[str], encode_fn: Callable[[str], List[str]]) -> float:
    """
    Fertility = average number of subword tokens per whitespace/word token.

    Lower is usually better, assuming the tokenizer still preserves meaning.
    """
    total_subwords = 0
    total_words = 0

    for text in texts:
        words = simple_words(text)
        if not words:
            continue

        tokens = encode_fn(strip_language_tag(text))
        total_subwords += len(tokens)
        total_words += len(words)

    return total_subwords / max(total_words, 1)


def char_per_token_ratio(texts: List[str], encode_fn: Callable[[str], List[str]]) -> float:
    """
    Average number of characters represented per token.

    Higher means better compression, but extremely high values may hide
    overlong or semantically messy tokens.
    """
    total_chars = 0
    total_tokens = 0

    for text in texts:
        clean = strip_language_tag(text)
        tokens = encode_fn(clean)
        total_chars += len(clean)
        total_tokens += len(tokens)

    return total_chars / max(total_tokens, 1)


def unk_rate(texts: List[str], encode_fn: Callable[[str], List[str]], unk_token: str = "[UNK]") -> float:
    """
    Percentage of produced tokens that are unknown tokens.
    For byte-level or SentencePiece models this may be zero or near zero.
    """
    total = 0
    unk = 0

    for text in texts:
        tokens = encode_fn(strip_language_tag(text))
        total += len(tokens)
        unk += sum(1 for tok in tokens if tok == unk_token or tok == "<unk>")

    return unk / max(total, 1)


def vocabulary_coverage(texts: List[str], vocab: set[str]) -> float:
    """
    Word-level vocabulary coverage:
    percentage of word types that appear as complete entries in tokenizer vocab.

    This is strict. A tokenizer can still be good if full-word coverage is low,
    because subword tokenization is expected.
    """
    word_types = set()

    for text in texts:
        word_types.update(simple_words(text))

    if not word_types:
        return 0.0

    covered = sum(1 for word in word_types if word in vocab)
    return covered / len(word_types)


def token_length_stats(texts: List[str], encode_fn: Callable[[str], List[str]]) -> Dict[str, float]:
    lengths = []

    for text in texts:
        tokens = encode_fn(strip_language_tag(text))
        lengths.extend([len(tok) for tok in tokens])

    if not lengths:
        return {
            "mean_token_chars": 0.0,
            "median_token_chars": 0.0,
            "p95_token_chars": 0.0,
        }

    return {
        "mean_token_chars": float(np.mean(lengths)),
        "median_token_chars": float(np.median(lengths)),
        "p95_token_chars": float(np.percentile(lengths, 95)),
    }


def boundary_f1(
    word: str,
    gold_segments: List[str],
    predicted_tokens: List[str],
) -> Dict[str, float]:
    """
    Compare gold morpheme boundaries against tokenizer boundaries.

    Example:
        word = "someword"
        gold_segments = ["some", "word"]
        predicted_tokens = ["so", "me", "word"]

    We convert segmentations into boundary positions.
    """
    def boundaries(segments):
        positions = set()
        cursor = 0
        for seg in segments[:-1]:
            cursor += len(seg)
            positions.add(cursor)
        return positions

    gold_b = boundaries(gold_segments)
    pred_b = boundaries(predicted_tokens)

    if not gold_b and not pred_b:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    tp = len(gold_b & pred_b)
    precision = tp / max(len(pred_b), 1)
    recall = tp / max(len(gold_b), 1)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def summarize_token_counts(texts: List[str], encode_fn: Callable[[str], List[str]]) -> Counter:
    counter = Counter()

    for text in texts:
        counter.update(encode_fn(strip_language_tag(text)))

    return counter