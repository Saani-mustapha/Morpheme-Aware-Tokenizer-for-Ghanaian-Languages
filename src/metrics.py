from collections import Counter
from typing import Callable, Dict, List

import numpy as np
import regex as re


def strip_language_tag(text: str) -> str:
    """
    Remove a language tag such as <twi>, <ewe>, <ga>, or <dagbani>
    from the beginning of a line.
    """
    return re.sub(r"^<[^>]+>\s*", "", text).strip()


def get_language_tag(text: str) -> str:
    """
    Extract the language tag from a line.

    Example:
        <twi> Awurade yɛ me hwɛfoɔ

    Returns:
        twi
    """
    if text.startswith("<") and ">" in text:
        return text[1:text.index(">")]
    return "unknown"


def simple_words(text: str) -> List[str]:
    """
    Extract words using Unicode-aware matching.

    This is better than English-only splitting because Ghanaian languages
    may contain characters such as ɛ, ɔ, ŋ, ɣ, and other non-ASCII letters.
    """
    text = strip_language_tag(text)

    return re.findall(
        r"\p{L}+(?:['’\-]\p{L}+)*",
        text,
    )


def tokenization_fertility(
    texts: List[str],
    encode_fn: Callable[[str], List[str]],
) -> float:
    """
    Fertility = average number of tokenizer pieces per word.

    Lower fertility usually means less tokenization bloat.

    Example:
        word count = 100
        subword token count = 230

        fertility = 230 / 100 = 2.3
    """
    total_subwords = 0
    total_words = 0

    for text in texts:
        clean_text = strip_language_tag(text)
        words = simple_words(clean_text)

        if not words:
            continue

        tokens = encode_fn(clean_text)

        total_subwords += len(tokens)
        total_words += len(words)

    return total_subwords / max(total_words, 1)


def char_per_token_ratio(
    texts: List[str],
    encode_fn: Callable[[str], List[str]],
) -> float:
    """
    Average number of characters represented per tokenizer token.

    Higher usually means better compression, but it should be interpreted
    together with fertility and morphology scores.
    """
    total_chars = 0
    total_tokens = 0

    for text in texts:
        clean_text = strip_language_tag(text)
        tokens = encode_fn(clean_text)

        total_chars += len(clean_text)
        total_tokens += len(tokens)

    return total_chars / max(total_tokens, 1)


def unk_rate(
    texts: List[str],
    encode_fn: Callable[[str], List[str]],
    unk_tokens: set[str] | None = None,
) -> float:
    """
    Percentage of output tokens that are unknown tokens.

    Hugging Face BPE/unigram tokenizers may use [UNK].
    SentencePiece-like tokenizers may use <unk>.
    """
    if unk_tokens is None:
        unk_tokens = {"[UNK]", "<unk>", "<UNK>"}

    total_tokens = 0
    total_unk = 0

    for text in texts:
        clean_text = strip_language_tag(text)
        tokens = encode_fn(clean_text)

        total_tokens += len(tokens)
        total_unk += sum(1 for token in tokens if token in unk_tokens)

    return total_unk / max(total_tokens, 1)


def vocabulary_coverage(
    texts: List[str],
    vocab: set[str],
) -> float:
    """
    Strict word-level vocabulary coverage.

    This checks how many full word types appear directly in the tokenizer vocabulary.

    Important:
        This metric is strict. A tokenizer can still be useful even when
        full-word coverage is low, because subword tokenization is expected.
    """
    word_types = set()

    for text in texts:
        word_types.update(simple_words(text))

    if not word_types:
        return 0.0

    covered = sum(1 for word in word_types if word in vocab)

    return covered / len(word_types)


def token_length_stats(
    texts: List[str],
    encode_fn: Callable[[str], List[str]],
) -> Dict[str, float]:
    """
    Compute average, median, and 95th percentile token length in characters.
    """
    lengths = []

    for text in texts:
        clean_text = strip_language_tag(text)
        tokens = encode_fn(clean_text)

        for token in tokens:
            cleaned_token = clean_subword_marker(token)
            if cleaned_token:
                lengths.append(len(cleaned_token))

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


def clean_subword_marker(token: str) -> str:
    """
    Remove common subword boundary markers.

    Examples:
        ##word -> word
        ▁word  -> word
        Ġword  -> word
    """
    return (
        token.replace("##", "")
        .replace("▁", "")
        .replace("Ġ", "")
        .strip()
    )


def boundary_f1(
    gold_segments: List[str],
    predicted_tokens: List[str],
) -> Dict[str, float]:
    """
    Compare gold morpheme boundaries against tokenizer boundaries.

    Example:
        gold_segments = ["sukuu", "foɔ"]
        predicted_tokens = ["sukuu", "foɔ"]

    This gives a high score because the tokenizer boundary matches
    the human morpheme boundary.
    """

    def boundaries(segments: List[str]) -> set[int]:
        positions = set()
        cursor = 0

        for segment in segments[:-1]:
            cursor += len(segment)
            positions.add(cursor)

        return positions

    cleaned_predicted_tokens = [
        clean_subword_marker(token)
        for token in predicted_tokens
        if clean_subword_marker(token)
    ]

    gold_boundaries = boundaries(gold_segments)
    predicted_boundaries = boundaries(cleaned_predicted_tokens)

    if not gold_boundaries and not predicted_boundaries:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        }

    true_positive = len(gold_boundaries & predicted_boundaries)

    precision = true_positive / max(len(predicted_boundaries), 1)
    recall = true_positive / max(len(gold_boundaries), 1)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def summarize_token_counts(
    texts: List[str],
    encode_fn: Callable[[str], List[str]],
) -> Counter:
    """
    Count tokenizer output pieces across a list of texts.
    """
    counter = Counter()

    for text in texts:
        clean_text = strip_language_tag(text)
        counter.update(encode_fn(clean_text))

    return counter