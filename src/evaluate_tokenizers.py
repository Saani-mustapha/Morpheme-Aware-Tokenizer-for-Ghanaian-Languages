import argparse
from pathlib import Path
from typing import Callable

import pandas as pd
import yaml
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from metrics import (
    char_per_token_ratio,
    get_language_tag,
    token_length_stats,
    tokenization_fertility,
    unk_rate,
    vocabulary_coverage,
)


def read_lines(path: str) -> list[str]:
    """
    Read non-empty lines from a text file.
    """
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_hf_json_tokenizer(tokenizer_json_path: str) -> tuple[Callable[[str], list[str]], set[str]]:
    """
    Load a custom Hugging Face tokenizer saved as tokenizer.json.

    This works for both:
        tokenizers/bpe/tokenizer.json
        tokenizers/unigram/tokenizer.json
    """
    tokenizer = Tokenizer.from_file(tokenizer_json_path)

    def encode(text: str) -> list[str]:
        return tokenizer.encode(text).tokens

    vocab = set(tokenizer.get_vocab().keys())

    return encode, vocab


def load_transformers_tokenizer(model_name: str) -> tuple[Callable[[str], list[str]], set[str]]:
    """
    Load a pretrained multilingual baseline tokenizer from Hugging Face.

    Examples:
        bert-base-multilingual-cased
        xlm-roberta-base
        google/mt5-small
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def encode(text: str) -> list[str]:
        return tokenizer.tokenize(text)

    vocab = set(tokenizer.get_vocab().keys())

    return encode, vocab


def evaluate_one_tokenizer(
    tokenizer_name: str,
    texts: list[str],
    encode_fn: Callable[[str], list[str]],
    vocab: set[str],
) -> dict:
    """
    Compute all tokenizer metrics for one tokenizer over one text split.
    """
    result = {
        "tokenizer": tokenizer_name,
        "num_examples": len(texts),
        "fertility": tokenization_fertility(texts, encode_fn),
        "char_per_token": char_per_token_ratio(texts, encode_fn),
        "unk_rate": unk_rate(texts, encode_fn),
        "vocab_coverage": vocabulary_coverage(texts, vocab),
    }

    result.update(token_length_stats(texts, encode_fn))

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    test_file = cfg["data"]["test_file"]
    test_texts = read_lines(test_file)

    print(f"Loaded {len(test_texts)} test examples from {test_file}")

    tokenizers = {}

    # Custom BPE tokenizer.
    bpe_tokenizer_path = Path(cfg["outputs"]["bpe_dir"]) / "tokenizer.json"
    if bpe_tokenizer_path.exists():
        encode_fn, vocab = load_hf_json_tokenizer(str(bpe_tokenizer_path))
        tokenizers["custom_bpe"] = {
            "encode_fn": encode_fn,
            "vocab": vocab,
        }
        print(f"Loaded custom BPE tokenizer from {bpe_tokenizer_path}")
    else:
        print(f"Warning: BPE tokenizer not found at {bpe_tokenizer_path}")

    # Custom Hugging Face unigram tokenizer.
    unigram_tokenizer_path = Path(cfg["outputs"]["unigram_dir"]) / "tokenizer.json"
    if unigram_tokenizer_path.exists():
        encode_fn, vocab = load_hf_json_tokenizer(str(unigram_tokenizer_path))
        tokenizers["custom_unigram"] = {
            "encode_fn": encode_fn,
            "vocab": vocab,
        }
        print(f"Loaded custom unigram tokenizer from {unigram_tokenizer_path}")
    else:
        print(f"Warning: unigram tokenizer not found at {unigram_tokenizer_path}")

    # Multilingual baseline tokenizers.
    for baseline_name in cfg["baselines"]:
        encode_fn, vocab = load_transformers_tokenizer(baseline_name)
        tokenizers[baseline_name] = {
            "encode_fn": encode_fn,
            "vocab": vocab,
        }
        print(f"Loaded baseline tokenizer: {baseline_name}")

    rows = []

    # Overall evaluation across all languages.
    for tokenizer_name, tokenizer_data in tokenizers.items():
        rows.append(
            {
                "language": "all",
                **evaluate_one_tokenizer(
                    tokenizer_name=tokenizer_name,
                    texts=test_texts,
                    encode_fn=tokenizer_data["encode_fn"],
                    vocab=tokenizer_data["vocab"],
                ),
            }
        )

    # Per-language evaluation.
    languages = sorted(set(get_language_tag(text) for text in test_texts))

    for language in languages:
        language_texts = [
            text for text in test_texts
            if get_language_tag(text) == language
        ]

        for tokenizer_name, tokenizer_data in tokenizers.items():
            rows.append(
                {
                    "language": language,
                    **evaluate_one_tokenizer(
                        tokenizer_name=tokenizer_name,
                        texts=language_texts,
                        encode_fn=tokenizer_data["encode_fn"],
                        vocab=tokenizer_data["vocab"],
                    ),
                }
            )

    df = pd.DataFrame(rows)

    output_dir = Path(cfg["outputs"]["reports_dir"]) / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "tokenizer_metrics.csv"
    df.to_csv(output_file, index=False)

    print()
    print(df)
    print()
    print(f"Saved tokenizer metrics to {output_file}")


if __name__ == "__main__":
    main()