import argparse
from pathlib import Path
from typing import List, Set

import pandas as pd
from tokenizers import Tokenizer
from transformers import AutoTokenizer


def clean_token(token: str) -> str:
    """
    Remove common tokenizer boundary markers.

    Examples:
        ##foɔ -> foɔ
        ▁foɔ  -> foɔ
        Ġfoɔ  -> foɔ
    """
    return (
        token.replace("##", "")
        .replace("▁", "")
        .replace("Ġ", "")
        .strip()
    )


def boundary_positions(segments: List[str]) -> Set[int]:
    """
    Convert a segmentation into boundary positions.

    Example:
        ["sukuu", "foɔ"]

    sukuu has length 5, so the boundary is {5}.
    """
    boundaries = set()
    cursor = 0

    for segment in segments[:-1]:
        cursor += len(segment)
        boundaries.add(cursor)

    return boundaries


def tokens_to_boundaries(tokens: List[str]) -> Set[int]:
    """
    Convert tokenizer output tokens into boundary positions.

    Example:
        ["sukuu", "foɔ"] -> {5}
        ["su", "kuu", "foɔ"] -> {2, 5}
    """
    cleaned_tokens = [clean_token(token) for token in tokens]
    cleaned_tokens = [token for token in cleaned_tokens if token]

    return boundary_positions(cleaned_tokens)


def score_boundaries(gold_boundaries: Set[int], predicted_boundaries: Set[int]) -> dict:
    """
    Compute precision, recall, and F1 for one word.
    """
    if not gold_boundaries and not predicted_boundaries:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "true_positive": 0,
            "false_positive": 0,
            "false_negative": 0,
        }

    true_positive = len(gold_boundaries & predicted_boundaries)
    false_positive = len(predicted_boundaries - gold_boundaries)
    false_negative = len(gold_boundaries - predicted_boundaries)

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def load_custom_tokenizer(tokenizer_path: str):
    tokenizer = Tokenizer.from_file(tokenizer_path)

    def tokenize(word: str) -> List[str]:
        return tokenizer.encode(word).tokens

    return tokenize


def load_baseline_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def tokenize(word: str) -> List[str]:
        return tokenizer.tokenize(word)

    return tokenize


def evaluate_tokenizer_on_gold(
    tokenizer_name: str,
    tokenize_fn,
    gold_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for _, row in gold_df.iterrows():
        word = str(row["word"]).strip()
        gold_segments = str(row["segments"]).strip().split("|")

        predicted_tokens = tokenize_fn(word)

        gold_boundaries = boundary_positions(gold_segments)
        predicted_boundaries = tokens_to_boundaries(predicted_tokens)

        scores = score_boundaries(
            gold_boundaries=gold_boundaries,
            predicted_boundaries=predicted_boundaries,
        )

        rows.append(
            {
                "tokenizer": tokenizer_name,
                "word": word,
                "gold_segments": "|".join(gold_segments),
                "predicted_tokens": "|".join(predicted_tokens),
                "gold_boundaries": ",".join(map(str, sorted(gold_boundaries))),
                "predicted_boundaries": ",".join(map(str, sorted(predicted_boundaries))),
                **scores,
            }
        )

    return pd.DataFrame(rows)


def macro_average(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average precision, recall, and F1 across words.
    """
    return (
        df.groupby("tokenizer", as_index=False)
        .agg(
            num_words=("word", "count"),
            boundary_precision=("precision", "mean"),
            boundary_recall=("recall", "mean"),
            boundary_f1=("f1", "mean"),
            total_true_positive=("true_positive", "sum"),
            total_false_positive=("false_positive", "sum"),
            total_false_negative=("false_negative", "sum"),
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--bpe_tokenizer", default="tokenizers/bpe/tokenizer.json")
    parser.add_argument("--unigram_tokenizer", default="tokenizers/unigram/tokenizer.json")
    parser.add_argument("--output_dir", default="reports/tables")
    args = parser.parse_args()

    gold_path = Path(args.gold)

    if not gold_path.exists():
        raise FileNotFoundError(f"Gold morpheme file not found: {gold_path}")

    gold_df = pd.read_csv(gold_path, sep="\t")

    required_columns = {"word", "segments"}
    missing = required_columns - set(gold_df.columns)

    if missing:
        raise ValueError(f"Gold file is missing columns: {missing}")

    tokenizers = {}

    if Path(args.bpe_tokenizer).exists():
        tokenizers["custom_bpe"] = load_custom_tokenizer(args.bpe_tokenizer)
    else:
        print(f"Warning: BPE tokenizer not found: {args.bpe_tokenizer}")

    if Path(args.unigram_tokenizer).exists():
        tokenizers["custom_unigram"] = load_custom_tokenizer(args.unigram_tokenizer)
    else:
        print(f"Warning: unigram tokenizer not found: {args.unigram_tokenizer}")

    # Optional baselines.
    baseline_names = [
        "bert-base-multilingual-cased",
        "xlm-roberta-base",
        "google/mt5-small",
    ]

    for name in baseline_names:
        tokenizers[name] = load_baseline_tokenizer(name)

    all_word_rows = []

    for tokenizer_name, tokenize_fn in tokenizers.items():
        result_df = evaluate_tokenizer_on_gold(
            tokenizer_name=tokenizer_name,
            tokenize_fn=tokenize_fn,
            gold_df=gold_df,
        )
        all_word_rows.append(result_df)

    word_level_df = pd.concat(all_word_rows, ignore_index=True)
    summary_df = macro_average(word_level_df)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    language_name = gold_path.stem.replace("_gold", "")

    word_output = output_dir / f"{language_name}_morph_boundary_word_level.csv"
    summary_output = output_dir / f"{language_name}_morph_boundary_summary.csv"

    word_level_df.to_csv(word_output, index=False)
    summary_df.to_csv(summary_output, index=False)

    print()
    print("Morpheme boundary summary:")
    print(summary_df)

    print()
    print(f"Saved word-level results to: {word_output}")
    print(f"Saved summary results to: {summary_output}")


if __name__ == "__main__":
    main()