import argparse
from pathlib import Path

import pandas as pd
import sentencepiece as spm
import yaml
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from metrics import (
    tokenization_fertility,
    char_per_token_ratio,
    unk_rate,
    vocabulary_coverage,
    token_length_stats,
)


def read_lines(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def get_language(text: str) -> str:
    if text.startswith("<") and ">" in text:
        return text[1:text.index(">")]
    return "unknown"


def load_custom_bpe(path: str):
    tok = Tokenizer.from_file(path)

    def encode(text: str):
        return tok.encode(text).tokens

    vocab = set(tok.get_vocab().keys())
    return encode, vocab


def load_custom_unigram(path: str):
    sp = spm.SentencePieceProcessor(model_file=path)

    def encode(text: str):
        return sp.encode(text, out_type=str)

    vocab = set(sp.id_to_piece(i) for i in range(sp.get_piece_size()))
    return encode, vocab


def load_hf_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def encode(text: str):
        return tok.tokenize(text)

    vocab = set(tok.get_vocab().keys())
    return encode, vocab


def evaluate_one(name, texts, encode_fn, vocab):
    result = {
        "tokenizer": name,
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

    test_texts = read_lines(cfg["data"]["test_file"])

    tokenizers = {}

    bpe_encode, bpe_vocab = load_custom_bpe(
        str(Path(cfg["outputs"]["bpe_dir"]) / "tokenizer.json")
    )
    tokenizers["custom_bpe"] = (bpe_encode, bpe_vocab)

    unigram_encode, unigram_vocab = load_custom_unigram(
        str(Path(cfg["outputs"]["unigram_dir"]) / "unigram.model")
    )
    tokenizers["custom_unigram"] = (unigram_encode, unigram_vocab)

    for baseline in cfg["baselines"]:
        encode, vocab = load_hf_tokenizer(baseline)
        tokenizers[baseline] = (encode, vocab)

    rows = []

    # Overall evaluation.
    for name, (encode_fn, vocab) in tokenizers.items():
        rows.append({
            "language": "all",
            **evaluate_one(name, test_texts, encode_fn, vocab)
        })

    # Per-language evaluation.
    languages = sorted(set(get_language(t) for t in test_texts))

    for lang in languages:
        lang_texts = [t for t in test_texts if get_language(t) == lang]

        for name, (encode_fn, vocab) in tokenizers.items():
            rows.append({
                "language": lang,
                **evaluate_one(name, lang_texts, encode_fn, vocab)
            })

    df = pd.DataFrame(rows)

    output_dir = Path(cfg["outputs"]["reports_dir"]) / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "tokenizer_metrics.csv"
    df.to_csv(output_file, index=False)

    print(df)
    print(f"Saved metrics to {output_file}")


if __name__ == "__main__":
    main()