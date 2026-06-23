import argparse
from pathlib import Path

import sentencepiece as spm
import yaml


def train_unigram_tokenizer(
    train_file: str,
    output_dir: str,
    vocab_size: int,
):
    """
    Train a SentencePiece unigram tokenizer.

    SentencePiece can train directly from raw sentences and supports unigram
    language-model tokenization. This is useful because we do not need perfect
    whitespace pre-tokenization.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_prefix = str(output_path / "unigram")

    spm.SentencePieceTrainer.Train(
        input=train_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=1.0,
        unk_id=0,
        pad_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols="[CLS],[SEP],[MASK]",
        input_sentence_size=2_000_000,
        shuffle_input_sentence=True,
    )

    print(f"Saved unigram model to {model_prefix}.model")
    print(f"Saved unigram vocab to {model_prefix}.vocab")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_unigram_tokenizer(
        train_file=cfg["data"]["train_file"],
        output_dir=cfg["outputs"]["unigram_dir"],
        vocab_size=cfg["tokenizer"]["vocab_size"],
    )


if __name__ == "__main__":
    main()