import argparse
from pathlib import Path

import yaml
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFC, Sequence
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing


def train_bpe_tokenizer(
    train_file: str,
    output_dir: str,
    vocab_size: int,
    min_frequency: int,
    special_tokens: list[str],
):
    """
    Train a BPE tokenizer on Ghanaian language text.
    BPE starts with smaller units and learns frequent merges.
    This should reduce tokenization bloat for frequent morphemes and word pieces.
    """
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    tokenizer.normalizer = Sequence([NFC()])
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )

    tokenizer.train(files=[train_file], trainer=trainer)

    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tokenizer.save(str(output_path / "tokenizer.json"))
    print(f"Saved BPE tokenizer to {output_path / 'tokenizer.json'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_bpe_tokenizer(
        train_file=cfg["data"]["train_file"],
        output_dir=cfg["outputs"]["bpe_dir"],
        vocab_size=cfg["tokenizer"]["vocab_size"],
        min_frequency=cfg["tokenizer"]["min_frequency"],
        special_tokens=cfg["special_tokens"],
    )


if __name__ == "__main__":
    main()