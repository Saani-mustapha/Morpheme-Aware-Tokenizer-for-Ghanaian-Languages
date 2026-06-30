import argparse
from pathlib import Path

import yaml
from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.normalizers import NFC, Sequence
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import UnigramTrainer
from tokenizers.processors import TemplateProcessing


def train_unigram_tokenizer(
    train_file: str,
    output_dir: str,
    vocab_size: int,
    special_tokens: list[str],
):
    """
    Train a Hugging Face Unigram tokenizer and save it as tokenizer.json.

    This makes the unigram tokenizer usable with PreTrainedTokenizerFast,
    so we can train a masked language model with it, just like the BPE tokenizer.
    """

    tokenizer = Tokenizer(Unigram())

    tokenizer.normalizer = Sequence([NFC()])
    tokenizer.pre_tokenizer = Whitespace()

    trainer = UnigramTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        unk_token="[UNK]",
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

    print(f"Saved Hugging Face unigram tokenizer to {output_path / 'tokenizer.json'}")


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
        special_tokens=cfg["special_tokens"],
    )


if __name__ == "__main__":
    main()