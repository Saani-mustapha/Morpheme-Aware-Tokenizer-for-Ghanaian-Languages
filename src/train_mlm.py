import argparse
import json
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from transformers import (
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)


def build_fast_tokenizer(tokenizer_json_path: str) -> PreTrainedTokenizerFast:
    """
    Wrap a Hugging Face tokenizers tokenizer.json as a Transformers tokenizer.

    This works for both:
        tokenizers/bpe/tokenizer.json
        tokenizers/unigram/tokenizer.json
    """
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_json_path,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )

    return tokenizer


def create_training_arguments(output_dir: str) -> TrainingArguments:
    """
    Create TrainingArguments in a way that works across different
    Transformers versions.

    Some versions use eval_strategy.
    Older versions use evaluation_strategy.
    """
    common_args = dict(
        output_dir=output_dir,
        save_strategy="epoch",
        learning_rate=5e-4,
        weight_decay=0.01,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=10,
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        report_to="none",
        save_total_limit=2,
    )

    try:
        return TrainingArguments(
            **common_args,
            eval_strategy="epoch",
        )
    except TypeError:
        return TrainingArguments(
            **common_args,
            evaluation_strategy="epoch",
        )


def create_trainer(
    model,
    training_args,
    train_dataset,
    eval_dataset,
    data_collator,
    tokenizer,
) -> Trainer:
    """
    Create a Trainer in a version-safe way.

    Newer Transformers versions use processing_class.
    Older versions use tokenizer.
    """
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    try:
        return Trainer(
            **trainer_kwargs,
            processing_class=tokenizer,
        )
    except TypeError:
        return Trainer(
            **trainer_kwargs,
            tokenizer=tokenizer,
        )


def save_metrics(metrics: dict, output_dir: str):
    """
    Save MLM evaluation metrics to a JSON file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics_file = output_path / "eval_metrics.json"

    with metrics_file.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved evaluation metrics to {metrics_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--tokenizer_json", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    tokenizer_path = Path(args.tokenizer_json)

    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer file not found: {tokenizer_path}"
        )

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = build_fast_tokenizer(str(tokenizer_path))

    print(f"Loaded tokenizer from: {tokenizer_path}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")

    dataset = load_dataset(
        "text",
        data_files={
            "train": cfg["data"]["train_file"],
            "validation": cfg["data"]["valid_file"],
        },
    )

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding=False,
            max_length=128,
        )

    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"],
    )

    config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=1024,
        max_position_embeddings=256,
        pad_token_id=tokenizer.pad_token_id,
        type_vocab_size=2,
    )

    model = BertForMaskedLM(config)
    model.to(device)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    training_args = create_training_arguments(args.output_dir)

    trainer = create_trainer(
        model=model,
        training_args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    metrics = trainer.evaluate()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    save_metrics(metrics, args.output_dir)

    print()
    print("Final evaluation metrics:")
    print(metrics)


if __name__ == "__main__":
    main()