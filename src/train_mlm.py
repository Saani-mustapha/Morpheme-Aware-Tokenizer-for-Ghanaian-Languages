import argparse
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from tokenizers import Tokenizer
from transformers import (
    BertConfig,
    BertForMaskedLM,
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def build_fast_tokenizer(tokenizer_json_path: str):
    """
    Wrap the custom Hugging Face tokenizers tokenizer.json as a Transformers tokenizer.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--tokenizer_json", default="tokenizers/bpe/tokenizer.json")
    parser.add_argument("--output_dir", default="models/custom_bpe_mlm")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = build_fast_tokenizer(args.tokenizer_json)

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

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
        processing_class=tokenizer,
    )

    trainer.train()
    metrics = trainer.evaluate()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(metrics)


if __name__ == "__main__":
    main()