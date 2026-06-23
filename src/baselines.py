from transformers import AutoTokenizer


def load_baseline_tokenizers(model_names: list[str]):
    """
    Load multilingual baseline tokenizers.

    Examples:
    - mBERT: bert-base-multilingual-cased
    - XLM-R: xlm-roberta-base
    - mT5: google/mt5-small
    """
    tokenizers = {}

    for name in model_names:
        print(f"Loading baseline tokenizer: {name}")
        tokenizers[name] = AutoTokenizer.from_pretrained(name, use_fast=True)

    return tokenizers