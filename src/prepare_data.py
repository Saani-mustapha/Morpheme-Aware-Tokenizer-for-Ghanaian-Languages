import argparse
import random
from pathlib import Path
from typing import List

import yaml
from tqdm import tqdm

from clean_text import clean_line, is_good_line


def read_language_files(raw_dir: Path, languages: List[str], lowercase: bool) -> List[str]:
    lines = []

    for lang in languages:
        lang_dir = raw_dir / lang
        if not lang_dir.exists():
            print(f"Warning: missing directory {lang_dir}")
            continue

        for file_path in lang_dir.glob("*.txt"):
            print(f"Reading {file_path}")
            with file_path.open("r", encoding="utf-8") as f:
                for line in tqdm(f, desc=f"{lang}/{file_path.name}"):
                    cleaned = clean_line(line, lowercase=lowercase)
                    if is_good_line(cleaned):
                        # Prefix language tag to enable language-aware analysis later.
                        lines.append(f"<{lang}> {cleaned}")

    return lines


def split_lines(lines: List[str], train_ratio=0.8, valid_ratio=0.1, seed=42):
    random.seed(seed)
    random.shuffle(lines)

    n = len(lines)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)

    train = lines[:n_train]
    valid = lines[n_train:n_train + n_valid]
    test = lines[n_train + n_valid:]

    return train, valid, test


def write_lines(lines: List[str], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(cfg["data"]["raw_dir"])
    processed_dir = Path(cfg["data"]["processed_dir"])
    lowercase = cfg["tokenizer"]["lowercase"]

    lines = read_language_files(
        raw_dir=raw_dir,
        languages=cfg["languages"],
        lowercase=lowercase,
    )

    print(f"Total cleaned lines: {len(lines)}")

    train, valid, test = split_lines(lines)

    write_lines(train, processed_dir / "train.txt")
    write_lines(valid, processed_dir / "valid.txt")
    write_lines(test, processed_dir / "test.txt")

    print("Saved:")
    print(f"  train: {len(train)}")
    print(f"  valid: {len(valid)}")
    print(f"  test:  {len(test)}")


if __name__ == "__main__":
    main()