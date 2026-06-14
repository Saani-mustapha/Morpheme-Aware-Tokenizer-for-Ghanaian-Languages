#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mavp.common import clean_text_line, count_words, ensure_dir, iter_lines, iter_text_files, read_config, set_seed, stable_shuffle, write_lines


def main():
    parser = argparse.ArgumentParser(description="Clean, deduplicate, and split local-language corpora.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = read_config(args.config)
    set_seed(int(cfg.get("seed", 13)))

    processed_dir = ensure_dir(ROOT / cfg["paths"]["processed_dir"])
    splits_dir = ensure_dir(ROOT / cfg["paths"]["splits_dir"])
    results_dir = ensure_dir(ROOT / cfg["paths"]["results_dir"])

    min_words = int(cfg["corpus"].get("min_words_per_line", 3))
    max_chars = int(cfg["corpus"].get("max_chars_per_line", 2000))
    train_ratio = float(cfg["corpus"].get("train_ratio", 0.8))
    valid_ratio = float(cfg["corpus"].get("valid_ratio", 0.1))

    all_splits = {"train": [], "valid": [], "test": []}
    stats = []

    for lang in cfg["languages"]:
        code = lang["code"]
        raw_pattern = str(ROOT / lang["raw_glob"])
        files = list(iter_text_files(raw_pattern))
        seen = set()
        cleaned = []
        for raw in iter_lines(files):
            line = clean_text_line(raw, max_chars=max_chars)
            if not line or count_words(line) < min_words:
                continue
            key = line.casefold()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(line)

        cleaned = stable_shuffle(cleaned, int(cfg.get("seed", 13)))
        n = len(cleaned)
        n_train = int(n * train_ratio)
        n_valid = int(n * valid_ratio)
        split_map = {
            "train": cleaned[:n_train],
            "valid": cleaned[n_train : n_train + n_valid],
            "test": cleaned[n_train + n_valid :],
        }

        write_lines(processed_dir / f"{code}.txt", cleaned)
        lang_split_dir = ensure_dir(splits_dir / code)
        for split, lines in split_map.items():
            write_lines(lang_split_dir / f"{split}.txt", lines)
            all_splits[split].extend(lines)

        stats.append({
            "language": code,
            "name": lang.get("name", code),
            "raw_files": len(files),
            "clean_lines": n,
            "train_lines": len(split_map["train"]),
            "valid_lines": len(split_map["valid"]),
            "test_lines": len(split_map["test"]),
            "words": sum(count_words(x) for x in cleaned),
        })
        print(f"{code}: {n} cleaned lines from {len(files)} files")

    for split, lines in all_splits.items():
        write_lines(splits_dir / f"all_{split}.txt", lines)

    with open(results_dir / "corpus_stats.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats[0].keys()) if stats else ["language"])
        writer.writeheader()
        writer.writerows(stats)

    print(f"Wrote splits under {splits_dir}")


if __name__ == "__main__":
    main()
