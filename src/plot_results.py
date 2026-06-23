import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml


def plot_metric(df, metric, output_dir):
    overall = df[df["language"] == "all"].sort_values(metric)

    plt.figure(figsize=(10, 6))
    plt.bar(overall["tokenizer"], overall[metric])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(metric)
    plt.title(f"Tokenizer comparison: {metric}")
    plt.tight_layout()

    output_path = output_dir / f"{metric}.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    reports_dir = Path(cfg["outputs"]["reports_dir"])
    table_path = reports_dir / "tables" / "tokenizer_metrics.csv"
    output_dir = reports_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(table_path)

    for metric in [
        "fertility",
        "char_per_token",
        "unk_rate",
        "vocab_coverage",
        "mean_token_chars",
    ]:
        plot_metric(df, metric, output_dir)


if __name__ == "__main__":
    main()