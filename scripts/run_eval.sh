#!/usr/bin/env bash
set -e

python src/evaluate_tokenizers.py --config config.yaml
python src/plot_results.py --config config.yaml