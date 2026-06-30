#!/usr/bin/env bash
set -e

python src/train_bpe.py --config config.yaml
python src/train_unigram.py --config config.yaml