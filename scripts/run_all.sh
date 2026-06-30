#!/usr/bin/env bash
set -e

bash scripts/run_prepare.sh
bash scripts/run_train_tokenizers.sh
bash scripts/run_eval.sh