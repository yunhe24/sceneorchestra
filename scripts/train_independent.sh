#!/usr/bin/env bash
set -euo pipefail

mode="${1:-all}"
if [[ "$mode" == "sft" || "$mode" == "all" ]]; then
  llamafactory-cli train configs/train/01_stepwise_sft.yaml
  llamafactory-cli train configs/train/02_trajectory_sft.yaml
fi

if [[ "$mode" == "rest" || "$mode" == "all" ]]; then
  if [[ ! -f data/generated/sceneorchestra_stepwise_dpo.jsonl ]]; then
    echo "Missing stepwise DPO data. Run sample/execute/finalize-stepwise-dpo first." >&2
    exit 1
  fi
  llamafactory-cli train configs/train/03_stepwise_dpo.yaml
  llamafactory-cli train configs/train/04_trajectory_dpo.yaml
  llamafactory-cli train configs/train/discriminator.yaml
fi
