#!/usr/bin/env bash
set -euo pipefail

llamafactory-cli train configs/train/05_interleaved_discriminator.yaml
llamafactory-cli train configs/train/06_interleaved_orchestrator.yaml
