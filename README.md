# SceneOrchestra

Official training, data-generation, and inference code for **SceneOrchestra: Efficient Agentic 3D Scene Synthesis via Full Tool-Call Trajectory Generation**.

- [Paper](https://arxiv.org/abs/2604.19907)
- [Project page](https://yunhe24.github.io/sceneorchestra/)
- [Video](https://www.youtube.com/watch?v=nPEjesP1ikw)
- [Official SceneWeaver repository](https://github.com/Scene-Weaver/SceneWeaver)

SceneOrchestra learns to predict an entire SceneWeaver tool-call trajectory from a text instruction. At inference time the predicted calls are validated and executed end to end, avoiding SceneWeaver's repeated execute-review-reflect planning loop. SceneWeaver still supplies the scene-generation tools and assets.

## What is included

This source release contains:

- the LLaMAFactory training core used for Qwen3 LoRA SFT and DPO;
- paper-aligned construction of stepwise SFT, trajectory SFT, stepwise DPO, trajectory DPO, and discriminator data;
- the one-cycle interleaved discriminator/orchestrator training pipeline;
- a wrapper that launches the original SceneWeaver agent to collect rollout trajectories;
- a parser that converts SceneWeaver outputs into a portable scored-rollout format;
- SceneWeaver execution adapters for an alternative next step and a full predicted trajectory;
- inference, trajectory validation, and complete training configurations.

This repository intentionally contains **no training instructions, rollouts, generated training datasets, checkpoints, model weights, scenes, renders, logs, or inference outputs**. Those paths and large SceneWeaver artefacts are excluded by `.gitignore`.

SceneWeaver itself is also not vendored. Clone its official repository separately and pass its path with `--sceneweaver-root` (or set `SCENEWEAVER_ROOT`).

## Method-to-code map

| Paper component | Implementation |
| --- | --- |
| Quality and composition scores, Eqs. (3)-(4) | `src/sceneorchestra/scoring.py` |
| Stepwise and trajectory SFT | `build_stepwise_sft`, `build_trajectory_sft` in `datasets.py` |
| Stepwise and trajectory DPO | `build_stepwise_dpo*`, `build_trajectory_dpo` in `datasets.py` |
| Independent discriminator | `sample_discriminator_groups`, `build_discriminator_sft` |
| Interleaved S2 execution and discriminator update | `interleaved.py`, `05_interleaved_discriminator.yaml` |
| Interleaved S3 ranking and orchestrator distillation | `rank-candidates`, `build_interleaved_dpo`, `06_interleaved_orchestrator.yaml` |
| Full-trajectory inference | `generate-trajectory`, `execute-trajectory`, and `infer` |
| SceneWeaver integration | `sceneweaver.py` |

The paper values are the defaults in `constants.py`:

```text
alpha = 4, lambda = 0.1, gamma = 0.05
tau_1 = 3, tau_2 = 7.5, tau_3 = 3, tau_4 = 3
```

For step `t`, the implementation computes

```text
Q_phy = N_obj - alpha * (N_ob + N_col)
Q_vis = (S_real + S_func + S_lay + S_comp) / 4
Q     = lambda * Q_phy + Q_vis
C     = Q - gamma * T
```

`T` is cumulative runtime in minutes. Object count is read from the SceneWeaver physics metric; when older SceneWeaver outputs contain `"Unknown"`, it is recovered from `record_scene/layout_<t>.json`.

## Repository layout

```text
configs/
  train/                 all independent and interleaved training stages
  inference/             orchestrator/discriminator model loading
scripts/                 ordered training launchers
src/llamafactory/        retained LLaMAFactory training framework
src/sceneorchestra/      scoring, data construction, execution, and inference
tests/                   synthetic unit tests (no research data)
```

## Installation

### 1. Training and model-generation environment

Python 3.11 and a CUDA-compatible PyTorch installation are recommended.

```bash
git clone <SCENEORCHESTRA_REPOSITORY_URL> SceneOrchestra
cd SceneOrchestra
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

The repository ships the LLaMAFactory core, so no second LLaMAFactory clone is needed. Verify both command-line entry points:

```bash
sceneorchestra --help
llamafactory-cli version
```

### 2. SceneWeaver environment

Follow all setup, asset, Blender, API-key, and executor instructions in the [official SceneWeaver README](https://github.com/Scene-Weaver/SceneWeaver). In outline:

```bash
git clone https://github.com/Scene-Weaver/SceneWeaver.git /path/to/SceneWeaver
export SCENEWEAVER_ROOT=/path/to/SceneWeaver
```

Install this repository's lightweight commands into the SceneWeaver planner environment without replacing SceneWeaver's dependency versions:

```bash
conda activate sceneweaver
pip install -e /path/to/SceneOrchestra --no-deps
pip install PyYAML
```

The commands that load Qwen adapters run in the training environment. The commands that execute SceneWeaver tools run in the SceneWeaver environment. A combined environment is supported, but is not required.

## Input instructions

Instruction files are user-provided and are never committed. Three formats are accepted:

```text
# instructions.txt: one instruction per non-empty line
Design a functional home office with two desks.
```

```json
["Design a functional home office with two desks."]
```

```json
{"instruction": "Design a functional home office with two desks."}
```

The last form is JSONL (one object per line).

### Optional GPT-assisted instruction preparation

The paper uses GPT-4 to propose detailed instructions, followed by manual checking and filtering. Install the optional client, provide its standard API credentials through the environment, and pass the exact model deployment/name available to you:

```bash
pip install -e '.[data]'

sceneorchestra generate-instructions \
  --num 100 \
  --model YOUR_GPT4_MODEL_NAME \
  --existing /path/to/existing_instructions.json \
  --output outputs/instructions/candidates.jsonl
```

Generated records have `"approved": false` by design. Review them manually and create the private S1/S2/S3 instruction files from accepted candidates.
The generator retains the development script's diversity constraints, no-style/theme constraint, common-room preference, semantic duplicate avoidance, batches of at most 30, JSON-only parsing, completion-token API fallback, and checkpointed output. `scripts/generate_instructions.py` provides its original `-n/-o` style entry point.

To reproduce instruction-rephrasing augmentation, generate candidates and manually change `approved` to `true` only after checking that all counts, objects, positions, relations, and constraints are unchanged:

```bash
sceneorchestra rephrase-instructions \
  --instructions /path/to/S1.jsonl \
  --model YOUR_GPT4_MODEL_NAME \
  --variants 3 \
  --output outputs/instructions/s1_rephrases_review.jsonl
```

## Phase 1: independent training

The commands below use `S1` for the paper's initial training instruction set.

### A. Collect original SceneWeaver rollouts

Run this from the training environment while pointing `--python` to the Python executable of the fully configured SceneWeaver planner environment:

```bash
sceneorchestra generate-rollouts \
  --instructions /path/to/S1.jsonl \
  --sceneweaver-root "$SCENEWEAVER_ROOT" \
  --python /path/to/sceneweaver/env/bin/python \
  --repeats 4 \
  --output-dir outputs/s1_raw
```

This invokes SceneWeaver's original `Pipeline/main.py`, so each rollout uses its native execute-review-reflect loop. A resumable `outputs/s1_raw/manifest.jsonl` records the instruction, run directory, log, wall time, and return code.

### B. Parse and score rollouts

```bash
sceneorchestra parse-rollouts \
  --manifest outputs/s1_raw/manifest.jsonl \
  --output data/s1_rollouts.jsonl
```

The parser combines `pipeline/trajs_<t>.json`, `pipeline/metric_<t>.json`, `record_scene/layout_<t>.json`, and timestamped step markers in the rollout log. It produces one normalized rollout per JSONL line. Failed or incomplete runs stop the command by default; use `--skip-failed` only after inspecting them.

### C. Build the first-pass datasets

```bash
sceneorchestra build-independent \
  --rollouts data/s1_rollouts.jsonl \
  --output-dir data/generated \
  --rephrases outputs/instructions/s1_rephrases_review.jsonl \
  --seed 0 \
  --pairs-per-instruction 4 \
  --discriminator-candidates 4
```

This writes:

- `sceneorchestra_stepwise_sft.jsonl`: steps satisfying `C_t - C_(t-1) > tau_1`;
- `sceneorchestra_trajectory_sft.jsonl`: one random truncation per rollout satisfying `C_t > tau_2`;
- `sceneorchestra_trajectory_dpo.jsonl`: independently truncated rollout pairs satisfying `|C_i-C_j| > tau_4`;
- `sceneorchestra_discriminator.jsonl`: candidate sets labeled with the highest composition-score index;
- `stepwise_dpo_tasks.jsonl`: histories satisfying `|C_t-C_(t-1)| > tau_1` and awaiting a model-generated alternative;
- `dataset_info.json`: the LLaMAFactory dataset registry.

Random operations use the explicit seed. A truncated trajectory is closed with `terminate(status="success")`, so its target has the same complete-trajectory grammar used during inference.
The `--rephrases` option is optional; only records explicitly marked `"approved": true` are used.

### D. Train the two SFT stages

```bash
bash scripts/train_independent.sh sft
```

The second stage initializes from the first stage adapter.

### E. Construct stepwise DPO by executing alternatives

Use the trajectory-SFT orchestrator to predict a new next call for every selected history:

```bash
sceneorchestra sample-stepwise-alternatives \
  --tasks data/generated/stepwise_dpo_tasks.jsonl \
  --config configs/inference/orchestrator_trajectory_sft.yaml \
  --output outputs/stepwise_dpo/alternatives.jsonl
```

Activate the SceneWeaver environment and execute every alternative from a copied `memory_<t-1>.pkl` state:

```bash
conda activate sceneweaver
sceneorchestra execute-stepwise-alternatives \
  --alternatives outputs/stepwise_dpo/alternatives.jsonl \
  --sceneweaver-root "$SCENEWEAVER_ROOT" \
  --output-dir outputs/stepwise_dpo/executions \
  --comparisons outputs/stepwise_dpo/comparisons.jsonl
```

Return to the training environment and keep pairs whose executed score gap exceeds `tau_3`:

```bash
sceneorchestra finalize-stepwise-dpo \
  --comparisons outputs/stepwise_dpo/comparisons.jsonl \
  --rephrases outputs/instructions/s1_rephrases_review.jsonl \
  --output-dir data/generated
```

The raw S1 output directories must remain available until this step completes because their serialized SceneWeaver state is the counterfactual execution starting point. Alternatives are executed only for the original instruction; manually approved semantic rephrases are applied after scoring as training augmentation.

### F. Train both DPO stages and the independent discriminator

```bash
bash scripts/train_independent.sh rest
```

Each orchestrator stage initializes from the previous checkpoint. In both DPO configs the previous checkpoint is also the reference adapter, matching the paper.

## Phase 2: one interleaved cycle

The paper performs one S2/S3 cycle.

### S2: adapt the discriminator to the orchestrator distribution

Sample multiple trajectories per S2 instruction with the independently trained orchestrator:

```bash
sceneorchestra generate-candidates \
  --instructions /path/to/S2.jsonl \
  --config configs/inference/orchestrator_independent.yaml \
  --candidates 4 \
  --output outputs/interleaved/s2_candidates.jsonl
```

Execute and score them in the SceneWeaver environment:

```bash
sceneorchestra execute-candidates \
  --candidates outputs/interleaved/s2_candidates.jsonl \
  --sceneweaver-root "$SCENEWEAVER_ROOT" \
  --output-dir outputs/interleaved/s2_executions \
  --groups outputs/interleaved/s2_scored_groups.jsonl
```

Create the updated discriminator data and train it:

```bash
sceneorchestra build-discriminator-from-groups \
  --groups outputs/interleaved/s2_scored_groups.jsonl \
  --output-dir data/interleaved_s2

llamafactory-cli train configs/train/05_interleaved_discriminator.yaml
```

### S3: distill the updated discriminator into the orchestrator

S3 must be a new instruction set. Candidate generation and discriminator ranking require no SceneWeaver execution:

```bash
sceneorchestra generate-candidates \
  --instructions /path/to/S3.jsonl \
  --config configs/inference/orchestrator_independent.yaml \
  --candidates 4 \
  --output outputs/interleaved/s3_candidates.jsonl

sceneorchestra rank-candidates \
  --candidates outputs/interleaved/s3_candidates.jsonl \
  --config configs/inference/discriminator.yaml \
  --output outputs/interleaved/s3_rankings.jsonl

sceneorchestra build-interleaved-dpo \
  --rankings outputs/interleaved/s3_rankings.jsonl \
  --output-dir data/interleaved_s3

llamafactory-cli train configs/train/06_interleaved_orchestrator.yaml
```

For each S3 group the discriminator-selected best trajectory is paired against every other candidate. Only the final orchestrator is needed after this stage.

## Inference

The recommended two-environment workflow first generates and validates the trajectory, then executes it using SceneWeaver.

In the training/model environment:

```bash
sceneorchestra generate-trajectory \
  --instruction "Design me a bedroom." \
  --config configs/inference/orchestrator.yaml \
  --output inference_outputs/bedroom/trajectory.txt
```

In the SceneWeaver environment:

```bash
sceneorchestra execute-trajectory \
  --trajectory inference_outputs/bedroom/trajectory.txt \
  --instruction "Design me a bedroom." \
  --sceneweaver-root "$SCENEWEAVER_ROOT" \
  --output-dir inference_outputs/bedroom/scene
```

This executes the calls end to end and does not render/review between calls. Add `--evaluate-final` only when a final research metric is needed. If one environment contains both dependency sets, the combined convenience command is:

```bash
sceneorchestra infer \
  --instruction "Design me a bedroom." \
  --config configs/inference/orchestrator.yaml \
  --sceneweaver-root "$SCENEWEAVER_ROOT" \
  --output-dir inference_outputs/bedroom/scene
```

Generated text is parsed through Python's AST and only literal keyword arguments are accepted. The tool name, first initializer, final terminator, and terminator position are validated. Invalid generations are retried (three attempts by default), following the mitigation discussed in the paper's limitations.

## Training configuration notes

- Base model: `Qwen/Qwen3-4B-Instruct-2507` for both orchestrator and discriminator.
- Fine-tuning: LoRA rank 8 with all linear targets.
- Context length: 4096; prompt template: `qwen3_nothink`.
- DPO: sigmoid loss with beta 0.1.
- The optimizer, epoch, batching, and sampling defaults reproduce the working configuration from the development repository. They are editable YAML values; paper-defined scoring constants and selection thresholds are implemented separately in `constants.py`.
- Outputs are stage-separated. The final independent orchestrator is `saves/orchestrator/04_trajectory_dpo`; the final interleaved orchestrator is `saves/orchestrator/06_interleaved_dpo`.

For multi-GPU training, launch the same YAML through the normal LLaMAFactory/torchrun mechanism supported by your cluster. Do not change dataset names unless the matching `dataset_info.json` entries are also changed.

## Normalized rollout schema

Each line of `s1_rollouts.jsonl` has this structure (values below are schematic, not released data):

```json
{
  "instruction": "...",
  "rollout_id": "i00000-r000",
  "source_dir": "/absolute/path/to/raw/run",
  "steps": [
    {
      "index": 0,
      "call": {"name": "init_gpt", "arguments": {"ideas": "..."}},
      "cumulative_minutes": 12.3,
      "metric": {},
      "score": {"physical": 20.0, "visual": 8.0, "quality": 10.0, "composition": 9.385}
    }
  ]
}
```

LLaMAFactory SFT files contain `prompt` and `response`. DPO files contain `prompt`, `chosen`, and `rejected`, with `ranking: true` in `dataset_info.json`.

## Tests

The test suite uses only synthetic metrics and trajectories:

```bash
pip install -e '.[dev]'
pytest -q
```

For a real end-to-end smoke test, use one private instruction and one rollout. SceneWeaver execution is expensive and requires its external assets/API services, so it is intentionally not part of unit tests.

## Reproducibility and data hygiene

- All random dataset sampling accepts an explicit seed and uses a local RNG.
- The normalized record stores the raw run path and rollout ID for provenance.
- The rollout manifest is updated after every run.
- Stepwise counterfactual execution refuses to overwrite an existing candidate workspace.
- Configuration, code, and synthetic tests are tracked; generated data and binary artefacts are not.
- Before publishing, run `git status --ignored` and verify that no instruction set, dataset, scene, render, log, API key, adapter, or checkpoint is staged.

## Third-party code

The retained training framework is derived from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and remains under the Apache License 2.0. See `LICENSE` and `NOTICE`.

SceneWeaver is a separate external project under its own license. This repository imports its public Python interfaces at runtime but does not redistribute its source, assets, or generated outputs.

## Citation

```bibtex
@article{he2026sceneorchestra,
  title   = {SceneOrchestra: Efficient Agentic 3D Scene Synthesis via Full Tool-Call Trajectory Generation},
  author  = {He, Yun and Yu, Kelin and Zwicker, Matthias},
  journal = {arXiv preprint arXiv:2604.19907},
  year    = {2026}
}
```
