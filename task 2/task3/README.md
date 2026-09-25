# Task 3 — Domain Generalization on PACS

This directory contains the implementation and results for Task 3 of ATML PA1. The task compares three approaches to generalization from the labeled PACS source domains—Art Painting, Cartoon, and Photo—to the unseen Sketch domain:

- **ERM:** the Source-only checkpoint from Task 2, reused without retraining.
- **DAN-DG:** ERM classification loss plus pairwise multi-kernel MMD between all observed source-domain pairs.
- **SAM:** ERM trained with non-adaptive Sharpness-Aware Minimization.

Sketch is excluded from training, source-side diagnostics, checkpoint selection, and hyperparameter selection. It is loaded only by the final locked evaluation script.

## Experimental protocol

- Dataset: PACS
- Source domains: `art_painting`, `cartoon`, `photo`
- Withheld target domain: `sketch`
- Classes: dog, elephant, giraffe, guitar, horse, house, person
- Seed: `6304`
- Backbone: ImageNet-initialized ResNet-18
- Feature dimension: `512`
- Source batch size: 8 images per domain
- BatchNorm running statistics: frozen
- Split file: `shared/splits/pacs_sketch_seed6304.json`
- Images root: `data/pacs/images`
- Checkpoint-selection metric: mean source-validation macro-F1
- Main DAN-DG setting: `lambda_dg = 1`
- Main SAM setting: `rho = 0.05`, non-adaptive
- Controlled study: `lambda_dg` in `{0.1, 1, 10}`

The same source splits, preprocessing, augmentation, optimizer settings, epoch budget, early-stopping rule, and seed used in Task 2 are reused here. No Task 2 Sketch result is used to select a Task 3 model or setting.

## Repository structure

```text
task3/
├── configs/
│   ├── erm.yaml
│   ├── dan_dg.yaml
│   └── sam.yaml
├── models/
├── methods/
│   ├── erm.py
│   ├── dan_dg.py
│   └── sam.py
├── selection/
│   └── source_validation.py
├── evaluation/
│   ├── domain_metrics.py
│   ├── source_domain_separability.py
│   └── sharpness.py
├── train.py
├── evaluate_sketch.py
├── plot_results.py
├── combine_report_plots.py
└── results/
```

Shared PACS loading and split logic are located in `shared/pacs.py` and `shared/pacs_protocol.py`. The Task 2 backbone and classifier definitions are reused where required.

## Environment

The experiments were run on Windows with Anaconda and an NVIDIA GeForce RTX 3060 Laptop GPU. The recorded PyTorch installation was `2.13.0+cu130` with CUDA enabled.

Required Python packages include:

```text
torch
torchvision
numpy
pandas
scikit-learn
matplotlib
PyYAML
Pillow
```

Use the repository-level `requirements.txt` or `environment.yml` for pinned versions. Verify the installation with:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Data preparation

Place PACS outside version control using the following structure:

```text
data/pacs/images/
├── art_painting/
├── cartoon/
├── photo/
└── sketch/
```

The expected image counts are:

| Domain | Images |
|---|---:|
| Art Painting | 2,048 |
| Cartoon | 2,344 |
| Photo | 1,670 |
| Sketch | 3,929 |
| Total | 9,991 |

Validate the source-only protocol before training:

```powershell
python -m task3.selection.source_validation
```

The command should report only Art Painting, Cartoon, and Photo as domains available to Task 3.

## Method checks

The individual objectives can be smoke-tested before full training:

```powershell
python -m task3.methods.erm
python -m task3.methods.dan_dg
python -m task3.methods.sam
```

For DAN-DG, the check should report classification loss, alignment loss, total loss, the three pairwise MMD values, and gradients for all source domains. The SAM check should report original loss, perturbed loss, and gradient norm.

## Main experiments

Run all commands from the repository root—the directory containing `task2`, `task3`, `shared`, and `data`.

### ERM baseline

The Task 2 Source-only checkpoint is reused without retraining:

```powershell
python -m task3.train --method erm
```

Expected run directory:

```text
task3/results/erm_task2_checkpoint/
```

### DAN-DG main setting

```powershell
python -m task3.train --method dan_dg --lambda-dg 1 --run-name dan_dg_lambda_1
```

### SAM main setting

```powershell
python -m task3.train --method sam --rho 0.05 --run-name sam_rho_0_05
```

These are the settings required for the main comparison. They must not be replaced by a post-hoc winner based on Sketch performance.

## Controlled DAN-DG study

The controlled study varies only `lambda_dg`; every other setting remains fixed:

```powershell
python -m task3.train --method dan_dg --lambda-dg 0.1 --run-name dan_dg_lambda_0_1
python -m task3.train --method dan_dg --lambda-dg 1 --run-name dan_dg_lambda_1
python -m task3.train --method dan_dg --lambda-dg 10 --run-name dan_dg_lambda_10
```

The `lambda_dg = 1` run is shared with the main comparison and does not need to be trained twice.

## Source-only diagnostics

Run the common diagnostics before unlocking Sketch evaluation:

```powershell
python -m task3.evaluation.domain_metrics `
  --runs erm_task2_checkpoint dan_dg_lambda_1 sam_rho_0_05 dan_dg_lambda_0_1 dan_dg_lambda_10 `
  --num-workers 0
```

This command evaluates:

- accuracy and macro-F1 for each source-validation domain;
- mean and worst-source accuracy and macro-F1;
- three-way source-domain separability using a balanced multinomial logistic-regression probe with `C = 1`, seed 6304, and a 70/30 split;
- the fixed-radius local sharpness proxy using 32 validation examples from each source and radius `0.05`.

Chance accuracy for the source-domain probe is `1/3 = 33.3%`. Lower separability indicates that observed source domains are harder to distinguish, but does not by itself establish better class preservation or Sketch generalization.

Expected output directory:

```text
task3/results/source_diagnostics/
```

## Final locked Sketch evaluation

Only run this command after all Task 3 configurations and checkpoint choices have been fixed:

```powershell
python -m task3.evaluate_sketch `
  --runs erm_task2_checkpoint dan_dg_lambda_1 sam_rho_0_05 dan_dg_lambda_0_1 dan_dg_lambda_10 `
  --erm-run erm_task2_checkpoint `
  --output-name final_sketch_evaluation `
  --num-workers 0 `
  --confirm-settings-locked
```

The explicit confirmation flag documents that target evaluation was unlocked only after source-only decisions were complete.

Expected output directory:

```text
task3/results/final_sketch_evaluation/
```

The final evaluation records Sketch accuracy, Sketch macro-F1, change in Sketch accuracy relative to ERM, per-class accuracy, and confusion matrices.

## Generate tables and plots

Generate the complete Task 3 result set with:

```powershell
python -m task3.plot_results `
  --main-runs erm_task2_checkpoint dan_dg_lambda_1 sam_rho_0_05 `
  --controlled-runs dan_dg_lambda_0_1 dan_dg_lambda_1 dan_dg_lambda_10 `
  --source-folder source_diagnostics `
  --sketch-folder final_sketch_evaluation `
  --output-folder plots
```

To create the compact report figure:

```powershell
python -m task3.combine_report_plots
```

Generated artifacts include the main comparison table, method comparison, sharpness comparison, training curves, per-class Sketch accuracy, controlled-lambda study, and Sketch confusion matrices.

## Recorded main results

| Method | Mean source accuracy | Worst source accuracy | Mean source macro-F1 | Worst source macro-F1 | Sketch accuracy | Sketch macro-F1 | Source separability | Sharpness proxy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ERM | 0.9359 | 0.9195 | 0.9377 | 0.9262 | 0.6750 | 0.7009 | 0.8472 | 0.557835 |
| DAN-DG (`lambda_dg=1`) | 0.2166 | 0.1727 | 0.0507 | 0.0421 | 0.0407 | 0.0112 | 0.3322 | 0.013543 |
| SAM (`rho=0.05`) | 0.9495 | 0.9244 | 0.9508 | 0.9273 | 0.6477 | 0.6541 | 0.8771 | 0.187686 |

The main DAN-DG run collapsed to predicting the `person` class for every example. This is a recorded experimental outcome, not a reason to replace the required `lambda_dg = 1` main setting. The controlled `lambda_dg = 0.1` run remained class-discriminative and is reported only as part of the controlled study.

## Recorded controlled-study results

| `lambda_dg` | Mean source macro-F1 | Worst source macro-F1 | Sketch accuracy | Source separability | Sharpness proxy | Status |
|---:|---:|---:|---:|---:|---:|---|
| 0.1 | 0.9563 | 0.9391 | 0.5836 | 0.8405 | 0.143719 | Stable |
| 1 | 0.0507 | 0.0421 | 0.0407 | 0.3322 | 0.013543 | One-class collapse |
| 10 | 0.0507 | 0.0421 | 0.0407 | 0.3322 | 0.031542 | One-class collapse |

## Reproducibility and leakage safeguards

- Use seed 6304 for splitting, training, diagnostic sampling, and probe evaluation.
- Preserve `shared/splits/pacs_sketch_seed6304.json` with the submission.
- Keep eight examples from each source domain in every training batch.
- Select checkpoints only by mean source-validation macro-F1.
- Do not load Sketch in training, source validation, checkpoint selection, source-domain separability, sharpness measurement, or hyperparameter selection.
- Do not use Task 2 Sketch results to revise Task 3 settings.
- Keep `lambda_dg = 1` and `rho = 0.05` as the main settings even if another controlled setting performs better on Sketch.
- Run `evaluate_sketch.py` only after settings are locked.
- Retain configuration files, split indices, training histories, machine-readable CSV/JSON results, and the code version used for the report.
- Do not commit PACS images, virtual environments, caches, or unnecessary large checkpoints.

## Submission notes

This file is the Task 3-specific reproduction guide. The repository-level `README.md` should link to it and provide the execution order for all four tasks. The repository should also contain a pinned `requirements.txt`, `environment.yml`, or `pyproject.toml`, and a `.gitignore` excluding raw datasets and unnecessary model checkpoints.
