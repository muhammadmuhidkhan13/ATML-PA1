# Task 2 — Unsupervised Domain Adaptation on PACS

This directory contains the implementation and results for Task 2 of ATML PA1. The task compares a source-only baseline with DAN, DANN, and CDAN for adaptation from three labelled PACS source domains to the unlabelled Sketch target domain.

## Experimental protocol

- Dataset: PACS
- Labelled source domains: `art_painting`, `cartoon`, and `photo`
- Unlabelled target domain during training: `sketch`
- Classes: dog, elephant, giraffe, guitar, horse, house, and person
- Backbone: ImageNet-initialized ResNet-18
- Classifier: seven-class linear head
- Random seed: `6304`
- Split file: `shared/splits/pacs_sketch_seed6304.json`
- PACS image root: `data/pacs/images`
- BatchNorm running statistics are frozen.
- Training uses domain-balanced source batches.
- Checkpoints are selected using mean source-validation macro-F1.
- Sketch labels must not be used for training, checkpoint selection, or hyperparameter selection. They are unlocked only by the final evaluation script after all settings are fixed.

The expected PACS image counts are:

| Domain | Images |
|---|---:|
| Art Painting | 2,048 |
| Cartoon | 2,344 |
| Photo | 1,670 |
| Sketch | 3,929 |
| Total | 9,991 |

## Repository layout

```text
task2/
├── configs/                  # Method configuration files
├── evaluation/               # Metrics and evaluation utilities
├── methods/                  # DAN, DANN, CDAN, and related losses
├── models/                   # ResNet-18 backbone and classifier head
├── results/                  # Run folders and final evaluation outputs
├── train.py                  # Main training entry point
├── evaluate_final.py         # Locked final Sketch evaluation
├── plot_results.py           # General result plots
├── plot_report_diagnostics.py # Compact report diagnostics
└── README.md

shared/
├── pacs.py                   # PACS datasets and transformations
├── pacs_protocol.py          # Fixed split creation and validation
└── splits/
    └── pacs_sketch_seed6304.json
```

## Environment

The experiments were run on Windows with Anaconda, PyTorch with CUDA support, and an NVIDIA RTX 3060 Laptop GPU. Required Python packages include:

- `torch`
- `torchvision`
- `numpy`
- `pandas`
- `scikit-learn`
- `PyYAML`
- `matplotlib`

Run every command below from the repository root:

```powershell
cd "C:\Users\HP\Desktop\ATML\PA1\task 2"
```

Confirm that CUDA is available:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Data preparation

Place the extracted PACS images at:

```text
data/pacs/images/
├── art_painting/
├── cartoon/
├── photo/
└── sketch/
```

The split file is created once using seed 6304 and reused across all methods. Training output should report `Using existing split file.` when the saved split is found.

## Main training runs

### Source-only baseline

```powershell
python -m task2.train --method source_only --run-name source_only_final
```

### DAN

```powershell
python -m task2.train --method dan --run-name dan_final
```

### DANN

The recorded full DANN run used discriminator learning rate `0.0003` and maximum GRL strength `1`:

```powershell
python -m task2.train --method dann --maximum-grl-strength 1 --discriminator-learning-rate 0.0003 --run-name dann_disc_lr_3e_4_full
```

### CDAN

The recorded full CDAN run used discriminator learning rate `0.001` and maximum GRL strength `1`:

```powershell
python -m task2.train --method cdan --maximum-grl-strength 1 --discriminator-learning-rate 0.001 --run-name cdan_disc_lr_1e_3_full
```

Each run creates a folder under `task2/results/<run-name>/` containing the resolved configuration, training history, best checkpoint, and source-validation results.

## Controlled DANN study

The controlled study varies only the maximum GRL strength while retaining discriminator learning rate `0.0003`.

```powershell
python -m task2.train --method dann --maximum-grl-strength 0.25 --discriminator-learning-rate 0.0003 --run-name dann_grl_0_25_lr_3e_4_full

python -m task2.train --method dann --maximum-grl-strength 0.5 --discriminator-learning-rate 0.0003 --run-name dann_grl_0_5_lr_3e_4_full
```

The maximum-GRL-1 result is supplied by `dann_disc_lr_3e_4_full`.

## Locked final evaluation

Run the final evaluation only after all training settings and selected checkpoints are fixed:

```powershell
python -m task2.evaluate_final `
  --runs source_only_final dan_final dann_disc_lr_3e_4_full cdan_disc_lr_1e_3_full dann_grl_0_25_lr_3e_4_full dann_grl_0_5_lr_3e_4_full `
  --output-name final_evaluation `
  --num-workers 0 `
  --confirm-settings-locked
```

This stage reports:

- mean source-validation accuracy and macro-F1;
- Sketch accuracy and macro-F1;
- Sketch accuracy change relative to Source-only;
- source–target domain separability;
- per-class Sketch accuracy and confusion matrices.

The final outputs are saved under:

```text
task2/results/final_evaluation/
```

Important files include:

```text
all_results.json
method_comparison.csv
per_class_changes.csv
```

## Report plots

Generate the compact Task 2 report diagnostics with:

```powershell
python -m task2.plot_report_diagnostics
```

The final report uses `task2_combined_diagnostics.pdf`. Additional confusion matrices and detailed plots can be placed in the appendix.

## Recorded training behaviour

- Source-only and DAN trained stably under their recorded configurations.
- The main DANN checkpoint was selected before later training instability. Its best checkpoint occurred at epoch 2.
- The CDAN checkpoint was selected at epoch 7 before later deterioration.
- DANN with maximum GRL strengths `0.25` and `0.5` avoided the full collapse observed later in the maximum-GRL-1 training trajectory.
- The saved best checkpoint, rather than the final epoch, must be used for evaluation.

These outcomes should be reported transparently. The stable controlled configurations must not be substituted post hoc for the prescribed main configuration using Sketch results.

## Reproducibility notes

To reproduce the recorded experiment:

1. Use the saved split file and seed 6304.
2. Keep the same PACS image root and class ordering.
3. Retain the same preprocessing, augmentation, optimizer, epoch budget, early-stopping rule, and frozen BatchNorm policy.
4. Use domain-balanced batches.
5. Select checkpoints using source-validation macro-F1 only.
6. Do not inspect Sketch labels before final locked evaluation.
7. Preserve each run's configuration, history CSV, and best checkpoint.
8. Record the Python, PyTorch, torchvision, CUDA, and GPU versions used.

## Result directories

Do not delete the following recorded runs:

```text
task2/results/source_only_final/
task2/results/dan_final/
task2/results/dann_disc_lr_3e_4_full/
task2/results/cdan_disc_lr_1e_3_full/
task2/results/dann_grl_0_25_lr_3e_4_full/
task2/results/dann_grl_0_5_lr_3e_4_full/
task2/results/final_evaluation/
```

Large datasets and generated checkpoints should normally be excluded from Git. Commit the source code, configuration files, fixed split JSON, README, and lightweight result summaries needed to identify and reproduce the reported experiments.
