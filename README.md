# ATML PA1 — Representation, Domain Shift, and Open-Set Recognition

This repository contains the code and recorded outputs for four experiments in ATML PA1. Task 1 studies visual representations under image interventions. Task 2 studies unsupervised domain adaptation on PACS. Task 3 studies domain generalization to the withheld PACS Sketch domain. Task 4 studies open-set recognition with CIFAR-10 known classes and CIFAR-100 unknown classes. The experiments use random seed `6304`.

The commands below show where to run each task. The task-specific guides contain the full settings, controlled studies, interpretation, and known limitations.

## Repository layout

```text
PA1/
├── README.md                         # This guide
├── task1.ipynb                       # Task 1 notebook; run from PA1/
├── data/                             # Downloaded CIFAR/STL-10 data (local)
│   └── test_subset_seed6304.csv      # Task 1 selected test identities
├── results/                          # Task 1 saved results
├── external/pytorch-AdaIN/           # External style-transfer implementation, if retained
├── task 2/                            # Run Task 2 and Task 3 commands here
│   ├── data/pacs/images/             # PACS images (local)
│   ├── shared/splits/pacs_sketch_seed6304.json
│   ├── task2/                        # Adaptation code, configs, README, results
│   └── task3/                        # Generalization code, configs, README, results
└── task4/                             # Open-set code, configs, split, README, results
    └── data/cifar10_split_seed6304.json
```

The PACS image directory and raw CIFAR/STL-10 downloads are local inputs, not submission files. The `results/` directories above refer to saved experiment outputs; they are separate from the raw datasets. Do not confuse the root `data/` directory, used by Tasks 1 and 4, with `task 2/data/`, used by Tasks 2 and 3.

## Environment

The experiments were run on Windows with an NVIDIA GeForce RTX 3060 Laptop GPU and CUDA-enabled PyTorch. Use a Python environment containing `torch`, `torchvision`, `numpy`, `pandas`, `scikit-learn`, `PyYAML`, `matplotlib`, `Pillow`, `tqdm`, `openpyxl`, `open_clip_torch`, and `jupyter`. Install PyTorch and torchvision builds appropriate for the machine before installing the remaining packages. Pretrained model downloads and dataset downloads need internet access on the first run.

A project-wide `requirements.txt` is included for the final submission. Install PyTorch and torchvision builds appropriate for the local machine first, then use `requirements.txt` for the remaining Python dependencies. Exact numerical equality across different PyTorch, CUDA, GPU, and operating-system versions is not guaranteed.

## Task 1 — Visual inductive biases

From `PA1/`, open the notebook:

```powershell
jupyter notebook task1.ipynb
```

The notebook compares frozen ResNet-50, ViT-B/16, and OpenCLIP ViT-B/32 representations on STL-10, using linear heads and a CLIP zero-shot comparison. It evaluates colour changes, translations, patch shuffling, and reviewed AdaIN cue-conflict images. Its saved results are in root `results/`; the selected 500-image test subset is `data/test_subset_seed6304.csv`.

The notebook is stateful. Its reviewed cue-conflict evaluation depends on the original images, manifest, and manually tagged review workbook in `results/cue_conflicts/`. Keep `REBUILD_CUE_CONFLICT_ASSETS = False` when inspecting the reported run. Rerunning cue-conflict generation or selection can change the set of examples and the reported results. Its stored OpenCLIP configuration and patch-grid reporting limitations are documented in the existing Task 1 guide; preserve that guide when replacing the old root `README.md` with this overall guide.

## Task 2 — Unsupervised domain adaptation

Extract PACS to `PA1/task 2/data/pacs/images/`, with `art_painting`, `cartoon`, `photo`, and `sketch` beneath `images/`. Run the following from `PA1/task 2/`:

```powershell
python -m task2.train --method source_only --run-name source_only_final
python -m task2.train --method dan --run-name dan_final
python -m task2.train --method dann --maximum-grl-strength 1 --discriminator-learning-rate 0.0003 --run-name dann_disc_lr_3e_4_full
python -m task2.train --method cdan --maximum-grl-strength 1 --discriminator-learning-rate 0.001 --run-name cdan_disc_lr_1e_3_full
```

The fixed split is `shared/splits/pacs_sketch_seed6304.json`. Checkpoints are selected by mean source-validation macro-F1. After the configurations and selected checkpoints are fixed, the final Sketch evaluation is:

```powershell
python -m task2.evaluate_final --runs source_only_final dan_final dann_disc_lr_3e_4_full cdan_disc_lr_1e_3_full dann_grl_0_25_lr_3e_4_full dann_grl_0_5_lr_3e_4_full --output-name final_evaluation --num-workers 0 --confirm-settings-locked
```

The evaluation command includes two controlled DANN runs. Their training commands and the complete protocol are in [`task 2/task2/README.md`](task%202/task2/README.md). Principal saved outputs are `task 2/task2/results/final_evaluation/all_results.json`, `method_comparison.csv`, and `per_class_changes.csv`. Sketch labels are reserved for final evaluation.

## Task 3 — Domain generalization

Task 3 uses the same PACS images and split. It first reuses `task2/results/source_only_final/best_checkpoint.pt` as its ERM baseline. From `PA1/task 2/`, run:

```powershell
python -m task3.train --method erm
python -m task3.train --method dan_dg --lambda-dg 1 --run-name dan_dg_lambda_1
python -m task3.train --method sam --rho 0.05 --run-name sam_rho_0_05
python -m task3.evaluation.domain_metrics --runs erm_task2_checkpoint dan_dg_lambda_1 sam_rho_0_05 dan_dg_lambda_0_1 dan_dg_lambda_10 --num-workers 0
```

The diagnostics command includes controlled DAN-DG runs with `lambda_dg = 0.1` and `10`; train those first using the commands in [`task 2/task3/README.md`](task%202/task3/README.md). Task 3 keeps Sketch out of training, diagnostics, and model selection. Once settings are locked, run:

```powershell
python -m task3.evaluate_sketch --runs erm_task2_checkpoint dan_dg_lambda_1 sam_rho_0_05 dan_dg_lambda_0_1 dan_dg_lambda_10 --erm-run erm_task2_checkpoint --output-name final_sketch_evaluation --num-workers 0 --confirm-settings-locked
```

Principal saved outputs are `task 2/task3/results/source_diagnostics/summary.csv`, `task 2/task3/results/final_sketch_evaluation/summary.csv`, and their corresponding `full_results.json` files. The main DAN-DG `lambda_dg = 1` run is retained as recorded, including its one-class collapse.

## Task 4 — Open-set recognition

Run Task 4 commands from `PA1/`. The CIFAR-10 train/validation indices are saved at `task4/data/cifar10_split_seed6304.json`. If that file is absent, create it **once** before training with `python -m task4.data.make_splits`; use the saved file for all reported runs. Dataset loaders download CIFAR-10 and the CIFAR-100 test set into root `data/`.

```powershell
python -m task4.train --config task4/configs/vanilla.yaml
python -m task4.train --config task4/configs/gcsc.yaml
python -m task4.train_proser --config task4/configs/proser.yaml
```

Vanilla and GCSC use separate CIFAR-adapted ResNet-18 models; PROSER starts from the selected Vanilla checkpoint. The default checkpoint paths are `task4/checkpoints/vanilla_best.pt`, `gcsc_best.pt`, and `proser_best.pt`. Extract frozen outputs and evaluate each model:

```powershell
python -m task4.extract_outputs --checkpoint task4/checkpoints/vanilla_best.pt --model-name vanilla
python -m task4.extract_outputs --checkpoint task4/checkpoints/gcsc_best.pt --model-name gcsc
python -m task4.extract_outputs --checkpoint task4/checkpoints/proser_best.pt --model-name proser --num-outputs 15
python -m task4.evaluate_osr --model-name vanilla
python -m task4.evaluate_osr --model-name gcsc
python -m task4.evaluate_osr --model-name proser
python -m task4.evaluation.build_tables
python -m task4.evaluation.failure_analysis --model-name vanilla
```

Vanilla's frozen outputs provide MSP, MLS, Energy, and Mahalanobis scores. GCSC and PROSER are compared using MLS; PROSER also has a placeholder score. Each rejection threshold is calibrated on CIFAR-10 validation examples. The near and far CIFAR-100 unknowns are used only for final evaluation and failure analysis. Results are saved under `task4/results/`, including `final_tables/table1_vanilla_scores.csv`, `final_tables/table2_model_comparison.csv`, and each model's `posthoc_osr_metrics.csv`. See [`task4/README.md`](task4/README.md) for the task protocol and reported results.

## Submission and provenance

Keep the code, configurations, fixed split files, run histories, and lightweight machine-readable results needed to trace the report. Keep PACS/CIFAR/STL-10 raw datasets, downloaded model weights, extracted output caches, smoke-run artifacts, backup archives, and unnecessary large checkpoints out of the GitHub submission. Task 3's ERM reconstruction requires the Task 2 source-only checkpoint; if that checkpoint is not shared, its original training command and run metadata must remain available so it can be regenerated.

The reviewed Task 1 cue-conflict images and their review workbook require separate handling: the saved metrics can be shared without them, but the cue-conflict evaluation cannot be reproduced from its manifest alone. Preserve these assets and document where a reproducer can obtain them. The Task 1 notebook uses the external PyTorch AdaIN implementation by Naoto Usuyama, based on Huang and Belongie, *Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization* (ICCV 2017); keep attribution and the upstream license if bundling that implementation.


