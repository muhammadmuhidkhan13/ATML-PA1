# Task 4: Open-Set Recognition on CIFAR-10 and CIFAR-100

This task compares a Vanilla closed-set classifier, a stronger closed-set classifier trained with RandAugment (GCSC), and PROSER. All ten CIFAR-10 classes are known. The selected CIFAR-100 test classes are unknown and are used only for final evaluation and failure analysis.

## Data and fixed split

- Split the official CIFAR-10 training set into stratified 90% training and 10% validation subsets with seed `6304`. Use the complete CIFAR-10 test set for known-class testing.
- Near unknowns: `bus`, `pickup_truck`, `motorcycle`, `tractor`, `wolf`, `fox`, `leopard`, `camel`.
- Far unknowns: `bottle`, `bowl`, `chair`, `clock`, `keyboard`, `mushroom`, `sunflower`, `wardrobe`.
- Use the official CIFAR-100 **test** partition for both unknown groups. Each group has 800 examples. Do not use CIFAR-100 images for training, checkpoint selection, score design, or threshold calibration.
- Keep the split indices and the two class lists fixed across all methods. Do not commit the raw datasets.

## Models and training

The assignment specifies a CIFAR-adapted ResNet-18: a `3 × 3`, stride-1 first convolution, no initial max-pooling, and original `32 × 32` inputs. The Vanilla and GCSC models start from random initialization; PROSER starts from the selected Vanilla checkpoint. Select checkpoints using **CIFAR-10 validation accuracy only**.

| Method | Training setup | Evaluation scores |
| --- | --- | --- |
| Vanilla | Ten-class cross-entropy classifier | MSP, MLS, Energy, Mahalanobis |
| GCSC | Same classifier recipe with `RandAugment(num_ops=2, magnitude=9)` after crop and flip | MLS |
| PROSER | Fine-tune Vanilla with five dummy classifiers, classifier placeholders, and different-class manifold mixup after `layer2` | MLS on the ten known-class logits; placeholder score |

The prescribed Vanilla/GCSC recipe uses random crop with four-pixel padding, random horizontal flip, SGD with learning rate `0.1`, momentum `0.9`, weight decay `5e-4`, cosine decay, batch size `128`, `100` epochs, and seed `6304`. The prescribed PROSER fine-tuning uses SGD with learning rate `1e-3`, momentum `0.9`, weight decay `5e-4`, cosine decay, batch size `128`, `50` epochs, and seed `6304`. Its classifier-placeholder weight is `β = 1`; its data-placeholder weight is `γ = 0.1`. Manifold mixup uses `λ ~ Beta(2, 2)` between examples of different known classes.

## Scores and evaluation

Use an unknownness convention in which **higher means more likely to be unknown**:

| Score | Unknownness |
| --- | --- |
| MSP | `1 - max(softmax(logits))` |
| MLS | `-max(logits)` |
| Energy | `-logsumexp(logits)` |
| Mahalanobis | Minimum class-mean squared feature distance using a shared diagonal covariance |

Fit Mahalanobis class means and the shared diagonal covariance on unaugmented CIFAR-10 **training** features, with `1e-6` added to each covariance diagonal entry. Compare all four post-hoc scores using the **same frozen Vanilla outputs**. PROSER's MLS uses only the ten known-class logits, while its separate placeholder score uses the learned dummy responses.

For each model/score combination, set its rejection threshold to the **95th percentile of its CIFAR-10 validation unknownness scores**. Accept an example if its unknownness is at or below that threshold. Compute CIFAR-10 test closed-set accuracy (CSA) using only known-class logits; also report known-class test acceptance, near/far/all-unknown AUROC, and near/far/all-unknown rejection at the calibrated threshold. AUROC measures ranking across thresholds; the rejection rate measures performance at the selected threshold.

## Reproduction order

1. Create and save the stratified CIFAR-10 split and fixed CIFAR-100 test-class lists.
2. Train Vanilla and save the checkpoint chosen by CIFAR-10 validation accuracy.
3. Extract the Vanilla logits and penultimate features once; compute MSP, MLS, Energy, and Mahalanobis from these saved outputs.
4. Train GCSC from a fresh initialization, changing only the specified augmentation, and evaluate it with MLS.
5. Initialize PROSER from the selected Vanilla checkpoint, train its classifier and data placeholders, and evaluate both its MLS and placeholder-based scores.
6. Calibrate each score on CIFAR-10 validation data; evaluate on the fixed known, near-unknown, and far-unknown test sets. Inspect accepted unknown examples **after** all models and thresholds are fixed.

The exact shell commands and output paths depend on the Task 4 scripts in the repository. They should be added here from those scripts before claiming that this README alone reproduces the runs.

## Results reported in the report

All values below are percentages. The four post-hoc scores share the same frozen Vanilla model.

| Vanilla score | Near AUROC | Far AUROC | All AUROC | Near rejection | Far rejection | All rejection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MSP | 80.8 | 90.0 | 85.4 | 29.0 | 46.5 | 37.7 |
| MLS | 78.7 | 90.0 | 84.4 | 32.1 | 57.0 | 44.6 |
| Energy | 78.7 | 90.1 | 84.4 | 31.5 | 58.2 | 44.9 |
| Mahalanobis | 79.2 | 91.8 | 85.5 | 27.8 | 52.1 | 39.9 |

| Model | Score | CSA | Near AUROC | Far AUROC | All AUROC | Known acceptance | Near rejection | Far rejection | All rejection |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Vanilla | MLS | 94.57 | 78.7 | 90.0 | 84.4 | 94.98 | 32.1 | 57.0 | 44.6 |
| GCSC | MLS | 95.20 | 82.0 | 89.4 | 85.7 | 94.70 | 37.5 | 58.0 | 47.7 |
| PROSER | MLS | 94.31 | 79.6 | 88.0 | 83.8 | 94.77 | 30.5 | 48.1 | 39.3 |
| PROSER | Placeholder | 94.31 | 77.3 | 87.6 | 82.4 | 95.27 | 26.2 | 45.1 | 35.7 |

## Items to verify before submission

- The report draft says **ResNet-19** in Task 4, while the assignment specifies **CIFAR-adapted ResNet-18**. Check the actual model definition and make the report and README agree with the code.
- Add the actual executable commands, script paths, saved split path, checkpoint paths, environment specification, and machine-readable results paths once checked against the repository. These cannot be established from the report alone.
- Include at least three accepted near unknowns and three accepted far unknowns in the failure-analysis output, each with the unknown class, predicted CIFAR-10 class, MLS score, and calibrated threshold.
