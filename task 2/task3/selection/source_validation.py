from pathlib import Path

import torch
from torch.utils.data import DataLoader

from shared.pacs import PACSLabeledDataset, get_image_transforms
from shared.pacs_protocol import load_split_data


def verify_source_only_protocol(split_data):
    source_domains = split_data["source_domains"]
    target_domain = split_data["target_domain"]
    available_domains = list(split_data["domains"].keys())

    if target_domain in available_domains:
        raise ValueError(
            f"Target domain '{target_domain}' appears in the Task 3 split."
        )

    if set(source_domains) != set(available_domains):
        raise ValueError(
            "The split domains do not match the declared source domains."
        )

    for domain in source_domains:
        for section in ["train", "validation"]:
            for sample in split_data["domains"][domain][section]:
                sample_domain = Path(sample["path"]).parts[0]

                if sample_domain != domain:
                    raise ValueError(
                        f"Sample {sample['path']} is stored under the "
                        f"wrong domain: {domain}."
                    )

                if sample_domain == target_domain:
                    raise ValueError(
                        "A target-domain sample entered Task 3."
                    )


def create_loader(
    images_root,
    samples,
    transform,
    batch_size,
    shuffle,
    num_workers,
    seed,
    drop_last
):
    dataset = PACSLabeledDataset(
        images_root=images_root,
        samples=samples,
        transform=transform
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        generator=generator
    )

    return loader


def build_source_data_loaders(
    images_root,
    split_path,
    training_batch_size=8,
    validation_batch_size=64,
    num_workers=0,
    seed=6304
):
    split_data = load_split_data(split_path)
    verify_source_only_protocol(split_data)

    training_transform, evaluation_transform = get_image_transforms()

    training_loaders = {}
    validation_loaders = {}

    for domain_number, domain in enumerate(
        split_data["source_domains"]
    ):
        domain_data = split_data["domains"][domain]

        training_loaders[domain] = create_loader(
            images_root=images_root,
            samples=domain_data["train"],
            transform=training_transform,
            batch_size=training_batch_size,
            shuffle=True,
            num_workers=num_workers,
            seed=seed + domain_number,
            drop_last=True
        )

        validation_loaders[domain] = create_loader(
            images_root=images_root,
            samples=domain_data["validation"],
            transform=evaluation_transform,
            batch_size=validation_batch_size,
            shuffle=False,
            num_workers=num_workers,
            seed=seed,
            drop_last=False
        )

    return training_loaders, validation_loaders, split_data


def confusion_matrix_from_predictions(
    targets,
    predictions,
    number_of_classes
):
    positions = targets * number_of_classes + predictions

    confusion_matrix = torch.bincount(
        positions,
        minlength=number_of_classes ** 2
    )

    return confusion_matrix.reshape(
        number_of_classes,
        number_of_classes
    )


def metrics_from_confusion_matrix(confusion_matrix):
    confusion_matrix = confusion_matrix.float()

    correct = confusion_matrix.diag()
    total = confusion_matrix.sum()

    accuracy = correct.sum() / total

    predicted_per_class = confusion_matrix.sum(dim=0)
    actual_per_class = confusion_matrix.sum(dim=1)

    precision = correct / predicted_per_class.clamp(min=1)
    recall = correct / actual_per_class.clamp(min=1)

    f1 = (
        2 * precision * recall
        / (precision + recall).clamp(min=1e-12)
    )

    macro_f1 = f1.mean()
    per_class_accuracy = recall

    return {
        "accuracy": accuracy.item(),
        "macro_f1": macro_f1.item(),
        "per_class_accuracy": per_class_accuracy.tolist(),
        "confusion_matrix": confusion_matrix.int().tolist()
    }


@torch.no_grad()
def evaluate_one_domain(
    backbone,
    classifier,
    data_loader,
    device,
    number_of_classes=7
):
    backbone.eval()
    classifier.eval()

    all_targets = []
    all_predictions = []

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        features = backbone(images)
        logits = classifier(features)
        predictions = logits.argmax(dim=1)

        all_targets.append(labels.cpu())
        all_predictions.append(predictions.cpu())

    targets = torch.cat(all_targets)
    predictions = torch.cat(all_predictions)

    confusion_matrix = confusion_matrix_from_predictions(
        targets=targets,
        predictions=predictions,
        number_of_classes=number_of_classes
    )

    return metrics_from_confusion_matrix(confusion_matrix)


def evaluate_source_domains(
    backbone,
    classifier,
    validation_loaders,
    device,
    number_of_classes=7
):
    results = {}

    for domain, data_loader in validation_loaders.items():
        results[domain] = evaluate_one_domain(
            backbone=backbone,
            classifier=classifier,
            data_loader=data_loader,
            device=device,
            number_of_classes=number_of_classes
        )

    accuracies = [
        domain_results["accuracy"]
        for domain_results in results.values()
    ]

    macro_f1_scores = [
        domain_results["macro_f1"]
        for domain_results in results.values()
    ]

    results["summary"] = {
        "mean_accuracy": sum(accuracies) / len(accuracies),
        "worst_accuracy": min(accuracies),
        "mean_macro_f1": (
            sum(macro_f1_scores) / len(macro_f1_scores)
        ),
        "worst_macro_f1": min(macro_f1_scores)
    }

    return results


def main():
    training_loaders, validation_loaders, split_data = (
        build_source_data_loaders(
            images_root="data/pacs/images",
            split_path=(
                "shared/splits/pacs_sketch_seed6304.json"
            ),
            training_batch_size=8,
            validation_batch_size=64,
            num_workers=0,
            seed=6304
        )
    )

    print("Source-only protocol verified.")
    print(f"Declared target: {split_data['target_domain']}")
    print(
        "Domains available to Task 3:",
        list(split_data["domains"].keys())
    )

    print("\nTraining loaders:")

    for domain, loader in training_loaders.items():
        images, labels = next(iter(loader))

        print(
            f"{domain:15s} | "
            f"examples: {len(loader.dataset):4d} | "
            f"batch images: {tuple(images.shape)} | "
            f"batch labels: {tuple(labels.shape)}"
        )

    print("\nValidation loaders:")

    for domain, loader in validation_loaders.items():
        print(
            f"{domain:15s} | "
            f"examples: {len(loader.dataset):4d} | "
            f"batches: {len(loader)}"
        )


if __name__ == "__main__":
    main()