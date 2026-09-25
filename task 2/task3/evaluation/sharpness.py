import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from task3.models.backbone import ResNet18Backbone
from task3.models.classifier_head import ClassifierHead
from task3.selection.source_validation import (
    build_source_data_loaders
)


def select_fixed_validation_batch(
    validation_loaders,
    examples_per_domain=32,
    seed=6304
):
    generator = torch.Generator()
    generator.manual_seed(seed)

    selected_images = []
    selected_labels = []
    selected_indices = {}

    for domain, data_loader in validation_loaders.items():
        dataset = data_loader.dataset

        if len(dataset) < examples_per_domain:
            raise ValueError(
                f"{domain} has fewer than "
                f"{examples_per_domain} validation examples."
            )

        indices = torch.randperm(
            len(dataset),
            generator=generator
        )[:examples_per_domain]

        domain_images = []
        domain_labels = []

        for index in indices.tolist():
            image, label = dataset[index]
            domain_images.append(image)
            domain_labels.append(label)

        selected_images.append(
            torch.stack(domain_images)
        )

        selected_labels.append(
            torch.tensor(
                domain_labels,
                dtype=torch.long
            )
        )

        selected_indices[domain] = indices.tolist()

    images = torch.cat(selected_images, dim=0)
    labels = torch.cat(selected_labels, dim=0)

    return images, labels, selected_indices


def collect_trainable_parameters(backbone, classifier):
    return [
        parameter
        for parameter in list(backbone.parameters())
        + list(classifier.parameters())
        if parameter.requires_grad
    ]


def calculate_gradient_norm(parameters):
    gradient_norms = [
        parameter.grad.norm(p=2)
        for parameter in parameters
        if parameter.grad is not None
    ]

    if not gradient_norms:
        raise RuntimeError(
            "No gradients were created for the "
            "sharpness diagnostic."
        )

    return torch.stack(gradient_norms).norm(p=2)


@torch.no_grad()
def add_normalized_perturbation(
    parameters,
    gradient_norm,
    radius
):
    perturbations = []

    scale = radius / (gradient_norm + 1e-12)

    for parameter in parameters:
        if parameter.grad is None:
            continue

        perturbation = parameter.grad * scale
        parameter.add_(perturbation)

        perturbations.append(
            (parameter, perturbation)
        )

    return perturbations


@torch.no_grad()
def restore_parameters(perturbations):
    for parameter, perturbation in perturbations:
        parameter.sub_(perturbation)


def calculate_sharpness_proxy(
    backbone,
    classifier,
    images,
    labels,
    device,
    radius=0.05
):
    backbone.eval()
    classifier.eval()

    images = images.to(device)
    labels = labels.to(device)

    parameters = collect_trainable_parameters(
        backbone=backbone,
        classifier=classifier
    )

    for parameter in parameters:
        parameter.grad = None

    features = backbone(images)
    logits = classifier(features)

    original_loss = F.cross_entropy(
        logits,
        labels
    )

    original_loss.backward()

    gradient_norm = calculate_gradient_norm(
        parameters
    )

    perturbations = add_normalized_perturbation(
        parameters=parameters,
        gradient_norm=gradient_norm,
        radius=radius
    )

    try:
        with torch.no_grad():
            perturbed_features = backbone(images)

            perturbed_logits = classifier(
                perturbed_features
            )

            perturbed_loss = F.cross_entropy(
                perturbed_logits,
                labels
            )

    finally:
        restore_parameters(perturbations)

    for parameter in parameters:
        parameter.grad = None

    sharpness_increase = (
        perturbed_loss.item()
        - original_loss.item()
    )

    return {
        "original_validation_loss": (
            original_loss.item()
        ),
        "perturbed_validation_loss": (
            perturbed_loss.item()
        ),
        "sharpness_proxy": sharpness_increase,
        "gradient_norm": gradient_norm.item(),
        "perturbation_radius": radius,
        "examples_per_source_domain": (
            images.shape[0] // 3
        ),
        "total_examples": images.shape[0],
        "seed": 6304
    }


def load_models(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    backbone = ResNet18Backbone().to(device)

    classifier = ClassifierHead(
        feature_dimension=512,
        number_of_classes=7
    ).to(device)

    backbone.load_state_dict(
        checkpoint["backbone_state_dict"]
    )

    classifier.load_state_dict(
        checkpoint["classifier_state_dict"]
    )

    return backbone, classifier


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Measure the Task 3 local sharpness proxy."
        )
    )

    parser.add_argument(
        "--checkpoint",
        required=True
    )

    parser.add_argument(
        "--output",
        required=True
    )

    parser.add_argument(
        "--radius",
        type=float,
        default=0.05
    )

    parser.add_argument(
        "--images-root",
        default="data/pacs/images"
    )

    parser.add_argument(
        "--split-path",
        default=(
            "shared/splits/"
            "pacs_sketch_seed6304.json"
        )
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0
    )

    return parser.parse_args()


def main():
    arguments = parse_arguments()

    seed = 6304

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    _, validation_loaders, _ = (
        build_source_data_loaders(
            images_root=arguments.images_root,
            split_path=arguments.split_path,
            training_batch_size=8,
            validation_batch_size=64,
            num_workers=arguments.num_workers,
            seed=seed
        )
    )

    images, labels, selected_indices = (
        select_fixed_validation_batch(
            validation_loaders=validation_loaders,
            examples_per_domain=32,
            seed=seed
        )
    )

    backbone, classifier = load_models(
        checkpoint_path=arguments.checkpoint,
        device=device
    )

    results = calculate_sharpness_proxy(
        backbone=backbone,
        classifier=classifier,
        images=images,
        labels=labels,
        device=device,
        radius=arguments.radius
    )

    results["selected_validation_indices"] = (
        selected_indices
    )

    output_path = Path(arguments.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(results, file, indent=2)

    print(
        "Original validation loss: "
        f"{results['original_validation_loss']:.6f}"
    )

    print(
        "Perturbed validation loss: "
        f"{results['perturbed_validation_loss']:.6f}"
    )

    print(
        "Sharpness proxy: "
        f"{results['sharpness_proxy']:.6f}"
    )

    print("Results saved to:")
    print(output_path.resolve())


if __name__ == "__main__":
    main()