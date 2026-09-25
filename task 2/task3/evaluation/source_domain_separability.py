import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from task3.models.backbone import ResNet18Backbone
from task3.selection.source_validation import (
    build_source_data_loaders
)


@torch.no_grad()
def collect_domain_features(
    backbone,
    validation_loaders,
    device
):
    backbone.eval()
    features_by_domain = {}

    for domain, data_loader in validation_loaders.items():
        domain_features = []

        for images, _ in data_loader:
            images = images.to(device)
            features = backbone(images)
            domain_features.append(features.cpu())

        features_by_domain[domain] = torch.cat(
            domain_features,
            dim=0
        )

    return features_by_domain


def balance_domain_features(features_by_domain, seed=6304):
    generator = torch.Generator()
    generator.manual_seed(seed)

    number_per_domain = min(
        features.shape[0]
        for features in features_by_domain.values()
    )

    balanced_features = []
    domain_labels = []
    domain_names = list(features_by_domain.keys())

    for domain_number, domain in enumerate(domain_names):
        features = features_by_domain[domain]

        selected_indices = torch.randperm(
            features.shape[0],
            generator=generator
        )[:number_per_domain]

        selected_features = features[selected_indices]

        balanced_features.append(selected_features)

        domain_labels.append(
            torch.full(
                size=(number_per_domain,),
                fill_value=domain_number,
                dtype=torch.long
            )
        )

    all_features = torch.cat(
        balanced_features,
        dim=0
    )

    all_domain_labels = torch.cat(
        domain_labels,
        dim=0
    )

    return all_features, all_domain_labels, domain_names


def calculate_source_domain_separability(
    backbone,
    validation_loaders,
    device,
    seed=6304
):
    features_by_domain = collect_domain_features(
        backbone=backbone,
        validation_loaders=validation_loaders,
        device=device
    )

    features, domain_labels, domain_names = (
        balance_domain_features(
            features_by_domain=features_by_domain,
            seed=seed
        )
    )

    features = features.numpy()
    domain_labels = domain_labels.numpy()

    (
        training_features,
        test_features,
        training_labels,
        test_labels
    ) = train_test_split(
        features,
        domain_labels,
        test_size=0.30,
        random_state=seed,
        stratify=domain_labels
    )

    domain_classifier = LogisticRegression(
        C=1.0,
        max_iter=2000,
        random_state=seed
    )

    domain_classifier.fit(
        training_features,
        training_labels
    )

    predictions = domain_classifier.predict(
        test_features
    )

    separability_accuracy = accuracy_score(
        test_labels,
        predictions
    )

    number_per_domain = min(
        features_by_domain[domain].shape[0]
        for domain in domain_names
    )

    return {
        "source_domain_separability": float(
            separability_accuracy
        ),
        "chance_accuracy": 1.0 / len(domain_names),
        "source_domains": domain_names,
        "balanced_examples_per_domain": int(
            number_per_domain
        ),
        "total_balanced_examples": int(
            number_per_domain * len(domain_names)
        ),
        "training_examples": int(
            training_features.shape[0]
        ),
        "test_examples": int(
            test_features.shape[0]
        ),
        "split_seed": seed,
        "test_fraction": 0.30,
        "classifier": "multinomial logistic regression",
        "classifier_c": 1.0
    }


def load_backbone(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    backbone = ResNet18Backbone().to(device)

    backbone.load_state_dict(
        checkpoint["backbone_state_dict"]
    )

    return backbone


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Measure source-domain separability for "
            "a Task 3 checkpoint."
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

    backbone = load_backbone(
        checkpoint_path=arguments.checkpoint,
        device=device
    )

    results = calculate_source_domain_separability(
        backbone=backbone,
        validation_loaders=validation_loaders,
        device=device,
        seed=seed
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
        "Source-domain separability: "
        f"{results['source_domain_separability']:.4f}"
    )

    print(
        "Chance accuracy: "
        f"{results['chance_accuracy']:.4f}"
    )

    print("Results saved to:")
    print(output_path.resolve())


if __name__ == "__main__":
    main()