import argparse
from pathlib import Path

import torch

from task4.data.cifar10 import get_cifar10_loaders
from task4.data.cifar100_unknowns import (
    get_cifar100_unknown_loaders
)
from task4.models.resnet_cifar import CIFARResNet18


@torch.inference_mode()
def extract_from_loader(model, loader, device):
    model.eval()

    all_logits = []
    all_features = []
    all_labels = []
    all_indices = []

    for images, labels, indices in loader:
        images = images.to(device)

        logits, features = model(
            images,
            return_features=True
        )

        all_logits.append(logits.cpu())
        all_features.append(features.cpu())
        all_labels.append(labels.cpu())
        all_indices.append(indices.cpu())

    return {
        "logits": torch.cat(all_logits),
        "features": torch.cat(all_features),
        "labels": torch.cat(all_labels),
        "indices": torch.cat(all_indices)
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True
    )

    parser.add_argument(
        "--model-name",
        type=str,
        required=True
    )

    parser.add_argument(
        "--num-outputs",
        type=int,
        default=10
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2
    )

    arguments = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        arguments.checkpoint,
        map_location=device,
        weights_only=False
    )

    model = CIFARResNet18(
        num_classes=arguments.num_outputs
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    cifar10_loaders = get_cifar10_loaders(
        batch_size=arguments.batch_size,
        num_workers=arguments.num_workers,
        use_randaugment=False,
        seed=6304
    )

    unknown_loaders = get_cifar100_unknown_loaders(
        batch_size=arguments.batch_size,
        num_workers=arguments.num_workers
    )

    selected_loaders = {
        "train": cifar10_loaders[
            "train_unaugmented"
        ],
        "validation": cifar10_loaders["val"],
        "test": cifar10_loaders["test"],
        "near_unknown": unknown_loaders["near"],
        "far_unknown": unknown_loaders["far"]
    }

    output_directory = (
        Path("task4/cache")
        / arguments.model_name
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    print(f"Model: {arguments.model_name}")
    print(f"Device: {device}")

    for split_name, loader in selected_loaders.items():
        print(f"Extracting {split_name}...")

        outputs = extract_from_loader(
            model,
            loader,
            device
        )

        output_path = (
            output_directory
            / f"{split_name}_outputs.pt"
        )

        torch.save(outputs, output_path)

        print(
            f"  logits: {tuple(outputs['logits'].shape)}"
        )
        print(
            f"  features: {tuple(outputs['features'].shape)}"
        )
        print(f"  saved: {output_path}")

    print("\nExtraction complete.")


if __name__ == "__main__":
    main()