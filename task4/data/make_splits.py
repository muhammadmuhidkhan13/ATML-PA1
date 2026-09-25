import json
from pathlib import Path

from sklearn.model_selection import train_test_split
from torchvision.datasets import CIFAR10


SEED = 6304
DATA_ROOT = Path("data")
OUTPUT_PATH = Path("task4/data/cifar10_split_seed6304.json")


def main():
    dataset = CIFAR10(
        root=DATA_ROOT,
        train=True,
        download=True
    )

    all_indices = list(range(len(dataset)))
    all_labels = dataset.targets

    train_indices, val_indices = train_test_split(
        all_indices,
        test_size=0.10,
        random_state=SEED,
        stratify=all_labels
    )

    split_data = {
        "seed": SEED,
        "train_indices": train_indices,
        "val_indices": val_indices
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w") as file:
        json.dump(split_data, file, indent=2)

    train_labels = [
        all_labels[index]
        for index in train_indices
    ]

    val_labels = [
        all_labels[index]
        for index in val_indices
    ]

    print(f"Total images: {len(dataset)}")
    print(f"Training images: {len(train_indices)}")
    print(f"Validation images: {len(val_indices)}")

    print("\nTraining images per class:")
    for class_id in range(10):
        print(
            class_id,
            train_labels.count(class_id)
        )

    print("\nValidation images per class:")
    for class_id in range(10):
        print(
            class_id,
            val_labels.count(class_id)
        )

    print(f"\nSplit saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()