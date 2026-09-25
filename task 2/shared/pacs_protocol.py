import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from .pacs import (
    IMAGE_EXTENSIONS,
    PACS_CLASSES,
)


SOURCE_DOMAINS = [
    "art_painting",
    "cartoon",
    "photo",
]

TARGET_DOMAIN = "sketch"

SPLIT_SEED = 6304

VALIDATION_FRACTION = 0.20


def collect_domain_samples(
    images_root,
    domain,
):
    images_root = Path(images_root)
    domain_folder = images_root / domain

    if not domain_folder.exists():
        raise FileNotFoundError(
            f"Domain folder not found: {domain_folder}"
        )

    samples = []

    for label, class_name in enumerate(PACS_CLASSES):
        class_folder = (
            domain_folder
            / class_name
        )

        if not class_folder.exists():
            raise FileNotFoundError(
                f"Class folder not found: {class_folder}"
            )

        image_paths = sorted(
            path
            for path in class_folder.rglob("*")
            if path.suffix.lower() in IMAGE_EXTENSIONS
        )

        for image_path in image_paths:
            relative_path = image_path.relative_to(
                images_root
            )

            samples.append({
                "path": relative_path.as_posix(),
                "label": label,
                "class_name": class_name,
            })

    return samples


def create_stratified_split(samples):
    labels = [
        sample["label"]
        for sample in samples
    ]

    train_samples, validation_samples = train_test_split(
        samples,
        test_size=VALIDATION_FRACTION,
        random_state=SPLIT_SEED,
        shuffle=True,
        stratify=labels,
    )

    train_samples = sorted(
        train_samples,
        key=lambda sample: sample["path"],
    )

    validation_samples = sorted(
        validation_samples,
        key=lambda sample: sample["path"],
    )

    return train_samples, validation_samples


def verify_domain_split(
    all_samples,
    train_samples,
    validation_samples,
):
    all_paths = {
        sample["path"]
        for sample in all_samples
    }

    train_paths = {
        sample["path"]
        for sample in train_samples
    }

    validation_paths = {
        sample["path"]
        for sample in validation_samples
    }

    if len(train_paths) != len(train_samples):
        raise ValueError(
            "The training split contains duplicate paths."
        )

    if len(validation_paths) != len(validation_samples):
        raise ValueError(
            "The validation split contains duplicate paths."
        )

    if train_paths.intersection(validation_paths):
        raise ValueError(
            "Training and validation splits overlap."
        )

    if train_paths.union(validation_paths) != all_paths:
        raise ValueError(
            "The split does not contain every source image."
        )


def build_split_data(images_root):
    split_data = {
        "seed": SPLIT_SEED,
        "validation_fraction": VALIDATION_FRACTION,
        "source_domains": SOURCE_DOMAINS,
        "target_domain": TARGET_DOMAIN,
        "classes": PACS_CLASSES,
        "domains": {},
    }

    for domain in SOURCE_DOMAINS:
        all_samples = collect_domain_samples(
            images_root,
            domain,
        )

        train_samples, validation_samples = (
            create_stratified_split(all_samples)
        )

        verify_domain_split(
            all_samples,
            train_samples,
            validation_samples,
        )

        split_data["domains"][domain] = {
            "train": train_samples,
            "validation": validation_samples,
        }

    return split_data


def save_split_data(
    split_data,
    split_path,
):
    split_path = Path(split_path)

    split_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with split_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            split_data,
            file,
            indent=2,
        )


def load_split_data(split_path):
    split_path = Path(split_path)

    if not split_path.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_path}"
        )

    with split_path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        split_data = json.load(file)

    return split_data


def validate_split_data(
    split_data,
    images_root,
):
    if split_data["seed"] != SPLIT_SEED:
        raise ValueError(
            "The split file uses the wrong seed."
        )

    if (
        split_data["validation_fraction"]
        != VALIDATION_FRACTION
    ):
        raise ValueError(
            "The split file uses the wrong validation fraction."
        )

    if split_data["source_domains"] != SOURCE_DOMAINS:
        raise ValueError(
            "The split file uses unexpected source domains."
        )

    if split_data["target_domain"] != TARGET_DOMAIN:
        raise ValueError(
            "The split file uses the wrong target domain."
        )

    if split_data["classes"] != PACS_CLASSES:
        raise ValueError(
            "The split file uses the wrong class mapping."
        )

    for domain in SOURCE_DOMAINS:
        all_samples = collect_domain_samples(
            images_root,
            domain,
        )

        domain_data = split_data["domains"][domain]

        train_samples = domain_data["train"]
        validation_samples = domain_data["validation"]

        verify_domain_split(
            all_samples,
            train_samples,
            validation_samples,
        )

        for sample in train_samples + validation_samples:
            image_path = (
                Path(images_root)
                / sample["path"]
            )

            if not image_path.exists():
                raise FileNotFoundError(
                    f"Split image not found: {image_path}"
                )

            label = int(sample["label"])
            expected_class = PACS_CLASSES[label]

            if sample["class_name"] != expected_class:
                raise ValueError(
                    "A sample has an inconsistent "
                    "class name and label."
                )


def load_or_create_split(
    images_root,
    split_path,
):
    split_path = Path(split_path)

    if split_path.exists():
        split_data = load_split_data(
            split_path
        )

        print("Using existing split file.")

    else:
        split_data = build_split_data(
            images_root
        )

        save_split_data(
            split_data,
            split_path,
        )

        print("Created a new split file.")

    validate_split_data(
        split_data,
        images_root,
    )

    return split_data


def print_split_summary(split_data):
    total_train = 0
    total_validation = 0

    for domain in SOURCE_DOMAINS:
        domain_data = split_data["domains"][domain]

        train_count = len(
            domain_data["train"]
        )

        validation_count = len(
            domain_data["validation"]
        )

        total_train += train_count
        total_validation += validation_count

        print(
            f"{domain:15s} | "
            f"train: {train_count:4d} | "
            f"validation: {validation_count:3d}"
        )

    print(
        f"\nTotal source training images: "
        f"{total_train}"
    )

    print(
        f"Total source validation images: "
        f"{total_validation}"
    )


def main():
    project_root = Path(__file__).resolve().parents[1]

    images_root = (
        project_root
        / "data"
        / "pacs"
        / "images"
    )

    split_path = (
        project_root
        / "shared"
        / "splits"
        / "pacs_sketch_seed6304.json"
    )

    split_data = load_or_create_split(
        images_root,
        split_path,
    )

    print_split_summary(split_data)


if __name__ == "__main__":
    main()