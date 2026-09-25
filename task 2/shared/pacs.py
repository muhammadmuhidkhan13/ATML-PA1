from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights


PACS_DOMAINS = [
    "art_painting",
    "cartoon",
    "photo",
    "sketch",
]

PACS_CLASSES = [
    "dog",
    "elephant",
    "giraffe",
    "guitar",
    "horse",
    "house",
    "person",
]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


def list_image_paths(folder):
    folder = Path(folder)

    image_paths = [
        path
        for path in folder.rglob("*")
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    return sorted(image_paths)


def validate_pacs(pacs_root):
    pacs_root = Path(pacs_root)
    images_root = pacs_root / "images"

    if not images_root.exists():
        raise FileNotFoundError(
            f"PACS image folder was not found: {images_root}"
        )

    total_images = 0

    print(f"PACS images: {images_root}\n")

    for domain in PACS_DOMAINS:
        domain_folder = images_root / domain

        if not domain_folder.exists():
            raise FileNotFoundError(
                f"Missing domain: {domain_folder}"
            )

        found_classes = sorted(
            folder.name
            for folder in domain_folder.iterdir()
            if folder.is_dir()
        )

        if found_classes != PACS_CLASSES:
            raise ValueError(
                f"Unexpected classes in {domain}.\n"
                f"Expected: {PACS_CLASSES}\n"
                f"Found: {found_classes}"
            )

        domain_count = len(
            list_image_paths(domain_folder)
        )

        total_images += domain_count

        print(
            f"{domain:15s}: "
            f"{domain_count} images"
        )

    print(f"\nTotal images: {total_images}")


def get_image_transforms():
    weights = ResNet18_Weights.IMAGENET1K_V1
    pretrained_transform = weights.transforms()

    normalize = transforms.Normalize(
        mean=pretrained_transform.mean,
        std=pretrained_transform.std,
    )

    train_transform = transforms.Compose([
        transforms.Resize(
            (256, 256),
            antialias=True,
        ),
        transforms.RandomCrop(
            (224, 224),
        ),
        transforms.RandomHorizontalFlip(
            p=0.5,
        ),
        transforms.ToTensor(),
        normalize,
    ])

    evaluation_transform = transforms.Compose([
        transforms.Resize(
            (256, 256),
            antialias=True,
        ),
        transforms.CenterCrop(
            (224, 224),
        ),
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, evaluation_transform


def open_rgb_image(image_path):
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")

    return rgb_image


class PACSLabeledDataset(Dataset):
    def __init__(
        self,
        images_root,
        samples,
        transform,
    ):
        self.images_root = Path(images_root)
        self.samples = list(samples)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        image_path = (
            self.images_root
            / sample["path"]
        )

        image = open_rgb_image(image_path)

        if self.transform is not None:
            image = self.transform(image)

        label = int(sample["label"])

        return image, label


class PACSUnlabeledDataset(Dataset):
    def __init__(
        self,
        images_root,
        domain,
        transform,
    ):
        self.images_root = Path(images_root)
        self.domain = domain
        self.transform = transform

        domain_folder = (
            self.images_root
            / self.domain
        )

        if not domain_folder.exists():
            raise FileNotFoundError(
                f"Domain folder not found: {domain_folder}"
            )

        self.image_paths = list_image_paths(
            domain_folder
        )

        if not self.image_paths:
            raise ValueError(
                f"No images found in: {domain_folder}"
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = open_rgb_image(image_path)

        if self.transform is not None:
            image = self.transform(image)

        return image


def main():
    project_root = Path(__file__).resolve().parents[1]
    pacs_root = project_root / "data" / "pacs"
    images_root = pacs_root / "images"

    validate_pacs(pacs_root)

    train_transform, _ = get_image_transforms()

    target_dataset = PACSUnlabeledDataset(
        images_root=images_root,
        domain="sketch",
        transform=train_transform,
    )

    example_image = target_dataset[0]

    print(
        "\nUnlabeled target images: "
        f"{len(target_dataset)}"
    )

    print(
        "Example target tensor: "
        f"{example_image.shape}"
    )


if __name__ == "__main__":
    main()