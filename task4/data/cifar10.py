import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class IndexedSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        dataset_index = self.indices[position]
        image, label = self.dataset[dataset_index]

        return image, label, dataset_index


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_train_transform(use_randaugment=False):
    operations = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip()
    ]

    if use_randaugment:
        operations.append(
            transforms.RandAugment(
                num_ops=2,
                magnitude=9
            )
        )

    operations.extend([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

    return transforms.Compose(operations)


def get_evaluation_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])


def get_cifar10_datasets(
    data_root="data",
    split_path="task4/data/cifar10_split_seed6304.json",
    use_randaugment=False
):
    with open(split_path, "r") as file:
        split = json.load(file)

    train_indices = split["train_indices"]
    val_indices = split["val_indices"]

    augmented_training_data = CIFAR10(
        root=data_root,
        train=True,
        transform=get_train_transform(use_randaugment),
        download=True
    )

    evaluation_training_data = CIFAR10(
        root=data_root,
        train=True,
        transform=get_evaluation_transform(),
        download=True
    )

    test_data = CIFAR10(
        root=data_root,
        train=False,
        transform=get_evaluation_transform(),
        download=True
    )

    train_dataset = IndexedSubset(
        augmented_training_data,
        train_indices
    )

    train_unaugmented_dataset = IndexedSubset(
        evaluation_training_data,
        train_indices
    )

    val_dataset = IndexedSubset(
        evaluation_training_data,
        val_indices
    )

    test_dataset = IndexedSubset(
        test_data,
        list(range(len(test_data)))
    )

    return {
        "train": train_dataset,
        "train_unaugmented": train_unaugmented_dataset,
        "val": val_dataset,
        "test": test_dataset
    }


def get_cifar10_loaders(
    batch_size=128,
    num_workers=2,
    use_randaugment=False,
    seed=6304
):
    datasets = get_cifar10_datasets(
        use_randaugment=use_randaugment
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader_settings = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
        "persistent_workers": num_workers > 0
    }

    train_loader = DataLoader(
        datasets["train"],
        shuffle=True,
        generator=generator,
        **loader_settings
    )

    train_unaugmented_loader = DataLoader(
        datasets["train_unaugmented"],
        shuffle=False,
        **loader_settings
    )

    val_loader = DataLoader(
        datasets["val"],
        shuffle=False,
        **loader_settings
    )

    test_loader = DataLoader(
        datasets["test"],
        shuffle=False,
        **loader_settings
    )

    return {
        "train": train_loader,
        "train_unaugmented": train_unaugmented_loader,
        "val": val_loader,
        "test": test_loader
    }