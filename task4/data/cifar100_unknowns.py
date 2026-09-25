import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR100

from task4.data.cifar10 import get_evaluation_transform


NEAR_UNKNOWN_CLASSES = [
    "bus",
    "pickup_truck",
    "motorcycle",
    "tractor",
    "wolf",
    "fox",
    "leopard",
    "camel"
]

FAR_UNKNOWN_CLASSES = [
    "bottle",
    "bowl",
    "chair",
    "clock",
    "keyboard",
    "mushroom",
    "sunflower",
    "wardrobe"
]


class CIFAR100UnknownSubset(Dataset):
    def __init__(self, dataset, indices, group):
        self.dataset = dataset
        self.indices = indices
        self.group = group

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        dataset_index = self.indices[position]
        image, label = self.dataset[dataset_index]

        return image, label, dataset_index


def select_class_indices(dataset, class_names):
    selected_class_ids = {
        dataset.class_to_idx[class_name]
        for class_name in class_names
    }

    return [
        index
        for index, label in enumerate(dataset.targets)
        if label in selected_class_ids
    ]


def get_cifar100_unknown_datasets(data_root="data"):
    dataset = CIFAR100(
        root=data_root,
        train=False,
        transform=get_evaluation_transform(),
        download=True
    )

    near_indices = select_class_indices(
        dataset,
        NEAR_UNKNOWN_CLASSES
    )

    far_indices = select_class_indices(
        dataset,
        FAR_UNKNOWN_CLASSES
    )

    all_indices = near_indices + far_indices

    return {
        "near": CIFAR100UnknownSubset(
            dataset,
            near_indices,
            "near"
        ),
        "far": CIFAR100UnknownSubset(
            dataset,
            far_indices,
            "far"
        ),
        "all": CIFAR100UnknownSubset(
            dataset,
            all_indices,
            "all"
        )
    }


def get_cifar100_unknown_loaders(
    batch_size=128,
    num_workers=2
):
    datasets = get_cifar100_unknown_datasets()

    loader_settings = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0
    }

    return {
        name: DataLoader(dataset, **loader_settings)
        for name, dataset in datasets.items()
    }