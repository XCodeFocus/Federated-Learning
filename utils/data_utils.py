"""
Utilities for loading MNIST and partitioning it across N federated clients.

Two partitioning strategies are provided:

  IID   — each client receives a random, balanced subset of all digit classes.
           Represents the "ideal" case where data is uniformly distributed.

  Non-IID — each client receives data predominantly from K digit classes,
             simulating realistic heterogeneous data across hospitals / devices.
             Controlled via `num_classes_per_client`.
"""

import random
from collections import defaultdict
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Shared transform: normalise to match MNIST statistics
# ---------------------------------------------------------------------------
MNIST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),                     # pixel values to [0, 1]
    transforms.Normalize((0.1307,), (0.3081,)) # MNIST mean / std
])


def load_mnist_datasets(data_dir: str = "./data") -> Tuple[Dataset, Dataset]:
    """
    Download (if necessary) and return (train_dataset, test_dataset).

    Parameters
    ----------
    data_dir : path where MNIST files will be cached.

    Returns
    -------
    Tuple of (train, test) torchvision Dataset objects.
    """
    train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=MNIST_TRANSFORM
    )
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=MNIST_TRANSFORM
    )
    print(
        f"[Data] MNIST loaded — train: {len(train_dataset)} samples, "
        f"test: {len(test_dataset)} samples"
    )
    return train_dataset, test_dataset


# ---------------------------------------------------------------------------
# IID partitioning
# ---------------------------------------------------------------------------

def iid_partition(
    dataset: Dataset,
    num_clients: int,
    seed: int = 42,
) -> List[Subset]:
    """
    Randomly split `dataset` into `num_clients` equal parts (IID).

    Parameters
    ----------
    dataset     : full training dataset (e.g. MNIST train split).
    num_clients : number of FL clients.
    seed        : random seed for reproducibility.

    Returns
    -------
    List of torch.utils.data.Subset, one per client.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    num_samples = len(dataset)
    indices = list(range(num_samples))
    random.shuffle(indices)

    # Split indices into equal chunks
    chunk_size = num_samples // num_clients
    client_subsets: List[Subset] = []

    for i in range(num_clients):
        start = i * chunk_size
        # Give any remainder samples to the last client
        end = start + chunk_size if i < num_clients - 1 else num_samples
        client_subsets.append(Subset(dataset, indices[start:end]))

    print(
        f"[IID] Split {num_samples} samples into {num_clients} clients "
        f"(~{chunk_size} each)"
    )
    return client_subsets


# ---------------------------------------------------------------------------
# Non-IID partitioning  (label-skew approach)
# ---------------------------------------------------------------------------

def non_iid_partition(
    dataset: Dataset,
    num_clients: int,
    num_classes_per_client: int = 2,
    seed: int = 42,
) -> List[Subset]:
    """
    Non-IID split: each client receives samples from only
    `num_classes_per_client` digit classes.

    This simulates realistic label distribution skew — e.g. one hospital
    treating patients with a specific demographic, or a device used by
    someone who writes only certain digits.

    Parameters
    ----------
    dataset                : full training dataset.
    num_clients            : number of FL clients.
    num_classes_per_client : how many distinct labels each client sees.
                             Default 2 → each client only sees 2 digit classes.
    seed                   : random seed.

    Returns
    -------
    List of torch.utils.data.Subset, one per client.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    # Group indices by class label
    label_to_indices: dict = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        label_to_indices[int(label)].append(idx)

    # Shuffle within each class
    for label in label_to_indices:
        random.shuffle(label_to_indices[label])

    # Assign classes to clients in a round-robin fashion
    all_labels = list(label_to_indices.keys())   # [0, 1, ..., 9]
    num_classes = len(all_labels)

    client_subsets: List[Subset] = []
    for client_id in range(num_clients):
        # Pick `num_classes_per_client` labels for this client
        client_labels = [
            all_labels[(client_id * num_classes_per_client + k) % num_classes]
            for k in range(num_classes_per_client)
        ]

        # Collect indices from those classes, split evenly among clients sharing them
        client_indices: List[int] = []
        for lbl in client_labels:
            lbl_indices = label_to_indices[lbl]
            # How many clients share this label?
            sharing_clients = [
                c for c in range(num_clients)
                if lbl in [
                    all_labels[(c * num_classes_per_client + k) % num_classes]
                    for k in range(num_classes_per_client)
                ]
            ]
            n_sharing = max(1, len(sharing_clients))
            chunk = len(lbl_indices) // n_sharing
            pos = sharing_clients.index(client_id) if client_id in sharing_clients else 0
            client_indices.extend(lbl_indices[pos * chunk: (pos + 1) * chunk])

        random.shuffle(client_indices)
        client_subsets.append(Subset(dataset, client_indices))
        print(
            f"  Client {client_id:02d} → labels {client_labels}, "
            f"samples: {len(client_indices)}"
        )

    print(
        f"[Non-IID] {num_clients} clients, "
        f"{num_classes_per_client} classes each"
    )
    return client_subsets


# ---------------------------------------------------------------------------
# DataLoader helpers
# ---------------------------------------------------------------------------

def make_dataloader(
    subset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Wrap a Subset (or Dataset) in a DataLoader."""
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )
