"""
Low-level training and evaluation routines used by every FL client.

These are intentionally separated from the Flower client class so that
the same functions can be used for:
  • Local client training inside Flower callbacks
  • Standalone evaluation / debugging outside the FL loop
  • Step 2 (gradient leakage attack) — computing gradients directly
"""

from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Run a single epoch of local SGD on `model`.

    Parameters
    ----------
    model      : the neural network (MNISTNet).
    dataloader : local client DataLoader.
    optimizer  : e.g. SGD or Adam.
    criterion  : loss function (CrossEntropyLoss).
    device     : 'cpu' or 'cuda'.

    Returns
    -------
    (avg_loss, accuracy) over the full epoch.
    """
    model.train()
    model.to(device)

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()              # clear accumulated gradients
        outputs = model(images)            # forward pass → logits
        loss = criterion(outputs, labels)  # compute cross-entropy loss
        loss.backward()                    # backprop → fill .grad on params
        optimizer.step()                   # update weights

        # --- bookkeeping ---
        total_loss += loss.item() * images.size(0)   # accumulate weighted loss
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def local_train(
    model: nn.Module,
    dataloader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
    optimizer_name: str = "sgd",
) -> Tuple[float, float, int]:
    """
    Train `model` locally for `num_epochs` epochs.

    This is the function called inside the Flower client's `fit()` callback.

    Parameters
    ----------
    model          : the local model (weights already set from server).
    dataloader     : client's local DataLoader.
    num_epochs     : number of local epochs before sending updates back.
    learning_rate  : step size for the local optimizer.
    device         : computation device.
    optimizer_name : 'sgd' (default) or 'adam'.

    Returns
    -------
    (final_loss, final_accuracy, num_examples)
    """
    criterion = nn.CrossEntropyLoss()

    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        # SGD with momentum — standard in FL literature (FedAvg paper uses SGD)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate, momentum=0.9
        )

    for epoch in range(num_epochs):
        loss, acc = train_one_epoch(model, dataloader, optimizer, criterion, device)

    num_examples = len(dataloader.dataset)
    return loss, acc, num_examples


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate `model` on `dataloader` without updating weights.

    Parameters
    ----------
    model      : the (potentially aggregated) model.
    dataloader : test / validation DataLoader.
    criterion  : loss function.
    device     : cpu or cuda.

    Returns
    -------
    (avg_loss, accuracy)
    """
    model.eval()
    model.to(device)

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():   # disable gradient tracking for speed / memory
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Gradient extraction helper (used in Step 2 — gradient leakage attack)
# ---------------------------------------------------------------------------

def compute_gradients(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> list:
    """
    Compute per-parameter gradients for a SINGLE mini-batch.

    This returns the raw gradient tensors that would normally be averaged
    and sent to the FL server. In a gradient leakage attack (Step 2),
    an adversary intercepts exactly these values to reconstruct the
    original training images.

    Parameters
    ----------
    model  : model (weights must already be set).
    images : input batch (B, 1, 28, 28).
    labels : corresponding class labels (B,).
    device : computation device.

    Returns
    -------
    List of gradient tensors, one per model parameter, in parameter order.
    """
    model.train()
    model.to(device)
    images, labels = images.to(device), labels.to(device)

    model.zero_grad()
    criterion = nn.CrossEntropyLoss()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()

    # Detach gradients and move back to CPU for inspection / storage
    gradients = [p.grad.detach().cpu().clone() for p in model.parameters()]
    return gradients


def compute_attack_gradients(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    create_graph: bool = False,
    detach: bool = True,
) -> List[torch.Tensor]:
    """
    Compute deterministic per-parameter gradients for leakage experiments.

    Unlike ``compute_gradients``, this helper puts the model in eval mode before
    the forward pass. That disables dropout, which makes gradient matching
    stable enough for reconstruction attacks. Set ``create_graph=True`` and
    ``detach=False`` when the returned gradients must remain differentiable.
    """
    model.eval()
    model.to(device)
    images, labels = images.to(device), labels.to(device)

    model.zero_grad(set_to_none=True)
    criterion = nn.CrossEntropyLoss()
    outputs = model(images)
    loss = criterion(outputs, labels)
    gradients = torch.autograd.grad(
        loss,
        tuple(model.parameters()),
        create_graph=create_graph,
        retain_graph=create_graph,
    )

    if detach:
        return [grad.detach().clone() for grad in gradients]
    return list(gradients)
