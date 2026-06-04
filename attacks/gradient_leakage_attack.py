"""
Standalone Gradient Leakage Attack (DLG/iDLG) on the MNIST FL model.

This script is intended for local privacy evaluation and teaching. It uses the
repo's own MNIST partitions and model, then reconstructs a selected local batch
from the raw gradients that a federated server could observe in a toy setup.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.mnist_model import get_model
from utils.data_utils import iid_partition, load_mnist_datasets, non_iid_partition
from utils.train_utils import compute_attack_gradients


MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
IMAGE_SHAPE = (1, 28, 28)
NUM_CLASSES = 10
NORMALIZED_PIXEL_MIN = (0.0 - MNIST_MEAN) / MNIST_STD
NORMALIZED_PIXEL_MAX = (1.0 - MNIST_MEAN) / MNIST_STD


def seed_everything(seed: int) -> None:
    """Make the standalone experiment reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    """Resolve auto/cpu/cuda into a torch device."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return torch.device(requested)


def normalized_to_pixel(images: torch.Tensor) -> torch.Tensor:
    """Convert MNIST-normalized tensors back to displayable [0, 1] pixels."""
    return (images.detach().cpu() * MNIST_STD + MNIST_MEAN).clamp(0.0, 1.0)


def pixel_to_normalized(images: torch.Tensor) -> torch.Tensor:
    """Convert [0, 1] image tensors to the model's MNIST normalization."""
    return (images - MNIST_MEAN) / MNIST_STD


def get_target_batch(
    dataset: Dataset,
    sample_index: int,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Read a deterministic contiguous batch from a client dataset."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if sample_index < 0:
        raise ValueError("sample_index must be non-negative.")
    if sample_index + batch_size > len(dataset):
        raise ValueError(
            f"Requested sample_index={sample_index}, batch_size={batch_size}, "
            f"but the client dataset has only {len(dataset)} samples."
        )

    images: List[torch.Tensor] = []
    labels: List[int] = []
    for offset in range(batch_size):
        image, label = dataset[sample_index + offset]
        images.append(image)
        labels.append(int(label))
    return torch.stack(images, dim=0), torch.tensor(labels, dtype=torch.long)


def gradient_distance(
    candidate_gradients: Sequence[torch.Tensor],
    target_gradients: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Gradient matching objective used by DLG."""
    loss = torch.zeros((), device=candidate_gradients[0].device)
    for candidate, target in zip(candidate_gradients, target_gradients):
        loss = loss + F.mse_loss(candidate, target)
    return loss


def soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    """Cross entropy for optimized soft labels."""
    return -(soft_targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def compute_soft_label_attack_gradients(
    model: nn.Module,
    images: torch.Tensor,
    label_logits: torch.Tensor,
    device: torch.device,
    create_graph: bool = False,
    detach: bool = True,
) -> List[torch.Tensor]:
    """Compute attack gradients when labels are optimized as soft logits."""
    model.eval()
    model.to(device)
    images = images.to(device)
    label_probs = torch.softmax(label_logits.to(device), dim=1)

    model.zero_grad(set_to_none=True)
    outputs = model(images)
    loss = soft_cross_entropy(outputs, label_probs)
    gradients = torch.autograd.grad(
        loss,
        tuple(model.parameters()),
        create_graph=create_graph,
        retain_graph=create_graph,
    )
    if detach:
        return [grad.detach().clone() for grad in gradients]
    return list(gradients)


def infer_labels_from_last_bias_gradient(
    target_gradients: Sequence[torch.Tensor],
    batch_size: int,
) -> Optional[torch.Tensor]:
    """
    Infer labels with iDLG's last-layer bias rule.

    For batch size 1, the true class has the most negative final bias gradient.
    For larger batches, this gradient only reveals class aggregate information,
    so the caller should fall back to optimizing soft labels.
    """
    if batch_size != 1:
        return None

    last_bias_gradient = target_gradients[-1]
    inferred_label = int(torch.argmin(last_bias_gradient).item())
    return torch.tensor([inferred_label], dtype=torch.long)


def make_optimizer(
    params: Sequence[torch.Tensor],
    optimizer_name: str,
    lr: float,
) -> torch.optim.Optimizer:
    """Build the requested reconstruction optimizer."""
    name = optimizer_name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "lbfgs":
        return torch.optim.LBFGS(
            params,
            lr=lr,
            max_iter=20,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def run_reconstruction(
    model: nn.Module,
    target_gradients: Sequence[torch.Tensor],
    batch_size: int,
    device: torch.device,
    num_iterations: int,
    lr: float,
    optimizer_name: str,
    labels: Optional[torch.Tensor] = None,
    optimize_labels: bool = False,
    seed: int = 42,
) -> Dict:
    """
    Reconstruct a batch by matching gradients from a dummy image.

    If ``optimize_labels`` is true, labels are optimized as soft label logits.
    Otherwise, ``labels`` must contain the hard labels used for the attack.
    """
    if not optimize_labels and labels is None:
        raise ValueError("Hard-label reconstruction requires labels.")

    torch.manual_seed(seed)
    initial_pixels = torch.rand((batch_size, *IMAGE_SHAPE), device=device)
    dummy_images = pixel_to_normalized(initial_pixels).detach().clone()
    dummy_images.requires_grad_(True)

    params: List[torch.Tensor] = [dummy_images]
    dummy_label_logits: Optional[torch.Tensor] = None
    hard_labels: Optional[torch.Tensor] = None
    if optimize_labels:
        dummy_label_logits = torch.randn(
            (batch_size, NUM_CLASSES),
            device=device,
            requires_grad=True,
        )
        params.append(dummy_label_logits)
    else:
        hard_labels = labels.to(device)

    optimizer = make_optimizer(params, optimizer_name, lr)
    target_gradients = [grad.to(device) for grad in target_gradients]
    losses: List[float] = []

    def current_dummy_images() -> torch.Tensor:
        return dummy_images

    def clamp_dummy_images() -> None:
        with torch.no_grad():
            dummy_images.clamp_(NORMALIZED_PIXEL_MIN, NORMALIZED_PIXEL_MAX)

    def build_candidate_gradients(create_graph: bool, detach: bool) -> List[torch.Tensor]:
        dummy_images = current_dummy_images()
        if optimize_labels:
            assert dummy_label_logits is not None
            return compute_soft_label_attack_gradients(
                model=model,
                images=dummy_images,
                label_logits=dummy_label_logits,
                device=device,
                create_graph=create_graph,
                detach=detach,
            )
        assert hard_labels is not None
        return compute_attack_gradients(
            model=model,
            images=dummy_images,
            labels=hard_labels,
            device=device,
            create_graph=create_graph,
            detach=detach,
        )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        candidate_gradients = build_candidate_gradients(create_graph=True, detach=False)
        loss = gradient_distance(candidate_gradients, target_gradients)
        loss.backward()
        return loss

    for _ in range(num_iterations):
        if optimizer_name.lower() == "lbfgs":
            loss = optimizer.step(closure)
        else:
            loss = closure()
            optimizer.step()
        clamp_dummy_images()
        losses.append(float(loss.detach().cpu()))

    with torch.no_grad():
        reconstructed_pixels = normalized_to_pixel(dummy_images.detach())
        inferred_labels = None
        if optimize_labels:
            assert dummy_label_logits is not None
            inferred_labels = torch.argmax(dummy_label_logits.detach().cpu(), dim=1)
        elif hard_labels is not None:
            inferred_labels = hard_labels.detach().cpu()

    final_gradients = build_candidate_gradients(create_graph=False, detach=True)
    final_loss = float(gradient_distance(final_gradients, target_gradients).detach().cpu())

    return {
        "reconstructed_pixels": reconstructed_pixels,
        "labels": inferred_labels,
        "losses": losses,
        "final_gradient_loss": final_loss,
        "optimizer_used": optimizer_name.lower(),
        "used_adam_fallback": False,
    }


def gradient_loss_stagnated(result: Dict, min_relative_improvement: float = 0.05) -> bool:
    """Return true when reconstruction made too little progress to be useful."""
    losses = result.get("losses", [])
    if not losses:
        return False
    initial_loss = float(losses[0])
    final_loss = float(result["final_gradient_loss"])
    if initial_loss <= 0:
        return False
    return final_loss >= initial_loss * (1.0 - min_relative_improvement)


def run_reconstruction_with_fallback(
    model: nn.Module,
    target_gradients: Sequence[torch.Tensor],
    batch_size: int,
    device: torch.device,
    num_iterations: int,
    lr: float,
    optimizer_name: str,
    labels: Optional[torch.Tensor] = None,
    optimize_labels: bool = False,
    seed: int = 42,
    label: str = "attack",
) -> Dict:
    """
    Run the requested optimizer, then fall back to Adam if LBFGS stalls.

    LBFGS is the plan's default optimizer, but it can be brittle for this small
    CNN gradient-matching objective. Adam gives a reliable teaching baseline.
    """
    result = run_reconstruction(
        model=model,
        target_gradients=target_gradients,
        batch_size=batch_size,
        device=device,
        num_iterations=num_iterations,
        lr=lr,
        optimizer_name=optimizer_name,
        labels=labels,
        optimize_labels=optimize_labels,
        seed=seed,
    )

    if optimizer_name.lower() != "lbfgs" or not gradient_loss_stagnated(result):
        return result

    print(f"[Attack] {label}: LBFGS stalled, falling back to Adam.")
    fallback = run_reconstruction(
        model=model,
        target_gradients=target_gradients,
        batch_size=batch_size,
        device=device,
        num_iterations=num_iterations,
        lr=lr,
        optimizer_name="adam",
        labels=labels,
        optimize_labels=optimize_labels,
        seed=seed,
    )
    fallback["requested_optimizer"] = optimizer_name.lower()
    fallback["used_adam_fallback"] = True
    return fallback


def reconstruction_metrics(
    target_pixels: torch.Tensor,
    reconstructed_pixels: torch.Tensor,
) -> Dict[str, Optional[float]]:
    """Compute lightweight image quality metrics over [0, 1] tensors."""
    target = target_pixels.float()
    reconstructed = reconstructed_pixels.float()
    mse = float(F.mse_loss(reconstructed, target).item())
    psnr = None if mse == 0 else 20.0 * math.log10(1.0 / math.sqrt(mse))

    target_flat = target.reshape(target.shape[0], -1)
    reconstructed_flat = reconstructed.reshape(reconstructed.shape[0], -1)
    ssim_values: List[float] = []
    c1 = 0.01**2
    c2 = 0.03**2
    for original, recovered in zip(target_flat, reconstructed_flat):
        mu_x = original.mean()
        mu_y = recovered.mean()
        var_x = original.var(unbiased=False)
        var_y = recovered.var(unbiased=False)
        cov_xy = ((original - mu_x) * (recovered - mu_y)).mean()
        numerator = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
        denominator = (mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2)
        ssim_values.append(float((numerator / denominator).item()))

    return {
        "mse": mse,
        "psnr": psnr,
        "ssim": float(np.mean(ssim_values)),
    }


def save_batch_image(images: torch.Tensor, path: Path, title: str) -> None:
    """Save a compact image grid for a batch."""
    max_rows = min(images.shape[0], 8)
    fig, axes = plt.subplots(1, max_rows, figsize=(1.8 * max_rows, 2.1))
    if max_rows == 1:
        axes = [axes]

    for idx in range(max_rows):
        axes[idx].imshow(images[idx, 0].numpy(), cmap="gray", vmin=0.0, vmax=1.0)
        axes[idx].set_title(f"{idx}", fontsize=9)
        axes[idx].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_comparison(
    target_pixels: torch.Tensor,
    reconstructed_pixels: torch.Tensor,
    labels: Sequence[int],
    path: Path,
    title: str,
) -> None:
    """Save original/reconstructed/difference comparison rows."""
    max_rows = min(target_pixels.shape[0], 8)
    fig, axes = plt.subplots(max_rows, 3, figsize=(6.6, 2.1 * max_rows))
    if max_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    diff = (target_pixels - reconstructed_pixels).abs().clamp(0.0, 1.0)
    column_titles = ["Original", "Reconstructed", "Absolute diff"]
    for col, column_title in enumerate(column_titles):
        axes[0, col].set_title(column_title, fontsize=10)

    for row in range(max_rows):
        axes[row, 0].imshow(target_pixels[row, 0].numpy(), cmap="gray", vmin=0.0, vmax=1.0)
        axes[row, 1].imshow(
            reconstructed_pixels[row, 0].numpy(),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
        axes[row, 2].imshow(diff[row, 0].numpy(), cmap="magma", vmin=0.0, vmax=1.0)
        axes[row, 0].set_ylabel(f"label={labels[row]}", fontsize=9)
        for col in range(3):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_loss_curve(
    known_losses: Sequence[float],
    inferred_losses: Sequence[float],
    path: Path,
) -> None:
    """Save gradient matching loss curves."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(known_losses, label="Known label DLG", linewidth=2)
    ax.plot(inferred_losses, label="Inferred label DLG", linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Gradient matching loss")
    ax.set_title("Gradient Leakage Attack Optimization")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_client_partitions(args: argparse.Namespace) -> List[Dataset]:
    """Load MNIST and build the requested FL client partitions."""
    train_dataset, _ = load_mnist_datasets(args.data_dir)
    if args.partition == "non_iid":
        return non_iid_partition(
            train_dataset,
            num_clients=args.num_clients,
            num_classes_per_client=args.num_classes_per_client,
            seed=args.seed,
        )
    return iid_partition(
        train_dataset,
        num_clients=args.num_clients,
        seed=args.seed,
    )


def write_metrics(path: Path, metrics: Dict) -> None:
    """Write JSON metrics with stable formatting."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, allow_nan=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone DLG/iDLG gradient leakage attack on MNIST."
    )
    parser.add_argument("--client_id", type=int, default=0)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_iterations", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--optimizer", type=str, default="lbfgs", choices=["lbfgs", "adam"])
    parser.add_argument("--partition", type=str, default="iid", choices=["iid", "non_iid"])
    parser.add_argument("--results_dir", type=str, default="./results/gradient_leakage")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--num_classes_per_client", type=int, default=2)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Setup] Using device: {device}")
    print(f"[Setup] Writing attack outputs to: {results_dir}")

    client_datasets = make_client_partitions(args)
    if args.client_id < 0 or args.client_id >= len(client_datasets):
        raise ValueError(
            f"client_id must be in [0, {len(client_datasets) - 1}], got {args.client_id}."
        )

    target_images, target_labels = get_target_batch(
        client_datasets[args.client_id],
        sample_index=args.sample_index,
        batch_size=args.batch_size,
    )
    target_pixels = normalized_to_pixel(target_images)

    model = get_model().to(device)
    target_gradients = compute_attack_gradients(
        model=model,
        images=target_images,
        labels=target_labels,
        device=device,
        create_graph=False,
        detach=True,
    )

    print(
        "[Attack] Target labels: "
        f"{target_labels.tolist()} from client {args.client_id}, sample {args.sample_index}"
    )

    known_result = run_reconstruction_with_fallback(
        model=model,
        target_gradients=target_gradients,
        batch_size=args.batch_size,
        device=device,
        num_iterations=args.num_iterations,
        lr=args.lr,
        optimizer_name=args.optimizer,
        labels=target_labels,
        optimize_labels=False,
        seed=args.seed + 1,
        label="Known-label DLG",
    )

    inferred_labels = infer_labels_from_last_bias_gradient(
        target_gradients=target_gradients,
        batch_size=args.batch_size,
    )
    used_soft_label_fallback = inferred_labels is None
    if used_soft_label_fallback:
        print("[Attack] Batch size > 1, using optimized soft-label fallback.")
        inferred_result = run_reconstruction_with_fallback(
            model=model,
            target_gradients=target_gradients,
            batch_size=args.batch_size,
            device=device,
            num_iterations=args.num_iterations,
            lr=args.lr,
            optimizer_name=args.optimizer,
            labels=None,
            optimize_labels=True,
            seed=args.seed + 2,
            label="Soft-label DLG",
        )
    else:
        print(f"[Attack] iDLG inferred label: {inferred_labels.tolist()}")
        inferred_result = run_reconstruction_with_fallback(
            model=model,
            target_gradients=target_gradients,
            batch_size=args.batch_size,
            device=device,
            num_iterations=args.num_iterations,
            lr=args.lr,
            optimizer_name=args.optimizer,
            labels=inferred_labels,
            optimize_labels=False,
            seed=args.seed + 2,
            label="Inferred-label DLG",
        )

    known_pixels = known_result["reconstructed_pixels"]
    inferred_pixels = inferred_result["reconstructed_pixels"]
    known_metrics = reconstruction_metrics(target_pixels, known_pixels)
    inferred_metrics = reconstruction_metrics(target_pixels, inferred_pixels)

    final_inferred_labels = inferred_result["labels"]
    if final_inferred_labels is None:
        final_inferred_labels = torch.empty(0, dtype=torch.long)
    label_matches = (
        final_inferred_labels.tolist() == target_labels.tolist()
        if final_inferred_labels.numel() == target_labels.numel()
        else False
    )

    save_batch_image(target_pixels, results_dir / "original_batch.png", "Original target batch")
    save_batch_image(
        known_pixels,
        results_dir / "reconstruction_known_label.png",
        "Known-label reconstruction",
    )
    save_batch_image(
        inferred_pixels,
        results_dir / "reconstruction_inferred_label.png",
        "Inferred-label reconstruction",
    )
    save_comparison(
        target_pixels=target_pixels,
        reconstructed_pixels=known_pixels,
        labels=target_labels.tolist(),
        path=results_dir / "comparison_known_label.png",
        title="Known-label DLG reconstruction",
    )
    save_comparison(
        target_pixels=target_pixels,
        reconstructed_pixels=inferred_pixels,
        labels=target_labels.tolist(),
        path=results_dir / "comparison_inferred_label.png",
        title="Inferred-label DLG reconstruction",
    )
    save_loss_curve(
        known_result["losses"],
        inferred_result["losses"],
        results_dir / "attack_loss_curve.png",
    )

    metrics = {
        "config": vars(args),
        "device": str(device),
        "target_labels": target_labels.tolist(),
        "inferred_labels": final_inferred_labels.tolist(),
        "label_inference_method": "soft_label_optimization"
        if used_soft_label_fallback
        else "idlg_last_bias_argmin",
        "label_inference_correct": label_matches,
        "known_label": {
            "final_gradient_loss": known_result["final_gradient_loss"],
            "optimizer_used": known_result["optimizer_used"],
            "used_adam_fallback": known_result["used_adam_fallback"],
            **known_metrics,
            "initial_gradient_loss": known_result["losses"][0]
            if known_result["losses"]
            else None,
        },
        "inferred_label": {
            "final_gradient_loss": inferred_result["final_gradient_loss"],
            "optimizer_used": inferred_result["optimizer_used"],
            "used_adam_fallback": inferred_result["used_adam_fallback"],
            **inferred_metrics,
            "initial_gradient_loss": inferred_result["losses"][0]
            if inferred_result["losses"]
            else None,
        },
        "outputs": {
            "original_batch": str(results_dir / "original_batch.png"),
            "known_reconstruction": str(results_dir / "reconstruction_known_label.png"),
            "inferred_reconstruction": str(results_dir / "reconstruction_inferred_label.png"),
            "known_comparison": str(results_dir / "comparison_known_label.png"),
            "inferred_comparison": str(results_dir / "comparison_inferred_label.png"),
            "loss_curve": str(results_dir / "attack_loss_curve.png"),
        },
    }
    write_metrics(results_dir / "metrics.json", metrics)

    print("[Results] Attack outputs saved:")
    print(f"  - {results_dir / 'comparison_known_label.png'}")
    print(f"  - {results_dir / 'comparison_inferred_label.png'}")
    print(f"  - {results_dir / 'attack_loss_curve.png'}")
    print(f"  - {results_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
