"""
Main entry point for Federated Learning on MNIST.

To run:
    python main_simulation.py

To customise:
    Edit the CONFIG dict below, or pass --help for CLI flags.
"""

import argparse
from html import parser
import json
import os
import time
from typing import Dict

import flwr as flwr
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")    # non-interactive backend (works without a display)
import matplotlib.pyplot as plt

from model.mnist_model import get_model
from server.fl_server import build_strategy
from client.fl_client import MNISTFlowerClient, make_client_fn
from utils.data_utils import (
    load_mnist_datasets,
    iid_partition,
    non_iid_partition,
    make_dataloader,
)
from utils.train_utils import evaluate
import torch.nn as nn

# ============================================================
# Default experiment configuration
# ============================================================
CONFIG: Dict = {
    # --- Federated Learning ---
    "num_clients": 10,          # total number of FL clients
    "num_rounds": 10,           # number of FL communication rounds
    "fraction_fit": 0.5,        # fraction of clients selected each round
    "fraction_evaluate": 0.5,   # fraction used for distributed evaluation

    # --- Data ---
    "partition": "iid",         # 'iid' or 'non_iid'
    "num_classes_per_client": 2,# (non-IID only) classes per client
    "data_dir": "./data",       # MNIST download directory

    # --- Local training per client ---
    "batch_size": 32,
    "local_epochs": 2,          # epochs run on client before sending weights
    "learning_rate": 0.01,
    "optimizer": "sgd",         # 'sgd' or 'adam'

    # --- Output ---
    "results_dir": "./results",
    "seed": 42,

    # --- Differential Privacy (optional) ---
    "dp_enabled": False,        
    "clip_norm": 1.0,          
    "noise_multiplier": 0.0,   
}


# ============================================================
# Results logging & plotting
# ============================================================

def save_results(metrics: list, results_dir: str, config: Dict) -> None:
    """Save per-round metrics to JSON and generate accuracy/loss plots."""
    os.makedirs(results_dir, exist_ok=True)

    # --- Save metrics JSON ---
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"config": config, "rounds": metrics}, f, indent=2)
    print(f"[Results] Metrics saved to {metrics_path}")

    # --- Plot accuracy curve ---
    rounds = [m["round"] for m in metrics if "federated_test_accuracy" in m]
    accs   = [m["federated_test_accuracy"] for m in metrics if "federated_test_accuracy" in m]

    if rounds:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(rounds, accs, marker="o", linewidth=2, label="Federated Test Accuracy")
        ax.set_xlabel("Communication Round")
        ax.set_ylabel("Accuracy")
        ax.set_title(
            f"FedAvg on MNIST — {config['num_clients']} clients, "
            f"{config['partition'].upper()} partition"
        )
        ax.set_ylim(0, 1.0)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(results_dir, "accuracy_curve.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"[Results] Accuracy curve saved to {plot_path}")

    # --- Plot train loss curve ---
    rounds_loss = [m["round"] for m in metrics if "avg_train_loss" in m]
    losses      = [m["avg_train_loss"] for m in metrics if "avg_train_loss" in m]

    if rounds_loss:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(rounds_loss, losses, marker="s", color="tomato",
                linewidth=2, label="Avg Client Train Loss")
        ax.set_xlabel("Communication Round")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.set_title("Average Client Training Loss per Round")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        loss_plot_path = os.path.join(results_dir, "loss_curve.png")
        plt.savefig(loss_plot_path, dpi=150)
        plt.close()
        print(f"[Results] Loss curve saved to {loss_plot_path}")


def print_summary(metrics: list, config: Dict) -> None:
    """Print a human-readable experiment summary."""
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY — Step 1: Federated Learning on MNIST")
    print("="*60)
    print(f"  Clients         : {config['num_clients']}")
    print(f"  Rounds          : {config['num_rounds']}")
    print(f"  Partition       : {config['partition']}")
    print(f"  Local epochs    : {config['local_epochs']}")
    print(f"  Learning rate   : {config['learning_rate']}")
    print(f"  Optimizer       : {config['optimizer']}")
    print("-"*60)

    if metrics:
        final = metrics[-1]
        print(f"  Final round         : {final['round']}")
        if "avg_train_loss" in final:
            print(f"  Avg train loss      : {final['avg_train_loss']:.4f}")
        if "avg_train_accuracy" in final:
            print(f"  Avg train accuracy  : {final['avg_train_accuracy']:.4f}")
        if "federated_test_accuracy" in final:
            print(f"  Federated test acc  : {final['federated_test_accuracy']:.4f}")
    print("="*60 + "\n")


# ============================================================
# Main simulation loop
# ============================================================

def run_simulation(config: Dict) -> None:
    """Run the complete FL simulation with the given config."""
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Setup] Using device: {device}")

    # ------------------------------------------------------------------
    # 1. Load MNIST and partition across clients
    # ------------------------------------------------------------------
    train_dataset, test_dataset = load_mnist_datasets(config["data_dir"])

    if config["partition"] == "non_iid":
        client_datasets = non_iid_partition(
            train_dataset,
            num_clients=config["num_clients"],
            num_classes_per_client=config["num_classes_per_client"],
            seed=config["seed"],
        )
    else:
        client_datasets = iid_partition(
            train_dataset,
            num_clients=config["num_clients"],
            seed=config["seed"],
        )

    # ------------------------------------------------------------------
    # 2. Build strategy
    # ------------------------------------------------------------------
    min_clients = max(2, int(config["num_clients"] * config["fraction_fit"]))
    strategy = build_strategy(
        num_clients=config["num_clients"],
        fraction_fit=config["fraction_fit"],
        fraction_evaluate=config["fraction_evaluate"],
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=config["num_clients"],
        test_dataset=test_dataset,   # enables server-side central evaluation
        device=device,
    )

    # ------------------------------------------------------------------
    # 3. Build client factory (Flower 2.x: must accept Context)
    # ------------------------------------------------------------------
    client_config = {
        "batch_size": config["batch_size"],
        "local_epochs": config["local_epochs"],
        "learning_rate": config["learning_rate"],
        "optimizer": config["optimizer"],
        "dp_enabled": config["dp_enabled"],
        "clip_norm": config["clip_norm"],
        "noise_multiplier": config["noise_multiplier"],
    }
    _client_fn = make_client_fn(
        client_datasets=client_datasets,
        test_dataset=test_dataset,
        client_config=client_config,
    )

    # ------------------------------------------------------------------
    # 4. Run Flower simulation
    # ------------------------------------------------------------------
    print(f"\n[Simulation] Starting FL simulation ({config['num_rounds']} rounds)...")
    t_start = time.time()

    flwr.simulation.start_simulation(
        client_fn=_client_fn,
        num_clients=config["num_clients"],
        config=flwr.server.ServerConfig(num_rounds=config["num_rounds"]),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )

    elapsed = time.time() - t_start
    print(f"\n[Simulation] Finished in {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # 5. Save results
    # ------------------------------------------------------------------
    save_results(strategy.round_metrics, config["results_dir"], config)
    print_summary(strategy.round_metrics, config)


# ============================================================
# CLI interface
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Step 1: Federated Learning on MNIST with Flower + PyTorch"
    )
    parser.add_argument("--num_clients",   type=int,   default=CONFIG["num_clients"])
    parser.add_argument("--num_rounds",    type=int,   default=CONFIG["num_rounds"])
    parser.add_argument("--partition",     type=str,   default=CONFIG["partition"],
                        choices=["iid", "non_iid"])
    parser.add_argument("--local_epochs",  type=int,   default=CONFIG["local_epochs"])
    parser.add_argument("--learning_rate", type=float, default=CONFIG["learning_rate"])
    parser.add_argument("--optimizer",     type=str,   default=CONFIG["optimizer"],
                        choices=["sgd", "adam"])
    parser.add_argument("--batch_size",    type=int,   default=CONFIG["batch_size"])
    parser.add_argument("--fraction_fit", type=float, default=CONFIG["fraction_fit"])
    parser.add_argument("--data_dir",      type=str,   default=CONFIG["data_dir"])
    parser.add_argument("--results_dir",   type=str,   default=CONFIG["results_dir"])
    parser.add_argument("--dp_enabled", action="store_true")
    parser.add_argument("--clip_norm", type=float, default=CONFIG["clip_norm"])
    parser.add_argument("--noise_multiplier", type=float, default=CONFIG["noise_multiplier"])
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_config = {**CONFIG, **vars(args)}
    run_simulation(run_config)
