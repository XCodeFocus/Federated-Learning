"""
Flower server configuration using the FedAvg aggregation strategy.

FedAvg (McMahan et al., 2017) is the canonical FL aggregation algorithm:
  For each communication round:
    1. Broadcast global model weights to a fraction of available clients.
    2. Each selected client trains locally and returns updated weights.
    3. Server computes a weighted average of all returned weights.
       w_global = Σ (n_k / N) * w_k   where n_k = client k's sample count.
    4. Repeat for T rounds.

This file defines:
  - Custom FedAvg strategy with per-round logging
  - Server entry point (for subprocess mode)
  - Evaluation function run on the server side each round
"""

from typing import Dict, List, Optional, Tuple, Union

import flwr as flwr
import numpy as np
import torch
import torch.nn as nn
from flwr.common import Metrics, Parameters, Scalar
from flwr.server.strategy import FedAvg
from torch.utils.data import DataLoader

from model.mnist_model import get_model
from utils.train_utils import evaluate
from utils.data_utils import make_dataloader


# ---------------------------------------------------------------------------
# Custom FedAvg strategy with round-level logging
# ---------------------------------------------------------------------------

class FedAvgWithLogging(FedAvg):
    """
    Extends Flower's built-in FedAvg with:
      - Verbose per-round aggregation logging
      - Accumulation of per-round metrics for plotting/reporting
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Storage for post-training analysis (Step 2 / Step 3 will read these)
        self.round_metrics: List[Dict] = []

    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
    ):
        """
        Called after clients return their local weights.
        Performs weighted FedAvg aggregation.
        """
        print(
            f"\n{'='*50}\n"
            f"[Server] Round {server_round} — "
            f"aggregating {len(results)} client updates "
            f"({len(failures)} failures)\n"
            f"{'='*50}"
        )

        # Delegate actual aggregation to parent FedAvg
        aggregated = super().aggregate_fit(server_round, results, failures)

        if aggregated:
            aggregated_params, metrics = aggregated
            # Log client-side training metrics
            if results:
                train_losses = [r.metrics.get("train_loss", 0) for _, r in results]
                train_accs   = [r.metrics.get("train_accuracy", 0) for _, r in results]
                avg_loss = np.mean(train_losses)
                avg_acc  = np.mean(train_accs)
                print(
                    f"[Server] Round {server_round} client metrics — "
                    f"avg train loss: {avg_loss:.4f}, avg train acc: {avg_acc:.4f}"
                )
                self.round_metrics.append({
                    "round": server_round,
                    "avg_train_loss": avg_loss,
                    "avg_train_accuracy": avg_acc,
                })

        return aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results,
        failures,
    ):
        """
        Called after clients return their local evaluation results.
        Logs weighted average test accuracy.
        """
        if not results:
            return None, {}

        aggregated = super().aggregate_evaluate(server_round, results, failures)

        # Compute weighted average accuracy
        total_examples = sum(r.num_examples for _, r in results)
        weighted_acc = sum(
            r.metrics.get("test_accuracy", 0) * r.num_examples
            for _, r in results
        ) / total_examples

        print(
            f"[Server] Round {server_round} — "
            f"federated test accuracy: {weighted_acc:.4f}"
        )

        # Update the round metrics dict with eval results
        if self.round_metrics and self.round_metrics[-1]["round"] == server_round:
            self.round_metrics[-1]["federated_test_accuracy"] = weighted_acc

        return aggregated


# ---------------------------------------------------------------------------
# Server-side centralised evaluation function
# ---------------------------------------------------------------------------

def build_server_eval_fn(test_dataset, device: torch.device):
    """
    Returns a Flower-compatible evaluation function that runs on the server.

    The server uses the GLOBAL model weights (post-aggregation) to evaluate
    on a held-out central test set — giving us a ground truth accuracy curve.

    Parameters
    ----------
    test_dataset : torchvision MNIST test Dataset.
    device       : torch device for inference.

    Returns
    -------
    evaluate_fn  : function(server_round, parameters, config) → (loss, metrics)
    """
    model = get_model()
    criterion = nn.CrossEntropyLoss()
    test_loader = make_dataloader(test_dataset, batch_size=256, shuffle=False)

    def evaluate_fn(
        server_round: int,
        parameters: flwr.common.NDArrays,
        config: Dict[str, Scalar],
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # Load aggregated parameters into the server-side model
        for param, arr in zip(model.parameters(), parameters):
            param.data = torch.tensor(arr)

        loss, acc = evaluate(model, test_loader, criterion, device)
        print(
            f"[Server eval] Round {server_round} — "
            f"central test loss: {loss:.4f}, accuracy: {acc:.4f}"
        )
        return float(loss), {"central_test_accuracy": float(acc)}

    return evaluate_fn


# ---------------------------------------------------------------------------
# Metrics aggregation helpers (required by Flower)
# ---------------------------------------------------------------------------

def weighted_average_accuracy(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Compute weighted average of 'test_accuracy' from all clients."""
    total = sum(n for n, _ in metrics)
    weighted_acc = sum(n * m.get("test_accuracy", 0) for n, m in metrics) / total
    return {"test_accuracy": weighted_acc}


# ---------------------------------------------------------------------------
# Strategy factory
# ---------------------------------------------------------------------------

def build_strategy(
    num_clients: int,
    fraction_fit: float,
    fraction_evaluate: float,
    min_fit_clients: int,
    min_evaluate_clients: int,
    min_available_clients: int,
    test_dataset=None,
    device: torch.device = torch.device("cpu"),
) -> FedAvgWithLogging:
    """
    Build and return the FedAvgWithLogging strategy.

    Parameters
    ----------
    num_clients            : total number of FL clients.
    fraction_fit           : fraction selected for training each round (0–1).
    fraction_evaluate      : fraction selected for evaluation each round.
    min_fit_clients        : minimum clients required to proceed with a round.
    min_evaluate_clients   : minimum clients required for eval.
    min_available_clients  : minimum clients that must be connected.
    test_dataset           : if provided, enables server-side central eval.
    device                 : for server-side model inference.

    Returns
    -------
    Configured FedAvgWithLogging strategy instance.
    """
    eval_fn = None
    if test_dataset is not None:
        eval_fn = build_server_eval_fn(test_dataset, device)

    strategy = FedAvgWithLogging(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        evaluate_fn=eval_fn,
        evaluate_metrics_aggregation_fn=weighted_average_accuracy,
    )

    print(
        f"[Server] FedAvg strategy created:\n"
        f"  clients={num_clients}, fraction_fit={fraction_fit}, "
        f"fraction_eval={fraction_evaluate}"
    )
    return strategy


# ---------------------------------------------------------------------------
# Server entry-point (subprocess mode)
# ---------------------------------------------------------------------------

def start_server(
    server_address: str,
    num_rounds: int,
    strategy: FedAvgWithLogging,
) -> None:
    """
    Launch the Flower gRPC server (for multi-process deployment).

    Parameters
    ----------
    server_address : e.g. "0.0.0.0:8080"
    num_rounds     : number of FL communication rounds.
    strategy       : configured FedAvg strategy.
    """
    flwr.server.start_server(
        server_address=server_address,
        config=flwr.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
