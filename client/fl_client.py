"""
fl_client.py
============
Flower (flwr) 2.x client implementation for federated MNIST training.

Each instance of `MNISTFlowerClient` represents ONE participant in the
federated learning round.  It:

  1. Receives the global model weights from the server  (get_parameters / fit)
  2. Trains locally on its private data partition       (fit)
  3. Sends updated weights back to the server           (fit return)
  4. Evaluates the global model on its local test data  (evaluate)

The client never shares raw data — only model parameters.

Flower 2.x requires client_fn to have the signature:
    def client_fn(context: Context) -> Client
"""

from typing import Dict, List, Tuple

import flwr as flwr
from flwr.common import Context
from torch.utils.data import Dataset
import numpy as np
import torch
import torch.nn as nn

from model.mnist_model import MNISTNet, get_model
from utils.train_utils import evaluate, local_train
from utils.data_utils import make_dataloader


# ---------------------------------------------------------------------------
# Flower client
# ---------------------------------------------------------------------------

class MNISTFlowerClient(flwr.client.NumPyClient):
    """
    Federated Learning client wrapping a local MNISTNet.

    Parameters
    ----------
    client_id    : integer ID (used for logging).
    train_data   : local training Subset (private to this client).
    test_data    : shared test Dataset (for local evaluation).
    config       : dict with keys:
                     batch_size       (int, default 32)
                     local_epochs     (int, default 1)
                     learning_rate    (float, default 0.01)
                     optimizer        (str, 'sgd' or 'adam')
    """

    def __init__(
        self,
        client_id: int,
        train_data: Dataset,
        test_data: Dataset,
        config: Dict,
    ) -> None:
        self.client_id = client_id
        self.train_data = train_data
        self.test_data = test_data
        self.config = config

        # Determine computation device (GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialise the local model (weights will be replaced by server's global model)
        self.model: MNISTNet = get_model()
        self.criterion = nn.CrossEntropyLoss()

        print(f"[Client {self.client_id}] Initialised on device={self.device}")

    # ------------------------------------------------------------------
    # Flower required methods
    # ------------------------------------------------------------------

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """
        Return current local model parameters as a list of NumPy arrays.
        Called by Flower before and after training.
        """
        # Convert each parameter tensor → numpy array
        return [param.cpu().detach().numpy() for param in self.model.parameters()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Overwrite local model parameters with those received from the server.
        This is how the aggregated global model is distributed each round.
        """
        for local_param, server_param in zip(
            self.model.parameters(), parameters
        ):
            local_param.data = torch.tensor(server_param)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Receive global weights → train locally → return updated weights.

        This is the core FL loop:
          server → client (parameters)
          client trains locally
          client → server (updated parameters)

        Parameters
        ----------
        parameters : global model weights from server (post-aggregation).
        config     : round-level config sent from server strategy.

        Returns
        -------
        (updated_parameters, num_local_examples, metrics_dict)
        """
        # 1. Load server's global weights into our local model
        self.set_parameters(parameters)

        # 2. Build local DataLoader
        batch_size = self.config.get("batch_size", 32)
        train_loader = make_dataloader(self.train_data, batch_size=batch_size, shuffle=True)

        # 3. Train locally for `local_epochs` epochs
        local_epochs = self.config.get("local_epochs", 1)
        lr = self.config.get("learning_rate", 0.01)
        opt = self.config.get("optimizer", "sgd")

        # DP hyperparameters (optional)
        dp_enabled = self.config.get("dp_enabled", False)
        clip_norm = self.config.get("clip_norm", 1.0)
        noise_multiplier = self.config.get("noise_multiplier", 0.0)

        loss, acc, n_examples = local_train(
            model=self.model,
            dataloader=train_loader,
            num_epochs=local_epochs,
            learning_rate=lr,
            device=self.device,
            optimizer_name=opt,
            dp_enabled=dp_enabled,
            clip_norm=clip_norm,
            noise_multiplier=noise_multiplier,
        )

        print(
            f"[Client {self.client_id}] Round fit done — "
            f"loss={loss:.4f}, acc={acc:.4f}, n={n_examples}"
        )

        # 4. Return updated weights + metadata
        return (
            self.get_parameters(config={}),
            n_examples,
            {"train_loss": loss, "train_accuracy": acc},
        )

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[float, int, Dict]:
        """
        Evaluate the (received) global model on this client's local test data.

        Note: in practice clients may not have access to a global test set.
        Here we use the shared MNIST test set as a proxy for model quality.

        Returns
        -------
        (loss, num_test_examples, metrics_dict)
        """
        # Load global weights without overwriting training progress
        self.set_parameters(parameters)

        batch_size = self.config.get("batch_size", 32)
        test_loader = make_dataloader(self.test_data, batch_size=batch_size, shuffle=False)

        loss, acc = evaluate(
            model=self.model,
            dataloader=test_loader,
            criterion=self.criterion,
            device=self.device,
        )

        print(
            f"[Client {self.client_id}] Eval — loss={loss:.4f}, acc={acc:.4f}"
        )
        return (
            float(loss),
            len(self.test_data),
            {"test_loss": loss, "test_accuracy": acc},
        )


# ---------------------------------------------------------------------------
# Client factory — Flower 2.x requires this exact signature
# ---------------------------------------------------------------------------

def make_client_fn(client_datasets, test_dataset, client_config: Dict):
    """
    Returns a Flower 2.x compatible client_fn closure.

    Flower 2.x requires:  def client_fn(context: Context) -> Client
    The context carries the client's node_id which we use as the client ID.

    Parameters
    ----------
    client_datasets : list of per-client training Subsets.
    test_dataset    : shared evaluation dataset.
    client_config   : hyperparameter dict.
    """
    def client_fn(context: Context) -> flwr.client.Client:
        # In Flower 2.x, context.node_id is a unique integer per virtual client
        client_id = int(context.node_id) % len(client_datasets)
        client = MNISTFlowerClient(
            client_id=client_id,
            train_data=client_datasets[client_id],
            test_data=test_dataset,
            config=client_config,
        )
        return client.to_client()   # wrap NumPyClient → Client

    return client_fn
