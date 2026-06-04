## Installation

- Use python 3.10 enviroenment

- Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Running the Simulation

### Default run (10 clients, 10 rounds, IID)
```bash
python main_simulation.py
```

### Custom configuration
```bash
# Non-IID partition, 20 clients, 15 rounds, Adam optimiser
python main_simulation.py \
    --num_clients 20 \
    --num_rounds 15 \
    --partition non_iid \
    --optimizer adam \
    --learning_rate 0.001 \
    --local_epochs 3
```

### All available flags
|       Flag       |   Default   |                    Description                   |
|------------------|-------------|--------------------------------------------------|
| `--num_clients`  | 10          | Total number of FL clients                       |
| `--num_rounds`   | 10          | Number of FL communication rounds                |
| `--partition`    | `iid`       | Data split: `iid` or `non_iid`                   |
| `--local_epochs` | 2           | Epochs each client trains before sending weights |
| `--learning_rate`| 0.01        | Optimiser learning rate                          |
| `--optimizer`    | `sgd`       | `sgd` or `adam`                                  |
| `--batch_size`   | 32          | Local mini-batch size                            |
| `--fraction_fit` | 0.5         | Fraction of clients selected per round           |
| `--data_dir`     | `./data`    | MNIST download path                              |
| `--results_dir`  | `./results` | Output directory                                 |

---

## Architecture: MNISTNet

```
Input: (B, 1, 28, 28)
│
├─ Conv2d(1→32, k=5, pad=2)  → ReLU → MaxPool(2) → (B, 32, 14, 14)
├─ Conv2d(32→64, k=5, pad=2) → ReLU → MaxPool(2) → (B, 64, 7, 7)
│
├─ Flatten                                          → (B, 3136)
├─ Linear(3136→128)          → ReLU → Dropout(0.5) → (B, 128)
└─ Linear(128→10)                                   → (B, 10) logits
```

---

## FedAvg Algorithm

```
Initialise global model w₀

For round t = 1, ..., T:
    S_t ← random subset of clients (fraction_fit × N)
    
    For each client k ∈ S_t (in parallel):
        w_k ← local_train(w_t, data_k, local_epochs, lr)
        Send w_k back to server
    
    w_{t+1} ← Σ_{k ∈ S_t} (n_k / N_t) × w_k    ← weighted average
```

Where `n_k` is client k's local dataset size and `N_t = Σ n_k`.

---

## Data Partitioning

### IID
Each client receives a random, balanced subset of all 10 digit classes.
Represents ideal conditions where data is uniformly distributed.

### Non-IID (Label Skew)
Each client receives data from only `num_classes_per_client` digit classes.
Simulates real-world heterogeneity (e.g., different users write different digits).
This typically reduces convergence speed and final accuracy.

---

## Outputs

After training, `results/` will contain:
- `metrics.json` — per-round train loss, train accuracy, federated test accuracy
- `accuracy_curve.png` — test accuracy over rounds
- `loss_curve.png` — average client train loss over rounds

---

## Extending to Steps 2 & 3

### Step 2 — Gradient Leakage Attack
`utils/train_utils.py` exposes `compute_gradients(model, images, labels, device)`, which returns the raw gradient tensors for a single batch — exactly what a gradient leakage attack intercepts. Import and use this in your attack script.

### Step 3 — Differential Privacy Defense
Wrap the local optimizer with `opacus.PrivacyEngine` (or `tensorflow_privacy`) before calling `local_train()`. The `fl_client.py::fit()` method is the natural injection point.

---

## References

- McMahan et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data.* (FedAvg) — https://arxiv.org/abs/1602.05629
- Flower Framework — https://flower.ai/
- LeCun et al. (1998). *Gradient-based learning applied to document recognition.* (MNIST)
