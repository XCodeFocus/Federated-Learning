import subprocess
import json
import os
import shutil
import matplotlib.pyplot as plt
import sys


noise_values = [0.0, 0.001, 0.01, 0.05, 0.1]

results = []

base_results_dir = "./results_dp_experiment"
os.makedirs(base_results_dir, exist_ok=True)


for noise in noise_values:
    print(f"\n===== Running noise={noise} =====")

    run_dir = os.path.join(
        base_results_dir,
        f"noise_{str(noise).replace('.', '_')}"
    )

    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)

    cmd = [
        sys.executable,
        "main_simulation.py",
        "--num_clients", "5",
        "--num_rounds", "3",
        "--partition", "iid",
        "--results_dir", run_dir,
    ]

    if noise > 0:
        cmd += [
            "--dp_enabled",
            "--noise_multiplier", str(noise),
            "--clip_norm", "1.0",
        ]

    subprocess.run(cmd, check=True)

    metrics_path = os.path.join(run_dir, "metrics.json")

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    rounds = metrics["rounds"]
    final_round = rounds[-1]

    acc = final_round["federated_test_accuracy"]
    loss = final_round.get("avg_train_loss", None)

    results.append({
        "noise": noise,
        "accuracy": acc,
        "loss": loss,
    })

    print(f"noise={noise}, accuracy={acc:.4f}")


# Save summary JSON
summary_path = os.path.join(base_results_dir, "dp_accuracy_summary.json")
with open(summary_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSummary saved to {summary_path}")


# Plot
noise_labels = [str(r["noise"]) for r in results]
accuracies = [r["accuracy"] * 100 for r in results]

plt.figure(figsize=(8, 5))

plt.plot(
    noise_labels,
    accuracies,
    marker="o",
    linewidth=2
)

plt.xlabel("Noise Multiplier")
plt.ylabel("Federated Test Accuracy (%)")
plt.title("Effect of DP Noise on Federated Learning Accuracy")

plt.grid(True, alpha=0.3)

for x, y in zip(noise_labels, accuracies):
    plt.text(
        x,
        y + 0.5,
        f"{y:.2f}%",
        ha="center"
    )

plt.tight_layout()

plot_path = os.path.join(base_results_dir, "dp_accuracy_comparison.png")
plt.savefig(plot_path, dpi=300)
plt.show()

print(f"Plot saved to {plot_path}")