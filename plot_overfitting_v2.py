"""Analyse d'overfitting des modèles v2 U-TILISE.

Trace les courbes train vs val loss et calcule le gap pour détecter le surapprentissage.
Usage: python plot_overfitting_v2.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS = {
    "mix_closest_random_clouds": "v2 mix_closest, random_clouds",
    "mix_closest_random_fully_masked": "v2 mix_closest, random_fully_masked",
    "asc_desc_random_clouds": "v2 asc+desc, random_clouds",
    "asc_desc_random_fully_masked": "v2 asc+desc, random_fully_masked",
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
axes = axes.flatten()

for ax, (key, title) in zip(axes, MODELS.items()):
    df = pd.read_csv(f"/tmp/losses_v2_{key}.csv")
    # Ignore epoch 0 (pre-training validation on random init)
    df = df[df["epoch"] >= 1].reset_index(drop=True)

    ax.plot(df["epoch"], df["train_loss"], label="Train loss", linewidth=1.2)
    ax.plot(df["epoch"], df["val_loss"], label="Val loss", linewidth=1.2)

    # Gap = val - train (positif = overfitting)
    gap = df["val_loss"].values - df["train_loss"].values
    gap_pct = gap / df["train_loss"].values * 100

    # Highlight gap
    ax.fill_between(
        df["epoch"],
        df["train_loss"],
        df["val_loss"],
        alpha=0.15,
        color="red" if gap[-1] > 0 else "green",
    )

    # Annotations
    best_epoch = df.loc[df["val_loss"].idxmin(), "epoch"]
    best_val = df["val_loss"].min()
    ax.axvline(best_epoch, color="gray", linestyle="--", alpha=0.5, label=f"Best epoch: {best_epoch}")

    final_train = df["train_loss"].iloc[-1]
    final_val = df["val_loss"].iloc[-1]
    final_gap_pct = (final_val - final_train) / final_train * 100

    ax.set_title(f"{title}\nFinal gap: {final_gap_pct:+.1f}% | Best val: {best_val:.5f} (ep {best_epoch})")
    ax.set_ylabel("L1 Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

axes[2].set_xlabel("Epoch")
axes[3].set_xlabel("Epoch")

fig.suptitle("Overfitting Analysis — V2 Models", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig("overfitting_v2.png", dpi=150)
print("Plot saved to: overfitting_v2.png")

# Summary table
print("\n" + "=" * 80)
print(f"{'Model':<45} {'Train':>8} {'Val':>8} {'Gap%':>8} {'Best ep':>8} {'Best val':>10}")
print("=" * 80)
for key, title in MODELS.items():
    df = pd.read_csv(f"/tmp/losses_v2_{key}.csv")
    df = df[df["epoch"] >= 1]
    final_train = df["train_loss"].iloc[-1]
    final_val = df["val_loss"].iloc[-1]
    gap_pct = (final_val - final_train) / final_train * 100
    best_epoch = df.loc[df["val_loss"].idxmin(), "epoch"]
    best_val = df["val_loss"].min()
    print(f"{title:<45} {final_train:.5f}  {final_val:.5f}  {gap_pct:+6.1f}%  {best_epoch:>6}  {best_val:.5f}")
print("=" * 80)
