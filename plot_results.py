"""
Plots the ablation results from a completed train_vicreg.py run.
The values below are the validation embedding std logged at each epoch
(copy-pasted from the training run output on Google Colab).
"""

import matplotlib.pyplot as plt

val_std = {
    "vicreg_full": [
        0.619, 0.661, 0.708, 0.709, 0.748, 0.760, 0.776, 0.796, 0.787, 0.782,
        0.794, 0.841, 0.829, 0.842, 0.838, 0.846, 0.837, 0.849, 0.853, 0.872,
    ],
    "no_variance": [
        0.012, 0.007, 0.005, 0.004, 0.003, 0.003, 0.003, 0.003, 0.002, 0.002,
        0.002, 0.002, 0.003, 0.003, 0.002, 0.002, 0.002, 0.002, 0.005, 0.002,
    ],
    "no_covariance": [
        1.093, 1.036, 1.109, 0.981, 0.931, 0.951, 1.051, 0.999, 1.039, 0.977,
        1.008, 0.974, 1.017, 0.946, 1.023, 1.037, 0.935, 0.954, 0.886, 1.059,
    ],
}

colors = {"vicreg_full": "tab:blue", "no_variance": "tab:red", "no_covariance": "tab:green"}
labels = {
    "vicreg_full": "VICReg (Inv + Var + Cov)",
    "no_variance": "No variance term (Inv + Cov)",
    "no_covariance": "No covariance term (Inv + Var)",
}

plt.figure(figsize=(7, 4.5))
for name, values in val_std.items():
    plt.plot(range(len(values)), values, label=labels[name], color=colors[name], marker="o", markersize=3)

plt.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="target std (γ=1)")
plt.xlabel("Epoch")
plt.ylabel("Average embedding std (validation)")
plt.title("Effect of variance/covariance regularization on collapse")
plt.legend()
plt.tight_layout()
plt.savefig("ablation_std.png", dpi=150)
print("Saved ablation_std.png")
