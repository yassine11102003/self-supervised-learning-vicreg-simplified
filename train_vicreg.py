"""
VICReg training loop, with an ablation mode to demonstrate the effect of the
variance and covariance regularization terms on representation collapse.

Both branches share the same encoder + expander (Siamese, shared weights) and
are trained by plain backprop — no stop-gradient, no momentum encoder needed.
"""

import torch

from config import MODEL_CONFIG, TRAIN_CONFIG, LOSS_CONFIG, ABLATIONS
from model import Encoder, Expander
from vicreg_loss import vicreg_loss
from dataset import get_pretrain_loaders


def embedding_std(z_a, z_b):
    """Average per-dimension std over the batch, for both views (diagnostic only)."""
    return 0.5 * (z_a.std(dim=0).mean() + z_b.std(dim=0).mean())


def run_epoch(loader, encoder, expander, optimizer, loss_cfg, use_var, use_cov, device, train):
    encoder.train(train)
    expander.train(train)

    total_loss, total_std, n = 0.0, 0.0, 0
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for (view_a, view_b), _ in loader:
            view_a, view_b = view_a.to(device), view_b.to(device)

            z_a = expander(encoder(view_a))
            z_b = expander(encoder(view_b))

            loss = vicreg_loss(z_a, z_b, use_var=use_var, use_cov=use_cov, **loss_cfg)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_std  += embedding_std(z_a, z_b).item()
            n += 1

    return total_loss / n, total_std / n


def train(config_name, use_var, use_cov,
          model_cfg=MODEL_CONFIG, train_cfg=TRAIN_CONFIG, loss_cfg=LOSS_CONFIG):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder  = Encoder(channels=model_cfg["encoder_channels"]).to(device)
    expander = Expander(encoder.out_dim, model_cfg["expander_hidden"], model_cfg["expander_out"]).to(device)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(expander.parameters()), lr=train_cfg["lr"])

    train_loader, val_loader = get_pretrain_loaders(batch_size=train_cfg["batch_size"])

    best_val_loss = float("inf")

    for epoch in range(train_cfg["epochs"]):
        train_loss, train_std = run_epoch(train_loader, encoder, expander, optimizer,
                                          loss_cfg, use_var, use_cov, device, train=True)
        val_loss, val_std = run_epoch(val_loader, encoder, expander, optimizer,
                                      loss_cfg, use_var, use_cov, device, train=False)

        print(f"[{config_name}] epoch {epoch:3d} | "
              f"train loss={train_loss:8.3f} std={train_std:.3f} | "
              f"val loss={val_loss:8.3f} std={val_std:.3f}")

        if use_var and use_cov and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(encoder.state_dict(), "encoder.pth")

    return val_std  # final embedding std, for a quick summary across configs


if __name__ == "__main__":
    results = {}
    for name, use_var, use_cov in ABLATIONS:
        final_std = train(name, use_var, use_cov)
        results[name] = final_std

    print("\nFinal validation embedding std per config:")
    for name, std in results.items():
        print(f"  {name:16s}: {std:.4f}")
