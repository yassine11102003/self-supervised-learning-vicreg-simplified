"""
VICReg loss: invariance + variance + covariance regularization.
Direct translation of the paper's Algorithm 1 (Appendix A) into named,
reusable functions.
"""

import torch
import torch.nn.functional as F


def invariance_loss(z_a, z_b):
    """Mean squared distance between the two batches of embeddings."""
    return F.mse_loss(z_a, z_b)


def variance_loss(z, gamma=1.0, eps=1e-4):
    """Hinge loss pushing each embedding dimension's std (over the batch) towards gamma."""
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.mean(F.relu(gamma - std))


def _off_diagonal(matrix):
    """Return the off-diagonal elements of a square matrix, flattened."""
    n, m = matrix.shape
    assert n == m
    return matrix.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def covariance_loss(z):
    """Sum of squared off-diagonal covariance coefficients, scaled by 1/dim."""
    n, d = z.shape
    z = z - z.mean(dim=0)
    cov = (z.T @ z) / (n - 1)
    return _off_diagonal(cov).pow(2).sum() / d


def vicreg_loss(z_a, z_b, lam=25.0, mu=25.0, nu=1.0,
                use_var=True, use_cov=True, gamma=1.0, eps=1e-4):
    """
    Combined VICReg loss. use_var / use_cov let us ablate each regularization
    term independently, to demonstrate their effect on collapse.
    """
    loss = lam * invariance_loss(z_a, z_b)

    if use_var:
        loss = loss + mu * (variance_loss(z_a, gamma, eps) + variance_loss(z_b, gamma, eps))

    if use_cov:
        loss = loss + nu * (covariance_loss(z_a) + covariance_loss(z_b))

    return loss
