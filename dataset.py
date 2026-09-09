"""
CIFAR-10 data loading for VICReg.

VICReg pretraining needs two augmented views per image (no labels used).
Labels are only needed for the optional downstream linear-probe evaluation.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from augmentations import build_transform, TwoViewsTransform, CIFAR_MEAN, CIFAR_STD


def get_pretrain_loaders(root="./data", batch_size=256, val_fraction=0.1, num_workers=0, seed=42):
    """
    Train/val DataLoaders yielding pairs of augmented views (view_a, view_b).
    Held out val split (from CIFAR-10's train set) monitors the VICReg pretext
    loss, independent from CIFAR-10's test set (reserved for the linear probe).
    """
    two_view_transform = TwoViewsTransform(build_transform())
    full_train = datasets.CIFAR10(root=root, train=True, download=True, transform=two_view_transform)

    n_val   = int(len(full_train) * val_fraction)
    n_train = len(full_train) - n_val
    train_ds, val_ds = random_split(full_train, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=num_workers)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              drop_last=False, num_workers=num_workers)
    return train_loader, val_loader


def get_probe_loaders(root="./data", batch_size=128, num_workers=0):
    """Train/test loaders WITH labels, used only for the linear-probe evaluation."""
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    train_ds = datasets.CIFAR10(root=root, train=True,  download=True, transform=eval_transform)
    test_ds  = datasets.CIFAR10(root=root, train=False, download=True, transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
