"""
Two-view augmentation pipeline for VICReg, adapted for CIFAR-10 (32x32).

The paper's augmentation protocol (crop, color jitter, grayscale, Gaussian
blur, solarization) is designed for 224x224 ImageNet images. Gaussian blur
with a large kernel and heavy solarization make little sense on 32x32
images, so we drop them and keep the augmentations that still matter at
this resolution: crop, flip, color jitter, grayscale.
"""

from torchvision import transforms

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)


def build_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])


class TwoViewsTransform:
    """Applies the augmentation pipeline twice, independently, to the same image."""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image):
        return self.transform(image), self.transform(image)
