import torch
from torch.utils.data import DataLoader

from torchvision.transforms import v2
from torchvision.datasets import Food101

from pathlib import Path

import os

from config import get_config

# The paper mentioned some specific training steps to follow to avoid overfitting
# These include Mixup, Cutmix, RandAugment, Random Erasing
# The dataset is Food 101 (https://www.kaggle.com/datasets/dansbecker/food-101) it is also available in PyTorch

train_preproc = v2.Compose([
    v2.RandomResizedCrop(224, interpolation=v2.InterpolationMode.BICUBIC),
    v2.RandomHorizontalFlip(),
    v2.RandAugment(),    # paper mentioned magnitude of 9, default values checks out the same
    v2.ToImage(),
    v2.ToDtype(dtype=torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),    # ImageNet standard
    v2.RandomErasing(),
])

cutmix_or_mixup = v2.RandomChoice([v2.MixUp(alpha=0.8, num_classes=101), v2.CutMix(alpha=1.0, num_classes=101)])

test_preproc = v2.Compose([
    v2.Resize(256),
    v2.CenterCrop(224),     # Resizing then centercrop to preserve aspect ratio
    v2.ToImage(),
    v2.ToDtype(dtype=torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# function to load dataset
def load_dataloader(config):
    Path(config["dataset_folder"]).mkdir(exist_ok=True)
    train_data = Food101(root=config["dataset_folder"], split="train", transform=train_preproc, download=True)
    test_data = Food101(root=config["dataset_folder"], split="test", transform=test_preproc, download=True)

    workers = min(os.cpu_count(), 8) if os.cpu_count() else 2

    train_dataloader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True, num_workers=workers, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=int(config["batch_size"]*1.5), num_workers=workers, pin_memory=True)
    return train_dataloader, test_dataloader, cutmix_or_mixup

if __name__ == "__main__":
    config = get_config()
    train, test, cutmix_or_mixup = load_dataloader(config=config)
    for i in train:
        print(i[0])
        print(i[1])
        break