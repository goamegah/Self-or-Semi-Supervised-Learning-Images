import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import numpy as np


def get_dataloaders_mnist(batch_size,
                          eval_batch_size,
                          num_workers=0,
                          train_size=None,
                          train_transforms=None,
                          test_transforms=None):
    if train_transforms is None:
        train_transforms = transforms.ToTensor()

    if test_transforms is None:
        test_transforms = transforms.ToTensor()

    train_dataset = datasets.MNIST(root='data',
                                   train=True,
                                   transform=train_transforms,
                                   download=True)

    test_dataset = datasets.MNIST(root='data',
                                  train=False,
                                  transform=test_transforms)

    if train_size is not None:

        # sample 100 data with same classes proportion
        train_idx, validation_idx = train_test_split(np.arange(len(train_dataset)),
                                                     train_size=train_size,
                                                     random_state=1,
                                                     shuffle=True,
                                                     stratify=train_dataset.targets.numpy())

        # Subset dataset for train and val
        train_dataset = Subset(train_dataset, train_idx)
        valid_dataset = Subset(train_dataset, validation_idx)

        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  drop_last=False,
                                  shuffle=True)

        valid_loader = DataLoader(dataset=valid_dataset,
                                  batch_size=eval_batch_size,
                                  num_workers=8,
                                  drop_last=False,
                                  shuffle=True)

    else:
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  shuffle=True)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=False)

    if train_size is None:
        return train_loader, test_loader
    else:
        return train_loader, valid_loader, test_loader
