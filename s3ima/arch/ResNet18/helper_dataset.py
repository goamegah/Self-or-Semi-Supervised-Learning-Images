import torch
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from torchvision import transforms
from torchvision import datasets
# from typing import List


def get_dataloaders_mnist(args,
                          num_workers=0,
                          validation_fraction=None,
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

    valid_dataset = datasets.MNIST(root='data',
                                   train=True,
                                   transform=test_transforms)

    test_dataset = datasets.MNIST(root='data',
                                  train=False,
                                  transform=test_transforms)

    if validation_fraction is not None:
        num = int(validation_fraction * 100)
        train_indices = torch.arange(0, 100 - num)
        valid_indices = torch.arange(100 - num, 100)

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)

        valid_loader = DataLoader(dataset=valid_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=num_workers,
                                  sampler=valid_sampler)

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  sampler=train_sampler)

    else:
        # We dispose of 100 samples
        train_indices = torch.arange(0, 100)
        train_sampler = SubsetRandomSampler(train_indices)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=num_workers,
                                  sampler=train_sampler,
                                  drop_last=True,
                                  shuffle=True)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.eval_batch_size,
                             num_workers=num_workers,
                             shuffle=False)

    if validation_fraction is None:
        return train_loader, test_loader
    else:
        return train_loader, valid_loader, test_loader


"""
def get_dataloaders(args,
                    name: str = "MNIST",
                    num_workers=0,
                    train_valid_test_size: List[int] = None,
                    train_transforms=None,
                    test_transforms=None):

    dataset = datasets.MNIST if name == "MNIST" else datasets.CIFAR10

    if train_transforms is None:
        train_transforms = transforms.ToTensor()

    if test_transforms is None:
        test_transforms = transforms.ToTensor()

    train_dataset = dataset(root='data',
                            train=True,
                            transform=train_transforms,
                            download=True)

    test_dataset = dataset(root='data',
                           train=False,
                           transform=test_transforms)

    # 100
    num_examples = 100  # we only have 100 labelled samples

    assert train_valid_test_size[0] + train_valid_test_size[1] + train_valid_test_size[2] == num_examples, \
        "We only dispose of 100 samples, please ajust your chunk size"

    train_indices = torch.arange(0, train_valid_test_size[0])
    valid_indices = torch.arange(train_valid_test_size[0],
                                 train_valid_test_size[0] + train_valid_test_size[1])

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              num_workers=num_workers,
                              drop_last=True,
                              sampler=train_sampler)

    valid_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              num_workers=num_workers,
                              sampler=valid_sampler)

    if args.mode == 'train':
        # just for confusion matrix
        test_indices = torch.arange(100, 200)
        test_sampler = SubsetRandomSampler(test_indices)

        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=args.batch_size,
                                 num_workers=num_workers,
                                 sampler=test_sampler)

        return train_loader, valid_loader, test_loader

    # eval
    else:
        test_indices = torch.arange(train_valid_test_size[0] + train_valid_test_size[1],
                                    train_valid_test_size[0] + train_valid_test_size[1] + train_valid_test_size[2])
        test_sampler = SubsetRandomSampler(test_indices)

        test_loader = DataLoader(dataset=train_dataset,
                                 batch_size=args.eval_batch_size,
                                 num_workers=num_workers,
                                 sampler=test_sampler)
        return test_loader
"""


