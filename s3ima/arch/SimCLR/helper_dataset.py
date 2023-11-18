import torch
from torchvision import transforms
from torchvision import datasets
# from torch.utils.data import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.utils.data import Subset, ConcatDataset
from helper_augmentation import SimclrViewGenerator


class SimclrDataset:

    def __init__(self, root: str = "data"):
        self.root = root

    @staticmethod
    def get_transforms(size, s):
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        transform = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomApply(transforms=[color_jitter], p=0.8),
                                        transforms.RandomGrayscale(p=0.2),
                                        transforms.GaussianBlur(kernel_size=3),
                                        transforms.ToTensor()])

        return transform

    def get_dataset(self,
                    args,
                    dataset_name: str,
                    n_views: int,
                    finetune_validation_fraction=None):

        if dataset_name == "mnist":

            test_dataset = datasets.MNIST(
                root=self.root,
                train=False,
                download=False,
                transform=transforms.ToTensor()
            )

            # mnist_dataset = ConcatDataset([train_dataset, test_dataset])

            if args.mode == "train":
                # pretrain dataset
                mnist_pretrain_dataset = datasets.MNIST(
                    root=self.root,
                    train=True,
                    download=True,
                    transform=SimclrViewGenerator(
                        transform=self.get_transforms(size=28, s=1),
                        n_views=n_views
                    )
                )

                # which train mode

                if args.train_mode == "finetune":
                    if finetune_validation_fraction is not None:
                        num = int(finetune_validation_fraction * 100)
                        train_indices = torch.arange(0, 100 - num)
                        valid_indices = torch.arange(100 - num, 100)

                        # train_sampler = SubsetRandomSampler(train_indices)
                        # valid_sampler = SubsetRandomSampler(valid_indices)

                        mnist_train_dataset = datasets.MNIST(
                            root=self.root,
                            train=True,
                            download=False,
                        )

                        # 100 samples off of train dataset
                        mnist_train_dataset_100 = Subset(dataset=mnist_train_dataset,
                                                         indices=train_indices)

                        mnist_valid_dataset_100 = Subset(dataset=mnist_train_dataset,
                                                         indices=valid_indices)

                        return mnist_train_dataset_100, mnist_valid_dataset_100, test_dataset

                    else:   # not validation dataset
                        return None

                else:   # mode pretrain
                    return mnist_pretrain_dataset, None, None

            else:   # eval mode

                """
                # finetune train dataset (70)
                mnist_finetune_train_dataset = Subset(
                    dataset=test_dataset,
                    indices=torch.arange(end=70)
                )

                # finetune valid dataset (10)
                mnist_finetune_valid_dataset = Subset(
                    dataset=test_dataset,
                    indices=torch.arange(start=70, end=80)
                )

                # finetune test dataset (20)
                mnist_finetune_test_dataset = Subset(
                    dataset=test_dataset,
                    indices=torch.arange(start=80, end=100)
                )
                """
                # return mnist_finetune_train_dataset, mnist_finetune_valid_dataset, mnist_finetune_test_dataset
                return

        else:   # Other dataset
            pass

        return None
