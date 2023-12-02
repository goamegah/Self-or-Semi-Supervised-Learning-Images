import torch
from torchvision import transforms
from torchvision import datasets
# from torch.utils.data import SubsetRandomSampler
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data import Subset, ConcatDataset
from simclr_augmentation import SimclrViewGenerator


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
                    dataset_name: str,
                    n_views: int):

        if dataset_name == "mnist":

            # pretrain dataset
            dataset = datasets.MNIST(
                root=self.root,
                train=True,
                download=True,
                transform=SimclrViewGenerator(
                    transform=self.get_transforms(size=28, s=1),
                    n_views=n_views
                )
            )

        # Other dataset
        else:
            pass

        return dataset
