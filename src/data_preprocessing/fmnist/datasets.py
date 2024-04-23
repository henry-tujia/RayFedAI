import logging

import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision.datasets import FashionMNIST
import torchvision.transforms as tt
import torch


class FMNIST_truncated(data.Dataset):
    def __init__(
        self,
        root,
        dataidxs=None,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        repeate=False,
    ):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        # print("download = " + str(self.download))
        cifar_dataobj = FashionMNIST(
            self.root, self.train, self.transform, self.target_transform, self.download
        )

        data = cifar_dataobj.data.unsqueeze(1) / 255.0
        target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    ds = FMNIST_truncated(
        root="/vepfs/DI/user/haotan/FL/Multi-FL-Training-main/datasets/fmnist"
    )
    # dataset = FashionMNIST(
    #     root="/mnt/data/th/FedTH/data/emnist",
    #     split="balanced",
    #     download=True,
    #     train=True,
    #     transform=tt.Compose(
    #         [
    #             lambda img: tt.functional.rotate(img, -90),
    #             lambda img: tt.functional.hflip(img),
    #             tt.ToTensor(),
    #         ]
    #     ),
    # )
    # train_ds = EMNIST_truncated(dataidxs=None, train=True, dataset=dataset)
