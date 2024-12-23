"""
The CIFAR-10 dataset from the torchvision package.
"""
import logging
import os
import sys

from torchvision import datasets, transforms

from plato.config import Config
from plato.datasources import base

# AUGMIX已经被集成到pytorch里面了 不用自行实现了
# import numpy as np
# import torch
# from augmix import augmentations

# def aug(image, preprocess):
#   """Perform AugMix augmentations and compute mixture.

#   Args:
#     image: PIL.Image input image
#     preprocess: Preprocessing function which should return a torch tensor.

#   Returns:
#     mixed: Augmented and mixed image.
#   """
#   aug_list = augmentations.augmentations
#   aug_list = augmentations.augmentations_all

#   ws = np.float32(np.random.dirichlet([1] * 3))
#   m = np.float32(np.random.beta(1, 1))

#   mix = torch.zeros_like(preprocess(image))
#   for i in range(3):
#     image_aug = image.copy()
#     depth =  np.random.randint(1, 4)
#     for _ in range(depth):
#       op = np.random.choice(aug_list)
#       image_aug = op(image_aug, 3)
#     # Preprocessing commutes since all coefficients are convex
#     mix += ws[i] * preprocess(image_aug)

#   mixed = (1 - m) * preprocess(image) + m * mix

#   return mixed

# class AugMixDataset(torch.utils.data.Dataset):
#   """Dataset wrapper to perform AugMix augmentation."""

#   def __init__(self, dataset, preprocess, no_jsd=True):
#     self.dataset = dataset
#     self.preprocess = preprocess
#     self.no_jsd = no_jsd
#     self.targets = dataset.targets
#     self.classes = dataset.classes

#   def __getitem__(self, i):
#     x, y = self.dataset[i]
#     if self.no_jsd:
#       return aug(x, self.preprocess), y
#     else:
#       im_tuple = (self.preprocess(x), aug(x, self.preprocess),
#                   aug(x, self.preprocess))
#       return im_tuple, y

#   def __len__(self):
#     return len(self.dataset)

class DataSource(base.DataSource):
    """The augmented CIFAR-10 dataset."""

    def __init__(self, **kwargs):
        super().__init__()

        # _path = Config().params["aug_data_path"]
        _path = Config().data.aug_data_path

        if not os.path.exists(_path):
            if hasattr(Config().server, "do_test") and not Config().server.do_test:
                # If the server is not performing local tests for accuracy, concurrent
                # downloading on the clients may lead to PyTorch errors
                if Config().clients.total_clients > 1:
                    if not hasattr(Config().data, 'concurrent_download'
                                ) or not Config().data.concurrent_download:
                        raise ValueError(
                            "The dataset has not yet been downloaded from the Internet. "
                            "Please re-run with '-d' or '--download' first. ") 
                        
        train_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.AugMix(),
            transforms.ToTensor(),
            transforms.Normalize(
                            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
        
        test_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(
                            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        
        self.trainset = datasets.CIFAR10(
            root=_path, train=True, download=True, transform=train_transform
        )
        self.testset = datasets.CIFAR10(
            root=_path, train=False, download=True, transform=test_transform, 
                        )
        
        # self.trainset = AugMixDataset(self.trainset, preprocess)
        
        if Config().args.download:
            logging.info("The dataset has been successfully downloaded. "
                        "Re-run the experiment without '-d' or '--download'.")
            sys.exit()

    def num_train_examples(self):
        return 50000

    def num_test_examples(self):
        return 10000
