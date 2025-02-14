from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import random
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class RandomGamma:
    def __init__(self, amount):
        self.amount = amount

    def __call__(self, x):
        return tv.transforms.functional.adjust_gamma(x, random.uniform(max(0, 1 - self.amount), (1 + self.amount)))

class RandomConstrast:
    def __call__(self, x):
        return tv.transforms.functional.adjust_constrast(x, random.uniform(0.7,1.3))

class RandomBrightness:
    def __call__(self, x):
        return tv.transforms.functional.adjust_gamma(x, random.uniform(0.7,1.3))


class ChallengeDataset(Dataset):
    # TODO implement the Dataset class according to the description
    """
    Class to create the Cracks Dataset. It inherits from the torch Dataset class.
    """

    def __init__(self, data, mode):
        """
        Constructor definition.
        """
        super().__init__()

        self.data = data
        self.mode = mode

        ## Defining the augmentations for the images ##
        if mode == "train":
            self._transform = tv.transforms.Compose(
                [
                    tv.transforms.ToPILImage(),  # Converts to PILImage category
                    tv.transforms.RandomVerticalFlip(p=0.5),  # Flips the image horizontally randomly
                    tv.transforms.RandomHorizontalFlip(p=0.5),  # Flips the image horizontally randomly
                    # tv.transforms.RandomRotation(degrees=(0,30)),  # Gets random perspective of the image
                    # tv.transforms.RandomAutocontrast(p=0.5), # not supported in this ver
                    # tv.transforms.RandomAdjustSharpness(), # not supported in this ver
                    tv.transforms.ColorJitter(0.2,0.2,0,0),
                    RandomGamma(0.2),
                    tv.transforms.ToTensor(),  # Transforms to Pytorch Tensor
                    tv.transforms.Normalize(train_mean, train_std),  # Normalizes the images
                ]
            )

        if mode == "val":
            self._transform = tv.transforms.Compose(
                [
                    tv.transforms.ToPILImage(),  # Converts to PILImage category
                    tv.transforms.ToTensor(),  # Transforms to Pytorch Tensor
                    tv.transforms.Normalize(train_mean, train_std),  # Normalizes the images
                ]
            )

    def __len__(self):
        """
        Returns the length of the entire dataset.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Returns the image (feature) and the labels (targets).
        Args: idx - A random index.
        """
        ## Making the idx cyclic so in cases of idx values more than the length it still gets something ##

        idx = index % len(self.data)

        ## Fetching the sample at the index idx ##
        data_sample = self.data.iloc[idx]

        ## Getting the image path which is the 'filename' part of the sample ##
        img_path = data_sample.filename

        ## Reading the image from image path ##
        img = imread(img_path)

        ## Changing the rgb image to rgb ##
        img = gray2rgb(img)

        ## Finally applying the augmentations ##
        img_tensor = self._transform(img)

        ## Getting the two labels ##
        labels = torch.from_numpy(np.array([data_sample.crack, data_sample.inactive]))

        ## returning a dictionary of images and labels ##
        return img_tensor, labels
