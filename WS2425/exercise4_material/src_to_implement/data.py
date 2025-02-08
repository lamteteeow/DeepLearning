from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


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
                    # tv.transforms.RandomHorizontalFlip(p=0.2),  # Flips the image horizontally randomly
                    # tv.transforms.RandomRotation(degrees=(0,30)),  # Gets random perspective of the image
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

        ## Finally applyig the augmentations ##
        img_tensor = self._transform(img)

        ## Getting the two labels ##
        labels = torch.from_numpy(np.array([data_sample.crack, data_sample.inactive]))

        ## returning a dictionary of images and labels ##
        return img_tensor, labels
