from torch.utils.data import Dataset
import torch
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import random
import numbers
import cv2
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class EdgeEnhancer:
    """Enhances edges in grayscale images while preserving cracks/stains"""
    def __init__(self, sigma=1.5, kernel=(3,3), canny_thresh=(50, 250), blend_ratio=0.3):
        self.sigma = sigma            # Controls blur strength before edge detection
        self.canny_thresh = canny_thresh  # (low, high) thresholds for crack detection
        self.blend_ratio = blend_ratio    # Edge-to-original ratio (0-1)
        self.kernel = kernel

    def __call__(self, img):
        """
        Input: PIL Image (mode 'L' - grayscale)
        Output: PIL Image (mode 'L') with enhanced edges
        """
        # Convert PIL to numpy array (already grayscale?!?)
        img_np = np.array(img)
        # gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # 1. Noise reduction with Gaussian blur
        blurred = cv2.GaussianBlur(img_np, self.kernel, self.sigma)
        
        # 2. Crack/stain detection with Canny
        edges = cv2.Canny(blurred, *self.canny_thresh)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        # 3. Combine original with edges (grayscale-safe)
        # Use maximum intensity between original and edges
        enhanced = cv2.addWeighted(img_np, 1 - self.blend_ratio,
                                 edges_rgb, self.blend_ratio, 0)

        return tv.transforms.functional.to_pil_image(enhanced)

class GaussianBlur:
    """
    Apply Gaussian blur as a data augmentation transform.
    
    Args:
        kernel_size (int or tuple): Size of the Gaussian kernel (must be odd).
        sigma (float or tuple): Standard deviation of the Gaussian kernel. 
            If a tuple, randomly sample sigma between [sigma[0], sigma[1]].
    """
    def __init__(self, kernel_size, sigma=(0.1, 2.0)):
        self.kernel_size = self._check_kernel_size(kernel_size)
        self.sigma = self._check_sigma(sigma)

    def _check_kernel_size(self, kernel_size):
        if isinstance(kernel_size, numbers.Number):
            kernel_size = (int(kernel_size), int(kernel_size))
        elif isinstance(kernel_size, (list, tuple)) and len(kernel_size) == 1:
            kernel_size = (kernel_size[0], kernel_size[0])
        
        if len(kernel_size) != 2 or any(k % 2 == 0 or k <= 0 for k in kernel_size):
            raise ValueError("kernel_size must be a tuple of 2 odd positive integers")
        
        return kernel_size

    def _check_sigma(self, sigma):
        if isinstance(sigma, numbers.Number):
            if sigma <= 0:
                raise ValueError("sigma must be positive")
            return (sigma, sigma)
        elif isinstance(sigma, (tuple, list)) and len(sigma) == 2:
            if not 0.0 < sigma[0] <= sigma[1]:
                raise ValueError("sigma values must be positive and ordered")
            return sigma
        else:
            raise ValueError("sigma must be a float or tuple of 2 floats")

    def _get_gaussian_kernel(self, kernel_size, sigma):
        # Create 1D Gaussian distributions
        x = torch.arange(-kernel_size[0]//2 + 1, kernel_size[0]//2 + 1, dtype=torch.float)
        y = torch.arange(-kernel_size[1]//2 + 1, kernel_size[1]//2 + 1, dtype=torch.float)
        
        # Calculate outer product without torch.outer
        gaussian_x = torch.exp(-x.pow(2) / (2 * sigma[0]**2))
        gaussian_y = torch.exp(-y.pow(2) / (2 * sigma[1]**2))
        gaussian_x /= gaussian_x.sum()
        gaussian_y /= gaussian_y.sum()
        
        # Create 2D kernel using broadcasting
        kernel = gaussian_x.view(-1, 1) * gaussian_y.view(1, -1)  # Equivalent to outer product
        return kernel

    def __call__(self, img):
        # Convert PIL Image to tensor [C, H, W]
        img_tensor = tv.transforms.functional.to_tensor(img)
        channels = img_tensor.shape[0]
        
        # Randomly sample sigma if it's a range
        if self.sigma[0] == self.sigma[1]:
            sigma = self.sigma[0]
        else:
            sigma = torch.empty(1).uniform_(self.sigma[0], self.sigma[1]).item()
        
        # Generate Gaussian kernel
        kernel = self._get_gaussian_kernel(self.kernel_size, (sigma, sigma))
        kernel = kernel.expand(channels, 1, *kernel.size())  # Shape: [C, 1, H, W]
        
        # Apply convolution
        padding = [k // 2 for k in self.kernel_size]
        blurred = torch.nn.functional.conv2d(
            img_tensor.unsqueeze(0),  # Add batch dimension
            kernel,
            padding=padding,
            groups=channels
        ).squeeze(0)  # Remove batch dimension
        
        # Convert back to PIL Image
        return tv.transforms.functional.to_pil_image(blurred)

    def __repr__(self):
        return f"{self.__class__.__name__}(kernel_size={self.kernel_size}, sigma={self.sigma})"

class RandomGamma:
    def __init__(self, amount):
        self.amount = amount

    def __call__(self, x):
        return tv.transforms.functional.adjust_gamma(x, random.uniform(max(0, 1 - self.amount), (1 + self.amount)))

class RandomGammaIncrease:
    def __init__(self, amount):
        self.amount = amount

    def __call__(self, x):
        return tv.transforms.functional.adjust_gamma(x, random.uniform((1 + self.amount), 1))

class RandomContrastIncrease:
    def __init__(self, amount):
        self.amount = amount

    def __call__(self, x):
        return tv.transforms.functional.adjust_contrast(x, random.uniform(1, (1 + self.amount)))

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
                    tv.transforms.ToPILImage(),
                    tv.transforms.RandomHorizontalFlip(p=0.5),
                    tv.transforms.RandomVerticalFlip(p=0.5),
                    tv.transforms.RandomRotation(degrees=(0,90)),
                    # tv.transforms.RandomApply([RandomGammaIncrease(-0.5)], p=0.5),
                    tv.transforms.ColorJitter(0.3,0.5,0,0),
                    # tv.transforms.RandomApply([RandomContrastIncrease(0.5)], p=0.3),
                    # tv.transforms.RandomApply([GaussianBlur((5,5), 0.1)], p=0.2),
                    # tv.transforms.RandomApply([EdgeEnhancer()], p=0.2),
                    tv.transforms.ToTensor(),
                    tv.transforms.Normalize(train_mean, train_std),
                ]
            )

        if mode == "val":
            self._transform = tv.transforms.Compose(
                [
                    tv.transforms.ToPILImage(),
                    tv.transforms.ToTensor(),
                    tv.transforms.Normalize(train_mean, train_std),
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
