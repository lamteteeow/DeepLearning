import json
import os.path
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc


# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(
        self,
        file_path,
        label_path,
        batch_size,
        image_size,
        rotation=False,
        mirroring=False,
        shuffle=False,
    ):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.class_dict = {
            0: "airplane",
            1: "automobile",
            2: "bird",
            3: "cat",
            4: "deer",
            5: "dog",
            6: "frog",
            7: "horse",
            8: "ship",
            9: "truck",
        }
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.images = os.listdir(file_path)
        self.dataset_size = len(self.images)
        self.image_names = [name[:-4] for name in self.images]
        with open(self.label_path, "rb") as label_json:
            self.labels = json.load(label_json)
        if self.shuffle:
            # import random
            random.shuffle(self.image_names)
        self.epoch_no = 0
        self.batch_no = 0

        # TODO: implement constructor

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        # TODO: implement next method
        start_idx = self.batch_no * self.batch_size
        end_idx = start_idx + self.batch_size
        if end_idx > self.dataset_size:
            if self.shuffle:
                random.shuffle(self.image_names)
            self.batch_no = 0
            self.epoch_no += 1
            end_idx = self.dataset_size
        batch_img_names = self._process_batch_image_names(start_idx, end_idx)
        images, labels = self._load_images_and_labels(batch_img_names)
        self.batch_no += 1
        return images, labels

    def _process_batch_image_names(self, start_idx, end_idx):
        img_names = self.image_names[start_idx:end_idx]
        if end_idx == self.dataset_size and end_idx - start_idx != self.batch_size:
            reminder = self.batch_size - (end_idx - start_idx)
            img_names += self.image_names[0:reminder]
        return img_names

    def _load_images_and_labels(self, img_names):
        imgs, labels = [], []
        for name in img_names:
            img = np.load(f"{self.file_path}/{name}.npy")
            img = self.augment(img)
            # img = cv2.resize(img, self.image_size[ : 2])
            img = cv2.resize(img, tuple(self.image_size[:-1]))
            label = self.labels.get(name, None)
            if label is not None:
                imgs.append(img)
                labels.append(label)
        return np.array(imgs), np.array(labels)

    def augment(self, img):
        if self.rotation:
            angle = random.choice([90, 180, 270])
            img = np.rot90(img, angle // 90, axes=(0, 1))
        if self.mirroring:
            img = np.flip(img, axis=[0, 1])
        return img

    def current_epoch(self):
        # return the current epoch number
        return self.epoch_no

    def class_name(self, x):
        # This function returns the class name for a specific input
        # TODO: implement class name function
        return self.class_dict[x]

    def show(self):
        images, labels = self.next()
        batch_sqrt = int(np.ceil(np.sqrt(self.batch_size)))
        counter = 0
        for image, label in zip(images, labels):
            counter += 1
            plt.subplot(batch_sqrt, batch_sqrt, counter)
            plt.title(self.class_name(label))
            plt.imshow(image)
        plt.show()


if __name__ == "__main__":
    gen = ImageGenerator(
        "./exercise_data/",
        "./Labels.json",
        60,
        [32, 32, 3],
        rotation=False,
        mirroring=False,
        shuffle=False,
    )
    gen.show()
