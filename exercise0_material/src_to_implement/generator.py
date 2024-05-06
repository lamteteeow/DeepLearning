import json
import numpy as np
import matplotlib.pyplot as plt
import os.path
from skimage.transform import resize


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
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.epoch = 0
        self.num_images = 0
        self.image_index = 0

        with open(label_path) as jdata:
            self.labels = json.load(jdata)

        self.image_names = sorted(os.listdir(file_path))
        self.num_images = len(self.image_names)

        if self.shuffle:
            np.random.shuffle(self.image_names)

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

        # print(np.array(self.all_images).shape)

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        images = []
        lbs = []
        lost_images = 0

        for _ in range(self.batch_size):
            # this methods only works with batch_size < num_images
            # current_epoch = i // self.num_images
            # image_index = i % self.num_images

            # if current_epoch > self.epoch:
            #     self.epoch = current_epoch
            #     if self.shuffle:
            #         np.random.shuffle(self.image_names)

            if self.image_index >= self.num_images:
                self.image_index = 0
                self.epoch += 1
                if self.shuffle:
                    np.random.shuffle(self.image_names)

            # Get image filename and path
            image_path = os.path.join(
                self.file_path, self.image_names[self.image_index]
            )

            # Load image data
            try:
                taken_image = np.load(image_path)
            except FileNotFoundError:
                lost_images += 1
                print(
                    f"Error: File not found at {image_path}.\n{lost_images} images were not found."
                )
                continue

            # Check image size and resize if necessary
            # res = list(np.array(taken_image).shape)
            # if res != self.image_size:
            #   taken_image = resize(taken_image, tuple(np.array(self.image_size)[0:2]))
            taken_image = resize(
                taken_image,
                self.image_size,
                preserve_range=True,
                anti_aliasing=True,
            )

            taken_image = self.augment(taken_image)

            # Add taken image to images batch
            images.append(taken_image)
            lbs.append(
                self.labels[os.path.splitext(self.image_names[self.image_index])[0]]
            )

            self.image_index += 1

        return np.array(images), np.array(lbs)

    def augment(self, img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        if self.mirroring:
            if np.random.rand() >= 0.5:
                img = np.fliplr(img)
            else:
                img = np.flipud(img)
        if self.rotation:
            # images have to be rotated if the flag is set
            img = np.rot90(img, np.random.randint(0, 3))
        return img

    def current_epoch(self):
        # return the current epoch number
        return self.epoch

    def class_name(self, x):
        # This function returns the class name for a specific input
        return self.class_dict[x]

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        (batch_images, batch_lbs) = self.next()
        fig = plt.figure(figsize=(50, 50))  # width, height in inches
        columns = 10
        rows = 10

        # ax enables access to manipulate each of subplots
        ax = []

        for i in range(self.batch_size):
            img = batch_images[i]
            # create subplot and append to ax
            ax.append(fig.add_subplot(rows, columns, i + 1))
            # set title
            ax[-1].set_title(self.class_name(batch_lbs[i]))
            plt.imshow(img)

        plt.show()
