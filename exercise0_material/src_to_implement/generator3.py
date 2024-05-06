import os.path
import json
import numpy as np
import matplotlib.pyplot as plt
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
        with open(label_path, "r") as f:
            self.labels = json.load(f)

        self.image_paths = sorted(os.listdir(file_path))
        self.num_images = len(self.image_paths)
        self.index = 0
        self.epoch = 0

        if self.shuffle:
            np.random.shuffle(self.image_paths)
        # TODO: implement constructor
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

    def next(self):
        images = []
        labels = []

        for _ in range(self.batch_size):
            # Check if we need to start a new epoch
            if self.index >= self.num_images:
                self.index = 0
                self.epoch += 1
                if self.shuffle:
                    np.random.shuffle(self.image_paths)

            # Get image filename and path
            image_filename = self.image_paths[self.index]
            image_path = os.path.join(self.file_path, image_filename)

            # Load image data
            try:
                image = np.load(image_path)
            except FileNotFoundError:
                print(f"Error: File not found at {image_path}")
                continue

            # Resize image
            image = resize(
                image, self.image_size, preserve_range=True, anti_aliasing=True
            )

            # Apply data augmentation
            if self.mirroring and np.random.rand() > 0.5:
                image = np.fliplr(image)

            if self.rotation:
                angle = np.random.choice([0, 90, 180, 270])
                image = np.rot90(image, k=angle // 90)

            # Remove file extension to use as label key
            image_key = os.path.splitext(image_filename)[0]
            labels.append(
                self.labels[image_key]
            )  # Get label based on filename without extension

            # Append image to batch
            images.append(image)

            # Move to the next image
            self.index += 1

        return np.array(images), np.array(labels)

    def augment(self, img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        # TODO: implement augmentation function
        if self.mirroring and np.random.rand() > 0.5:
            img = np.fliplr(img)

        # Randomly choose rotation angle (90°, 180°, 270°)
        if self.rotation:
            angle = np.random.choice([0, 90, 180, 270])
            img = np.rot90(img, k=angle // 90)

        return img

    def current_epoch(self):
        # return the current epoch number
        return self.epoch

    def class_name(self, x):
        # This function returns the class name for a specific input
        # TODO: implement class name function
        class_names = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]

        return class_names[x]

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        # TODO: implement show method
        images, labels = self.next()
        num_images = len(images)

        fig, axes = plt.subplots(1, num_images, figsize=(num_images * 3, 3))
        for i in range(num_images):
            axes[i].imshow(images[i].astype(np.uint8))
            axes[i].axis("off")
            axes[i].set_title(self.class_name(labels[i]))
        plt.show()
