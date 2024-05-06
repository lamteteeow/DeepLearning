import json
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt


# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    file_path = ""
    label_path = ""
    image_path = ""
    batch_size = 0
    image_size = (0, 0)
    class_dict = {}
    all_images = []
    image_labels = {"a": 1, "b": 2}
    flag = 0
    number_of_image = 0
    rotation = False
    mirroring = False
    shuffle = False

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

        with open(label_path) as jdata:
            jsonlabels = json.load(jdata)
        self.image_labels = jsonlabels

        self.image_path = file_path + "/*.npy"
        images = glob.glob(self.image_path)

        for myFile in images:
            image_load = np.load(myFile)
            self.number_of_image += 1
            self.all_images.append(image_load)

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

        for i in range(self.batch_size):
            j = (self.flag + i) % self.number_of_image
            taken_image = self.all_images[j]
            # if j == 98:
            #     print(123123)
            # check image size and resize if necessary
            resolution = list(np.array(taken_image).shape)
            if resolution != self.image_size:
                taken_image = cv2.resize(
                    taken_image, tuple(np.array(self.image_size)[0:2])
                )

            # add taken image to images batch
            images.append(taken_image)
            lbs.append(self.image_labels[str(j)])

        # print(np.array(images).shape)
        # return images, labels
        return images, lbs

    def augment(self, img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        # TODO: implement augmentation function

        lbs = []

        # shuffling
        if self.shuffle:
            # s = np.arange(np.array(images).shape[0])
            # np.random.shuffle(s)
            # images = images[s[0]]
            # lbs = lbs[s[0]]
            # print(lbs)
            temp_img = []
            temp_lbs = []
            indices = np.arange(self.batch_size)
            np.random.shuffle(indices)
            # print(indices)

            for i in range(self.batch_size):
                temp_img.append(img[indices[i]])
                temp_lbs.append(lbs[indices[i]])

            img = temp_img
            lbs = temp_lbs

        # rotation
        if self.rotation:
            rdstart = self.flag
            rdend = self.flag + self.batch_size
            for i in np.random.randint(rdstart, rdend, self.batch_size):
                j = i % self.number_of_image
                d = np.random.randint(1, 4, 1)
                rotated_image = np.rot90(img[j], d[0])
                img[j] = rotated_image

        # mirroring
        if self.mirroring:
            rdstart = self.flag
            rdend = self.flag + self.batch_size
            for i in np.random.randint(rdstart, rdend, self.batch_size):
                j = i % self.number_of_image
                flipped_image = np.fliplr(img[j])
                img[j] = flipped_image

        # change flag
        self.flag = (self.flag + self.batch_size) % self.number_of_image

        img = np.asarray(img)

        return img

    def current_epoch(self):
        # return the current epoch number
        return 0

    def class_name(self, x):
        # This function returns the class name for a specific input
        return self.class_dict[x]

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        (batch_image, batch_lbls) = self.next()
        fig = plt.figure(figsize=(50, 50))  # width, height in inches
        columns = 10
        rows = 10

        # prep (x,y) for extra plotting
        xs = np.linspace(0, 2 * np.pi, 60)  # from 0 to 2pi
        ys = np.abs(np.sin(xs))  # absolute of sine

        # ax enables access to manipulate each of subplots
        ax = []

        for i in range(self.batch_size):
            img = batch_image[i]
            # create subplot and append to ax
            ax.append(fig.add_subplot(rows, columns, i + 1))
            ax[-1].set_title(self.class_name(batch_lbls[i]))  # set title
            plt.imshow(img)

        plt.show()  # finally, render the plot
