import matplotlib.pyplot as plt
import numpy as np


class Checker:
    def __init__(self, resolution: int, tile_size: int):
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = None
        # pass

    def draw(self):
        # res = np.zeros(shape = (self.resolution, self.resolution))
        row_idx, col_idx = np.indices((self.resolution, self.resolution))
        con = (row_idx // self.tile_size + col_idx // self.tile_size) % 2 == 0
        self.output = np.where(con, 0, 1)
        return self.output.copy()

    def show(self):
        plt.imshow(self.output, cmap="gray")
        plt.show()


class Circle:
    def __init__(self, resolution: int, radius: int, position: tuple):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = None

    def draw(self):
        row_idx, col_idx = np.indices((self.resolution, self.resolution))
        condition = (
            np.sqrt(np.square(row_idx - self.position[1]) + np.square(col_idx - self.position[0]))
            <= self.radius
        )
        self.output = np.where(condition, 1, 0)
        return self.output.copy()

    def show(self):
        plt.imshow(self.output, cmap="gray")
        plt.show()


class Spectrum:
    def __init__(self, resolution: int):
        assert isinstance(resolution, int), "resolution must be dynamically castable to int"
        self.resolution = resolution
        self.output = None

    def draw(self):
        self.output = np.zeros((self.resolution, self.resolution, 3))
        # red channel: ascending horizantally
        self.output[:, :, 0] = np.linspace(0, 1, self.resolution)
        # green channcel: descending vertically
        self.output[:, :, 1] = np.linspace(0, 1, self.resolution).reshape(self.resolution, 1)
        # blue channel: ascending horizantally
        self.output[:, :, 2] = np.linspace(1, 0, self.resolution)
        return self.output.copy()

    def show(self):
        plt.imshow(self.output)
        plt.show()


def main():
    checker = Checker(1024, 128)
    checker.draw()
    checker.show()

    circle = Circle(1024, 40, (256, 256))
    circle.draw()
    circle.show()

    spectrum = Spectrum(1024)
    spectrum.draw()
    spectrum.show()


if __name__ == "__main__":
    main()
