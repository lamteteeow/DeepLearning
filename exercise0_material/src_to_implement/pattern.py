from abc import abstractmethod
import numpy as np
import matplotlib.pyplot as plt


class Pattern:
    def __init__(self, res):
        self.res = res
        self.output = None

    @classmethod
    def create(cls, res, *args):
        if len(args) == 1 and isinstance(args[0], int):
            return Checker(res, args[0])
        elif len(args) == 2 and isinstance(args[0], int) and isinstance(args[1], tuple):
            return Circle(res, args[0], args[1])
        elif len(args) == 0:
            return Spectrum(res)
        else:
            raise ValueError("Invalid parameters for pattern creation")

    @abstractmethod
    def draw(self):
        raise NotImplementedError("Subclass must implement draw method")

    def show(self):
        plt.imshow(self.output, cmap="gray")
        plt.show()


class Checker(Pattern):
    def __init__(self, res, tile_size):
        super().__init__(res)
        self.tile_size = tile_size
        if res % (2 * self.tile_size) != 0:
            raise ValueError("Resolution must be evenly divisible by 2 * tile_size")

    def draw(self):
        if self.res % (2 * self.tile_size) != 0:
            raise ValueError("Please check resolution and tize_size arguments.")
        black_tile = np.zeros((self.tile_size, self.tile_size), dtype=int)
        white_tile = np.ones((self.tile_size, self.tile_size), dtype=int)
        # Black tile top left
        block_unit = np.concatenate(
            (
                np.concatenate((black_tile, white_tile), axis=1),
                np.concatenate((white_tile, black_tile), axis=1),
            ),
            axis=0,
        )

        self.output = np.tile(
            block_unit,
            (
                int(self.res // (2 * self.tile_size)),
                int(self.res // (2 * self.tile_size)),
            ),
        )
        return np.copy(self.output)


class Circle(Pattern):
    def __init__(self, res, radius, pos):
        super().__init__(res)
        self.radius = radius
        self.pos = pos
        self.output = np.zeros((res, res, 3))

    def draw(self):
        # if (self.pos[0] + self.radius) > (self.res // 2) or (
        #     self.pos[1] + self.radius
        # ) > (self.res // 2):
        #     raise ValueError("Please check res and radius arguments.")

        xval = np.linspace(
            self.pos[0] - self.radius,
            self.pos[0] + self.radius,
            num=self.radius * 2 + 1,
            dtype=int,
        )
        yval = np.linspace(
            self.pos[1] - self.radius,
            self.pos[1] + self.radius,
            num=self.radius * 2 + 1,
            dtype=int,
        )

        # Create a meshgrid
        x_v, y_v = np.meshgrid(xval, yval)

        max_radius = (
            np.square(x_v - self.pos[0]) + np.square(y_v - self.pos[1])
            <= self.radius**2
        )

        self.output[x_v[max_radius], y_v[max_radius]] = 1.0

        return np.copy(self.output)


class Spectrum(Pattern):
    def __init__(self, res):
        super().__init__(res)

    def draw(self):
        unit = np.linspace(0.0, 1.0, num=self.res)
        # Create the rbg planes
        r = np.tile(unit, (self.res, 1))
        g = np.rot90(np.rot90(np.rot90(r)))
        b = np.rot90(np.rot90(r))

        self.output = np.dstack((r, g, b))

        return np.copy(self.output)


if __name__ == "__main__":
    ch = Checker(512, 64)
    ch.draw()
    ch.show()

    c = Circle(512, 100, (250, 250))
    c.draw()
    c.show()

    s = Spectrum(512)
    s.draw()
    s.show()
    # pass
