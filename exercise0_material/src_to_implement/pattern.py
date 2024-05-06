import numpy as np
import matplotlib.pyplot as plt


class Pattern:
    def __init__(self, res, *args):
        # Determine the type of pattern based on args
        if len(args) == 1 and isinstance(args[0], int):
            self.__class__ = Checker
        elif len(args) == 2 and isinstance(args[0], int) and isinstance(args[1], tuple):
            self.__class__ = Circle
        elif len(args) == 0:
            self.__class__ = Spectrum
        else:
            raise ValueError("Invalid parameters for pattern creation")

        # Initialize the chosen subclass
        self.res = res
        self.output = np.zeros((self.res, self.res), dtype=int)
        if hasattr(self, "__post_init__"):
            self.__post_init__(*args)

    def draw(self):
        raise NotImplementedError("Subclasses must implement this method")

    def show(self):
        if self.output is not None:
            plt.imshow(self.output, cmap="gray" if self.output.ndim == 2 else None)
            plt.title(self.__class__.__name__)
            plt.axis("off")
            plt.show()
        else:
            print("No pattern generated to show.")


class Checker(Pattern):
    def __post_init__(self, tile_size):
        self.tile_size = tile_size
        if self.res % (2 * self.tile_size) != 0:
            raise ValueError("Resolution must be evenly divisible by 2 * tile_size")

    def draw(self):
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
    def __post_init__(self, radius, pos):
        self.radius = radius
        self.pos = pos

    def draw(self):
        # if (self.pos[0] + self.radius) > (self.res // 2) or (
        #     self.pos[1] + self.radius
        # ) > (self.res // 2):
        #     raise ValueError("Please check res and radius arguments.")

        xval = np.arange(self.res)
        yval = np.arange(self.res)

        # Create a meshgrid
        xgrid, ygrid = np.meshgrid(xval, yval)

        max_radius = (
            np.square(xgrid - self.pos[0]) + np.square(ygrid - self.pos[1])
            <= self.radius**2
        )

        self.output = max_radius

        return np.copy(self.output)


class Spectrum(Pattern):
    def __post_init__(self):
        pass

    def draw(self):
        unit = np.linspace(0.0, 1.0, num=self.res)
        # Create the rbg planes
        r = np.tile(unit, (self.res, 1))
        g = np.rot90(np.rot90(np.rot90(r)))
        b = np.rot90(np.rot90(r))

        self.output = np.dstack((r, g, b))

        return np.copy(self.output)
