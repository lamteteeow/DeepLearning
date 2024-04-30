import numpy as np
import matplotlib.pyplot as plt


class Checker:
    def __init__(self, res, tile_size):
        assert isinstance(res, int) and res >= 0
        self.res = res
        assert isinstance(tile_size, int) and tile_size >= 0
        self.tile_size = tile_size
        assert res % tile_size == 0
        self.output = np.empty((res, res))

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
            block_unit, self.res / self.tile_size, self.res / self.tile_size
        )
        return np.copy(self.output)

    def show(self):
        plt.imshow(self.output, cmap="gray")
        plt.show()


class Circle:
    def __init__(self, res, radius, pos):
        assert isinstance(res, int) and res >= 0
        self.res = res
        assert isinstance(radius, int) and radius >= 0
        self.radius = radius
        assert isinstance(pos, tuple)
        self.pos = pos
        self.out = np.empty()

    def draw(self):
        # if (self.pos[0] + self.radius) > (self.res // 2) or (
        #     self.pos[1] + self.radius
        # ) > (self.res // 2):
        #     raise ValueError("Please check res and radius arguments.")
        return
