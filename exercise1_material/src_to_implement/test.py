import numpy as np


def main():
    weights = np.random.random((5, 2))
    b_weights = np.append(weights, np.random.random((5, 1)), axis=1)
    print(b_weights)


main()
