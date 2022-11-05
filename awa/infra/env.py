from numpy import ndarray
from numpy.random import randint, rand
from numpy.linalg import norm


class Env:
    """
    2D grid of x \in [-100, 100] ^ 2

    We make a classification task using the following algorithm.
    Given hyperparameters k, c, and N,
    1. Pick $k$ points as our centers
    2. Assign each center a label $c$
    3. Pick $N$ points
    4. Assign the points a label $c$ depending on which center is closer

    This is a Voronoi diagram

    k ~ Uniform(100, 150)
    C = 2

    Each env has a dataset size of 100000 points, so D = R^{N x 2}, N = 100000
    """
    X_BOUNDS = [-100, 100]
    K_BOUNDS = [100, 150]
    N = 100000
    C = 2

    def __init__(self) -> None:
        # Pick k
        k_min, k_max = self.K_BOUNDS
        k = randint(k_min, k_max + 1)

        # 1. Pick k points as our centers
        x_min, x_max = self.X_BOUNDS
        x_loc, x_scale = (x_min + x_max) / 2, (x_max - x_min) / 2
        centers = (2 * rand(k, 2) - 1) * x_scale + x_loc

        # 2. Assign each center a label c
        center_labels = randint(self.C, size=(k,))

        # 3. Pick n points
        points = (2 * rand(self.N, 2) - 1) * x_scale + x_loc

        # 4. Assign the points a label c depending on which center is closer
        centers_rescaled = centers / (norm(centers, axis=1, keepdims=True) ** 2)
        dot_products = points @ centers_rescaled.T
        assert dot_products.shape == (self.N, k)
        closest_to_1 = (dot_products - 1) ** 2
        closest_center_indices = closest_to_1.argmin(axis=1)
        labels: ndarray = center_labels[closest_center_indices]

        self.centers = centers
        self.center_labels = center_labels
        self.points = points
        self.labels = labels
