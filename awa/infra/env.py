from numpy import ndarray, float32
from numpy.random import randint, rand
from numpy.linalg import norm


__all__ = ["Env"]


class Env:
    """
    Grid of x \in [-100, 100] ^ D

    We make a classification task using the following algorithm.
    Given hyperparameters k, c, and N,
    1. Pick $k$ points as our centers
    2. Assign each center a label $c$
    3. Pick $N$ points
    4. Assign the points a label $c$ depending on which center is closer

    This is a Voronoi diagram

    k ~ Uniform(100, 150)
    """
    X_BOUNDS = [-100, 100]
    K_BOUNDS = [100, 150]

    def __init__(self, N: int, C: int, D: int) -> None:
        self.N = N
        self.C = C
        self.D = D

        k_min, k_max = self.K_BOUNDS
        x_min, x_max = self.X_BOUNDS

        # Pick k
        k = randint(k_min, k_max + 1)

        # 1. Pick k points as our centers
        x_loc, x_scale = (x_min + x_max) / 2, (x_max - x_min) / 2
        centers = (2 * rand(k, D) - 1) * x_scale + x_loc

        # 2. Assign each center a label c
        center_labels = randint(C, size=(k,))

        # 3. Pick n points
        points = (2 * rand(N, D) - 1) * x_scale + x_loc

        # 4. Assign the points a label c depending on which center is closer
        centers_rescaled = centers / (norm(centers, axis=1, keepdims=True) ** 2)
        dot_products = points @ centers_rescaled.T
        assert dot_products.shape == (N, k)
        closest_to_1 = (dot_products - 1) ** 2
        closest_center_indices = closest_to_1.argmin(axis=1)
        labels: ndarray = center_labels[closest_center_indices]

        self.centers = centers.astype(float32)
        self.center_labels = center_labels
        self.points = points.astype(float32)
        self.labels = labels
