import os

import matplotlib.pyplot as plt
from seaborn import scatterplot

from awa import VIS_OUPTUT_DIR
from awa.infra import Env


def plot_env(env: Env) -> None:
    rows, cols = 1, 1
    fig, ax = plt.subplots(rows, cols, figsize=(10 * cols, 8 * rows))

    xs = env.points[:, 0]
    ys = env.points[:, 1]
    scatterplot(x=xs, y=ys, hue=env.labels, ax=ax, s=10, linewidth=0)

    fig.tight_layout()
    fig.savefig(os.path.join(VIS_OUPTUT_DIR, "env.png"))


if __name__ == "__main__":
    env = Env(N=100000, C=2, D=2)
    plot_env(env)
