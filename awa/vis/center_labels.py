import os

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from seaborn import scatterplot

from awa import RESULTS_DIR
from awa.infra import Env
from awa.vis.tensorboard_utils import get_property_and_steps


__all__ = ["CenterLabelsVisMixin"]


class CenterLabelsVisMixin:
    def visualize(self, env: Env) -> None:
        logdir = os.path.join(RESULTS_DIR, self.name)
        steps, center_logits_over_time = get_property_and_steps(logdir, "center_logits")

        logits_to_labels = lambda logits: logits.argmax(dim=1)
        center_labels_over_time = list(map(logits_to_labels, center_logits_over_time))

        points = env.points[:10000]
        get_point_labels = lambda center_labels: env.assign_labels_to_points(points, env.centers, center_labels)
        point_labels_over_time = list(map(get_point_labels, center_labels_over_time))

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        xs, ys = points[:, 0], points[:, 1]

        # animation function.  This is called sequentially
        def animate(i):
            return scatterplot(x=xs, y=ys, hue=point_labels_over_time[i], ax=ax, s=10, linewidth=0)

        fps = 5
        anim = FuncAnimation(fig, animate, frames=len(steps),interval=1000 / fps)
        anim.save(os.path.join(logdir, "output.png"), fps=fps)
