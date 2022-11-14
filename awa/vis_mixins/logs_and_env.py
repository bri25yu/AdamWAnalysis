import os

from tqdm.notebook import trange, tqdm

from matplotlib.pyplot import close, text, subplots
from matplotlib.animation import ArtistAnimation, PillowWriter

from numpy import unique

from torch import Tensor

from awa import RESULTS_DIR
from awa.infra.utils import to_numpy
from awa.modeling.base import ModelOutput


class LogsAndEnvVisMixin:
    def visualize(self, data: Tensor, labels: Tensor) -> None:
        data = to_numpy(data)
        labels = to_numpy(labels)

        logs_by_timestep = self.logs_to_plot_over_time
        logs = {k: [l[k] for l in logs_by_timestep] for k in logs_by_timestep[0]}

        n_total_plots = 1 + len(logs)
        rows, cols = (n_total_plots // 2) + (n_total_plots % 2), 2
        fig, axs = subplots(rows, cols, figsize=(10 * cols, 8 * rows))

        axs = axs.ravel()
        env_ax, axs = axs[0], axs[1:]
        if (n_total_plots % 2):
            fig.delaxes(axs[-1])
            axs = axs[:-1]

        # Set the axes labels
        for ax, value_name in zip(axs, logs):
            ax.set_xlabel("Train step")
            ax.set_title(value_name)

        xs, ys = data[:, 0], data[:, 1]

        artists = []
        for step in trange(0, len(self.eval_predictions_over_time), 100, leave=False, desc="Visualizing"):
            preds = self.eval_predictions_over_time[step]

            artists_current_step = []

            # Scatterplot of 2D env
            for group in unique(preds):
                mask = preds == group
                artists_current_step.append(
                    env_ax.scatter(xs[mask], ys[mask], label=group, color=f"C{group}")
                )

            # Other logs line plot
            steps = list(range(step+1))
            for ax, values in zip(axs, logs.values()):
                artists_current_step.extend(
                    ax.plot(steps, values[:step+1], color="C0")
                )

            # Title with extra info
            accuracy = (preds == labels).sum() * 100 / len(labels)
            step_text = text(
                x=.5, y=1.05,
                s=f"Train step: {step} / {len(self.eval_predictions_over_time)} | Accuracy: {accuracy:.2f}%",
                va="center", ha="center",
                transform=env_ax.transAxes,
            )
            artists_current_step.append(step_text)

            artists.append(artists_current_step)

        fig.suptitle(f"Final benchmark for {self.name}")

        with tqdm(total=len(artists), desc="Drawing and saving", leave=False) as pbar:
            update_pbar = lambda current_step, total_steps: pbar.update(1)

            animation = ArtistAnimation(fig, artists, interval=1, blit=True)

            writer = PillowWriter(fps=5)
            output_path = os.path.join(RESULTS_DIR, self.name, "benchmark.gif")
            animation.save(output_path, writer=writer, progress_callback=update_pbar)

        close()

    def setup_visualization_logging(self) -> None:
        self.eval_predictions_over_time = []
        self.logs_to_plot_over_time = []

    def store_eval_logs_to_visualize(self, output: ModelOutput, loss: Tensor) -> None:
        eval_preds = output.logits.argmax(dim=1)
        self.eval_predictions_over_time.append(to_numpy(eval_preds))

        logs = {
            "Eval loss": loss,
            **(output.logs if output.logs else {}),
        }
        logs = {k: to_numpy(v) for k, v in logs.items()}
        self.logs_to_plot_over_time.append(logs)
