import os

from tqdm.notebook import trange, tqdm

from matplotlib.pyplot import close, text, subplots
from matplotlib.animation import ArtistAnimation, PillowWriter

from numpy import unique, ndarray, exp

from torch import Tensor

from awa import RESULTS_DIR
from awa.infra.utils import to_numpy
from awa.modeling.base import ModelOutput


class LogsEnvParamsVisMixin:
    def visualize(self, data: Tensor, labels: Tensor) -> None:
        # Setup data for plotting
        data = to_numpy(data)
        labels = to_numpy(labels)
        xs, ys = data[:, 0], data[:, 1]

        n_steps = len(self.vectors_to_plot_over_time)
        plot_steps = 100

        # List[Dict] -> Dict[str, List]
        transpose = lambda l: {k: [d[k] for d in l] for k in l[0]}
        vector_logs = transpose(self.vectors_to_plot_over_time)
        scalar_logs = transpose(self.scalars_to_plot_over_time)

        def setup_fig_axs():
            n_total_plots = len(vector_logs) + len(scalar_logs)
            rows, cols = (n_total_plots // 2) + (n_total_plots % 2), 2
            fig, axs = subplots(rows, cols, figsize=(10 * cols, 8 * rows))

            axs = axs.ravel()
            if (n_total_plots % 2):
                fig.delaxes(axs[-1])
                axs = axs[:-1]

            vector_axs = axs[:len(vector_logs)]
            scalar_axs = axs[len(vector_logs):]

            fig.suptitle(f"Final benchmark for {self.name}")

            return fig, vector_axs, scalar_axs

        fig, vector_axs, scalar_axs = setup_fig_axs()

        # Set the scalar axs labels
        for ax, value_name in zip(scalar_axs, scalar_logs):
            ax.set_xlabel("Train step")
            ax.set_title(value_name)

        def plot_scalars(step: int):
            steps = list(range(step+1))
            return [
                ax.plot(steps, values[:step+1], color="C0")
                for ax, values in zip(scalar_axs, scalar_logs.values())
            ]

        def plot_logits(step: int):
            # Scatterplot of 2D env
            logits_ax = vector_axs[0]

            logits: ndarray = vector_logs["Eval logits"][step]

            preds = logits.argmax(axis=1)

            probs = exp(logits)
            probs: ndarray = probs / probs.sum(axis=1, keepdims=True)
            probs = probs.max(axis=1)

            artists = []
            for group in unique(preds):
                mask = preds == group
                artists.append(
                    logits_ax.scatter(xs[mask], ys[mask], label=group, color=f"C{group}", alpha=probs[mask])
                )

            # Title with extra info
            accuracy = (preds == labels).sum() * 100 / len(labels)
            step_text = text(
                x=.5, y=1.05,
                s=f"Train step: {step} / {n_steps} | Accuracy: {accuracy:.2f}%",
                va="center", ha="center",
                transform=logits_ax.transAxes,
            )

            return artists + [step_text]

        def plot_vectors(step: int):
            axs = vector_axs[1:]
            logs = {
                v[step] for k, v in vector_logs.items() if k != "Eval logits"
            }

            artists = []
            for ax, (value_name, values) in zip(axs, logs.items()):
                xs = values[:, 0]
                ys = values[:, 1]
                artists.append(ax.scatter(xs, ys))

                step_text = text(
                    x=.5, y=1.05,
                    s=f"{value_name} | Train step: {step} / {n_steps}",
                    va="center", ha="center",
                    transform=ax.transAxes,
                )
                artists.append(step_text)

            return artists

        artists = []
        for step in trange(0, n_steps, plot_steps, leave=False, desc="Visualizing"):
            artists.append([
                *plot_scalars(step),
                *plot_logits(step),
                *plot_vectors(step),
            ])

        with tqdm(total=n_steps, desc="Drawing and saving", leave=False) as pbar:
            update_pbar = lambda current_step, total_steps: pbar.update(1)

            animation = ArtistAnimation(fig, artists, interval=1, blit=True)

            writer = PillowWriter(fps=5)
            output_path = os.path.join(RESULTS_DIR, self.name, "benchmark.gif")
            animation.save(output_path, writer=writer, progress_callback=update_pbar)

        close()

    def setup_visualization_logging(self) -> None:
        self.vectors_to_plot_over_time = []  # 2D vectors
        self.scalars_to_plot_over_time = []

    def store_eval_logs_to_visualize(self, output: ModelOutput, loss: Tensor) -> None:
        logs = output.logs if output.logs else {}

        filter_by_size_len = lambda d, s: {k: to_numpy(v) for k, v in d.items() if len(v.size()) == s}

        self.vectors_to_plot_over_time.append({
            "Eval logits": to_numpy(output.logits),
            **filter_by_size_len(logs, 2),
        })
        self.scalars_to_plot_over_time.append({
            "Eval loss": loss,
            **filter_by_size_len(logs, 0),
            **filter_by_size_len(logs, 1),
        })
