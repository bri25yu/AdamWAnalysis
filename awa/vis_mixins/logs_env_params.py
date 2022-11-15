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
        writer = PillowWriter(fps=5)

        # List[Dict] -> Dict[str, List]
        transpose = lambda l: {k: [d[k] for d in l] for k in l[0]}

        def setup_fig_axs(logs):
            n_total_plots = len(logs)
            rows, cols = (n_total_plots // 2) + (n_total_plots % 2), 2
            fig, axs = subplots(rows, cols, figsize=(10 * cols, 8 * rows), dpi=200)

            axs = axs.ravel()
            if (n_total_plots % 2):
                fig.delaxes(axs[-1])
                axs = axs[:-1]

            fig.suptitle(f"Final benchmark for {self.name}")

            return fig, axs

        def plot_scalars(step: int, scalar_axs, scalar_logs):
            steps = list(range(step+1))
            artists = []
            for ax, (value_name, values) in zip(scalar_axs, scalar_logs.items()):
                ax.set_xlabel("Train step")
                ax.set_title(value_name)
                artists.extend(
                    ax.plot(steps, values[:step+1], color="C0")
                )

            return artists

        def plot_logits(step: int, logits_ax, vector_logs):
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
            artists.append(step_text)

            return artists

        def plot_vectors(step: int, vector_axs, vector_logs):
            logs = {
                k: v[step] for k, v in vector_logs.items() if k != "Eval logits"
            }

            artists = []
            for ax, (value_name, values) in zip(vector_axs, logs.items()):
                xs = values[:, 0]
                ys = values[:, 1]
                artists.append(ax.scatter(xs, ys, color="C0"))

                step_text = text(
                    x=.5, y=1.05,
                    s=f"{value_name} | Train step: {step} / {n_steps}",
                    va="center", ha="center",
                    transform=ax.transAxes,
                )
                artists.append(step_text)

            return artists

        def save_fig(fig, artists, output_path, desc):
            with tqdm(total=n_steps // plot_steps, desc=f"Drawing and saving {desc}", leave=False) as pbar:
                update_pbar = lambda current_step, total_steps: pbar.update(1)

                animation = ArtistAnimation(fig, artists, interval=1, blit=True)
                animation.save(output_path, writer=writer, progress_callback=update_pbar)

            close(fig)

        # Visualize scalar data
        scalar_logs = transpose(self.scalars_to_plot_over_time)
        scalar_fig, scalar_axs = setup_fig_axs(scalar_logs)
        scalars_output_path = os.path.join(RESULTS_DIR, self.name, "benchmark_scalars.gif")
        scalar_artists = [
            plot_scalars(step, scalar_axs, scalar_logs)
            for step in trange(0, n_steps, plot_steps, leave=False, desc="Visualizing scalars")
        ]
        save_fig(scalar_fig, scalar_artists, scalars_output_path, "scalars")

        # Visualize vector data
        vector_logs = transpose(self.vectors_to_plot_over_time)
        vector_fig, vector_axs = setup_fig_axs(vector_logs)
        vectors_output_path = os.path.join(RESULTS_DIR, self.name, "benchmark_vectors.gif")
        vector_artists = [
            [
                *plot_logits(step, vector_axs[0], vector_logs),
                *plot_vectors(step, vector_axs[1:], vector_logs),
            ]
            for step in trange(0, n_steps, plot_steps, leave=False, desc="Visualizing vectors")
        ]
        save_fig(vector_fig, vector_artists, vectors_output_path, "vectors")

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
            "Eval loss": to_numpy(loss),
            **filter_by_size_len(logs, 0),
            **filter_by_size_len(logs, 1),
        })
