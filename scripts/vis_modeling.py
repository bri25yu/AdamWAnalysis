from os.path import join

from glob import glob

from collections import defaultdict

from matplotlib.pyplot import subplots

from pandas import DataFrame

from tensorflow.python.summary.summary_iterator import summary_iterator

from awa import RESULTS_DIR, VIS_OUPTUT_DIR


TF_EVENTFILE_PREFIX = "events.out.tfevents"


def get_summary(experiment_dir: str) -> DataFrame:
    path = glob(f"{experiment_dir}/*/{TF_EVENTFILE_PREFIX}*", recursive=True)[0]

    data = defaultdict(dict)
    for event in summary_iterator(path):
        for v in event.summary.value:
            if not v.simple_value:
                continue
            data[event.step][v.tag] = v.simple_value

    return DataFrame(data=data.values(), index=data.keys())


def vis_modeling_losses() -> None:
    paths = [
        "CenterLabelsAdamWExperiment",
        "LearnOffsetAdamWExperiment",
        "OffsetScaleAdamWExperiment",
        "PlusMinusAdamWExperiment",
        "CentersAdamWExperiment",
    ]

    n_rows, n_cols = 1, 2
    fig, (loss_ax, accuracy_ax) = subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows))

    for path in paths:
        summary = get_summary(join(RESULTS_DIR, path))
        label = path.removesuffix("AdamWExperiment")
        loss_ax.plot(summary["loss_eval"], label=label)
        accuracy_ax.plot(summary["accuracy_eval"], label=label)

    loss_ax.legend()
    loss_ax.set_xlabel("Training step")
    loss_ax.set_ylabel("Eval loss")

    accuracy_ax.legend()
    accuracy_ax.set_xlabel("Training step")
    accuracy_ax.set_ylabel("Eval accuracy")

    fig.tight_layout()
    fig.savefig(join(VIS_OUPTUT_DIR, "modeling_losses.png"))


if __name__ == "__main__":
    vis_modeling_losses()
