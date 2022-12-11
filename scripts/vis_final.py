from typing import Any, Dict

from os.path import join

from numpy import array

from matplotlib.pyplot import subplots, rcParams

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from awa import RESULTS_DIR, VIS_OUPTUT_DIR
from awa.experiments.final import FinalModelExperiment


LEARNING_RATES = FinalModelExperiment.LEARNING_RATES
EVAL_SUFFIX = "_eval"
TEST_SCORE_PROPERTY = "accuracy_test"


def get_summary_dict(experiment_dir: str, learning_rate: float) -> Dict[str, Any]:
    event_accumulator = EventAccumulator(
        join(RESULTS_DIR, experiment_dir, f"lr={learning_rate:.0e}"),
        size_guidance={"histograms": 10000}
    )
    event_accumulator.Reload()

    tags = event_accumulator.Tags()

    steps = None
    data = dict()
    for value_name in tags["scalars"]:
        if not (value_name.endswith(EVAL_SUFFIX) or value_name == TEST_SCORE_PROPERTY):
            continue

        if steps is None:
            scalars = event_accumulator.Scalars(value_name)
            steps = array([s.step for s in scalars])

        scalars = event_accumulator.Scalars(value_name)
        data[value_name] = [s.value for s in scalars]

    test_score = data.pop(TEST_SCORE_PROPERTY)[0]

    histograms_dict = {}
    for hist in tags["histograms"]:
        histograms = event_accumulator.Histograms(hist)
        histograms_dict[hist] = {
            "min": array([h.histogram_value.min for h in histograms]),
            "max": array([h.histogram_value.max for h in histograms]),
            "mean": array([h.histogram_value.sum / h.histogram_value.num for h in histograms])
        }

    return {
        "test_score": test_score,
        "steps": steps,
        "summary_dict": data,
        "histograms_dict": histograms_dict,
    }


def comparison(fig_name: str, configs: Dict[str, str]) -> None:
    save_name = fig_name.lower().replace(" ", "_")

    plotting_data = dict()
    for experiment_name, experiment_path in configs.items():
        best_test_score, best_summary = float("-inf"), None
        for lr in LEARNING_RATES:
            summary = get_summary_dict(experiment_path, lr)
            test_score = summary["test_score"]
            if test_score > best_test_score:
                best_test_score, best_summary = test_score, summary

        plotting_data[experiment_name] = best_summary

    # Plot loss and accuracy
    n_rows, n_cols = 1, 2
    fig, (loss_ax, accuracy_ax) = subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows))

    for experiment_name, summary in plotting_data.items():
        steps = summary["steps"]
        summary_dict = summary["summary_dict"]

        loss_ax.plot(steps, summary_dict["loss_eval"], label=experiment_name)
        accuracy_ax.plot(steps, summary_dict["accuracy_eval"], label=experiment_name)

    loss_ax.legend()
    loss_ax.set_xlabel("Training step")
    loss_ax.set_ylabel("Eval loss")

    accuracy_ax.legend()
    accuracy_ax.set_xlabel("Training step")
    accuracy_ax.set_ylabel("Eval accuracy")

    fig.suptitle(fig_name)
    fig.tight_layout()
    fig.savefig(join(VIS_OUPTUT_DIR, f"{save_name}_stats.png"))

    # Plot histograms
    n_histograms = len(summary["histograms_dict"])
    n_rows, n_cols = 2, (n_histograms + 1) // 2
    fig, axs = subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows))
    hist_to_ax = {value_name: ax for value_name, ax in zip(summary["histograms_dict"], axs.ravel())}

    for experiment_name, summary in plotting_data.items():
        steps = summary["steps"]
        histograms_dict = summary["histograms_dict"]

        for value_name, values_dict in histograms_dict.items():
            histogram_ax = hist_to_ax[value_name]
            histogram_ax.plot(steps, values_dict["mean"], label=experiment_name)
            histogram_ax.fill_between(steps, values_dict["min"], values_dict["max"], alpha=.1)

    for value_name, histogram_ax in hist_to_ax.items():
        histogram_ax.legend()
        histogram_ax.set_xlabel("Training step")
        histogram_ax.set_ylabel("Weight values (shaded between min and max)")
        histogram_ax.set_title(value_name)

    fig.suptitle(fig_name)
    fig.tight_layout()
    fig.savefig(join(VIS_OUPTUT_DIR, f"{save_name}_weights.png"))


def baseline():
    fig_name = "Baseline"
    configs = {
        "Baseline": "FinalModelExperiment",
    }
    comparison(fig_name, configs)


def nonlinearities():
    fig_name = "Nonlinearities"
    configs = {
        "ReLU (Baseline)": "FinalModelExperiment",
        "Swish": "SwishNonlinearityExperiment",
        "GELU": "GELUNonlinearityExperiment",
    }
    comparison(fig_name, configs)


def weight_initialization():
    fig_name = "Weight initialization"
    configs = {
        "Uniform (Baseline)": "FinalModelExperiment",
        "Normal": "WeightInitializationExperiment",
    }
    comparison(fig_name, configs)


def layer_normalization():
    fig_name = "Layer normalization"
    configs = {
        "No layer normalization (Baseline)": "FinalModelExperiment",
        "Layer normalization": "LayerNormalizationExperiment",
    }
    comparison(fig_name, configs)


def weight_decay():
    fig_name = "Weight decay"
    configs = {
        "0.0": "WeightDecayConfig1Experiment",
        "0.001": "WeightDecayConfig2Experiment",
        "0.01 (Baseline)": "FinalModelExperiment",
        "0.1": "WeightDecayConfig3Experiment",
        "1": "WeightDecayConfig4Experiment",
    }
    comparison(fig_name, configs)


def ema_beta():
    rcParams["text.usetex"] = True

    fig_name = "EMA beta values"
    configs = {
        "$\\beta_1=0.9, \\beta_2=0.99$ (Baseline)": "FinalModelExperiment",
        "$\\beta_1=0.85, \\beta_2=0.99$": "EMABeta1Config1Experiment",
        "$\\beta_1=0.95, \\beta_2=0.99$": "EMABeta1Config2Experiment",
        "$\\beta_1=0.9, \\beta_2=0.98$": "EMABeta2Config1Experiment",
        "$\\beta_1=0.9, \\beta_2=0.995$": "EMABeta2Config2Experiment",
    }
    comparison(fig_name, configs)


def adamw_types():
    fig_name = "AdamW types"
    configs = {
        "AdamW L2 (Baseline)": "FinalModelExperiment",
        "AdamW L1": "AdamWTypesL1Experiment",
        "AdamW min": "AdamWTypesMinExperiment",
    }
    comparison(fig_name, configs)


def hidden_dims():
    fig_name = "Hidden dim"
    configs = {
        "H = 256": "HiddenDim256Experiment",
        "H = 512 (Baseline)": "FinalModelExperiment",
        "H = 1024": "HiddenDim1024Experiment",
        "H = 2048": "HiddenDim2048Experiment",
    }
    comparison(fig_name, configs)


if __name__ == "__main__":
    hidden_dims()
