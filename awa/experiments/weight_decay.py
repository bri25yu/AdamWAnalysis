from awa.experiments.final import FinalModelExperiment


__all__ = [
    "WeightDecayConfig1Experiment",
    "WeightDecayConfig2Experiment",
    "WeightDecayConfig3Experiment",
    "WeightDecayConfig4Experiment",
]


class WeightDecayConfig1Experiment(FinalModelExperiment):
    WEIGHT_DECAY = 0.0


class WeightDecayConfig2Experiment(FinalModelExperiment):
    WEIGHT_DECAY = 1e-3


class WeightDecayConfig3Experiment(FinalModelExperiment):
    WEIGHT_DECAY = 1e-1


class WeightDecayConfig4Experiment(FinalModelExperiment):
    WEIGHT_DECAY = 1e0
