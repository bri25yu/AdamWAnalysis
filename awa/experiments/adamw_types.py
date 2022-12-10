from awa.optimizer_mixins import AdamWL1OptimizerMixin, AdamWMinOptimizerMixin
from awa.experiments.final import FinalModelExperiment


__all__ = ["AdamWTypesL1Experiment", "AdamWTypesMinExperiment"]


class AdamWTypesExperimentBase(FinalModelExperiment):
    pass


class AdamWTypesL1Experiment(AdamWL1OptimizerMixin, AdamWTypesExperimentBase):
    pass


class AdamWTypesMinExperiment(AdamWMinOptimizerMixin, AdamWTypesExperimentBase):
    pass
