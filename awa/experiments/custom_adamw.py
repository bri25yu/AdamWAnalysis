from torch.nn import Module

from awa.infra import TrainingPipeline, Env
from awa.modeling import CentersModel
from awa.optimizer_mixins import (
    CustomAdamWOptimizerMixin,
    AdamWL1OptimizerMixin,
    AdamWL1L2OptimizerMixin,
    AdamWL1WithGradOptimizerMixin,
)


class CustomAdamWExperimentBase(TrainingPipeline):
    def get_model(self, env: Env) -> Module:
        return CentersModel(env)


class CustomAdamWExperiment(CustomAdamWOptimizerMixin, CustomAdamWExperimentBase):
    LR = 1e-2
    WEIGHT_DECAY = 1e-2


class AdamWL1Experiment(AdamWL1OptimizerMixin, CustomAdamWExperimentBase):
    LR = 1e-2
    WEIGHT_DECAY = 1e-2


class AdamWL1L2Experiment(AdamWL1L2OptimizerMixin, CustomAdamWExperimentBase):
    LR = 1e-2
    WEIGHT_DECAY = 1e-2


class AdamWL1WithGradExperiment(AdamWL1WithGradOptimizerMixin, CustomAdamWExperimentBase):
    LR = 1e-2
    WEIGHT_DECAY = 1e-2
