from typing import Union

from torch.nn import Module

from awa.infra import TrainingPipeline, Env
from awa.vis_mixins.logs_and_env import LogsAndEnvVisMixin
from awa.optimizer_mixins import AdamWOptimizerMixin
from awa.modeling.v1 import *


class AdamWExperimentBase(AdamWOptimizerMixin, LogsAndEnvVisMixin, TrainingPipeline):
    MODEL_CLS: Union[None, type] = None

    def get_model(self, env: Env) -> Module:
        model_cls = self.MODEL_CLS

        return model_cls(env)


class TestAdamWExperiment(AdamWExperimentBase):
    LR = 5e-3
    WEIGHT_DECAY = 1e-2
    MODEL_CLS = TestModel


class ExactAdamWExperiment(AdamWExperimentBase):
    LR = 1e-100
    WEIGHT_DECAY = 1e-2
    MODEL_CLS = ExactModel


class CenterLabelsAdamWExperiment(AdamWExperimentBase):
    LR = 1e-2
    WEIGHT_DECAY = 1e-2
    MODEL_CLS = CenterLabelsModel


class LearnOffsetAdamWExperiment(AdamWExperimentBase):
    LR = 2e-3
    WEIGHT_DECAY = 1e-2
    MODEL_CLS = LearnOffsetModel


class OffsetScaleAdamWExperiment(AdamWExperimentBase):
    LR = 1e-2
    WEIGHT_DECAY = 1e-2
    MODEL_CLS = OffsetScaleModel


class AbsUsingReLUAdamWExperiment(AdamWExperimentBase):
    LR = 1e-2
    WEIGHT_DECAY = 1e-2
    MODEL_CLS = AbsUsingReLUModel


class PlusMinusAdamWExperiment(AdamWExperimentBase):
    LR = 5e-3
    WEIGHT_DECAY = 1e-2
    MODEL_CLS = PlusMinusModel


class CentersAdamWExperiment(AdamWExperimentBase):
    LR = 1e-2
    WEIGHT_DECAY = 1e-2
    MODEL_CLS = CentersModel


class SwishAdamWExperiment(AdamWExperimentBase):
    LR = 2e-2
    WEIGHT_DECAY = 0.0
    MODEL_CLS = SwishModel
