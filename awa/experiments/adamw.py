from typing import Union

from torch.nn import Module

from awa.infra import TrainingPipeline, Env
from awa.optimizer_mixins import AdamWOptimizerMixin
from awa.modeling import (
    ExactModel,
    CenterLabelsModel,
    LearnOffsetModel,
    OffsetScaleModel,
)


class AdamWExperimentBase(AdamWOptimizerMixin, TrainingPipeline):
    MODEL_CLS: Union[None, type] = None

    def get_model(self, env: Env) -> Module:
        model_cls = self.MODEL_CLS

        return model_cls(env)


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
    LR = 1e-3
    WEIGHT_DECAY = 1e-2
    MODEL_CLS = OffsetScaleModel
