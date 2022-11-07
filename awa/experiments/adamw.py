from typing import Union

from torch.nn import Module

from awa.infra import TrainingPipeline, Env
from awa.optimizer_mixins import AdamWOptimizerMixin
from awa.modeling import (
    ExactModel,
)


class AdamWExperimentBase(AdamWOptimizerMixin, TrainingPipeline):
    MODEL_CLS: Union[None, type] = None

    def get_model(self, env: Env) -> Module:
        model_cls = self.MODEL_CLS

        return model_cls(env)


class ExactAdamWExperiment(AdamWExperimentBase):
    MODEL_CLS = ExactModel
