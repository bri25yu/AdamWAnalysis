from typing import Union

from torch.nn import Module

from awa.infra import TrainingPipeline, Env
from awa.optimizer_mixins import CustomAdamWOptimizerMixin
from awa.modeling import *


class CustomAdamWExperimentBase(CustomAdamWOptimizerMixin, TrainingPipeline):
    MODEL_CLS: Union[None, type] = None

    def get_model(self, env: Env) -> Module:
        model_cls = self.MODEL_CLS

        return model_cls(env)


class CentersCustomAdamWExperiment(CustomAdamWExperimentBase):
    LR = 1e-2
    WEIGHT_DECAY = 1e-2
    MODEL_CLS = CentersModel
