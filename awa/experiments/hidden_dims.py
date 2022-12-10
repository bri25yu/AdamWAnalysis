from typing import Union

from torch.nn import Module

from awa.infra import Env
from awa.modeling.v2 import ModelConfig, FinalModel

from awa.experiments.final import FinalModelExperiment


__all__ = [
    "HiddenDim256Experiment",
    "HiddenDim1024Experiment",
    "HiddenDim2048Experiment",
]


class HiddenDimExperimentBase(FinalModelExperiment):
    HIDDEN_DIM: Union[None, int] = None

    def get_model(self, env: Env) -> Module:
        config = ModelConfig(hidden_dim=self.HIDDEN_DIM)
        return FinalModel(env, config)


class HiddenDim256Experiment(HiddenDimExperimentBase):
    HIDDEN_DIM = 256


class HiddenDim1024Experiment(HiddenDimExperimentBase):
    HIDDEN_DIM = 1024


class HiddenDim2048Experiment(HiddenDimExperimentBase):
    HIDDEN_DIM = 2048
