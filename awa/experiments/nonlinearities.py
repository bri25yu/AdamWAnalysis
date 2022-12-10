from typing import Union

from torch.nn import Module

from awa.infra import Env
from awa.modeling.v2 import ModelConfig, FinalModel

from awa.experiments.final import FinalModelExperiment


__all__ = ["SwishNonlinearityExperiment", "GELUNonlinearityExperiment"]


class NonlinearitiesExperimentBase(FinalModelExperiment):
    NONLINEARITY: Union[None, str] = None

    def get_model(self, env: Env) -> Module:
        config = ModelConfig(nonlinearity=self.NONLINEARITY)
        return FinalModel(env, config)


class SwishNonlinearityExperiment(NonlinearitiesExperimentBase):
    NONLINEARITY = "swish"


class GELUNonlinearityExperiment(NonlinearitiesExperimentBase):
    NONLINEARITY = "gelu"
