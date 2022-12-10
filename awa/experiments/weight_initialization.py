from torch.nn import Module

from awa.infra import Env
from awa.modeling.v2 import ModelConfig, FinalModel

from awa.experiments.final import FinalModelExperiment


__all__ = ["WeightInitializationExperiment"]


class WeightInitializationExperiment(FinalModelExperiment):
    def get_model(self, env: Env) -> Module:
        config = ModelConfig(weight_initialization="normal")
        return FinalModel(env, config)
