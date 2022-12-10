from torch.nn import Module

from awa.infra import Env
from awa.modeling.v2 import ModelConfig, FinalModel

from awa.experiments.final import FinalModelExperiment


__all__ = ["LayerNormalizationExperiment"]


class LayerNormalizationExperiment(FinalModelExperiment):
    def get_model(self, env: Env) -> Module:
        config = ModelConfig(layernorm_normalization=True)
        return FinalModel(env, config)
