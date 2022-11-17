from torch.nn import Module

from awa.infra import TrainingPipeline, Env
from awa.modeling import CentersWithParamsModel, TestModel
from awa.vis_mixins.logs_env_params import LogsEnvParamsVisMixin
from awa.optimizer_mixins import (
    CustomAdamWOptimizerMixin,
    AdamWL1OptimizerMixin,
    TestAdamWOptimizerMixin,
)


class CustomAdamWExperimentBase(LogsEnvParamsVisMixin, TrainingPipeline):
    def get_model(self, env: Env) -> Module:
        return CentersWithParamsModel(env)


class TestAdamWExperiment(TestAdamWOptimizerMixin, LogsEnvParamsVisMixin, TrainingPipeline):
    LR = 1e-2
    WEIGHT_DECAY = 1e-2

    def get_model(self, env: Env) -> Module:
        return TestModel(env)


class CustomAdamWExperiment(CustomAdamWOptimizerMixin, CustomAdamWExperimentBase):
    LR = 1e-2
    WEIGHT_DECAY = 1e-2


class AdamWL1Experiment(AdamWL1OptimizerMixin, CustomAdamWExperimentBase):
    LR = 1e-2
    WEIGHT_DECAY = 1e-2
