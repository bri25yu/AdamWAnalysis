from typing import Union

from torch.nn import Module

from awa.infra import TrainingPipeline, Env
from awa.optimizer_mixins import AdamWOptimizerMixin
from awa.modeling import (
    ExactModel,
    ClassesModel,
    ScoresAndClassesModel,
    CenterNormScoresClassesModel,
    NumCentersModel,
    EnvDimAndCentersModel,
    EnvDimAndNoCenterNormModel,
    CrossAttnModel,
)


class AdamWExperimentBase(AdamWOptimizerMixin, TrainingPipeline):
    MODEL_CLS: Union[None, type] = None

    def get_model(self, env: Env) -> Module:
        model_cls = self.MODEL_CLS

        return model_cls(env)


class ExactAdamWExperiment(AdamWExperimentBase):
    MODEL_CLS = ExactModel


class ClassesAdamWExperiment(AdamWExperimentBase):
    MODEL_CLS = ClassesModel


class ScoresAndClassesAdamWExperiment(AdamWExperimentBase):
    MODEL_CLS = ScoresAndClassesModel


class CenterNormScoresClassesAdamWExperiment(AdamWExperimentBase):
    MODEL_CLS = CenterNormScoresClassesModel


class NumCentersAdamWExperiment(AdamWExperimentBase):
    MODEL_CLS = NumCentersModel


class EnvDimAndCentersAdamWExperiment(AdamWExperimentBase):
    MODEL_CLS = EnvDimAndCentersModel


class EnvDimAndNoCenterNormAdamWExperiment(AdamWExperimentBase):
    MODEL_CLS = EnvDimAndNoCenterNormModel


class CrossAttnAdamWExperiment(AdamWExperimentBase):
    MODEL_CLS = CrossAttnModel
