from awa.infra import TrainingPipeline
from awa.optimizer_mixins import AdamWOptimizerMixin
from awa.modeling import (
    ScoresAndClassesModel,
    CenterNormScoresClassesModel,
    NumCentersModel,
    EnvDimAndCentersModel,
    EnvDimAndNoCenterNormModel,
    CrossAttnModel,
)


class AdamWExperimentBase(AdamWOptimizerMixin, TrainingPipeline):
    pass


class ScoresAndClassesAdamWExperiment(ScoresAndClassesModel, AdamWExperimentBase):
    pass


class CenterNormScoresClassesAdamWExperiment(CenterNormScoresClassesModel, AdamWExperimentBase):
    pass


class NumCentersAdamWExperiment(NumCentersModel, AdamWExperimentBase):
    pass


class EnvDimAndCentersAdamWExperiment(EnvDimAndCentersModel, AdamWExperimentBase):
    pass


class EnvDimAndNoCenterNormAdamWExperiment(EnvDimAndNoCenterNormModel, AdamWExperimentBase):
    pass


class CrossAttnAdamWExperiment(CrossAttnModel, AdamWExperimentBase):
    pass
