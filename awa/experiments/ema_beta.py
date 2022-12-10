from typing import Union

from torch.optim import Optimizer, AdamW

from awa.experiments.final import FinalModelExperiment


__all__ = [
    "EMABeta1Config1Experiment",
    "EMABeta1Config2Experiment",
    "EMABeta2Config1Experiment",
    "EMABeta2Config2Experiment",
]


class EMABetaExperimentBase(FinalModelExperiment):
    BETA_1: Union[None, float] = None
    BETA_2: Union[None, float] = None

    def get_optimizer(self, params) -> Optimizer:
        assert (self.BETA_1 is not None) and (self.BETA_2 is not None)
        return AdamW(params, lr=self.LR, betas=(self.BETA_1, self.BETA_2))


class EMABeta1Config1Experiment(EMABetaExperimentBase):
    BETA_1 = 0.85
    BETA_2 = 0.99


class EMABeta1Config2Experiment(EMABetaExperimentBase):
    BETA_1 = 0.95
    BETA_2 = 0.99


class EMABeta2Config1Experiment(EMABetaExperimentBase):
    BETA_1 = 0.9
    BETA_2 = 0.98


class EMABeta2Config2Experiment(EMABetaExperimentBase):
    BETA_1 = 0.9
    BETA_2 = 0.995
