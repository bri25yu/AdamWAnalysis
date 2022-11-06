from torch.optim import Optimizer, AdamW

from awa.infra import TrainingPipeline


class AdamWExperiment(TrainingPipeline):
    def get_optimizer(self, params) -> Optimizer:
        return AdamW(params, lr=1e-4)
