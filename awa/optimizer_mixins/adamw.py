from torch.optim import Optimizer, AdamW


__all__ = ["AdamWOptimizerMixin"]


class AdamWOptimizerMixin:
    def get_optimizer(self, params) -> Optimizer:
        return AdamW(params, lr=1e-6)
