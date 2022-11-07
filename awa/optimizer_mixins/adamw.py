from typing import Union

from torch.optim import Optimizer, AdamW


__all__ = ["AdamWOptimizerMixin"]


class AdamWOptimizerMixin:
    LR: Union[None, float] = None

    def get_optimizer(self, params) -> Optimizer:
        lr = self.LR
        assert lr

        return AdamW(params, lr=lr)
