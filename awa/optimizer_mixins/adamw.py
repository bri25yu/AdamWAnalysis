from typing import Union

from torch.optim import Optimizer, AdamW


__all__ = ["AdamWOptimizerMixin"]


class AdamWOptimizerMixin:
    LR: Union[None, float] = None
    WEIGHT_DECAY: Union[None, float] = None

    def get_optimizer(self, params) -> Optimizer:
        lr = self.LR
        weight_decay = self.WEIGHT_DECAY
        assert lr and weight_decay

        return AdamW(params, lr=lr, weight_decay=weight_decay)
