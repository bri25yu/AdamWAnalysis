from typing import Union

from torch.optim import Optimizer, AdamW
from awa.modeling.optimizers import *


__all__ = [
    "AdamWOptimizerMixin",
    "CustomAdamWOptimizerMixin",
    "AdamWL1OptimizerMixin",
    "AdamWL1L2OptimizerMixin",
]


class AdamWOptimizerMixinBase:
    OPTIMIZER_CLS: Union[None, type] = None
    LR: Union[None, float] = None
    WEIGHT_DECAY: Union[None, float] = None

    def get_optimizer(self, params) -> Optimizer:
        optimizer_cls = self.OPTIMIZER_CLS
        lr = self.LR
        weight_decay = self.WEIGHT_DECAY
        assert optimizer_cls and lr and weight_decay

        return optimizer_cls(params, lr=lr, weight_decay=weight_decay)


class AdamWOptimizerMixin(AdamWOptimizerMixinBase):
    OPTIMIZER_CLS = AdamW


class CustomAdamWOptimizerMixin(AdamWOptimizerMixinBase):
    OPTIMIZER_CLS = CustomAdamW


class AdamWL1OptimizerMixin(AdamWOptimizerMixinBase):
    OPTIMIZER_CLS = AdamWL1


class AdamWL1L2OptimizerMixin(AdamWOptimizerMixinBase):
    OPTIMIZER_CLS = AdamWL1L2
