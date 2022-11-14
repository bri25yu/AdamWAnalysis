"""
This is a reduced custom implementation of the AdamW optimization algorithm
"""
from typing import Callable, List

import math

from torch import Tensor

from awa.modeling.optimizers.custom_adamw_base import CustomAdamWBase


__all__ = ["AdamWL1WithGrad"]


def adamw_l1_withgrad(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    *,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
) -> None:
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        # update step
        step_t += 1

        # Perform L1 stepweight decay
        param.sub_(param.sign(), alpha=lr * weight_decay)

        # Perform L1 grad decay
        grad.sub_(grad.sign(), alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        step = step_t.item()

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        step_size = lr / bias_correction1

        bias_correction2_sqrt = math.sqrt(bias_correction2)

        denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

        param.addcdiv_(exp_avg, denom, value=-step_size)


class AdamWL1WithGrad(CustomAdamWBase):
    def get_adamw_function(self) -> Callable:
        return adamw_l1_withgrad
