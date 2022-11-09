from typing import Dict, Union

from dataclasses import dataclass

from torch import Tensor
from torch.nn import Module

from awa.infra import Env


__all__ = ["ModelOutput", "ModelBase"]


@dataclass
class ModelOutput:
    logits: Tensor  # Tensor of shape (batch_size, env.C)
    logs: Union[None, Dict[str, Tensor]] = None


class ModelBase(Module):
    def __init__(self, env: Env) -> None:
        super().__init__()

        self.env = env

    def forward(self, inputs: Tensor) -> ModelOutput:
        """
        Parameters
        ----------
        inputs: Tensor of shape (batch_size, env.D)

        Returns
        -------
        output: ModelOutput

        """
        raise NotImplementedError
