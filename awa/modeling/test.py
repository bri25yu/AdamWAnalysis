from torch import Tensor, softmax
from torch.nn import Linear, ReLU, Sequential, Module

from awa.infra import Env
from awa.modeling.base import ModelBase, ModelOutput


__all__ = ["TestModel"]


class Dense(Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.dense = Sequential(
            Linear(in_dim, hidden_dim, bias=False),
            ReLU(),
            Linear(hidden_dim, out_dim, bias=False),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.dense(inputs)


class TestModel(ModelBase):
    def __init__(self, env: Env) -> None:
        super().__init__(env)

        hidden_dim = 2048

        self.dense = Dense(env.D, hidden_dim, hidden_dim)
        self.classification_head = Dense(hidden_dim, hidden_dim, env.C)

    def forward(self, inputs: Tensor) -> Tensor:
        inputs = inputs / 100

        center_scores = self.dense(inputs)
        center_probs = softmax(center_scores, dim=1)
        logits = self.classification_head(center_probs)

        return ModelOutput(
            logits=logits,
            logs={
                "value_center_probs": center_probs.max(dim=1)[0].mean(),
            }
        )
