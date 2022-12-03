from torch import Tensor, softmax, ones
from torch.nn import Linear, ReLU, Module, Parameter

from awa.infra import Env
from awa.modeling.base import ModelBase, ModelOutput


__all__ = ["FullyParameterizedModel"]


class ReducedLayerNorm(Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.weights = Parameter(ones(1, hidden_dim))

    def forward(self, inputs: Tensor) -> Tensor:
        return self.weights * inputs


class Dense(Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.layernorm = ReducedLayerNorm(in_dim)
        self.linear1 = Linear(in_dim, hidden_dim, bias=False)
        self.nonlinearity = ReLU()
        self.linear2 = Linear(hidden_dim, out_dim, bias=False)

    def forward(self, inputs: Tensor) -> Tensor:
        inputs = self.layernorm(inputs)
        inputs = self.linear1(inputs)
        inputs = self.nonlinearity(inputs)
        inputs = self.linear2(inputs)
        return inputs


class ReducedAttention(Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.layernorm = ReducedLayerNorm(hidden_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        inputs = self.layernorm(inputs)
        return softmax(inputs, dim=1)


class FullyParameterizedModel(ModelBase):
    def __init__(self, env: Env) -> None:
        super().__init__(env)

        hidden_dim = 512

        self.dense = Dense(env.D, hidden_dim, hidden_dim)
        self.attention = ReducedAttention(hidden_dim)
        self.classification_head = Dense(hidden_dim, hidden_dim, env.C)
        self.final_layer_norm = ReducedLayerNorm(env.C)

    def forward(self, inputs: Tensor) -> Tensor:
        inputs = inputs / 100

        center_scores = self.dense(inputs)
        center_probs = self.attention(center_scores)
        logits = self.classification_head(center_probs)
        logits = self.final_layer_norm(logits)

        return ModelOutput(
            logits=logits,
            logs={
                "value_center_probs": center_probs.max(dim=1)[0].mean(),
            }
        )
