from dataclasses import dataclass

from torch import Tensor, softmax, ones
from torch.nn import Linear, ReLU, Module, Parameter, SiLU, GELU

from awa.infra import Env
from awa.modeling.base import ModelBase, ModelOutput


__all__ = ["ModelConfig", "FinalModel"]


@dataclass
class ModelConfig:
    hidden_dim: int = 512
    nonlinearity: str = "relu"  # ["relu", "swish", "gelu"]
    weight_initialization: str = "uniform"  # ["uniform", "normal"]
    layernorm_normalization: bool = False


class ReducedLayerNorm(Module):
    def __init__(self, config: ModelConfig, hidden_dim: int) -> None:
        super().__init__()
        self.weights = Parameter(ones(1, hidden_dim))
        self.layernorm_normalization = config.layernorm_normalization

    def forward(self, inputs: Tensor) -> Tensor:
        if self.layernorm_normalization:
            variance = inputs.pow(2).mean(-1, keepdim=True)
            inputs = inputs * (variance + 1e-6).rsqrt()

        return self.weights * inputs


class Dense(Module):
    NONLINEARITIES = {
        "relu": ReLU,
        "swish": SiLU,
        "gelu": GELU,
    }

    def __init__(
        self, config: ModelConfig, in_dim: int, hidden_dim: int, out_dim: int
    ) -> None:
        super().__init__()
        self.layernorm = ReducedLayerNorm(config, in_dim)
        self.linear1 = Linear(in_dim, hidden_dim, bias=False)
        self.nonlinearity = self.NONLINEARITIES[config.nonlinearity]()
        self.linear2 = Linear(hidden_dim, out_dim, bias=False)

        if config.weight_initialization == "uniform":
            # This is the default init
            pass
        elif config.weight_initialization == "normal":
            self.linear1.weight.data.normal_(mean=0.0, std=in_dim ** -0.5)
            self.linear2.weight.data.normal_(mean=0.0, std=hidden_dim ** -0.5)
        else:
            raise ValueError(f"Weight initialization {config.weight_initialization} not recognized")

    def forward(self, inputs: Tensor) -> Tensor:
        inputs = self.layernorm(inputs)
        inputs = self.linear1(inputs)
        inputs = self.nonlinearity(inputs)
        inputs = self.linear2(inputs)
        return inputs


class ReducedAttention(Module):
    def __init__(self, config: ModelConfig, hidden_dim: int) -> None:
        super().__init__()
        self.layernorm = ReducedLayerNorm(config, hidden_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        inputs = self.layernorm(inputs)
        return softmax(inputs, dim=1)


class FinalModel(ModelBase):
    def __init__(self, env: Env, config: ModelConfig) -> None:
        super().__init__(env)

        self.dense = Dense(config, env.D, config.hidden_dim, config.hidden_dim)
        self.attention = ReducedAttention(config, config.hidden_dim)
        self.classification_head = Dense(config, config.hidden_dim, config.hidden_dim, env.C)
        self.final_layer_norm = ReducedLayerNorm(config, env.C)

    def forward(self, inputs: Tensor) -> Tensor:
        inputs = inputs / 100

        center_scores = self.dense(inputs)
        center_probs = self.attention(center_scores)
        logits = self.classification_head(center_probs)
        logits = self.final_layer_norm(logits)

        return ModelOutput(logits=logits)
