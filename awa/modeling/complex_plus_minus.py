from torch import Tensor, from_numpy, softmax, norm, rand, ones
from torch.nn import Parameter, Linear, Sequential, ReLU, Dropout

from awa.infra import Env
from awa.modeling.base import ModelBase, ModelOutput


__all__ = ["ComplexPlusMinusModel"]


class ComplexPlusMinusModel(ModelBase):
    def __init__(self, env: Env) -> None:
        super().__init__(env)

        self.num_centers = env.centers.shape[0]

        centers_normalized = from_numpy(env.centers)  # (num_centers, D)
        centers_normalized = centers_normalized / (norm(centers_normalized, dim=1, keepdim=True) ** 2)
        assert centers_normalized.size() == (self.num_centers, env.D)
        self.centers_normalized = Parameter(centers_normalized, requires_grad=False)

        self.plus_minus = Sequential(
            Linear(2, 4, bias=False),
            Dropout(p=0.1),
            ReLU(),
            Linear(4, 2, bias=False),
        )

        self.scale = Parameter(ones((1,)))
        self.offset = Parameter(rand((1,)))

        self.center_logits = Linear(self.num_centers, env.C, bias=False)

    def forward(self, inputs: Tensor) -> Tensor:
        batch_size = inputs.size()[0]
        env = self.env
        num_centers = self.num_centers

        assert inputs.size() == (batch_size, env.D)

        dot_products = inputs @ self.centers_normalized.T
        assert dot_products.size() == (batch_size, num_centers)

        dot_products = dot_products - self.offset

        # Learned abs
        dot_products = self.plus_minus(dot_products)  # (batch_size, num_centers)

        center_probs = softmax(-self.scale * dot_products, dim=1)  # (batch_size, num_centers)

        logits = self.center_logits(center_probs)  # (batch_size, C)

        return ModelOutput(
            logits=logits,
            logs={
                "Offset": self.offset.data,
                "Scale": self.scale.data,
                "Plus minus 0": self.plus_minus.data[0, 0],
                "Plus minus 1": self.plus_minus.data[0, 1],
            }
        )