from torch import Tensor, from_numpy, softmax, norm, rand, ones
from torch.nn import Parameter, Linear, Sequential, ReLU

from awa.infra import Env
from awa.modeling.base import ModelBase, ModelOutput


__all__ = ["AbsModel"]


class AbsModel(ModelBase):
    def __init__(self, env: Env) -> None:
        super().__init__(env)

        self.num_centers = env.centers.shape[0]

        centers_normalized = from_numpy(env.centers)  # (num_centers, D)
        centers_normalized = centers_normalized / (norm(centers_normalized, dim=1, keepdim=True) ** 2)
        assert centers_normalized.size() == (self.num_centers, env.D)
        self.centers_normalized = Parameter(centers_normalized, requires_grad=False)

        self.abs_network = Sequential(
            Linear(self.num_centers, self.num_centers, bias=False),
            ReLU(),
            Linear(self.num_centers, self.num_centers, bias=False),
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

        center_scores = -self.scale * self.abs_network(dot_products - self.offset)
        center_probs = softmax(center_scores, dim=1)  # (batch_size, num_centers)

        logits = self.center_logits(center_probs)  # (batch_size, C)

        return ModelOutput(
            logits=logits,
            logs={
                "Offset": self.offset.data,
                "Scale": self.scale.data,
            }
        )
