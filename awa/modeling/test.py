from torch import Tensor, softmax, rand, ones, relu, randn, norm
from torch.nn import Parameter, Linear

from awa.infra import Env
from awa.modeling.base import ModelBase, ModelOutput


__all__ = ["TestModel"]


class TestModel(ModelBase):
    def __init__(self, env: Env) -> None:
        super().__init__(env)

        self.num_centers = 2048

        self.centers = Parameter(randn((self.num_centers, env.D)))
        self.centers_dense = Linear(env.D, env.D, bias=False)

        # We manually initialize plus minus to have positive and negative values
        # We multiply by 0.4 to be farther away from 0, but not as strong as 1
        self.plus_minus = Parameter(0.4 * Tensor([-1.0, 1.0]).reshape((1, 2)))

        self.scale = Parameter(ones((1,)))
        self.offset = Parameter(rand((1,)))

        self.center_logits = Linear(self.num_centers, env.C, bias=False)

    def forward(self, inputs: Tensor) -> Tensor:
        batch_size = inputs.size()[0]
        env = self.env
        num_centers = self.num_centers
        centers = self.centers

        assert inputs.size() == (batch_size, env.D)

        centers = self.centers_dense(centers)

        dot_products = inputs @ centers.T
        assert dot_products.size() == (batch_size, num_centers)

        # Learned abs
        center_scores = dot_products - self.offset
        center_scores = center_scores.unsqueeze(2)  # (batch_size, num_centers, 1)
        center_scores = center_scores @ self.plus_minus  # (batch_size, num_centers, 2)
        center_scores = relu(center_scores).sum(dim=2)  # (batch_size, num_centers)

        center_probs = softmax(-self.scale * center_scores, dim=1)  # (batch_size, num_centers)

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
