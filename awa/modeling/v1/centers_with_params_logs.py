from torch import Tensor, softmax, rand, ones, relu, randn, norm
from torch.nn import Parameter, Linear

from awa.infra import Env
from awa.modeling.base import ModelBase, ModelOutput


__all__ = ["CentersWithParamsModel"]


class CentersWithParamsModel(ModelBase):
    def __init__(self, env: Env) -> None:
        super().__init__(env)

        num_centers = 2048

        # We manually initialize centers to be on the scale of [-100, 100]
        self.centers = Parameter(100 * randn((num_centers, 2)))

        # We manually initialize plus minus to have positive and negative values
        # We multiply by 0.4 to be farther away from 0, but not as strong as 1
        self.plus_minus = Parameter(0.4 * Tensor([-1.0, 1.0]).reshape((1, 2)))

        self.scale = Parameter(ones((1,)))
        self.offset = Parameter(rand((1,)))

        self.center_logits = Linear(num_centers, env.C, bias=False)

    def forward(self, inputs: Tensor) -> Tensor:
        centers = self.centers

        centers = centers / norm(centers, dim=1, keepdim=True).pow(2)

        dot_products = inputs @ centers.T

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
                "Centers": self.centers.data,
            }
        )
