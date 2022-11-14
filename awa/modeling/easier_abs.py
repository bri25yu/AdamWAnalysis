from torch import Tensor, softmax, rand, ones, relu, randn, norm
from torch.nn import Parameter, Linear

from awa.infra import Env
from awa.modeling.base import ModelBase, ModelOutput


__all__ = ["EasierAbsModel"]


class EasierAbsModel(ModelBase):
    def __init__(self, env: Env) -> None:
        super().__init__(env)

        self.num_centers = 2048

        # We manually initialize centers to be on the scale of [-100, 100]
        self.centers = Parameter(100 * randn((self.num_centers, 2)))

        self.scale = Parameter(ones((1,)))
        self.offset = Parameter(rand((1,)))

        self.center_logits = Linear(self.num_centers, env.C, bias=False)

    def forward(self, inputs: Tensor) -> Tensor:
        batch_size = inputs.size()[0]
        env = self.env
        num_centers = self.num_centers
        centers = self.centers

        assert inputs.size() == (batch_size, env.D)

        centers = centers / norm(centers, dim=1, keepdim=True).pow(2)

        dot_products = inputs @ centers.T
        assert dot_products.size() == (batch_size, num_centers)

        center_scores = dot_products - self.offset
        center_scores = relu(center_scores)
        center_probs = softmax(-self.scale * center_scores, dim=1)  # (batch_size, num_centers)

        logits = self.center_logits(center_probs)  # (batch_size, C)

        return ModelOutput(
            logits=logits,
            logs={
                "Offset": self.offset.data,
                "Scale": self.scale.data,
            }
        )
