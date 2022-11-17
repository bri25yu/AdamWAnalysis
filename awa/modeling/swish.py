from torch import Tensor, softmax, ones
from torch.nn import Parameter, Linear, SiLU

from awa.infra import Env
from awa.modeling.base import ModelBase, ModelOutput


__all__ = ["SwishModel"]


class SwishModel(ModelBase):
    def __init__(self, env: Env) -> None:
        super().__init__(env)

        num_centers = 2048

        self.centers = Linear(env.D, num_centers)
        self.activation = SiLU()

        self.scale = Parameter(ones((1, num_centers)))

        self.center_logits = Linear(num_centers, env.C, bias=False)

    def forward(self, inputs: Tensor) -> Tensor:
        inputs = inputs / 100

        dot_products = self.centers(inputs)

        center_scores = self.activation(dot_products)
        center_probs = softmax(-self.scale * center_scores, dim=1)  # (batch_size, num_centers)

        logits = self.center_logits(center_probs)  # (batch_size, C)

        return ModelOutput(
            logits=logits,
            logs={
                "Scale": self.scale.data.mean(),
                "Centers": self.centers.weight.data.T,
            }
        )
