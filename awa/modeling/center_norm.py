from torch import Tensor, from_numpy, softmax, norm, rand, ones
from torch.nn import Parameter, Linear

from awa.infra import Env
from awa.modeling.base import ModelBase, ModelOutput


__all__ = ["CenterNormModel"]


class CenterNormModel(ModelBase):
    def __init__(self, env: Env) -> None:
        super().__init__(env)

        self.num_centers = env.centers.shape[0]

        self.centers = Parameter(from_numpy(env.centers), requires_grad=False)  # (num_centers, D)
        self.center_norm = Parameter(ones((1, self.num_centers)))

        self.scale = Parameter(ones((1,)))
        self.offset = Parameter(rand((1,)))

        self.center_logits = Linear(self.num_centers, env.C, bias=False)

    def forward(self, inputs: Tensor) -> Tensor:
        batch_size = inputs.size()[0]
        env = self.env
        num_centers = self.num_centers
        centers = self.centers

        assert inputs.size() == (batch_size, env.D)

        # Rescale inputs from [-100, 100] to [-1, 1]
        inputs = inputs / 100

        # Rescale centers from [-100, 100] to [-1, 1] for better conditioning on center_norm
        centers = centers / 100
        centers = centers * (self.center_norm ** 2)

        dot_products = inputs @ centers.T
        assert dot_products.size() == (batch_size, num_centers)

        closest_to_1 = -self.scale * (dot_products - self.offset).abs()  # Target is 0, (batch_size, num_centers)
        center_probs = softmax(closest_to_1, dim=1)  # (batch_size, num_centers)

        logits = self.center_logits(center_probs)  # (batch_size, C)

        return ModelOutput(
            logits=logits,
            logs={
                "Offset": self.offset.data,
                "Scale": self.scale.data,
                "value_centers_max_avg": centers.max(dim=1)[0].mean(),
            }
        )
