from torch import Tensor, from_numpy, softmax
from torch.linalg import norm
from torch.nn import Parameter, Linear

from awa.infra import Env
from awa.modeling.base import ModelBase, ModelOutput


__all__ = ["CenterLabelsModel"]


class CenterLabelsModel(ModelBase):
    def __init__(self, env: Env) -> None:
        super().__init__(env)

        self.num_centers = env.centers.shape[0]

        centers_normalized = from_numpy(env.centers)  # (num_centers, D)
        centers_normalized = centers_normalized / (norm(centers_normalized, dim=1, keepdim=True) ** 2)
        assert centers_normalized.size() == (self.num_centers, env.D)
        self.centers_normalized = Parameter(centers_normalized, requires_grad=False)

        self.center_logits = Linear(self.num_centers, env.C, bias=False)

    def forward(self, inputs: Tensor) -> Tensor:
        batch_size = inputs.size()[0]
        env = self.env
        num_centers = self.num_centers

        assert inputs.size() == (batch_size, env.D)

        dot_products = inputs @ self.centers_normalized.T
        assert dot_products.size() == (batch_size, num_centers)

        closest_to_1 = (dot_products - 1).abs()  # Target is 0, (batch_size, num_centers)
        closest_to_1 = closest_to_1 - closest_to_1.min(dim=1, keepdim=True)[0]
        center_probs = softmax(-closest_to_1 * 10000, dim=1)  # (batch_size, num_centers)

        logits = self.center_logits(center_probs)  # (batch_size, C)

        return ModelOutput(
            logits=logits,
            logs={
                "Majority class weight": logits.max(dim=1)[0].mean(),
                "Minority class weight": logits.min(dim=1)[0].mean(),
                "Classification confidence": softmax(logits, dim=1).max(dim=1)[0].mean(),
            }
        )
