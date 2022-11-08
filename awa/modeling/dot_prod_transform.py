from torch import Tensor, from_numpy, softmax, norm
from torch.nn import Module, Parameter, Linear, Sequential, ReLU

from awa.infra import Env


__all__ = ["DotProdTransformModel"]


class DotProdTransformModel(Module):
    def __init__(self, env: Env) -> None:
        super().__init__()
        self.env = env

        self.num_centers = env.centers.shape[0]

        centers_normalized = from_numpy(env.centers)  # (num_centers, D)
        centers_normalized = centers_normalized / (norm(centers_normalized, dim=1, keepdim=True) ** 2)
        assert centers_normalized.size() == (self.num_centers, env.D)
        self.centers_normalized = Parameter(centers_normalized, requires_grad=False)

        self.center_logits = Sequential(Linear(env.D, env.D), ReLU(), Linear(env.D, env.D), ReLU())

        self.center_logits = Linear(self.num_centers, env.C, bias=False)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        inputs is a tensor of shape (batch_size, D)
        outputs is a tensor of shape (batch_size, C)
        """
        batch_size = inputs.size()[0]
        env = self.env
        num_centers = self.num_centers

        assert inputs.size() == (batch_size, env.D)

        dot_products = inputs @ self.centers_normalized.T
        assert dot_products.size() == (batch_size, num_centers)

        closest_to_1 = -self.scale * (dot_products - self.offset).abs()  # Target is 0, (batch_size, num_centers)
        center_probs = softmax(closest_to_1, dim=1)  # (batch_size, num_centers)

        logits = self.center_logits(center_probs)  # (batch_size, C)

        return logits
