from torch import Tensor, from_numpy, softmax
from torch.linalg import norm
from torch.nn import Linear, Module, Parameter

from awa.infra import Env


__all__ = ["ClassesModel"]


class ClassesModel(Module):
    """
    This model is required to learn:
    1) Which class each center corresponds to

    """

    def __init__(self, env: Env) -> None:
        super().__init__()
        self.env = env

        self.num_centers = env.centers.shape[0]

        centers_normalized = from_numpy(env.centers)  # (num_centers, D)
        centers_normalized = centers_normalized / (norm(centers_normalized, dim=1, keepdim=True) ** 2)
        assert centers_normalized.size() == (self.num_centers, env.D)
        self.centers_normalized = Parameter(centers_normalized, requires_grad=False)

        self.classification_head = Linear(self.num_centers, env.C)

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

        closest_to_1 = (dot_products - 1) ** 2  # Target is 0, (batch_size, num_centers)
        cross_attn = softmax(-closest_to_1, dim=1)  # (batch_size, num_centers)

        # 1) Which class each center corresponds to
        outputs = self.classification_head(cross_attn)  # (batch_size, n_classes)

        return outputs
