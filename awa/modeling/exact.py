from torch import Tensor, from_numpy, zeros
from torch.linalg import norm
from torch.nn import Module, Parameter

from awa.infra import Env


__all__ = ["ExactModel"]


class ExactModel(Module):
    def __init__(self, env: Env) -> None:
        super().__init__()
        self.env = env

        self.num_centers = env.centers.shape[0]

        centers_normalized = from_numpy(env.centers)  # (num_centers, D)
        centers_normalized = centers_normalized / (norm(centers_normalized, dim=1, keepdim=True) ** 2)
        assert centers_normalized.size() == (self.num_centers, env.D)
        self.centers_normalized = Parameter(centers_normalized, requires_grad=False)

        center_labels = from_numpy(env.center_labels)  # (num_centers,)
        self.center_labels = Parameter(center_labels, requires_grad=False)

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
        closest_center_idxs = closest_to_1.argmin(dim=1)  # (batch_size,)
        classes = self.center_labels[closest_center_idxs]  # (batch_size,)

        logits = zeros((batch_size, self.env.C), device=classes.device)
        logits.scatter_(1, classes.unsqueeze(1), 1)
        logits.requires_grad_()

        return logits
