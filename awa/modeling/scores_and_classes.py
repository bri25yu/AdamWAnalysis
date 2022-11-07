from torch import Tensor, from_numpy
from torch.linalg import norm
from torch.nn import Linear, Module
from torch.nn.functional import relu

from awa.infra import Env


__all__ = ["ScoresAndClassesModel"]


class ScoresAndClassesModel(Module):
    """
    This model is required to learn:
    1) Scores closer to 1 are better
    2) Which class each center corresponds to

    """

    def __init__(self, env: Env) -> None:
        super().__init__()
        self.env = env

        self.num_centers = env.centers.shape[0]
        self.centers_normalized = from_numpy(env.centers)  # (num_centers, D)
        self.centers_normalized = self.centers_normalized / (norm(self.centers_normalized, dim=1, keepdim=True) ** 2)
        assert self.centers_normalized.size() == (self.num_centers, env.D)

        self.o = Linear(self.num_centers, self.num_centers)
        self.score_map = Linear(self.num_centers, self.num_centers)

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

        cross_attn_scores = inputs @ self.centers_normalized.T
        assert cross_attn_scores.size() == (batch_size, num_centers)

        # 1) Scores closer to 1 are better
        cross_attn_output = self.o(cross_attn_scores)  # (batch_size, num_centers)
        cross_attn_output = relu(cross_attn_output)
        cross_attn_output = self.score_map(cross_attn_output)  # (batch_size, num_centers)

        # 2) Which class each center corresponds to
        outputs = self.classification_head(cross_attn_output)  # (batch_size, n_classes)

        return outputs
