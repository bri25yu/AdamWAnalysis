from torch import Tensor, rand
from torch.nn import Linear, Module, Parameter
from torch.nn.functional import relu

from awa.infra import Env


__all__ = ["NumCentersModel"]


class NumCentersModel(Module):
    """
    This model is required to learn:
    1) Location of centers
    2) Center normalization
    3) Scores closer to 1 are better
    4) Which class each center corresponds to

    """

    def __init__(self, env: Env) -> None:
        super().__init__()
        self.env = env

        self.num_centers = env.centers.shape[0]
        self.centers = Parameter(rand((self.num_centers, env.D)))

        self.center_normalization = Linear(env.D, env.D)

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

        # 1) Location of centers
        centers = self.centers

        # 2) Center normalization
        centers_normalized = self.center_normalization(centers)

        cross_attn_scores = inputs @ centers_normalized.T
        assert cross_attn_scores.size() == (batch_size, num_centers)

        # 3) Scores closer to 1 are better
        cross_attn_output = self.o(cross_attn_scores)  # (batch_size, num_centers)
        cross_attn_output = relu(cross_attn_output)
        cross_attn_output = self.score_map(cross_attn_output)  # (batch_size, num_centers)

        # 4) Which class each center corresponds to
        outputs = self.classification_head(cross_attn_output)  # (batch_size, n_classes)

        return outputs
