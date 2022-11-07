from torch import Tensor, rand
from torch.nn import Linear, Module, Parameter
from torch.nn.functional import relu

from awa.infra import Env


__all__ = ["EnvDimAndCentersModel"]


class EnvDimAndCentersModel(Module):
    """
    This model is required to learn:
    1) Number of centers
    2) Location of centers
    3) Center normalization
    4) Scores closer to 1 are better
    5) Which class each center corresponds to

    """

    def __init__(self, env: Env) -> None:
        super().__init__()
        self.env = env

        self.max_n_centers = 1024
        self.centers = Parameter(rand((self.max_n_centers, env.D)))

        self.center_normalization = Linear(env.D, env.D)

        self.o = Linear(self.max_n_centers, self.max_n_centers)
        self.score_map = Linear(self.max_n_centers, self.max_n_centers)

        self.classification_head = Linear(self.max_n_centers, env.C)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        inputs is a tensor of shape (batch_size, D)
        outputs is a tensor of shape (batch_size, C)
        """
        batch_size = inputs.size()[0]
        env = self.env
        max_n_centers = self.max_n_centers

        assert inputs.size() == (batch_size, env.D)

        # 1) Number of centers
        # 2) Location of centers
        centers = self.centers

        # 3) Center normalization
        centers_normalized = self.center_normalization(centers)

        cross_attn_scores = inputs @ centers_normalized.T
        assert cross_attn_scores.size() == (batch_size, max_n_centers)

        # 4) Scores closer to 1 are better
        cross_attn_output = self.o(cross_attn_scores)  # (batch_size, max_n_centers)
        cross_attn_output = relu(cross_attn_output)
        cross_attn_output = self.score_map(cross_attn_output)  # (batch_size, max_n_centers)

        # 5) Which class each center corresponds to
        outputs = self.classification_head(cross_attn_output)  # (batch_size, n_classes)

        return outputs
