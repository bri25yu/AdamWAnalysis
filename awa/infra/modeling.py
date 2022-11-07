from dataclasses import dataclass

from torch import Tensor, matmul
from torch.nn import Linear, Module
from torch.nn.functional import softmax, dropout, relu


__all__ = ["KMeansConfig", "KMeansModel"]


@dataclass
class KMeansConfig:
    input_dim: int
    num_classes: int
    hidden_dim: int
    n_heads: int
    head_dim: int
    dropout: float = 0.1

    @property
    def attn_dim(self) -> int:
        return self.n_heads * self.head_dim


class KMeansModel(Module):
    def __init__(self, config: KMeansConfig) -> None:
        super().__init__()
        self.config = config

        # Transform inputs to hidden_dim
        self.input_transformation = Linear(config.input_dim, config.hidden_dim)

        # Cross attention
        self.q = Linear(config.hidden_dim, config.head_dim, bias=False)
        self.k = Linear(config.hidden_dim, config.attn_dim, bias=False)
        self.v = Linear(config.hidden_dim, config.attn_dim, bias=False)
        self.o = Linear(config.head_dim, config.hidden_dim, bias=False)

        self.classification_head = Linear(config.hidden_dim, config.num_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        inputs is a tensor of shape (batch_size, input_dim)
        outputs is a tensor of shape (batch_size, num_classes)
        """
        config = self.config
        batch_size = inputs.size()[0]

        inputs = self.input_transformation(inputs)  # (batch_size, hidden_dim)
        inputs = relu(inputs)

        query_states = self.q(inputs)  # (batch_size, head_dim)
        key_states = self.k(inputs).view(batch_size, config.n_heads, config.head_dim)
        value_states = self.v(inputs).view(batch_size, config.n_heads, config.head_dim)

        cross_attn_scores = matmul(query_states.unsqueeze(1), key_states.transpose(1, 2)).squeeze(1)
        assert cross_attn_scores.size() == (batch_size, config.n_heads)

        cross_attn_weights = softmax(cross_attn_scores, dim=1)  # (batch_size, n_heads)
        cross_attn_weights = dropout(cross_attn_weights, p=config.dropout, training=self.training)

        cross_attn_output = matmul(cross_attn_weights.unsqueeze(1), value_states).squeeze(1)
        assert cross_attn_output.size() == (batch_size, config.head_dim)

        cross_attn_output = self.o(cross_attn_output)  # (batch_size, hidden_dim)
        cross_attn_output = relu(cross_attn_output)

        outputs = self.classification_head(cross_attn_output)  # (batch_size, n_classes)

        return outputs
