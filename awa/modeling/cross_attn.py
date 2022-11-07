from torch import Tensor, matmul, softmax, relu
from torch.nn import Linear, Module

from awa.infra import Env


__all__ = ["CrossAttnModel"]


class CrossAttnModel(Module):
    def __init__(self, env: Env) -> None:
        super().__init__()
        self.env = env

        self.hidden_dim = 64

        self.input_transform = Linear(env.D, self.hidden_dim)

        self.q = Linear(self.hidden_dim, self.hidden_dim)
        self.k = Linear(self.hidden_dim, self.hidden_dim)
        self.v = Linear(self.hidden_dim, self.hidden_dim)
        self.o = Linear(self.hidden_dim, self.hidden_dim)

        self.classification_head = Linear(self.max_n_centers, env.C)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        inputs is a tensor of shape (batch_size, D)
        outputs is a tensor of shape (batch_size, C)
        """
        batch_size = inputs.size()[0]
        env = self.env
        hidden_dim = self.hidden_dim

        assert inputs.size() == (batch_size, env.D)

        inputs = inputs.unsqueeze(1).tile((1, hidden_dim, 1))
        assert inputs.size() == (batch_size, hidden_dim, env.D)
        inputs = self.input_transform(inputs)  # (batch_size, hidden_dim, hidden_dim)
        inputs = relu(inputs)

        query_states = self.q(inputs)  # (batch_size, hidden_dim, hidden_dim)
        key_states = self.k(inputs)  # (batch_size, hidden_dim, hidden_dim)
        value_states = self.v(inputs)  # (batch_size, hidden_dim, hidden_dim)

        cross_attn_scores = matmul(query_states, key_states.T)
        assert cross_attn_scores.size() == (batch_size, hidden_dim, hidden_dim)

        cross_attn_weights = softmax(cross_attn_scores, dim=2)  # (batch_size, hidden_dim, hidden_dim)

        cross_attn_output = matmul(cross_attn_weights, value_states)  # (batch_size, hidden_dim, hidden_dim)
        cross_attn_output = cross_attn_output.sum(dim=2)  # (batch_size, hidden_dim)
        cross_attn_output = self.o(cross_attn_output)  # (batch_size, hidden_dim)
        cross_attn_output = relu(cross_attn_output)

        outputs = self.classification_head(cross_attn_output)  # (batch_size, n_classes)

        return outputs
