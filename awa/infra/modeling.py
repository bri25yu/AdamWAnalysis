from copy import deepcopy

from torch import Tensor
from torch.nn import Linear

from transformers import T5Config
from transformers.models.t5.modeling_t5 import T5Stack, T5PreTrainedModel


__all__ = ["T5ForClassification"]


class T5ForClassification(T5PreTrainedModel):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        config = T5Config()  # We just use the default T5 base config
        super().__init__(config)

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.d_model = config.d_model

        self.input_ffn = Linear(input_dim, config.d_model)
        self.output_ffn = Linear(config.d_model, num_classes)

        # This is an exact copy of `T5Model.__init__`
        encoder_config = deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config)

        decoder_config = deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, inputs: Tensor) -> Tensor:
        """
        inputs is a tensor of shape (batch_size, input_dim)
        outputs is a tensor of shape (batch_size, num_classes)
        """
        batch_size = inputs.size()[0]

        assert inputs.size() == (batch_size, self.input_dim)
        inputs = self.input_ffn(inputs)
        assert inputs.size() == (batch_size, self.d_model)
        inputs = inputs.unsqueeze(dim=1)
        assert inputs.size() == (batch_size, 1, self.d_model)

        encoder_outputs = self.encoder(inputs_embeds=inputs)
        last_hidden_state = encoder_outputs.last_hidden_state
        assert last_hidden_state.size() == (batch_size, 1, self.d_model)

        decoder_outputs = self.decoder(inputs_embeds=last_hidden_state)
        last_hidden_state: Tensor = decoder_outputs.last_hidden_state
        assert last_hidden_state.size() == (batch_size, 1, self.d_model)

        last_hidden_state = last_hidden_state.squeeze(dim=1)
        assert last_hidden_state.size() == (batch_size, self.d_model)

        outputs = self.output_ffn(last_hidden_state)
        assert outputs.size() == (batch_size, self.num_classes)

        return outputs
