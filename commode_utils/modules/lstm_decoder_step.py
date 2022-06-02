from typing import Tuple, Optional

import torch
from omegaconf import DictConfig
from torch import nn

from commode_utils.modules import LuongAttention
from commode_utils.modules.base_decoder_step import BaseDecoderStep, DecoderState


class LSTMDecoderStep(BaseDecoderStep):
    def __init__(self, config: DictConfig, output_size: int, pad_idx: Optional[int] = None):
        super().__init__()
        self._decoder_num_layers = config.decoder_num_layers

        self._target_embedding = nn.Embedding(output_size, config.embedding_size, padding_idx=pad_idx)

        self._attention = LuongAttention(config.decoder_size)

        self._decoder_lstm = nn.LSTM(
            config.embedding_size,
            config.decoder_size,
            num_layers=config.decoder_num_layers,
            dropout=config.rnn_dropout if config.decoder_num_layers > 1 else 0,
            batch_first=True,
        )
        self._dropout_rnn = nn.Dropout(config.rnn_dropout)

        self._concat_layer = nn.Linear(config.decoder_size * 2, config.decoder_size, bias=False)
        self._norm = nn.LayerNorm(config.decoder_size)
        self._projection_layer = nn.Linear(config.decoder_size, output_size, bias=False)

    def get_initial_state(self, encoder_output: torch.Tensor, attention_mask: torch.Tensor) -> DecoderState:
        initial_state: torch.Tensor = encoder_output.sum(dim=1)  # [batch size; encoder size]
        segment_sizes: torch.Tensor = (attention_mask == 0).sum(dim=1, keepdim=True)  # [batch size; 1]
        initial_state /= segment_sizes  # [batch size; encoder size]
        initial_state = initial_state.unsqueeze(0).repeat(self._decoder_num_layers, 1, 1)
        return initial_state, initial_state

    def forward(
        self,
        input_token: torch.Tensor,
        encoder_output: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_state: DecoderState,
    ) -> Tuple[torch.Tensor, torch.Tensor, DecoderState]:
        h_prev, c_prev = decoder_state

        # [batch size; 1; embedding size]
        embedded = self._target_embedding(input_token).unsqueeze(1)

        # hidden -- [n layers; batch size; decoder size]
        # output -- [batch size; 1; decoder size]
        rnn_output, (h_prev, c_prev) = self._decoder_lstm(embedded, (h_prev, c_prev))
        rnn_output = self._dropout_rnn(rnn_output)

        # [batch size; context size]
        attn_weights = self._attention(h_prev[-1], encoder_output, attention_mask)

        # [batch size; 1; decoder size]
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_output)

        # [batch size; 2 * decoder size]
        concat_input = torch.cat([rnn_output, context], dim=2).squeeze(1)

        # [batch size; decoder size]
        concat = self._concat_layer(concat_input)
        concat = self._norm(concat)
        concat = torch.tanh(concat)

        # [batch size; vocab size]
        output = self._projection_layer(concat)

        return output, attn_weights, (h_prev, c_prev)
