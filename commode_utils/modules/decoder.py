from typing import Tuple

import torch
from torch import nn, Tensor

from commode_utils.modules.base_decoder_step import BaseDecoderStep
from commode_utils.training import cut_into_segments
import numpy as np
from collections import defaultdict

class Decoder(nn.Module):

    _negative_value = -1e9

    def __init__(self, decoder_step: BaseDecoderStep, output_size: int, sos_token: int, teacher_forcing: float = 0.0):
        super().__init__()
        self._decoder_step = decoder_step
        self._teacher_forcing = teacher_forcing
        self._out_size = output_size
        self._sos_token = sos_token
    
    
    def forward(
        self,
        encoder_output: torch.Tensor,
        segment_sizes: torch.LongTensor,
        output_size: int,
        target_sequence: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate output sequence based on encoder output

        :param encoder_output: [n sequences; encoder size] -- stacked encoder output
        :param segment_sizes: [batch size] -- size of each segment in encoder output
        :param output_size: size of output sequence
        :param target_sequence: [batch size; max seq len] -- if passed can be used for teacher forcing
        :return:
            [output size; batch size; vocab size] -- sequence with logits for each position
            [output size; batch size; encoder seq length] -- sequence with attention weights for each position
        """
        batch_size = segment_sizes.shape[0]
        # encoder output -- [batch size; max context len; units]
        # attention mask -- [batch size; max context len]
        #weights = torch.squeeze(encoder_output)
        #print(f'encoder_output1 {weights} size {weights.shape}')
        
        batched_encoder_output, attention_mask = cut_into_segments(encoder_output, segment_sizes, self._negative_value)
        #print(f'batched_encoder_output {batched_encoder_output} {batched_encoder_output.shape}')
        decoder_state = self._decoder_step.get_initial_state(batched_encoder_output, attention_mask)
        #distance_matrix = dijkstras_matrix_gpu(build_weighted_graph_gpu(X))
        #print(f'distance_matrix  {distance_matrix}')
        
        # [output size; batch size; vocab size]
        output = batched_encoder_output.new_zeros((output_size, batch_size, self._out_size))
        #print('begin output ', output)
        #print('size', output.shape)
        output[0, :, self._sos_token] = 1
        
        output_probs = []
        
        #print('after output ', output)
        #print('after size', output.shape)
        # [output size; batch size; encoder seq size]
        attentions = batched_encoder_output.new_zeros((output_size, batch_size, attention_mask.shape[1]))

        # [batch size]
        current_input = batched_encoder_output.new_full((batch_size,), self._sos_token, dtype=torch.long)
        print('first current_input', current_input)
        print('first current_input size', current_input.shape)
        
        #if self.training and target_sequence is not None and torch.rand(1) <= self._teacher_forcing:
        for step in range(1, output_size):
            current_output, current_attention, decoder_state = self._decoder_step(
                current_input, batched_encoder_output, attention_mask, decoder_state
            )
            if step == 1:
                print('Random number', current_output[0][0:10], current_output.shape)
            print(f'before current_input {current_input}')
            #print(f'before current_output {current_output}')
            #print(f'before current_output size {current_output.shape}')
            
            
            output[step] = current_output
            attentions[step] = current_attention
            if self.training and target_sequence is not None and torch.rand(1) <= self._teacher_forcing:
                current_input = target_sequence[step]
            else:
                #print(output)
                #print(output.shape)
                current_input = output[step].argmax(dim=-1)
                #max_val = output[step].max().item()
                #output_probs.append(max_val)
                #print(f'after current_input {current_input}, max {max_val}')
                #print(output[step], '\n#################################')

        #print(output_probs)
        #else:
            #_beam_search(self._decoder_step, current_input, batched_encoder_output, attention_mask, decoder_state)
        return output, attentions
