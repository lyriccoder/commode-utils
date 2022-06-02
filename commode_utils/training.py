from typing import List, Tuple

import torch


def segment_sizes_to_slices(sizes: torch.Tensor) -> List:
    cum_sums = torch.cumsum(sizes, dim=0)
    slices = [slice(0, cum_sums[0])]
    slices += [slice(start, end) for start, end in zip(cum_sums[:-1], cum_sums[1:])]
    return slices


def cut_into_segments(
    data: torch.Tensor, sample_sizes: torch.LongTensor, mask_value: float = -1e9
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cut data from common tensor into samples part with padding to longest sequence.
    Also return mask with value on pad places.
    :param data: [n contexts; units]
    :param sample_sizes: [batch size]
    :param mask_value:
    :return: [batch size; max context len; units], [batch size; max context len]
    """
    batch_size = len(sample_sizes)
    max_context_len = max(sample_sizes)

    batched_contexts = data.new_zeros((batch_size, max_context_len, data.shape[-1]))
    attention_mask = data.new_zeros((batch_size, max_context_len))

    context_slices = segment_sizes_to_slices(sample_sizes)
    for i, (cur_slice, cur_size) in enumerate(zip(context_slices, sample_sizes)):
        batched_contexts[i, :cur_size] = data[cur_slice]
        attention_mask[i, cur_size:] = mask_value

    return batched_contexts, attention_mask
