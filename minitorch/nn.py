from typing import Tuple

from .tensor import Tensor
from numpy import random


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    new_height = height // kh
    new_width = width // kw

    six_d_tensor = input.contiguous().view(
        batch, channel, new_height, kh, new_width, kw
    )

    permuted_tensor = six_d_tensor.contiguous().permute(0, 1, 2, 4, 3, 5)

    out = permuted_tensor.contiguous().view(
        batch, channel, new_height, new_width, kh * kw
    )

    return out, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled average pooling 2D

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """
    input_tiled, new_height, new_width = tile(input, kernel)
    dim = len(input_tiled.shape) - 1

    return (
        input_tiled.mean(dim=dim)
        .contiguous()
        .view(input.shape[0], input.shape[1], new_height, new_width)
    )


# MAX is in the tensor_functions.py file
# max is tensor.py file but just calls Max.apply anyways


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled max pooling 2D

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """
    input_tiled, new_height, new_width = tile(input, kernel)
    dim = len(input_tiled.shape) - 1
    return input_tiled.max(dim).view(
        input.shape[0], input.shape[1], new_height, new_width
    )


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise, include an argument to turn off

    Args:
    ----
        input: batch x channel x height x width
        rate: probability of dropping a position
        ignore: if True, do not drop any positions

    Returns:
    -------
        Tensor of size batch x channel x height x width

    """
    if ignore:
        return input
    mask = input.zeros(input.shape)
    for pos in input._tensor.indices():
        if random.random() > rate:
            mask[pos] = 1.0

    return input * mask


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax

    Args:
    ----
        input: batch x channel x height x width
        dim: dimension to reduce

    Returns:
    -------
        Tensor of size batch x channel x height x width

    """
    # Subtract the max value in the dimension to prevent overflow
    max_vals = input.max(dim=dim)
    input_sub = input - max_vals
    input_exp = input_sub.exp()
    input_sum = input_exp.sum(dim=dim)
    return input_exp / input_sum


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax

    Args:
    ----
        input: batch x channel x height x width
        dim: dimension to reduce

    Returns:
    -------
        Tensor of size batch x channel x height x width

    """
    # Subtract the max value in the dimension to prevent overflow
    max_vals = input.max(dim=dim)
    input_sub = input - max_vals
    input_exp = input_sub.exp()
    input_sum = input_exp.sum(dim=dim)
    input_log = input_sum.log()
    return input_sub - input_log
