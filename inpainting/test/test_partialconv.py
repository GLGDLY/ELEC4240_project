import numpy as np
import torch
from torch_partialconv2d import PartialConv2d as torch_PartialConv2d

from partialconv import PartialConv2D


def test_partialconv():
    # Generate data
    x = np.random.randn(1, 3, 256, 256).astype(np.float32)
    mask = np.random.randint(0, 2, (1, 1, 256, 256)).astype(np.float32)

    # Torch
    torch_partial_conv2d = torch_PartialConv2d(
        3, 256, kernel_size=3, padding=1  # , multi_channel=True
    )
    torch_partial_conv2d.weight.data = torch.randn_like(torch_partial_conv2d.weight)
    torch_partial_conv2d.bias.data = torch.randn_like(torch_partial_conv2d.bias)
    torch_output = torch_partial_conv2d(torch.tensor(x), torch.tensor(mask))

    # Tensorflow
    partial_conv2d = PartialConv2D(
        256, kernel_size=(3, 3), padding="same"  # , multi_channel=True
    )
    partial_conv2d.build((None, 256, 256, 3))
    partial_conv2d.kernel.assign(
        np.transpose(torch_partial_conv2d.weight.detach().numpy(), (2, 3, 1, 0))
    )
    partial_conv2d.bias.assign(torch_partial_conv2d.bias.detach().numpy())

    output = partial_conv2d(
        np.transpose(x, (0, 2, 3, 1)), np.transpose(mask, (0, 2, 3, 1))
    )

    # Compare
    output_t = np.transpose(
        output.numpy(), (0, 3, 1, 2)
    )  # Channels-first for comparison

    np.testing.assert_allclose(output_t, torch_output.detach().numpy(), atol=1e-5)


if __name__ == "__main__":
    test_partialconv()
