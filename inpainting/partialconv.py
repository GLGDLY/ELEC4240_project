# This file is modified from https://github.com/NVIDIA/partialconv/blob/master/models/partialconv2d.py
import tensorflow as tf


class PartialConv2D(tf.keras.layers.Conv2D):
    def __init__(self, *args, multi_channel=False, return_mask=False, **kwargs):
        self.multi_channel = multi_channel
        self.return_mask = return_mask
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)

        kernel_height, kernel_width = self.kernel_size
        in_channels = input_shape[-1]

        if self.multi_channel:
            self.weight_mask_updater = tf.ones(
                (kernel_height, kernel_width, in_channels, self.filters),
                dtype=self.dtype,
            )
            self.slide_winsize = kernel_height * kernel_width * in_channels
        else:
            self.weight_mask_updater = tf.ones(
                (kernel_height, kernel_width, 1, 1), dtype=self.dtype
            )
            self.slide_winsize = kernel_height * kernel_width * 1

    def call(self, input_tensor, mask_in=None):
        input_shape = tf.shape(input_tensor)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]

        if mask_in is None:
            if self.multi_channel:
                mask = tf.ones(
                    (batch_size, height, width, input_shape[-1]),
                    dtype=input_tensor.dtype,
                )
            else:
                mask = tf.ones((batch_size, height, width, 1), dtype=input_tensor.dtype)
        else:
            mask = mask_in

        update_mask = tf.nn.conv2d(
            mask,
            self.weight_mask_updater,
            strides=self.strides,
            padding=self.padding.upper(),
            dilations=self.dilation_rate,
        )

        update_mask_clamped = tf.clip_by_value(update_mask, 0.0, 1.0)

        epsilon = 1e-8
        mask_ratio = (
            self.slide_winsize / (update_mask + epsilon)
        ) * update_mask_clamped

        if mask_in is not None:
            masked_input = input_tensor * mask
        else:
            masked_input = input_tensor

        raw_output = super().call(masked_input)

        if self.use_bias:
            bias = tf.reshape(self.bias, (1, 1, 1, self.filters))
            output = (raw_output - bias) * mask_ratio + bias
        else:
            output = raw_output * mask_ratio

        output *= update_mask_clamped

        return (output, update_mask_clamped) if self.return_mask else output


class StandardConv2DWithMaskUpdate(tf.keras.layers.Conv2D):
    def __init__(self, *args, multi_channel=False, return_mask=False, **kwargs):
        self.multi_channel = multi_channel
        self.return_mask = return_mask
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)

        kernel_height, kernel_width = self.kernel_size
        in_channels = input_shape[-1]

        if self.multi_channel:
            self.weight_mask_updater = tf.ones(
                (kernel_height, kernel_width, in_channels, self.filters),
                dtype=self.dtype,
            )
        else:
            self.weight_mask_updater = tf.ones(
                (kernel_height, kernel_width, 1, 1), dtype=self.dtype
            )

    def call(self, input_tensor, mask_in=None):
        input_shape = tf.shape(input_tensor)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]

        if mask_in is None:
            if self.multi_channel:
                mask = tf.ones(
                    (batch_size, height, width, input_shape[-1]),
                    dtype=input_tensor.dtype,
                )
            else:
                mask = tf.ones((batch_size, height, width, 1), dtype=input_tensor.dtype)
        else:
            mask = mask_in

        update_mask = tf.nn.conv2d(
            mask,
            self.weight_mask_updater,
            strides=self.strides,
            padding=self.padding.upper(),
            dilations=self.dilation_rate,
        )

        update_mask_clamped = tf.clip_by_value(update_mask, 0.0, 1.0)

        output = super().call(input_tensor)

        return (output, update_mask_clamped) if self.return_mask else output
