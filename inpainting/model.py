from enum import Enum
from typing import Tuple

import tensorflow as tf

from partialconv import PartialConv2D, StandardConv2DWithMaskUpdate


def p_downsample(
    in_channels: int,
    out_channels: int,
    size: int,
    apply_batchnorm: bool = True,
    stride=2,
):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    in_img = tf.keras.layers.Input(shape=[None, None, in_channels])
    in_mask = tf.keras.layers.Input(shape=[None, None, 1])

    x, mask = PartialConv2D(
        filters=out_channels,
        kernel_size=size,
        strides=stride,
        padding="same",
        return_mask=True,
        kernel_initializer=initializer,
        use_bias=False,
    )(in_img, in_mask)

    if apply_batchnorm:
        x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.LeakyReLU()(x)

    return tf.keras.Model(inputs=[in_img, in_mask], outputs=[x, mask])


def downsample_with_update_mask(
    in_channels: int,
    out_channels: int,
    size: int,
    apply_batchnorm: bool = True,
    stride=2,
):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    in_img = tf.keras.layers.Input(shape=[None, None, in_channels])
    in_mask = tf.keras.layers.Input(shape=[None, None, 1])

    x, mask = StandardConv2DWithMaskUpdate(
        filters=out_channels,
        kernel_size=size,
        strides=stride,
        padding="same",
        return_mask=True,
        kernel_initializer=initializer,
        use_bias=False,
    )(in_img, in_mask)

    if apply_batchnorm:
        x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.LeakyReLU()(x)

    return tf.keras.Model(inputs=[in_img, in_mask], outputs=[x, mask])


def downsample(
    in_channels: int, out_channels: int, size: int, apply_batchnorm: bool = True
):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(
            out_channels,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(out_channels: int, size: int, apply_dropout: bool = False):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(
            out_channels,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


class GeneratorType(Enum):
    STANDARD_CONV = 1
    PARTIAL_CONV = 2
    MIXED_CONV = 3


def Generator(
    generator_type: GeneratorType = GeneratorType.STANDARD_CONV,
) -> tf.keras.Model:
    in_img = tf.keras.layers.Input(shape=[None, None, 3], name="in_img")
    in_mask = tf.keras.layers.Input(shape=[None, None, 1], name="in_mask")

    if generator_type == GeneratorType.STANDARD_CONV:
        down_stack = [
            downsample(3, 64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
            downsample(64, 128, 4),  # (batch_size, 64, 64, 128)
            downsample(128, 256, 4),  # (batch_size, 32, 32, 256)
            downsample(256, 512, 4),  # (batch_size, 16, 16, 512)
            downsample(512, 512, 4),  # (batch_size, 8, 8, 512)
            downsample(512, 512, 4),  # (batch_size, 4, 4, 512)
            downsample(512, 512, 4),  # (batch_size, 2, 2, 512)
            downsample(512, 512, 4),  # (batch_size, 1, 1, 512)
        ]
    elif generator_type == GeneratorType.PARTIAL_CONV:
        down_stack = [
            p_downsample(3, 64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
            p_downsample(64, 128, 4),  # (batch_size, 64, 64, 128)
            p_downsample(128, 256, 4),  # (batch_size, 32, 32, 256)
            p_downsample(256, 512, 4),  # (batch_size, 16, 16, 512)
            p_downsample(512, 512, 4),  # (batch_size, 8, 8, 512)
            p_downsample(512, 512, 4),  # (batch_size, 4, 4, 512)
            p_downsample(512, 512, 4),  # (batch_size, 2, 2, 512)
            p_downsample(512, 512, 4),  # (batch_size, 1, 1, 512)
        ]
    elif generator_type == GeneratorType.MIXED_CONV:
        # TODO: implement mixed conv
        down_stack = [
            downsample_with_update_mask(
                3, 64, 4, apply_batchnorm=False
            ),  # (batch_size, 128, 128, 64)
            downsample_with_update_mask(64, 128, 4),  # (batch_size, 64, 64, 128)
            p_downsample(128, 256, 4),  # (batch_size, 32, 32, 256)
            p_downsample(256, 512, 4),  # (batch_size, 16, 16, 512)
            p_downsample(512, 512, 4),  # (batch_size, 8, 8, 512)
            p_downsample(512, 512, 4),  # (batch_size, 4, 4, 512)
            p_downsample(512, 512, 4),  # (batch_size, 2, 2, 512)
            p_downsample(512, 512, 4),  # (batch_size, 1, 1, 512)
        ]
    else:
        raise ValueError("Invalid generator_type")

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
        upsample(512, 4),  # (batch_size, 16, 16, 1024)
        upsample(256, 4),  # (batch_size, 32, 32, 512)
        upsample(128, 4),  # (batch_size, 64, 64, 256)
        upsample(64, 4),  # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0.0, 0.02)
    last = tf.keras.layers.Conv2DTranspose(
        3,
        4,
        strides=2,
        padding="same",
        kernel_initializer=initializer,
        activation="tanh",
    )  # (batch_size, 256, 256, 3)

    x, mask = in_img, in_mask  # i.e. in_mask is already 0 for holes

    # Downsampling through the model
    skips = []
    for down in down_stack:
        # print(down.get_config())
        if len(down.get_config().get("input_layers", [])) == 2:  # TODO: any better way?
            x, mask = down([x, mask])  # partial conv
        else:
            x = down(x)  # standard conv

        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    # output image = x inpainting mask + original image on non-masked area
    output = in_img * in_mask + x * (1 - in_mask)

    return tf.keras.Model(inputs=[in_img, in_mask], outputs=output)


def Discriminator() -> tf.keras.Model:
    initializer = tf.random_normal_initializer(0.0, 0.02)

    inp = tf.keras.layers.Input(shape=[None, None, 3], name="input_image")
    tar = tf.keras.layers.Input(shape=[None, None, 3], name="target_image")

    x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

    down1 = downsample(3, 64, 4, False)(x)  # (batch_size, 128, 128, 64)
    down2 = downsample(64, 128, 4)(down1)  # (batch_size, 64, 64, 128)
    down3 = downsample(128, 256, 4)(down2)  # (batch_size, 32, 32, 256)
    down4 = downsample(256, 512, 4)(down3)  # (batch_size, 16, 16, 512)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down4)  # (batch_size, 34, 34, 512)
    conv = tf.keras.layers.Conv2D(
        512 * 2, 4, strides=1, kernel_initializer=initializer, use_bias=False
    )(
        zero_pad1
    )  # (batch_size, 31, 31, 512 * 2)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(
        leaky_relu
    )  # (batch_size, 33, 33, 512 * 2)

    last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(
        zero_pad2
    )  # (batch_size, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)


LAMBDA = 50  # 100
criteriation = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(
    disc_generated_output, gen_output, target, mask
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    gan_loss = criteriation(tf.ones_like(disc_generated_output), disc_generated_output)

    # Mean absolute error
    l1_loss = tf.reduce_mean(tf.abs((target - gen_output) * (1 - mask)))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss


def discriminator_loss(disc_real_output, disc_generated_output) -> tf.Tensor:
    real_loss = criteriation(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = criteriation(
        tf.zeros_like(disc_generated_output), disc_generated_output
    )

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss
