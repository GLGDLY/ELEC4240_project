import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from PIL import Image

from data_loader import normalize_img_tensor
from model import Generator, GeneratorType


def display_tensor(tensor):
    out_image = tf.squeeze(tensor).numpy() * 0.5 + 0.5
    Image.fromarray((out_image * 255).astype(np.uint8)).show()


class Inpainting:
    def __init__(self, generator_path: str, generator_type: GeneratorType):
        self.generator = Generator(generator_type=generator_type)
        if generator_path:
            print("loading weights from:", generator_path)
            self.generator.load_weights(generator_path)

    @tf.function
    def predict(self, in_image, mask):
        return self.generator([in_image, mask], training=False)


if __name__ == "__main__":
    # find the latest model of {model_suffix}
    model_suffix = "stand_conv"

    model_dir = "./models"
    models = list(
        filter(
            lambda x: x.endswith(".h5")
            and x.startswith(f"best_generator_{model_suffix}"),
            os.listdir(model_dir),
        )
    )
    gen_type = GeneratorType.STANDARD_CONV
    if models:
        # if 0:
        models.sort()
        latest_model = models[-1]
        latest_model = os.path.join(model_dir, latest_model)

        inpainting = Inpainting(latest_model, gen_type)
    else:
        inpainting = Inpainting(None, gen_type)

    with open("./data/dataset/test/COCO_train2014_000000000030.jpg", "rb") as f:
        in_image = tf.image.decode_jpeg(f.read(), channels=3)
    ori_height, ori_width = in_image.shape[0], in_image.shape[1]

    in_image = tf.image.resize(in_image, (256, 256))
    in_image = normalize_img_tensor(in_image)  # map in_image to [-1, 1]
    in_image = in_image[None, :, :, :]

    height = in_image.shape[1]
    width = in_image.shape[2]
    height_f = tf.cast(height, tf.float32)
    width_f = tf.cast(width, tf.float32)

    mask_height = tf.random.uniform(
        [],
        minval=int(height_f * 0.25),
        maxval=int(height_f * 0.3),
        dtype=tf.int32,
    )
    mask_width = tf.random.uniform(
        [],
        minval=int(width_f * 0.25),
        maxval=int(width_f * 0.3),
        dtype=tf.int32,
    )

    # Random position for mask
    y = tf.random.uniform([], 0, height - mask_height, dtype=tf.int32)
    x = tf.random.uniform([], 0, width - mask_width, dtype=tf.int32)

    # Create mask with rectangle of zeros
    mask = tf.ones([mask_height, mask_width, 1], dtype=tf.float32)
    mask = tf.image.pad_to_bounding_box(mask, y, x, height, width)
    mask = 1.0 - mask
    mask = mask[None, :, :, :]

    in_image = in_image * mask + (1 - mask) * 1.0

    display_tensor(in_image)
    print("in_shape:", in_image.shape)
    print("mask_shape:", mask.shape)

    out_tensor = inpainting.predict(in_image, mask)
    print("out_shape:", out_tensor.shape)

    # return to original size
    out_tensor = tf.image.resize(out_tensor, (ori_height, ori_width))

    # print the max and min value of the output tensor
    # print("max:", tf.reduce_max(out_tensor).numpy())
    # print("min:", tf.reduce_min(out_tensor).numpy())

    # show the result
    display_tensor(out_tensor)
