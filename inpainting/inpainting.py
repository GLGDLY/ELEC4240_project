import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

from matplotlib import pyplot as plt

from data_loader import normalize_img_tensor
from model import Generator, GeneratorType


def display_result(in_image, mask, out_image):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(tf.squeeze(in_image).numpy() * 0.5 + 0.5)
    ax[0].set_title("Input Image")
    ax[1].imshow(tf.squeeze(mask).numpy(), cmap="gray")
    ax[1].set_title("Mask")
    ax[2].imshow(tf.squeeze(out_image).numpy() * 0.5 + 0.5)
    ax[2].set_title("Output Image")
    plt.show()


class Inpainting:
    def __init__(self, generator_path: str, generator_type: GeneratorType):
        self.generator = Generator(generator_type=generator_type)
        if generator_path:
            print("loading weights from:", generator_path)
            self.generator.load_weights(generator_path)

    @tf.function
    def predict(self, in_image, mask):
        return self.generator([in_image, mask], training=False)


def _load_mask(mask_path):
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_jpeg(mask, channels=1)
    mask = tf.image.resize(mask, (256, 256))
    mask = tf.image.convert_image_dtype(mask, tf.float32)
    mask /= 255.0

    return 1.0 - mask


def _gen_mask(in_image):
    height = in_image.shape[1]
    width = in_image.shape[2]
    height_f = tf.cast(height, tf.float32)
    width_f = tf.cast(width, tf.float32)

    out_mask = tf.zeros([height, width, 1], dtype=tf.float32)

    num_rectangles = tf.random.uniform([], 1, 4, dtype=tf.int32)
    for _ in tf.range(num_rectangles):
        # Generate random mask parameters
        mask_height = tf.random.uniform(
            [],
            minval=int(height_f * 0.25),
            maxval=int(height_f * 0.5),
            dtype=tf.int32,
        )
        mask_width = tf.random.uniform(
            [],
            minval=int(width_f * 0.25),
            maxval=int(width_f * 0.5),
            dtype=tf.int32,
        )

        # Random position for mask
        y = tf.random.uniform([], 0, height - mask_height, dtype=tf.int32)
        x = tf.random.uniform([], 0, width - mask_width, dtype=tf.int32)

        # Create mask with rectangle of zeros
        mask = tf.ones([mask_height, mask_width, 1], dtype=tf.float32)
        mask = tf.image.pad_to_bounding_box(mask, y, x, height, width)
        out_mask = tf.maximum(out_mask, mask)

    return 1.0 - mask


if __name__ == "__main__":
    # find the latest model of {model_suffix}
    # model_suffix = "stand_conv_fixmask"
    # gen_type = GeneratorType.STANDARD_CONV
    # model_suffix = "p_conv_fixmask_bugfix"
    # gen_type = GeneratorType.PARTIAL_CONV
    model_suffix = "mix_conv_fixmask"
    gen_type = GeneratorType.MIXED_CONV

    model_dir = "./models"
    models = list(
        filter(
            lambda x: x.endswith(".h5")
            and x.startswith(f"best_generator_{model_suffix}_202"),
            os.listdir(model_dir),
        )
    )  # i.e. best_generator_stand_conv_20250317-174942.h5
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

    mask = _gen_mask(in_image)
    # mask = _load_mask("./data/dataset/test/mask/COCO_train2014_000000000030.jpg")
    # mask = _load_mask("mask.jpg")
    mask = mask[None, :, :, :]

    in_image = in_image * mask + (1 - mask)

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
    display_result(in_image, mask, out_tensor)

    # save mask
    # plt.imsave("mask.jpg", tf.squeeze(1.0 - mask).numpy(), cmap="gray")
