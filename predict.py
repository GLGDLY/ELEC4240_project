from segmentation import model as segmentation_model
from inpainting import (
    Inpainting,
    display_result,
    normalize_img_tensor,
    GeneratorType,
)

import tensorflow as tf
import numpy as np

if __name__ == "__main__":
    segmentation_model_path = r"./segmentation/checkpoints/model_05_0.56.h5"
    inpainting_model_path = r"./inpainting/models/best_generator_p_conv_fixmask_50lambda_more_disc_layer_300epochs_20250430-005458.h5"
    image_path = r"./test.jpg"

    seg_model = segmentation_model.UNet()
    seg_model.build(input_shape=(1, 256, 256, 3))
    seg_model.load_weights(segmentation_model_path)

    inpainting_pipeline = Inpainting(
        generator_path=inpainting_model_path, generator_type=GeneratorType.PARTIAL_CONV
    )

    image = tf.io.read_file(image_path)
    if image_path.lower().endswith(".png"):
        image = tf.image.decode_png(image, channels=3)
    elif image_path.lower().endswith(".jpg") or image_path.lower().endswith(".jpeg"):
        image = tf.image.decode_jpeg(image, channels=3)
    else:
        raise ValueError("Unsupported image format. Please use PNG or JPEG.")
    image = tf.image.resize(image, (256, 256))
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = np.expand_dims(image, axis=0)
    image = normalize_img_tensor(image)

    mask = seg_model.predict(image)
    out_tensor = inpainting_pipeline.predict(image, mask)

    display_result(image, mask, out_tensor)
    # display_result(image * mask + (1 - mask), mask, out_tensor)
