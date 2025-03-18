import pathlib

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold


def normalize_img_tensor(image: tf.Tensor) -> tf.Tensor:
    return (image / 127.5) - 1


class InpaintingDataGenerator:
    def __init__(
        self,
        image_paths,
        img_size=(256, 256),
        batch_size=32,
        buffer_size=1000,
        k_fold=5,
    ):
        self.image_paths = image_paths
        self.img_size = img_size
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.k_fold = k_fold

    def _load_and_preprocess(self, image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, self.img_size)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = normalize_img_tensor(image)
        return image

    def _load_mask(self, image_path):
        parts = tf.strings.split(image_path, "/")
        if len(parts) == 1:
            parts = tf.strings.split(image_path, "\\")
        file = parts[-1]
        folder = tf.strings.reduce_join(parts[:-1], separator="/")
        mask_path = tf.strings.join([folder, "mask", file], separator="/")
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_jpeg(mask, channels=1)
        mask = tf.image.resize(mask, self.img_size)
        mask = tf.image.convert_image_dtype(mask, tf.float32)
        mask /= 255.0  # range [0, 1]
        return 1.0 - mask

    def _random_augmentation(self, image):
        # Random flip
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)

        # Random rotation
        k = tf.random.uniform([], 0, 4, dtype=tf.int32)
        image = tf.image.rot90(image, k)

        # Random brightness
        image = tf.image.random_brightness(image, 0.2)

        return image

    def _generate_random_mask(self, image):
        height = tf.shape(image)[0]
        width = tf.shape(image)[1]

        height_f = tf.cast(height, tf.float32)
        width_f = tf.cast(width, tf.float32)

        num_rectangles = tf.random.uniform([], 1, 4, dtype=tf.int32)

        out_mask = tf.zeros([height, width, 1], dtype=tf.float32)

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

        return 1.0 - out_mask

    def _apply_mask(self, image, mask):
        return image * mask + (1 - mask) * 1.0  # i.e. range [-1, 1]
        # return image * mask  # i.e. range [0, 1]

    def _train_process_path(self, image_path):
        image = self._load_and_preprocess(image_path)
        image = self._random_augmentation(image)
        mask = self._load_mask(image_path)
        masked_image = self._apply_mask(image, mask)
        return (masked_image, mask), image

    def _val_process_path(self, image_path):
        image = self._load_and_preprocess(image_path)
        mask = self._load_mask(image_path)
        masked_image = self._apply_mask(image, mask)
        return (masked_image, mask), image

    def get_dataset(self, training=True):
        dataset = tf.data.Dataset.from_tensor_slices(self.image_paths)

        if training:
            dataset = dataset.shuffle(self.buffer_size)
            dataset = dataset.map(
                self._train_process_path, num_parallel_calls=tf.data.AUTOTUNE
            )
        else:
            dataset = dataset.map(
                self._val_process_path, num_parallel_calls=tf.data.AUTOTUNE
            )

        # Note: I am not using step in training now, so reqpeat is not needed
        # if training:
        #     dataset = dataset.repeat()

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset


def prepare_kfold_data(image_dir, n_splits=5, batch_size=32, img_size=(256, 256)):
    data_root = pathlib.Path(image_dir)
    image_paths = [str(p) for p in data_root.glob("*.jpg")]
    image_paths = np.array(image_paths)

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_datasets = []

    for train_idx, val_idx in kfold.split(image_paths):
        train_paths = image_paths[train_idx]
        val_paths = image_paths[val_idx]

        train_gen = InpaintingDataGenerator(
            image_paths=train_paths, img_size=img_size, batch_size=batch_size
        )
        train_ds = train_gen.get_dataset(training=True)

        val_gen = InpaintingDataGenerator(
            image_paths=val_paths, img_size=img_size, batch_size=batch_size
        )
        val_ds = val_gen.get_dataset(training=False)

        fold_datasets.append((train_ds, val_ds))

    return fold_datasets
