import pathlib
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold


def normalize_img_tensor(image: tf.Tensor) -> tf.Tensor:
    return (image / 127.5) - 1


class SegmentationDataGenerator:
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

    def _load_segmentation_mask(self, image_path):
        mask_path = tf.strings.regex_replace(image_path, r"[\\/]images[\\/]", "/masks/")
        mask_path = tf.strings.regex_replace(mask_path, ".jpg", "_mask.jpg")

        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_jpeg(mask, channels=1)
        mask = tf.image.resize(mask, self.img_size, method="nearest")
        mask = tf.image.convert_image_dtype(mask, tf.float32)
        return mask

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

    # def _applying_augmentation(self, image, mask, augmented_image):
    #     mask = tf.image.random_flip_left_right(mask)
    #     mask = tf.image.random_flip_up_down(mask)

    #     k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    #     mask = tf.image.rot90(mask, k)

    #     return augmented_image, mask

    def _train_process_path(self, image_path):
        image = self._load_and_preprocess(image_path)
        mask = self._load_segmentation_mask(image_path)

        augmented_image = self._random_augmentation(image)

        # augmented_image, augmented_mask = self._applying_augmentation(
        #     image, mask, augmented_image
        # )

        return augmented_image, mask

    def _val_process_path(self, image_path):
        image = self._load_and_preprocess(image_path)
        mask = self._load_segmentation_mask(image_path)
        return image, mask

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

        train_gen = SegmentationDataGenerator(
            image_paths=train_paths,
            img_size=img_size,
            batch_size=batch_size,
        )
        train_ds = train_gen.get_dataset(training=True)

        val_gen = SegmentationDataGenerator(
            image_paths=val_paths,
            img_size=img_size,
            batch_size=batch_size,
        )

        val_ds = val_gen.get_dataset(training=False)

        fold_datasets.append((train_ds, val_ds))

    return fold_datasets


def test_generator():
    data_root = pathlib.Path("./data/train/images")
    image_paths = [str(p) for p in data_root.glob("*.jpg")]
    image_paths = np.array(image_paths)

    gen = SegmentationDataGenerator(
        image_paths=image_paths,
        img_size=(256, 256),
        batch_size=1,
    )

    dataset = gen.get_dataset(training=True)
    print(dataset)

    for (img, mask), _ in dataset.take(1):
        print(img.shape, mask.shape)


if __name__ == "__main__":
    test_generator()
