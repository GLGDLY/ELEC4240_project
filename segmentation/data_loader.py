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
        mask_paths,
        img_size=(256, 256),
        batch_size=32,
        buffer_size=1000,
        k_fold=5,
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.k_fold = k_fold

    def _load_and_preprocess_image(self, image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, self.img_size)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = normalize_img_tensor(image)
        return image

    def _load_and_preprocess_mask(self, mask_path):
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.image.resize(mask, self.img_size, method='nearest')
        mask = tf.image.convert_image_dtype(mask, tf.float32)
        return mask

    def _random_augmentation(self, image, mask):
        # Stack image and mask to apply same random operations
        combined = tf.concat([image, mask], axis=-1)
        
        # Random flip
        combined = tf.image.random_flip_left_right(combined)
        combined = tf.image.random_flip_up_down(combined)

        # Random rotation
        k = tf.random.uniform([], 0, 4, dtype=tf.int32)
        combined = tf.image.rot90(combined, k)

        # Split back into image and mask
        image, mask = combined[..., :3], combined[..., 3:]

        # Random brightness (only apply to image)
        image = tf.image.random_brightness(image, 0.2)
        image = tf.clip_by_value(image, -1.0, 1.0)

        return image, mask

    def _train_process_path(self, image_path, mask_path):
        image = self._load_and_preprocess_image(image_path)
        mask = self._load_and_preprocess_mask(mask_path)
        image, mask = self._random_augmentation(image, mask)
        return image, mask

    def _val_process_path(self, image_path, mask_path):
        image = self._load_and_preprocess_image(image_path)
        mask = self._load_and_preprocess_mask(mask_path)
        return image, mask

    def get_dataset(self, training=True):
        dataset = tf.data.Dataset.from_tensor_slices((self.image_paths, self.mask_paths))

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


def prepare_kfold_data(image_dir, mask_dir, n_splits=5, batch_size=32, img_size=(256, 256)):
    # Get image and mask paths
    image_root = pathlib.Path(image_dir)
    mask_root = pathlib.Path(mask_dir)
    
    image_paths = sorted([str(p) for p in image_root.glob("*.jpg")])
    mask_paths = sorted([str(p) for p in mask_root.glob("*.png")])
    
    image_paths = np.array(image_paths)
    mask_paths = np.array(mask_paths)

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_datasets = []

    for train_idx, val_idx in kfold.split(image_paths):
        train_image_paths = image_paths[train_idx]
        train_mask_paths = mask_paths[train_idx]
        val_image_paths = image_paths[val_idx]
        val_mask_paths = mask_paths[val_idx]

        train_gen = SegmentationDataGenerator(
            image_paths=train_image_paths,
            mask_paths=train_mask_paths,
            img_size=img_size,
            batch_size=batch_size
        )
        train_ds = train_gen.get_dataset(training=True)

        val_gen = SegmentationDataGenerator(
            image_paths=val_image_paths,
            mask_paths=val_mask_paths,
            img_size=img_size,
            batch_size=batch_size
        )
        val_ds = val_gen.get_dataset(training=False)

        fold_datasets.append((train_ds, val_ds))

    return fold_datasets