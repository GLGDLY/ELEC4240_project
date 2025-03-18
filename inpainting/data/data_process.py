import json
import os
import shutil

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def process_cocotext(out_folder: str) -> None:
    # filter out data without text
    img_folder = "./train2014"
    with open("./cocotext.v2.json") as f:
        data = json.load(f)

    filter_list = set()
    for ann in data["anns"].values():
        if ann["bbox"]:
            filter_list.add(ann["image_id"])

    for img in data["imgs"].values():
        if img["id"] not in filter_list:
            shutil.copy(os.path.join(img_folder, img["file_name"]), out_folder)


def check_image_ann(name: str) -> None:
    with open("./cocotext.v2.json") as f:
        data = json.load(f)

    img_id = None
    for img in data["imgs"].values():
        if img["file_name"].startswith(name):
            img_id = img["id"]
            break

    if img_id is None:
        print("Image not found")
        return

    for ann in data["anns"].values():
        if ann["image_id"] == img_id:
            print(ann)
            break
    else:
        print("No annotation found")


def split_dataset(folder: str) -> None:
    files = os.listdir(folder)
    train_files, test_files = train_test_split(files, test_size=0.2)

    os.makedirs(os.path.join(folder, "train"), exist_ok=True)
    os.makedirs(os.path.join(folder, "test"), exist_ok=True)

    for file in train_files:
        shutil.move(os.path.join(folder, file), os.path.join(folder, "train", file))

    for file in test_files:
        shutil.move(os.path.join(folder, file), os.path.join(folder, "test", file))


def generate_masks(img_size=(256, 256)):
    print("Generating masks...")

    data_roots = ["./dataset/train", "./dataset/test"]
    for data_root in data_roots:
        print(f"Processing {data_root}...")
        image_paths = tf.data.Dataset.list_files(os.path.join(data_root, "*.jpg"))

        # Process each file path individually
        for image_path in tqdm(image_paths):
            _generate_random_mask(image_path.numpy().decode(), img_size)


def _generate_random_mask(image_path, img_size):
    seed = hash(image_path) % (2**31)
    tf.random.set_seed(seed)

    height, width = img_size

    height_f = tf.cast(height, tf.float32)
    width_f = tf.cast(width, tf.float32)

    num_rectangles = tf.random.uniform([], 1, 7, dtype=tf.int32)

    out_mask = tf.zeros([height, width, 1], dtype=tf.float32)

    for _ in tf.range(num_rectangles):
        # Generate random mask parameters
        mask_height = tf.random.uniform(
            [],
            minval=int(height_f * 0.1),
            maxval=int(height_f * 0.4),
            dtype=tf.int32,
        )
        mask_width = tf.random.uniform(
            [],
            minval=int(width_f * 0.1),
            maxval=int(width_f * 0.4),
            dtype=tf.int32,
        )

        # Random position for mask
        y = tf.random.uniform([], 0, height - mask_height, dtype=tf.int32)
        x = tf.random.uniform([], 0, width - mask_width, dtype=tf.int32)

        # Create mask with rectangle of zeros
        mask = tf.ones([mask_height, mask_width, 1], dtype=tf.float32)
        mask = tf.image.pad_to_bounding_box(mask, y, x, height, width)
        out_mask = tf.maximum(out_mask, mask)

    # return 1.0 - out_mask

    folder, file = os.path.split(image_path)
    os.makedirs(os.path.join(folder, "mask"), exist_ok=True)
    mask_path = os.path.join(folder, "mask", file)

    out_mask_uint8 = tf.cast(out_mask * 255.0, tf.uint8)
    tf.io.write_file(mask_path, tf.io.encode_jpeg(out_mask_uint8))


if __name__ == "__main__":
    # process_cocotext("./dataset")
    # check_image_ann("COCO_train2014_000000004159")  # for testing
    # split_dataset("./dataset")
    generate_masks()
