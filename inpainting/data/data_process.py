import json
import os
import shutil

from sklearn.model_selection import train_test_split


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


if __name__ == "__main__":
    # process_cocotext("./dataset")
    # check_image_ann("COCO_train2014_000000004159")  # for testing
    split_dataset("./dataset")
