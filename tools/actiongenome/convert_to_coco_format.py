import argparse
import json
import os
import pickle
import shutil

import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", required=True, type=str)
parser.add_argument("--dst_dir", required=True, type=str)
parser.add_argument("--parse_frame", action="store_true")


def list_box_xyxy_to_xywh(x):
    x0, y0, x1, y1 = x
    b = [x0, y0, (x1 - x0), (y1 - y0)]
    return b


def main(args):
    print("Start process")
    annotations_train = {"info": [], "licenses": [], "images": [], "annotations": [], "categories": []}
    annotations_val = {"info": [], "licenses": [], "images": [], "annotations": [], "categories": []}

    image_names = []
    with open(os.path.join(args.root_dir, "annotations/frame_list.txt"), "r") as file:
        lines = file.readlines()
        image_names = [line.rstrip() for line in lines]

    object_classes = {}
    with open(os.path.join(args.root_dir, "annotations/object_classes.txt"), "r") as file:
        lines = file.readlines()
        class_id = 0
        for line in lines:
            class_name = line.rstrip()
            object_classes[class_name] = class_id
            categories = {
                "supercategory": class_name,
                "id": class_id,
                "name": class_name,
            }
            annotations_train["categories"].append(categories)
            annotations_val["categories"].append(categories)
            class_id += 1

    object_bbox_and_relationship = pickle.load(
        open(
            os.path.join(args.root_dir, "annotations/object_bbox_and_relationship.pkl"),
            "rb",
        )
    )
    person_bbox = pickle.load(open(os.path.join(args.root_dir, "annotations/person_bbox.pkl"), "rb"))

    image_names_with_subset = []
    for image_name in image_names:
        rels = object_bbox_and_relationship[image_name]
        image_names_with_subset.append(
            {
                "image_name": image_name,
                "subset": "train" if rels[0]["metadata"]["set"] == "train" else "val",
            }
        )

    if not os.path.exists(os.path.join(args.dst_dir, "train")):
        os.mkdir(os.path.join(args.dst_dir, "train"))

    if not os.path.exists(os.path.join(args.dst_dir, "val")):
        os.mkdir(os.path.join(args.dst_dir, "val"))

    if not os.path.exists(os.path.join(args.dst_dir, "annotations")):
        os.mkdir(os.path.join(args.dst_dir, "annotations"))

    image_id_map = {}

    print("Start copy frames")
    for i, image in tqdm(enumerate(image_names_with_subset)):
        if args.parse_frame:
            frame_path = os.path.join(args.root_dir, "frames", image["image_name"])
            dst_dir = os.path.join(args.dst_dir, image["subset"], f"{i}.png")
            shutil.copy(frame_path, dst_dir)
        image_id_map[image["image_name"]] = i
    if not args.parse_frame:
        print(">> Skip")

    print("Start convert annotations")
    count_id = 0
    for image_name_subset in tqdm(image_names_with_subset):
        image_name = image_name_subset["image_name"]
        image_subset = image_name_subset["subset"]

        rels = object_bbox_and_relationship[image_name]

        for rel in rels:
            bbox = rel["bbox"]
            class_id = object_classes[rel["class"].replace("/", "")]

            if bbox is None:
                continue

            annot = {
                "segmentation": [],
                "area": 0,
                "iscrowd": 0,
                "image_id": image_id_map[image_name],
                "bbox": list(np.array(bbox, dtype=np.float64)),
                "category_id": class_id,
                "id": count_id,
            }

            if image_subset == "train":
                annotations_train["annotations"].append(annot)
            elif image_subset == "val":
                annotations_val["annotations"].append(annot)

            count_id += 1

        person = person_bbox[image_name]
        for p_bbox in person["bbox"]:
            bbox = list_box_xyxy_to_xywh(p_bbox)
            class_id = object_classes["person"]
            annot = {
                #"segmentation": [],
                "area": 0,
                "iscrowd": 0,
                "image_id": image_id_map[image_name],
                "bbox": list(np.array(bbox, dtype=np.float64)),
                "category_id": class_id,
                "id": count_id,
            }

            if image_subset == "train":
                annotations_train["annotations"].append(annot)
            elif image_subset == "val":
                annotations_val["annotations"].append(annot)

            count_id += 1

        image_info = {
            "license": 1,
            "file_name": f"{image_id_map[image_name]}.png",
            "coco_url": "",
            "height": person["bbox_size"][1],
            "width": person["bbox_size"][0],
            "date_captured": "",
            "flickr_url": "",
            "id": image_id_map[image_name],
        }

        if image_subset == "train":
            annotations_train["images"].append(image_info)
        elif image_subset == "val":
            annotations_val["images"].append(image_info)

    json.dump(annotations_train, open(os.path.join(args.dst_dir, "annotations/instances_train2017.json"), "w"))
    json.dump(annotations_val, open(os.path.join(args.dst_dir, "annotations/instances_val2017.json"), "w"))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
