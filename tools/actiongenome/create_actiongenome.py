import argparse
import base64
import json
import os.path as op
import pickle

import cv2
from tqdm import tqdm

from maskrcnn_benchmark.structures.tsv_file_ops import (generate_linelist_file,
                                                        tsv_writer)

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", type=str, default="tools/actiongenome/ag")
parser.add_argument("--dst_dir", type=str, required=True)
parser.add_argument("--image_set", choices=["train", "test"], default="train")
parser.add_argument("--image_ext", choices=[".jpg", ".png"], default=".jpg")
args = parser.parse_args()

data_path = args.root_dir
with open(op.join(data_path, "annotations/frame_list.txt"), "r") as f:
    lines = f.readlines()
    img_list = [line.rstrip() for line in lines]

tsv_file = op.join(args.dst_dir, f"{args.image_set}.img.tsv")
label_file = op.join(args.dst_dir, f"{args.image_set}.label.tsv")
hw_file = op.join(args.dst_dir, f"{args.image_set}.hw.tsv")
linelist_file = op.join(args.dst_dir, f"{args.image_set}.linelist.tsv")
labelmap_file = op.join(args.dst_dir, "labelmap.json")

label_to_idx, idx_to_label = {}, {}
with open(op.join(data_path, "annotations/object_classes.txt"), "r") as f:
    lines = f.readlines()
    class_id = 1
    for line in lines:
        label_to_idx[line.rstrip()] = class_id
        idx_to_label[class_id] = line.rstrip()
        class_id += 1
predicate_to_idx, idx_to_predicate = {}, {}
with open(op.join(data_path, "annotations/relationship_classes.txt"), "r") as f:
    lines = f.readlines()
    class_id = 1
    for line in lines:
        predicate_to_idx[line.rstrip()] = class_id
        idx_to_predicate[class_id] = line.rstrip()
        class_id += 1
labelmap = {
    "label_to_idx": label_to_idx,
    "idx_to_label": idx_to_label,
    "predicate_to_idx": predicate_to_idx,
    "idx_to_predicate": idx_to_predicate,
}
with open(labelmap_file, "w+") as f:
    json.dump(labelmap, f)

person_bbox = pickle.load(open(op.join(data_path, "annotations/person_bbox.pkl"), "rb"))
object_bbox_and_relationship = pickle.load(
    open(op.join(data_path, "annotations/object_bbox_and_relationship.pkl"), "rb")
)

rows = []
rows_label = []
rows_hw = []
for i, img_p in enumerate(tqdm(img_list)):
    img_path = op.join(data_path, "frames", img_p)
    img = cv2.imread(img_path)
    img_encoded_str = base64.b64encode(cv2.imencode(args.image_ext, img)[1])

    # labels.tsv
    labels = {"objects": [], "relations": []}
    person = person_bbox[img_p]
    rels = object_bbox_and_relationship[img_p]

    if person["bbox"] is None or len(person["bbox"]) == 0:
        continue
    sub_bbox = [
        int(person["bbox"][0][0]),
        int(person["bbox"][0][1]),
        int(person["bbox"][0][2]),
        int(person["bbox"][0][3]),
    ]

    labels["objects"].append({"class": "person", "rect": list(sub_bbox)})

    image_set = ""
    for rel in rels:
        if rel["metadata"]["set"] != args.image_set:
            break
        image_set = rel["metadata"]["set"]

        obj_bbox = rel["bbox"]
        if obj_bbox is None:
            continue
        obj_bbox = [
            int(obj_bbox[0]),
            int(obj_bbox[1]),
            int(obj_bbox[0] + obj_bbox[2]),
            int(obj_bbox[1] + obj_bbox[3]),
        ]
        obj_class = rel["class"].replace("/", "")
        if obj_bbox is None:
            continue
        labels["objects"].append({"class": obj_class, "rect": list(obj_bbox)})

        if rel["attention_relationship"] is not None:
            for r in rel["attention_relationship"]:
                labels["relations"].append(
                    {"subj_id": 0, "obj_id": len(labels["objects"])-1, "class": r.replace("_", "")}
                )
        if rel["spatial_relationship"] is not None:
            for r in rel["spatial_relationship"]:
                labels["relations"].append(
                    {"subj_id": 0, "obj_id": len(labels["objects"])-1, "class": r.replace("_", "")}
                )
        if rel["contacting_relationship"] is not None:
            for r in rel["contacting_relationship"]:
                labels["relations"].append(
                    {"subj_id": 0, "obj_id": len(labels["objects"])-1, "class": r.replace("_", "")}
                )

    if image_set != args.image_set:
        continue

    if len(labels["objects"]) == 0 or len(labels["relations"]) == 0:
        continue

    row_label = [str(i), json.dumps(labels)]
    rows_label.append(row_label)

    # img.tsv
    row = [str(i), img_encoded_str]
    rows.append(row)

    # hw.tsv
    w, h = person["bbox_size"]
    row_hw = [str(i), json.dumps([{"height": h, "width": w}])]
    rows_hw.append(row_hw)

tsv_writer(rows, tsv_file)
tsv_writer(rows_label, label_file)
tsv_writer(rows_hw, hw_file)
generate_linelist_file(label_file, save_file=linelist_file)
