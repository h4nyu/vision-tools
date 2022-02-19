from __future__ import annotations
from torch.utils.data import Dataset
from typing_extensions import TypedDict
import json
import pandas as pd
import os
from pathlib import Path
from typing import Any
from toolz.curried import pipe, map, groupby, valmap, frequencies, sorted
from coco_annotator import CocoImage, CocoCategory
from PIL import Image as PILImage
from nanoid import generate as nanoid


Annotation = TypedDict(
    "Annotation",
    {
        "image_file": str,
        "species": str,
        "individual_id": str,
    },
)


def correct_species(
    annotation: Annotation,
) -> Annotation:
    species = annotation["species"]
    if species == "bottlenose_dolpin":
        print("aaa")
        annotation["species"] = "bottlenose_dolphin"
    elif species == "kiler_whale":
        annotation["species"] = "killer_whale"
    return annotation


def cleansing(
    annotations: list[Annotation],
) -> list[Annotation]:
    return pipe(
        annotations,
        map(correct_species),
        list,
    )


def summary(
    annotations: list[Annotation],
) -> dict:
    all_species = pipe(annotations, map(lambda x: x["species"]), set)
    individual_id_count = pipe(annotations, map(lambda x: x["individual_id"]), set, len)
    class_freq = pipe(
        annotations,
        groupby(lambda x: x["individual_id"]),
        valmap(len),
        lambda x: x.values(),
        frequencies,
        lambda x: x.items(),
        lambda x: sorted(x, key=lambda y: y[0]),
        map(lambda x: {"class_size": x[0], "count": x[1]}),
        list,
    )
    return {
        "count": len(annotations),
        "species_count": len(all_species),
        "individual_id_count": individual_id_count,
        "all_species": all_species,
        "class_freq": class_freq,
    }


def merge_to_coco_annotations(annotations: list[Annotation]) -> dict[str, list]:
    coco_categories = pipe(
        annotations,
        map(lambda x: x["species"]),
        set,
        enumerate,
        map(
            lambda x: {
                "id": x[0],
                "name": x[1],
                "supercategory": "animal",
            }
        ),
        list,
    )
    categories_id_map = pipe(
        coco_categories,
        map(lambda x: (x["name"], x["id"])),
        dict,
    )
    coco_annotations = []
    coco_images: list[dict] = []
    for i, annt in enumerate(annotations):
        coco_images.append(
            {
                "id": i,
                "file_name": annt["image_file"],
            }
        )
        coco_annotations.append(
            {
                "id": i,
                "image_id": i,
                "category_id": categories_id_map[annt["species"]],
            }
        )

    return {
        "images": coco_images,
        "categories": coco_categories,
        "annotations": coco_annotations,
    }


def read_annotations(file_path: str) -> list:
    df = pd.read_csv(file_path)
    rows: list[Annotation] = []
    for _, csv_row in df.iterrows():
        rows.append(
            Annotation(
                image_file=os.path.basename(csv_row["image"]),
                species=csv_row["species"],
                individual_id=csv_row["individual_id"],
            )
        )
    return rows


def read_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def create_croped_dataset(
    coco: dict, annotations: list[Annotation], source_dir: str, dist_dir: str
) -> list[Annotation]:
    image_map = pipe(
        coco["images"],
        map(lambda x: (x["id"], x)),
        dict,
    )
    annotation_map = pipe(
        annotations,
        map(lambda x: (x["image_file"], x)),
        dict,
    )
    print(annotation_map)
    croped_annots: list[Annotation] = []
    for annot in coco["annotations"]:
        image = image_map[annot["image_id"]]
        image_annot = annotation_map[image["file_name"]]
        source_path = os.path.join(source_dir, image["file_name"])
        im = PILImage.open(source_path)
        coco_bbox = annot["bbox"]
        bbox = [
            coco_bbox[0],
            coco_bbox[1],
            coco_bbox[0] + coco_bbox[2],
            coco_bbox[1] + coco_bbox[3],
        ]
        im_crop = im.crop(bbox)
        stem = Path(image_annot["image_file"]).stem
        dist_file_name = f"{stem}-{nanoid(size=4)}.png"
        dist_path = os.path.join(dist_dir, dist_file_name)
        im_crop.save(dist_path)
        croped_annots.append(
            Annotation(
                {
                    "image_file": dist_file_name,
                    "species": image_annot["species"],
                    "individual_id": image_annot["individual_id"],
                }
            )
        )
    return croped_annots


# # class HWADDataset(Dataset):
# #     def __init__(
# #         self,
# #         rows: List[Row],
# #         transform: Any,
# #         random_cut_and_paste: Optional[RandomCutAndPaste] = None,
# #     ) -> None:
# #         self.rows = rows
# #         self.transform = transform
# #         self.random_cut_and_paste = random_cut_and_paste

# #     def __str__(self) -> str:
# #         string = ""
# #         string += "\tlen = %d\n" % len(self)
# #         zero_count = pipe(self.rows, filter(lambda x: x["boxes"].shape[0] == 0), count)
# #         non_zero_count = pipe(
# #             self.rows, filter(lambda x: x["boxes"].shape[0] != 0), count
# #         )
# #         string += "\tzero_count     = %5d (%0.2f)\n" % (
# #             zero_count,
# #             zero_count / len(self.rows),
# #         )
# #         string += "\tnon_zero_count = %5d (%0.2f)\n" % (
# #             non_zero_count,
# #             non_zero_count / len(self.rows),
# #         )
# #         return string

# #     def __len__(self) -> int:
# #         return len(self.rows)

# #     def __getitem__(self, idx: int) -> Tuple[Detection, Row]:
# #         row = self.rows[idx]
# #         id = row["image_id"]
# #         video_id = row["video_id"]
# #         video_frame = row["video_frame"]
# #         image_path = row["image_path"]
# #         img_arr = np.array(PIL.Image.open(image_path))
# #         labels = torch.zeros(len(row["boxes"]))
# #         transformed = self.transform(
# #             image=img_arr,
# #             bboxes=clip_boxes_to_image(row["boxes"], img_arr.shape[:2]),
# #             labels=labels,
# #         )
# #         image = (transformed["image"] / 255).float()
# #         boxes = torch.tensor(transformed["bboxes"]).float()
# #         labels = torch.zeros(len(boxes)).long()
# #         confs = torch.ones(len(labels)).float()
# #         sample = Detection(
# #             image=image,
# #             boxes=boxes,
# #             labels=labels,
# #             confs=confs,
# #         )
# #         if self.random_cut_and_paste is not None:
# #             sample = self.random_cut_and_paste(sample)
# #         return sample, row
