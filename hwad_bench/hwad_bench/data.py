from __future__ import annotations
from torch.utils.data import Dataset
from typing_extensions import TypedDict
import pandas as pd
import os
from pathlib import Path
from typing import Any
from toolz.curried import pipe, map, groupby
from coco_annotator import CocoAnnotator


Annotation = TypedDict(
    "Annotation",
    {
        "image_file": str,
        "species": str,
        "individual_id": str,
    },
)

CocoImage = TypedDict(
    "CocoImage",
    {
        "id": int,
        "file_name": str,
    },
)


def correct_species(
    annotation: Annotation,
) -> Annotation:
    species = annotation["species"]
    if(species == "bottlenose_dolpin"):
        print('aaa')
        annotation["species"] = "bottlenose_dolphin"
    elif(species == "kiler_whale"):
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

def fetch_coco_images(annotations:list[Annotation]) -> list[CocoImage]:
    annotator = CocoAnnotator()
    annotator.login(username="admin", password="admin")
    images:list[Any] = annotator.image.filter(limit=len(annotations))
    return images

def summary(
    annotations: list[Annotation],
) -> dict:
    all_species = pipe(annotations, map(lambda x: x["species"]), set)
    individual_id_count = pipe(annotations, map(lambda x: x["individual_id"]), set, len)
    return {
        "count": len(annotations),
        "species_count": len(all_species),
        "individual_id_count": individual_id_count,
        "all_species": all_species,
    }

def merge_to_coco_annotations(annotations:list[Annotation], coco_images:list[CocoImage]) -> dict[str, list]:
    relation = pipe(coco_images, map(lambda x: (x["file_name"], x['id'])), dict)
    coco_categories = []
    coco_annotations = []
    for i, annt in enumerate(annotations):
        coco_categories.append({
            "id": i,
            "name": annt["individual_id"],
            "supercategory": annt["species"],
        })
        coco_annotations.append({
            "id": i,
            "image_id": relation[annt["image_file"]],
            "category_id": i,
        })

    return {
        "images": coco_categories,
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
