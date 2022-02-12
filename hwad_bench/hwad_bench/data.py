from __future__ import annotations
from torch.utils.data import Dataset
from typing_extensions import TypedDict
import pandas as pd
import os


Annotation = TypedDict(
    "Annotation",
    {
        "image_id": str,
        "species": str,
        "individual_id": str,
    },
)


def read_annotations(file_path: str) -> list:
    df = pd.read_csv(file_path)
    rows: list[Annotation] = []
    for _, csv_row in df.iterrows():
        rows.append(
            Annotation(
                image_id=os.path.basename(csv_row["image"]),
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
