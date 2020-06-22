import typing as t
import pandas as pd
from app.entities import PredBoxes


def to_df(preds: t.List[PredBoxes]) -> t.Any:
    ...
    #  rows: t.Any = []
    #  for confidences, boxes in preds:
    #      strs = []
    #      for box, confidence in zip(annot.boxes, annot.confidences):
    #          x0, y0, x1, y1 = box.unbind()
    #          x0 = x0 * annot.w
    #          x1 = x1 * annot.w
    #          y0 = y0 * annot.h
    #          y1 = y1 * annot.h
    #          w = x1 - x0
    #          h = y1 - y0
    #          cx = x0 + w / 2
    #          cy = y0 + h / 2
    #          strs.append(f"{confidence} {int(cx)} {int(cy)} {int(w)} {int(h)}")
    #
    #      row = {
    #          "image_id": annot.id,
    #          "PredictionString": " ".join(strs),
    #      }
    #      rows.append(row)
    #  df = pd.DataFrame.from_records(rows)
    #  return df


def save_csv(annots: t.Any, path: str) -> None:
    ...
    #  df = to_df(annots)
    #  df.to_csv(path, index=False)
