import typing as t
import pandas as pd
from app.entities import Annotations


def to_df(annots: Annotations) -> t.Any:
    rows: t.Any = []
    for annot in annots:
        annot = annot.to_xyxy()
        for box, confidence in zip(annot.boxes, annot.confidences):
            x0, y0, x1, y1 = box.unbind()
            x0 = x0 * annot.w
            x1 = x1 * annot.w
            y0 = y0 * annot.h
            y1 = y1 * annot.h
            w = x1 - x0
            h = y1 - y0
            cx = x0 + w / 2
            cy = y0 + h / 2
            row = {
                "image_id": annot.id,
                "PredictionString": f"{confidence} {int(cx)} {int(cy)} {int(w)} {int(h)}",
            }
            rows.append(row)
    print(rows)
    df = pd.DataFrame.from_records(rows)
    df = df.set_index("image_id")
    print(df)
    return df


def save_csv(annots: Annotations, path: str) -> None:
    df = to_df(annots)
    df.to_csv(path)
