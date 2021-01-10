import numpy as np
import torch
from object_detection.entities import Labels, PascalBoxes
from object_detection.metrics.average_precision import AveragePrecision, auc


# def test_average_precision() -> None:
#     metrics = MeanAveragePrecision(classes=[0],iou_threshold=0.3)
#     boxes = PascalBoxes(
#         torch.tensor(
#             [
#                 [15, 15, 25, 25],
#                 [0, 0, 15, 15],
#                 [25, 25, 35, 35],
#             ]
#         )
#     )

#     gt_boxes = PascalBoxes(
#         torch.tensor(
#             [
#                 [0, 0, 10, 10],
#                 [20, 20, 30, 30],
#             ]
#         )
#     )

#     res = metrics(
#         boxes,
#         gt_boxes,
#     )
#     assert round(res, 4) == round((1 / 2 * 1 / 2 + 1 / 3 * 1 / 2), 4)
