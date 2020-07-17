import torch
from typing import Tuple, List
from object_detection.entities import (
    YoloBoxes,
    Confidences,
    yolo_to_pascal,
    pascal_to_yolo,
    PascalBoxes,
)
from ensemble_boxes import weighted_boxes_fusion


class BoxMerge:
    def __init__(
        self, iou_threshold: float = 0.55, confidence_threshold: float = 0.0,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold

    @torch.no_grad()
    def __call__(
        self, *args: Tuple[List[YoloBoxes], List[Confidences]]
    ) -> Tuple[List[YoloBoxes], List[Confidences]]:
        if len(args) == 0:
            return [], []
        length = len(args[0][0])
        device = args[0][0][0].device
        pbox_batch: List[PascalBoxes] = []
        box_batch: List[YoloBoxes] = []
        conf_batch: List[Confidences] = []
        for i in range(length):
            pboxes, confs, _ = weighted_boxes_fusion(
                boxes_list=[yolo_to_pascal(x[0][i], (1, 1)) for x in args],
                scores_list=[x[1][i] for x in args],
                labels_list=[torch.zeros(x[1][i].shape) for x in args],
                iou_thr=self.iou_threshold,
            )
            indices = confs > self.confidence_threshold
            pboxes = pboxes[indices]
            confs = confs[indices]
            box_batch.append(
                pascal_to_yolo(PascalBoxes(torch.from_numpy(pboxes).to(device)), (1, 1))
            )
            conf_batch.append(Confidences(torch.from_numpy(confs).to(device)))

        return box_batch, conf_batch
