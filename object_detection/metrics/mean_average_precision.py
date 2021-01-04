from torchvision.ops import box_iou
from object_detection.entities.box import PascalBoxes, Labels, Confidences
from .mean_precision import precision


class MeanAveragePrecision:
    def __init__(
        self,
    ) -> None:
        ...

    def __call__(
        self,
        boxes: PascalBoxes,
        labels: Labels,
        gt_boxes: PascalBoxes,
        gt_labels: Labels,
    ) -> float:
        ...


class AveragePrecision:
    def __init__(
        self,
        iou_threshold: float,
    ) -> None:
        self.iou_threshold = iou_threshold

    def __call__(
        self,
        boxes: PascalBoxes,
        confidences: Confidences,
        gt_boxes: PascalBoxes,
    ) -> float:
        sorted_confidences, sort_idx = confidences.sort(descending=True)
        sorted_boxes = boxes[sort_idx]
        iou_matrix, _ = box_iou(sorted_boxes, gt_boxes).max(1)
        tp = (iou_matrix > self.iou_threshold).to("cpu").numpy()
        ap = 0.0
        gt_len = len(gt_boxes)
        count = 0
        for i, v in enumerate(tp):
            if v:
                count += 1
            p = count / (i + 1)
            ap += p / min(gt_len, (i + 1))
        return ap
