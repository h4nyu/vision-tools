import torch
from vnet import Boxes, Labels, Points


class Metrics:
    def __init__(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def add(
        self,
        points: Points,
        labels: Labels,
        gt_boxes: Boxes,
        gt_labels: Labels,
    ) -> tuple[int, int, int]:  # tp, fp, fn
        gt_count = len(gt_labels)
        pred_count = len(labels)
        tp = 0
        fp = 0
        fn = 0
        if pred_count == gt_count == 0:
            return tp, fp, fn
        elif gt_count == 0 and pred_count > 0:
            fp = pred_count
            return tp, fp, fn
        elif pred_count == 0 and gt_count > 0:
            fn = gt_count
            return tp, fp, fn
        x, y = points.unbind(-1)
        preds_unused = torch.ones(pred_count, dtype=torch.bool, device=points.device)

        for gt_box, gt_label in zip(gt_boxes, gt_labels):
            gt_x0, gt_y0, gt_x1, gt_y1 = gt_box.unbind(-1)
            matched = (
                (gt_x0 < x)
                * (gt_x1 > x)
                * (gt_y0 < y)
                * (gt_y1 > y)
                * (gt_label == labels)
                * preds_unused
            )
            if matched.sum() == 0:
                fn += 1
            else:
                tp += 1
                preds_unused[torch.argmax(matched.int())] = False
        fp = int(preds_unused.sum())
        self.tp += tp
        self.fp += fp
        self.fn += fn
        return tp, fp, fn

    def __call__(
        self,
    ) -> float:
        tp, fp, fn = self.tp, self.fp, self.fn
        if (tp + fp) == 0 or (tp + fn) == 0:
            return 0.0
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if precision > 0 and recall > 0:
            f1 = (2 * precision * recall) / (precision + recall)
        else:
            f1 = 0
        return f1
