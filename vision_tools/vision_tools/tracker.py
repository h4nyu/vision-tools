import torch
from .interface import TrainSample
import torch.nn.functional as F
from ensemble_boxes import weighted_boxes_fusion
from .box import resize_boxes


class Tracker:
    def __init__(
        self, iou_thr: float = 0.5, diff_thr: float = 0.05, conf_thr: float = 0.5
    ) -> None:
        self.iou_thr = iou_thr
        self.diff_thr = diff_thr
        self.conf_thr = conf_thr
        self.prev_sample = None

    def __call__(
        self,
        sample: TrainSample,
    ) -> TrainSample:
        if self.prev_sample is None:
            self.prev_sample = sample
            return sample
        image = sample["image"]
        prev_image = self.prev_sample["image"]
        dist = F.mse_loss(image, prev_image)
        if dist > self.diff_thr:
            self.prev_sample = sample
            return sample

        _, h, w = image.shape
        device = image.device
        np_boxes, np_confs, np_lables = weighted_boxes_fusion(
            [
                resize_boxes(sample['boxes'], (1 / w, 1 / h)),
                resize_boxes(self.prev_sample["boxes"], (1 / w, 1 / h)),
            ],
            [
                sample['confs'],
                self.prev_sample["confs"],
            ],
            [
                sample['labels'],
                self.prev_sample["labels"],
            ],
            weights=[2, 1],
            iou_thr = 0.3
        )

        next_sample = TrainSample(
            image=sample["image"],
            boxes=resize_boxes(torch.from_numpy(np_boxes), (w, h)).to(device),
            labels=torch.from_numpy(np_lables).to(device),
            confs=torch.from_numpy(np_confs).to(device),
        )
        print(next_sample["boxes"].shape, sample["boxes"].shape)
        self.prev_sample = sample
        return next_sample
