from vision_tools.metric import BoxMAP
from vision_tools.interface import TrainBatch


class EvaluateFn:
    def __init__(self) -> None:
        self.map = BoxMAP()


#     def __call__(self, pred_batch:TrainBatch, gt_batch:TrainBatch) -> tuple[float, dict[str, float]]:
#         return self.map(pred_batch["box_batch"],
