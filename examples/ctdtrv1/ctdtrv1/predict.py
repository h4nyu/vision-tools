from object_detection.models.centernetv1 import Predicter
from . import train

predictor = Predicter(
    model=train.model,
    loader=train.test_loader,
    model_loader=train.model_loader,
    device="cuda",
    box_merge=train.box_merge,
    to_boxes=train.to_boxes,
)
