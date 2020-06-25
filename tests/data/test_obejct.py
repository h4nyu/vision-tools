import cv2
from object_detection.data.object import PolyImage, ObjectDataset
from object_detection.utils import DetectionPlot


def test_polyimage() -> None:
    plot = DetectionPlot(figsize=(10, 10))
    poly = PolyImage(width=400, height=300,)
    for _ in range(10):
        poly.add(32)
    img, boxes = poly()
    plot.with_image(img)
    plot.with_yolo_boxes(boxes, color="red")
    plot.save("/store/test-poly-0.png")


def test_objectdataset() -> None:
    dataset = ObjectDataset(
        image_size=(300, 400), object_count_range=(1, 10), object_size_range=(64, 128),
    )
    for i in range(10):
        _, img, boxes = dataset[0]
        plot = DetectionPlot()
        plot.with_image(img)
        plot.with_yolo_boxes(boxes, color="red")
        plot.save(f"/store/test-poly-{i}.png")
