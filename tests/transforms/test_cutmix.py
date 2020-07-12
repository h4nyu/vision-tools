from object_detection.transforms.cutmix import Cutmix
from object_detection.data.object import PolyImage, ObjectDataset
from object_detection.utils import DetectionPlot
from object_detection.models.centernetv1 import collate_fn
from torch.utils.data import DataLoader


def test_cutmix() -> None:

    dataset = ObjectDataset(
        image_size=(300, 400), object_count_range=(1, 10), object_size_range=(64, 128),
    )
    dataloader = DataLoader(dataset=dataset, collate_fn=collate_fn, batch_size=8,)
    sample = next(iter(dataloader))
    images, box_batch, label_batch, id_batch = sample
    fn = Cutmix()
    images, box_batch, label_batch = fn(images, box_batch, label_batch)
    plot = DetectionPlot()
    plot.with_image(images[0])
    plot.with_yolo_boxes(box_batch[0], color="red")
    plot.save(f"store/test-cutmix.png")
