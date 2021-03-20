from logging import (
    getLogger,
)
import os, torch, tqdm, vnet

from bench.kuzushiji.data import (
    KuzushijiDataset,
    read_test_rows,
    inv_normalize,
    save_submission,
    read_code_map,
    SubRow,
)

from vnet.utils import DetectionPlot
from torch.utils.data import DataLoader
from bench.kuzushiji.effdet.config import Config
from bench.kuzushiji.effdet.train import collate_fn

logger = getLogger(__name__)


@torch.no_grad()
def predict() -> None:
    config = Config()
    device = config.device
    model = config.model
    to_boxes = config.to_boxes
    to_points = config.to_points
    config.model_loader.load_if_needed(model)
    model.to(device)
    rows = read_test_rows(config.root_dir)
    code_map = read_code_map(config.code_map_path)
    dataset = KuzushijiDataset(
        rows=rows,
    )
    loader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=config.batch_size * 2,
        num_workers=min(os.cpu_count() or 1, config.batch_size * 2),
        shuffle=False,
    )

    submissions: list[SubRow] = []
    for (
        image_batch,
        _,
        _,
        original_img_list,
        row_list,
    ) in tqdm.tqdm(loader):
        image_batch = image_batch.to(device)
        netout = model(image_batch)
        box_batch, confidence_batch, label_batch = to_boxes(netout)
        _, _, h, w = image_batch.shape

        for boxes, labels, img, original_img, row in zip(
            box_batch, label_batch, image_batch, original_img_list, row_list
        ):
            original = (row["width"], row["height"])
            padded = (w, h)
            points = to_points(boxes)
            scale, pad = vnet.inv_scale_and_pad(original, padded)
            points = vnet.shift_points(points, (-pad[0], -pad[1]))
            points = vnet.resize_points(points, scale, scale)
            # plot = DetectionPlot(original_img)
            # plot.draw_points(points, color="red")
            # plot.save(os.path.join(config.out_dir, f"{row['id']}.png"))
            submissions.append(
                {
                    "id": row["id"],
                    "points": points,
                    "labels": labels,
                }
            )
    save_submission(submissions, code_map, config.submission_path)


if __name__ == "__main__":
    predict()
