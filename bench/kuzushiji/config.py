import os, torch
from vnet.transforms import normalize_mean, normalize_std

device = torch.device("cuda")
root_dir = "/store/kuzushiji"
image_dir = os.path.join(root_dir, "images")
num_classes = 4787
image_size = 512
n_splits = 5
log_path = os.path.join(image_dir, "app.log")
