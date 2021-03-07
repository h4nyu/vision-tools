import os
from vnet.transforms import normalize_mean, normalize_std

root_dir = "/store/kuzushiji"
image_dir = os.path.join(root_dir, "images")
num_classes = 4787
image_size = 512
