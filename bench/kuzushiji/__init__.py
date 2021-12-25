from vision_tools.utils import seed_everything
import os
from logging import (
    getLogger,
    StreamHandler,
    Formatter,
    INFO,
    FileHandler,
)
from .config import Config

config = Config()

handler_format = Formatter("%(asctime)s,%(name)s,%(message)s")
file_handler = FileHandler(config.log_path)

logger = getLogger()
logger.setLevel(INFO)
stream_handler = StreamHandler()
handler_format = Formatter("%(asctime)s,%(name)s,%(message)s")
stream_handler.setFormatter(handler_format)
file_handler.setFormatter(handler_format)

logger.addHandler(stream_handler)
logger.addHandler(file_handler)
seed_everything(10)
