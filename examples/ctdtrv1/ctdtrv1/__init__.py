from logging import getLogger, StreamHandler, Formatter, INFO, FileHandler
from .config import out_dir

logger = getLogger()
logger.setLevel(INFO)
stream_handler = StreamHandler()
handler_format = Formatter("%(asctime)s,%(name)s,%(message)s")
stream_handler.setFormatter(handler_format)
logger.addHandler(stream_handler)
file_handler = FileHandler(filename=f"{out_dir}/app.log")
logger.addHandler(file_handler)
