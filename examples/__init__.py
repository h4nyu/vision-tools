from logging import (
    getLogger,
    StreamHandler,
    Formatter,
    INFO,
    FileHandler,
)

logger = getLogger()
logger.setLevel(INFO)
stream_handler = StreamHandler()
handler_format = Formatter("%(asctime)s,%(name)s,%(message)s")
stream_handler.setFormatter(handler_format)
logger.addHandler(stream_handler)
