from loguru import logger
from hifinet.loader import load_data

from hifinet.config import InjectConfig

def inject_from_config(config: InjectConfig):
    logger.info("Loading")
    data = load_data(config.dataset)
