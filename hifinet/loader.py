from loguru import logger

from hifinet.adapter.intel import IntelAdaptor

ADAPTORS = {"intel": IntelAdaptor}


def load_data(dataset: str):
    logger.info(f"Loading {dataset} dataset")

    if dataset in ADAPTORS:
        logger.info(f"Using default {dataset} adapter.")
        adaptor = ADAPTORS[dataset]()

    data = adaptor.read()
    return data
