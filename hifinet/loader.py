from loguru import logger
from hifinet.adapter.intel import IntelAdaptor

ADAPTORS = {"intel": IntelAdaptor}
def load_data(dataset: str):
    if dataset in ADAPTORS.keys():
        logger.info(f"Using default {dataset} adapter.")
        adaptor = ADAPTORS[dataset]()

    data = adaptor.read()
    return data
