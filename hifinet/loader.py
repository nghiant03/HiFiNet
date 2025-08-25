from pathlib import Path

import pandas as pd
from loguru import logger
from pandera.errors import SchemaErrors

from hifinet.adapter.base import DataSchema
from hifinet.adapter.intel import IntelAdaptor

ADAPTORS = {"intel": IntelAdaptor}


def load_data(dataset: str):
    logger.info(f"Loading {dataset} dataset")

    if dataset in ADAPTORS:
        logger.info(f"Using default {dataset} adapter.")
        adaptor = ADAPTORS[dataset]()
        data = adaptor.read()
    else:
        dataset_path = Path(dataset)
        logger.info(f"Loading raw dataset from {dataset_path}")
        data = pd.read_csv(dataset_path)

        try:
            data = DataSchema.validate(data, lazy=True)
        except SchemaErrors:
            logger.error("Failed to validated data")
            raise

    return data
