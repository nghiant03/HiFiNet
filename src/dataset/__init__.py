import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict

from src.dataset.args import IntelArgs
from src.dataset.loader import load_intel_dataset

logger = logging.getLogger(__name__)

DATA_PATH = Path.cwd() / "data/raw"
assert DATA_PATH.is_dir, "Data path {DATA_PATH.as_posix()} does not exist!"

def load_dataset(datasets: List[str], args: Dict[str, Dict]) -> List[pd.DataFrame]:
    loaded = []
    for dataset in datasets:
        match dataset:
            case 'intel':
                if 'intel' not in args:
                    logger.debug("No arguments provided for intel dataset. Using default arguments.")
                    intel_args = IntelArgs()
                else:
                    intel_args = IntelArgs(**args['intel'])

                loaded.append(load_intel_dataset(DATA_PATH / "data.txt", intel_args))

    logger.info("Total {len(loaded)} datasets loaded.")
    return loaded
