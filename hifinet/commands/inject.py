from typing import Optional
from hydra_zen import builds, store, zen

from hifinet.loader import load_data

def inject(dataset: str, chance: float, seed: Optional[int] = None):
    data = load_data(dataset)
    return data

InjectConf = builds(
    inject,
    chance=0.2,
    seed=None,
    populate_full_signature=True
)

store(InjectConf)

inject_zen = zen(inject)
