import pandas as pd
from loguru import logger
from hifinet.config import InjectorConfig
from hifinet.fault.type import DriftFault, ErraticFault, HardoverFault, SpikeFault, StuckFault 

NAME_MAPPING = {
    "hardover": HardoverFault,
    "drift": DriftFault,
    "spike": SpikeFault,
    "erratic": ErraticFault,
    "stuck": StuckFault
}

class FaultInjector:
    def __init__(self, config: InjectorConfig):
        self.config = config

    def inject(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info("Injecting fault to data")
        logger.debug(f"Current injector config: {self.config.model_dump()}")
        data["type"] = 0
        clean = pd.DataFrame()
        if self.config.exclude:
            clean = data[data["node_id"].isin(self.config.exclude)].copy().reset_index(drop=True)

        faulty = []
        node_ids = data["node_id"].unique()

        match self.config.mode:
            case "singular":
                for idx, (fault_name, type_idx) in enumerate(self.config.type_mapping.items()):
                    node_data = data[data["node_id"] == node_ids[idx]].copy().reset_index(drop=True)
                    logger.debug(f"Injecting to {node_ids[idx]}")
                    fault_class = NAME_MAPPING[fault_name]
                    fault = fault_class(**self.config.fault_config[fault_name].model_dump())
                    node_data = fault.apply(node_data, "target", type_idx)

                    faulty.append(node_data)

            case "circular":
                type_list = list(self.config.type_mapping.items())
                for idx, node_id in enumerate(node_ids):
                    node_data = data[["node_id"] == node_id].copy().reset_index(drop=True)
                    fault_name, type_idx = type_list[idx % len(type_list)]
                    fault_class = NAME_MAPPING[fault_name]
                    fault = fault_class(**self.config.fault_config[fault_name].model_dump())
                    node_data = fault.apply(node_data, "target")

                    faulty.append(node_data)

        injected_data = pd.concat(faulty, axis=0)
        if not clean.empty:
            injected_data = pd.concat([injected_data, clean], axis=0)
        return injected_data
