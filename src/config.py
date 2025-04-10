import logging
from pathlib import Path
from dataclasses import InitVar, dataclass, field
from typing import Dict, Union, List, Optional

logger = logging.getLogger(__name__)
DATASET_IMPLEMENTED = ["intel"]

@dataclass
class Config:
    dataset: InitVar[Union[str, List[str]]]
    dataset_args: Optional[Dict[str, Dict]] = None
    inject_dir: Union[str, Path] = "inject"
    output_dir: Union[str, Path] = "result"
    enable_log: bool = False
    log_dir: Optional[Union[str, Path]] = None
    fault_type: Union[str, List[str]] = "all"
    fault_params: Optional[Dict[str, Dict]] = None

    _dataset_list: List[str] = field(init=False)

    def __post_init__(self, dataset):
        self.cwd = Path.cwd()

        if isinstance(dataset, str):
            self._dataset_list = [dataset]

        self._dataset_list = [s.lower() for s in dataset]
        assert set(self._dataset_list).issubset(set(DATASET_IMPLEMENTED)), f"{set(self._dataset_list) - set(DATASET_IMPLEMENTED)} is not implemented."

        if isinstance(self.output_dir, str):
            self.output_dir = self.cwd / self.output_dir 
        if not self.output_dir.is_dir():
            self.output_dir.mkdir(parents=True)

        if self.enable_log:
            if self.log_dir is None:
                self.log_dir = Path.cwd() / "log"
                logger.warning("Logging in enabled with no log_dir provided. Default to {self.log_dir}")
            if isinstance(self.log_dir, str):
                self.log_dir = self.cwd / self.log_dir
            if not self.log_dir.is_dir():
                self.log_dir.mkdir(parents=True)

    @property
    def dataset_list(self) -> List[str]:
        return self._dataset_list
