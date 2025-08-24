import json
import typer
from typing import Annotated, Optional, List
from loguru import logger
from rich.text import Text
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from pathlib import Path

from hifinet.fault.injector import FaultInjector
from hifinet.config import DEFAULT_DRIFT, DEFAULT_ERRATIC, DEFAULT_HARDOVER, DEFAULT_STUCK, DEFAULT_SPIKE, NAME_CONFIG_MAPPING, InjectorConfig
from hifinet.loader import load_data

console = Console()

logger.remove()

min_level = logger.level("INFO").no

def _filter_by_level(record):
    return record["level"].no >= min_level

logger.configure(
    handlers=[
        {
            "sink": lambda s: console.print(Text.from_ansi(s)),
            "colorize": console.is_terminal,
            "level": 0,
            "filter": _filter_by_level
        }
    ]
)

app = typer.Typer(help="HiFiNet CLI", no_args_is_help=True)


@app.command(help="Inject faults to the dataset")
def inject(
    dataset: Annotated[str, typer.Argument(help="The name of a default dataset or path to custom dataset")] = "intel",
    name_string: Annotated[Optional[str], typer.Option("--output", help="Output file name")] = None,
    dir_string: Annotated[str, typer.Argument(help="Directory to output file")] = "data/inject",
    chance: Annotated[float, typer.Argument(help="Percent of fault to be injected")] = 0.2,
    exclude: Annotated[Optional[List[int]], typer.Option(help="Node to exclude from injection")] = None,
    seed: Annotated[Optional[int], typer.Option(help="Random seed to set")] = None,
    fault_json: Annotated[Optional[str], typer.Option("--fault-config", help="Json string for fault type params")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show debugging information")] = False
    ):
    global min_level
    if verbose:
        min_level = logger.level("DEBUG").no
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True, console=console) as progress:
        progress.add_task(description="Injecting...", total=None)

        data = load_data(dataset)

        init_config = {"hardover": DEFAULT_HARDOVER, "drift": DEFAULT_DRIFT, "erratic": DEFAULT_ERRATIC, "spike": DEFAULT_SPIKE, "stuck": DEFAULT_STUCK}
        if fault_json:
            fault_config = json.loads(fault_json)
            for fault_name, cli_overide in fault_config.items():
                config = {**init_config[fault_name].model_dump(), **cli_overide}
                init_config[fault_name] = NAME_CONFIG_MAPPING[fault_name](**config)

        injector_config = InjectorConfig(fault_config=init_config, exclude=exclude)
        injector = FaultInjector(injector_config)

        injected_data = injector.inject(data)
        output_dir = Path(dir_string)
        output_path = output_dir / (name_string or f"{dataset}.csv")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        injected_data.to_csv(output_path, index=False)

    logger.info(f"Data write to {output_path}")


@app.command()
def train():
    pass

if __name__ == "__main__":
    app()
