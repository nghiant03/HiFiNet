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
    chance: Annotated[float, typer.Argument(help="Percent of fault to be injected")] = 0.2,
    seed: Annotated[Optional[int], typer.Option(help="Random seed to set")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show debugging information")] = False
    ):
    global min_level
    if verbose:
        min_level = logger.level("DEBUG").no
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True, console=console) as progress:
        progress.add_task(description="Injecting...", total=None)


@app.command()
def train():
    pass

if __name__ == "__main__":
    app()
