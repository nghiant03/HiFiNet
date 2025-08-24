import typer
from typing import Annotated, Optional
from loguru import logger
from rich.text import Text
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn



console = Console()

logger.remove()
logger.configure(
    handlers=[
        {
            "sink": lambda s: console.print(Text.from_ansi(s)),
            "colorize": console.is_terminal,
        }
    ]
)

app = typer.Typer(help="HiFiNet CLI", no_args_is_help=True)


@app.command(help="Inject faults to the dataset")
def inject(
    dataset: Annotated[str, typer.Argument(help="The name of a default dataset or path to custom dataset")] = "intel",
    chance: Annotated[float, typer.Argument(help="Percent of fault to be injected")] = 0.2,
    seed: Annotated[Optional[int], typer.Option(help="Random seed to set")] = None,
    ):
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True, console=console) as progress:
        progress.add_task(description="Injecting...", total=None)


@app.command()
def train():
    pass

if __name__ == "__main__":

    app()
