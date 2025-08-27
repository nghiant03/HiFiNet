import json
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text

from hifinet.config import CONFIG_CLASS_MAPPING, DEFAULT_CONFIG_MAPPING, InjectorConfig
from hifinet.fault import FaultInjector
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
            "filter": _filter_by_level,
        }
    ]
)

app = typer.Typer(help="HiFiNet CLI", no_args_is_help=True)


@app.command(help="Inject faults to the dataset")
def inject(
    dataset: Annotated[
        str,
        typer.Argument(help="The name of a default dataset or path to custom dataset"),
    ] = "intel",
    name_string: Annotated[
        str | None, typer.Option("--output", help="Output file name")
    ] = None,
    dir_string: Annotated[
        str, typer.Argument(help="Directory to output file")
    ] = "data/inject",
    chance: Annotated[
        float, typer.Argument(help="Percent of fault to be injected")
    ] = 0.2,
    exclude: Annotated[
        list[int] | None, typer.Option(help="Node to exclude from injection")
    ] = None,
    seed: Annotated[int | None, typer.Option(help="Random seed to set")] = None,
    fault_config: Annotated[
        str | None,
        typer.Option(
            help="Json string or path to Json file for fault type params",
        ),
    ] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show debugging information")
    ] = False,
):
    global min_level
    if verbose:
        min_level = logger.level("DEBUG").no
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        progress.add_task(description="Injecting...", total=None)

        data = load_data(dataset)

        init_config = {
            name: CONFIG_CLASS_MAPPING[name](
                **{**params, "seed": seed, "chance": chance}
            )
            for name, params in DEFAULT_CONFIG_MAPPING.items()
        }

        if fault_config:
            try:
                path = Path(fault_config)
                if path.is_file():
                    with path.open() as f:
                        config_override = json.load(f)
                else:
                    config_override = json.loads(fault_config)
            except (OSError, json.JSONDecodeError) as e:
                raise typer.BadParameter(f"Invalid fault config: {e}") from e

            for fault_name, override in config_override.items():
                config = {**init_config[fault_name].model_dump(), **override}
                init_config[fault_name] = CONFIG_CLASS_MAPPING[fault_name](**config)

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
