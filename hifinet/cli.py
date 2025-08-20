import typer
from hifinet.loader import load_data
from hifinet.commands.inject import inject_zen

app = typer.Typer(help="HiFiNet CLI")

@app.command()
def inject(dataset: str, chance: float, seed: int):
    inject_zen.hydra_main()

if __name__ == "__main__":
    app()
