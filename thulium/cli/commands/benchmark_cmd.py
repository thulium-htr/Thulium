import typer
from pathlib import Path

app = typer.Typer()

@app.command()
def main(
    config: Path = typer.Option(..., help="Path to benchmark config"),
    dataset: Path = typer.Option(..., help="Path to dataset root"),
):
    """
    Run evaluation benchmark suite.
    """
    typer.echo(f"Benchmarking on {dataset}...")
    # from thulium.evaluation.benchmarking import run_benchmark
    # run_benchmark(dataset, config)

if __name__ == "__main__":
    app()
