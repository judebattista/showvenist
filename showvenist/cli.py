"""Command-line interface for showvenist."""

import importlib.metadata

import typer

TMDB_NOTICE = "This product uses the TMDB API but is not endorsed or certified by TMDB."

app = typer.Typer()


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        try:
            ver = importlib.metadata.version("showvenist")
        except importlib.metadata.PackageNotFoundError:
            ver = "0.1.0"
        typer.echo(f"showvenist {ver}")
        typer.echo(TMDB_NOTICE)
        raise typer.Exit(0)


@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=version_callback,
        help="Print version and exit",
    )
) -> None:
    """Identifies season and episode of TV disc rips against a reference bundle."""
    pass


if __name__ == "__main__":
    app()

