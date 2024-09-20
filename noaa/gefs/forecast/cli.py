import typer

from noaa.gefs.forecast import template

app = typer.Typer()


@app.command()
def update_template() -> None:
    template.update_template()


@app.command()
def launch_processing() -> None:
    raise NotImplementedError()


if __name__ == "__main__":
    app()
