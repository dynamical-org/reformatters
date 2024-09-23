import pandas as pd
import typer

from noaa.gefs.forecast import reformat, template

app = typer.Typer()


@app.command()
def update_template() -> None:
    template.update_template()


@app.command()
def reformat_local(init_time_end: str) -> None:
    reformat.local_reformat(init_time_end)


if __name__ == "__main__":
    app()
