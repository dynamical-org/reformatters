from typing import Annotated

import typer

from reformatters.noaa.gfs.forecast import reformat, template
from reformatters.noaa.gfs.forecast.template import DATASET_ID as DATASET_ID

app = typer.Typer()


@app.command()
def update_template() -> None:
    template.update_template()


@app.command()
def reformat_local(init_time_end: str) -> None:
    reformat.reformat_local(init_time_end)


@app.command()
def reformat_kubernetes(
    init_time_end: str,
    jobs_per_pod: int = 1,
    max_parallelism: int = 32,
    docker_image: str | None = None,
) -> None:
    raise NotImplementedError("reformat_kubernetes not implemented")


@app.command()
def reformat_chunks(
    init_time_end: str,
    worker_index: Annotated[int, typer.Argument(envvar="WORKER_INDEX")],
    workers_total: Annotated[int, typer.Argument(envvar="WORKERS_TOTAL")],
) -> None:
    raise NotImplementedError("reformat_chunks not implemented")


@app.command()
def reformat_operational_update() -> None:
    raise NotImplementedError("reformat_operational_update not implemented")


@app.command()
def validate_zarr() -> None:
    raise NotImplementedError("validate_zarr not implemented")


if __name__ == "__main__":
    app()
