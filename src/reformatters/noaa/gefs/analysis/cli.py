from typing import Annotated

import typer

from reformatters.noaa.gefs.analysis import reformat, template
from reformatters.noaa.gefs.analysis.template import DATASET_ID as DATASET_ID

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
    jobs_per_pod: int = 10,
    max_parallelism: int = 32,
    docker_image: str | None = None,
) -> None:
    reformat.reformat_kubernetes(
        init_time_end, jobs_per_pod, max_parallelism, docker_image=docker_image
    )


@app.command()
def reformat_chunks(
    init_time_end: str,
    worker_index: Annotated[int, typer.Argument(envvar="WORKER_INDEX")],
    workers_total: Annotated[int, typer.Argument(envvar="WORKERS_TOTAL")],
) -> None:
    reformat.reformat_chunks(
        init_time_end, worker_index=worker_index, workers_total=workers_total
    )


# Not implemented yet
# @app.command()
# def reformat_operational_update() -> None:
#     reformat.reformat_operational_update()


# @app.command()
# def validate_zarr() -> None:
#     reformat.validate_zarr()


if __name__ == "__main__":
    app()
