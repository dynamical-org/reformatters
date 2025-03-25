from typing import Annotated

import typer

from reformatters.noaa.gefs.forecast_35_day import reformat, template
from reformatters.noaa.gefs.forecast_35_day.template import DATASET_ID as DATASET_ID

app = typer.Typer()


@app.command()
def update_template() -> None:
    template.update_template()


@app.command()
def reformat_local(
    init_time_end: str,
    filter_init_time_start: str | None = None,
    filter_init_time_end: str | None = None,
    filter_variable_names: list[str] | None = None,
) -> None:
    reformat.reformat_local(
        init_time_end,
        chunk_filters=reformat.ChunkFilters(
            time_dim=template.APPEND_DIMENSION,
            time_start=filter_init_time_start,
            time_end=filter_init_time_end,
            variable_names=filter_variable_names,
        ),
    )


@app.command()
def reformat_kubernetes(
    init_time_end: str,
    jobs_per_pod: int = 10,
    max_parallelism: int = 32,
    docker_image: str | None = None,
    filter_init_time_start: str | None = None,
    filter_init_time_end: str | None = None,
    filter_variable_names: list[str] | None = None,
) -> None:
    reformat.reformat_kubernetes(
        init_time_end,
        jobs_per_pod,
        max_parallelism,
        docker_image=docker_image,
        chunk_filters=reformat.ChunkFilters(
            time_dim=template.APPEND_DIMENSION,
            time_start=filter_init_time_start,
            time_end=filter_init_time_end,
            variable_names=filter_variable_names,
        ),
    )


@app.command()
def reformat_chunks(
    init_time_end: str,
    worker_index: Annotated[int, typer.Argument(envvar="WORKER_INDEX")],
    workers_total: Annotated[int, typer.Argument(envvar="WORKERS_TOTAL")],
    filter_init_time_start: str | None = None,
    filter_init_time_end: str | None = None,
    filter_variable_names: list[str] | None = None,
) -> None:
    reformat.reformat_chunks(
        init_time_end,
        worker_index=worker_index,
        workers_total=workers_total,
        chunk_filters=reformat.ChunkFilters(
            time_dim=template.APPEND_DIMENSION,
            time_start=filter_init_time_start,
            time_end=filter_init_time_end,
            variable_names=filter_variable_names,
        ),
    )


@app.command()
def reformat_operational_update() -> None:
    reformat.reformat_operational_update()


@app.command()
def validate_zarr() -> None:
    reformat.validate_zarr()


if __name__ == "__main__":
    app()
