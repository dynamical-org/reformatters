from typing import Annotated

import typer

from reformatters.common.reformat_utils import ChunkFilters
from reformatters.noaa.gefs.analysis import reformat, template
from reformatters.noaa.gefs.analysis.template import DATASET_ID as DATASET_ID

app = typer.Typer()


@app.command()
def update_template() -> None:
    template.update_template()


@app.command()
def backfill_local(
    time_end: str,
    filter_time_start: str | None = None,
    filter_time_end: str | None = None,
    filter_variable_names: list[str] | None = None,
) -> None:
    reformat.reformat_local(
        time_end,
        chunk_filters=ChunkFilters(
            time_dim=template.APPEND_DIMENSION,
            time_start=filter_time_start,
            time_end=filter_time_end,
            variable_names=filter_variable_names,
        ),
    )


@app.command()
def backfill_kubernetes(
    time_end: str,
    jobs_per_pod: int = 1,
    max_parallelism: int = 32,
    docker_image: str | None = None,
    filter_time_start: str | None = None,
    filter_time_end: str | None = None,
    filter_variable_names: list[str] | None = None,
) -> None:
    reformat.reformat_kubernetes(
        time_end,
        jobs_per_pod,
        max_parallelism,
        docker_image=docker_image,
        chunk_filters=ChunkFilters(
            time_dim=template.APPEND_DIMENSION,
            time_start=filter_time_start,
            time_end=filter_time_end,
            variable_names=filter_variable_names,
        ),
    )


@app.command()
def reformat_chunks(
    time_end: str,
    worker_index: Annotated[int, typer.Argument(envvar="WORKER_INDEX")],
    workers_total: Annotated[int, typer.Argument(envvar="WORKERS_TOTAL")],
    filter_time_start: str | None = None,
    filter_time_end: str | None = None,
    filter_variable_names: list[str] | None = None,
) -> None:
    reformat.reformat_chunks(
        time_end,
        worker_index=worker_index,
        workers_total=workers_total,
        chunk_filters=ChunkFilters(
            time_dim=template.APPEND_DIMENSION,
            time_start=filter_time_start,
            time_end=filter_time_end,
            variable_names=filter_variable_names,
        ),
    )


@app.command()
def update(
    job_name: Annotated[str, typer.Argument(envvar="JOB_NAME")],
) -> None:
    reformat.reformat_operational_update(job_name)


@app.command()
def validate() -> None:
    reformat.validate_dataset()


if __name__ == "__main__":
    app()
