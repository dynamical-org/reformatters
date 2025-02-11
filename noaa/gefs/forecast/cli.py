from typing import Annotated

import typer

from noaa.gefs.forecast import reformat, template

app = typer.Typer()


@app.command()
def update_template() -> None:
    template.update_template()


@app.command()
def reformat_local(init_time_end: str) -> None:
    reformat.reformat_local(init_time_end)


@app.command()
def reformat_kubernetes(
    init_time_end: str, jobs_per_pod: int = 10, max_parallelism: int = 32
) -> None:
    reformat.reformat_kubernetes(init_time_end, jobs_per_pod, max_parallelism)


@app.command()
def reformat_chunks(
    init_time_end: str,
    worker_index: Annotated[int, typer.Argument(envvar="WORKER_INDEX")],
    workers_total: Annotated[int, typer.Argument(envvar="WORKERS_TOTAL")],
) -> None:
    reformat.reformat_chunks(
        init_time_end, worker_index=worker_index, workers_total=workers_total
    )


@app.command()
def reformat_operational_update() -> None:
    reformat.reformat_operational_update()


@app.command()
def deploy_operational_updates() -> None:
    reformat.deploy_operational_updates()


@app.command()
def validate_zarr(
    zarr_path: str = "https://data.dynamical.org/noaa/gefs/forecast/latest.zarr",
) -> None:
    reformat.validate_zarr(zarr_path)


if __name__ == "__main__":
    app()
