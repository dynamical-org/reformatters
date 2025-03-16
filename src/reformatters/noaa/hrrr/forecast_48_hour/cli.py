from typing import Annotated

import typer

app = typer.Typer()


@app.command()
def update_template() -> None:
    raise NotImplementedError("update_template not implemented")


@app.command()
def reformat_local(init_time_end: str) -> None:
    raise NotImplementedError("reformat_local not implemented")


@app.command()
def reformat_kubernetes(
    init_time_end: str,
    jobs_per_pod: int = 10,
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
